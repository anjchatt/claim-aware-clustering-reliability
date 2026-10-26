#!/usr/bin/env python3
"""R1 diagnosis-encoding sensitivity on the saved development cohort.

Fits position-specific and position-independent representations on the same
10,000 patients. The new masking schedule is reported separately from the
historical R1 masking experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from scipy.sparse import csr_matrix, hstack
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import OneHotEncoder


DIAGNOSIS_COLUMNS = ["diag1_family", "diag2_family", "diag3_family"]
FEATURE_COLUMNS = ["age", *DIAGNOSIS_COLUMNS]


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_input(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame[FEATURE_COLUMNS].copy()
    for column in FEATURE_COLUMNS:
        normalized[column] = (
            normalized[column].astype("string").fillna("missing")
            .str.strip().replace({"": "missing", "?": "missing"})
        )
    return normalized


def fit_encoders(frame: pd.DataFrame):
    age_encoder = OneHotEncoder(handle_unknown="ignore", dtype=np.float64)
    age_encoder.fit(frame[["age"]])

    position_encoder = OneHotEncoder(handle_unknown="ignore", dtype=np.float64)
    position_encoder.fit(frame[DIAGNOSIS_COLUMNS])
    families = sorted(set(frame[DIAGNOSIS_COLUMNS].to_numpy().ravel()))
    return age_encoder, position_encoder, families


def encode_by_position(frame, age_encoder, position_encoder):
    return hstack(
        [age_encoder.transform(frame[["age"]]),
         position_encoder.transform(frame[DIAGNOSIS_COLUMNS])],
        format="csr",
    )


def encode_as_multihot(frame, age_encoder, families):
    family_to_column = {family: index for index, family in enumerate(families)}
    diagnosis = frame[DIAGNOSIS_COLUMNS].to_numpy()
    rows, columns = [], []
    for position in range(diagnosis.shape[1]):
        for row, family in enumerate(diagnosis[:, position]):
            column = family_to_column.get(family)
            if column is not None:
                rows.append(row)
                columns.append(column)

    # Duplicates across positions become one binary family indicator.
    family_matrix = csr_matrix(
        (np.ones(len(rows), dtype=np.float64), (rows, columns)),
        shape=(len(frame), len(families)),
        dtype=np.float64,
    )
    family_matrix.data[:] = 1.0
    return hstack([age_encoder.transform(frame[["age"]]), family_matrix], format="csr")


def fit_partition(features, seed: int, n_init: int):
    model = KMeans(n_clusters=2, random_state=int(seed), n_init=n_init, algorithm="lloyd")
    return model.fit_predict(features)


def mean_pairwise_ari(label_sets):
    scores = [adjusted_rand_score(a, b) for a, b in combinations(label_sets, 2)]
    return float(np.mean(scores)), float(np.std(scores)), len(scores)


def mask_diagnosis_fields(frame, seed: int, proportion: float):
    masked = frame.copy()
    values = frame[DIAGNOSIS_COLUMNS].to_numpy()
    eligible = values != "missing"
    selected = (np.random.default_rng(int(seed)).random(eligible.shape) < proportion) & eligible

    changed = masked[DIAGNOSIS_COLUMNS].to_numpy(copy=True)
    changed[selected] = "missing"
    for position, column in enumerate(DIAGNOSIS_COLUMNS):
        masked[column] = changed[:, position]
    return masked, int(selected.sum()), int(eligible.sum())


def evaluate_representation(features, transform, frame, age_encoder, encoder_spec,
                            mask_reps: int, n_init: int):
    reference = fit_partition(features, 42, n_init)
    seeded_labels = [fit_partition(features, seed, n_init) for seed in range(30)]
    seed_mean, seed_std, comparisons = mean_pairwise_ari(seeded_labels)

    masked_ari = []
    masked_counts = []
    eligible_count = 0
    for repetition in range(mask_reps):
        masked_frame, count, eligible_count = mask_diagnosis_fields(
            frame, 2000 + repetition, 0.20
        )
        refitted = fit_partition(
            transform(masked_frame, age_encoder, encoder_spec), 42 + repetition, n_init
        )
        masked_ari.append(float(adjusted_rand_score(reference, refitted)))
        masked_counts.append(count)

    metrics = {
        "seed_ari_pairwise_mean": seed_mean,
        "seed_ari_pairwise_sd": seed_std,
        "seed_comparisons": comparisons,
        "refit_masking_20pct_ari_mean": float(np.mean(masked_ari)),
        "refit_masking_20pct_ari_sd": float(np.std(masked_ari, ddof=1)),
        "refit_masking_20pct_ari_runs": masked_ari,
        "masked_diagnosis_entries_per_run": masked_counts,
        "eligible_diagnosis_entries": eligible_count,
        "random_seed_scheme": (
            "30 model seeds 0..29; baseline 42; masking seeds 2000..; "
            "masked fits use model seeds 42.."
        ),
    }
    return reference, metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mask-reps", type=int, default=20)
    parser.add_argument("--n-init", type=int, default=20)
    parser.add_argument("--allow-unmatched-control", action="store_true")
    args = parser.parse_args()

    source_path = args.project / "data" / "processed" / "development_R1_10k.parquet"
    if not source_path.is_file():
        parser.error(f"Missing saved development data: {source_path}")

    original = pd.read_parquet(source_path)
    missing = sorted(set(FEATURE_COLUMNS) - set(original.columns))
    if missing or len(original) != 10000:
        parser.error(f"Unexpected R1 cohort: rows={len(original)}, missing columns={missing}")

    frame = normalize_input(original)
    age_encoder, position_encoder, families = fit_encoders(frame)
    positional_features = encode_by_position(frame, age_encoder, position_encoder)
    multihot_features = encode_as_multihot(frame, age_encoder, families)

    positional_labels, positional = evaluate_representation(
        positional_features, encode_by_position, frame, age_encoder,
        position_encoder, args.mask_reps, args.n_init
    )
    multihot_labels, multihot = evaluate_representation(
        multihot_features, encode_as_multihot, frame, age_encoder,
        families, args.mask_reps, args.n_init
    )

    positional_indicator = frame["diag2_family"].eq("circulatory").to_numpy(dtype=np.int8)
    any_position_indicator = frame[DIAGNOSIS_COLUMNS].eq("circulatory").any(axis=1).to_numpy(dtype=np.int8)
    positional_ari = float(adjusted_rand_score(positional_labels, positional_indicator))
    multihot_ari = float(adjusted_rand_score(multihot_labels, any_position_indicator))
    cross_encoding_ari = float(adjusted_rand_score(positional_labels, multihot_labels))

    print("POSITIONAL R1 CONTROL (independent fit)")
    print(f"Position-2 Circulatory indicator ARI: {positional_ari:.6f}")
    print(f"Seed ARI: {positional['seed_ari_pairwise_mean']:.6f}")
    print(f"Refitted 20% masking ARI: {positional['refit_masking_20pct_ari_mean']:.6f}")
    print("\nPOSITION-INDEPENDENT R1")
    print(f"Seed ARI: {multihot['seed_ari_pairwise_mean']:.6f}")
    print(f"Any-position Circulatory indicator ARI: {multihot_ari:.6f}")
    print(f"Refitted 20% masking ARI: {multihot['refit_masking_20pct_ari_mean']:.6f}")
    print(f"Cross-encoding ARI: {cross_encoding_ari:.6f}")

    if not np.isclose(positional_ari, 1.0, atol=1e-12) and not args.allow_unmatched_control:
        print("Positional indicator check failed. No results written.")
        return 3

    output = args.output or args.project / "results" / "multihot_r1_sensitivity"
    output.mkdir(parents=True, exist_ok=True)
    record = {
        "experimental_status": "development-only encoding sensitivity; not a historical masking replication",
        "source_file": source_path.name,
        "source_sha256": checksum(source_path),
        "n_dev": len(original),
        "feature_columns": FEATURE_COLUMNS,
        "categorical_families": families,
        "k": 2,
        "n_init": args.n_init,
        "algorithm": "lloyd",
        "original_positional_control": {"diag2_indicator_ari": positional_ari, **positional},
        "new_multihot": {
            "any_position_circulatory_ari": multihot_ari,
            "vs_new_positional_ari": cross_encoding_ari,
            **multihot,
        },
        "masking_design": (
            "Each observed diagnosis position masked independently with p=0.20; "
            "encoders retained, K-means refitted"
        ),
        "protocol_note": "Masking seeds differ from the historical R1 analysis.",
        "scikit_learn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
    }
    destination = output / "multihot_r1_sensitivity_summary.json"
    destination.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"\nSaved: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
