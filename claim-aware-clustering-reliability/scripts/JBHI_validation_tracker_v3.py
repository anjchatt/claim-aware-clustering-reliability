#!/usr/bin/env python3
"""JBHI evidence tracker and optional targeted validation.

READ-ONLY with respect to source data, reference CSVs, and existing models.
New derived results are written under --out only.

Default: verify saved characterization, summarize 20-run primary masking,
        verify reconstructed label transition, and cross-tabulate primary R1
        against circulatory diagnosis family 2 when source profiles exist.
Optional --run-reconstruction-masking: reproduce development-fitted selected
        w=.6 and w=1.0 models, then test fixed-model diagnosis-position masking.

No primary/reconstruction patient-level equivalence is inferred.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import platform
import sys
import traceback

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from scipy.optimize import linear_sum_assignment

SCRIPT_VERSION = "3.0.0"
WEIGHTS = (0.6, 1.0)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def first_existing(*paths: Path) -> Path | None:
    return next((p for p in paths if p.is_file()), None)


def attr_value(column: pd.Series, label: str) -> str:
    return f"{label}: {column!r}"


def aligned_differences(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    ua, ai = np.unique(a, return_inverse=True)
    ub, bi = np.unique(b, return_inverse=True)
    if len(ua) != len(ub):
        raise ValueError("Cannot use a one-to-one label alignment with different K")
    contingency = np.zeros((len(ua), len(ub)), dtype=np.int64)
    np.add.at(contingency, (ai, bi), 1)
    ridx, cidx = linear_sum_assignment(-contingency)
    mapping = {ub[c]: ua[r] for r, c in zip(ridx, cidx)}
    mapped = np.array([mapping[v] for v in b])
    return int(np.sum(a != mapped)), contingency, mapping


def read_reference_labels(path: Path):
    dat = pd.read_csv(path, dtype={"patient_nbr": str})
    needed = {"cohort", "patient_nbr", "weight", "cluster"}
    if not needed.issubset(dat):
        raise ValueError(f"{path.name} missing {sorted(needed - set(dat))}")
    if dat.duplicated(["cohort", "patient_nbr", "weight"]).any():
        raise ValueError("Saved reconstruction assignments have duplicate patient/weight rows")
    return dat


def verify_reconstruction_labels(dat):
    d = dat[dat.cohort == "development"].copy()
    pivot = d.pivot(index="patient_nbr", columns="weight", values="cluster")
    if not {0.6, 1.0}.issubset(pivot.columns):
        raise ValueError("Selected development weights are absent from saved labels")
    if pivot[[.6, 1.0]].isna().any().any():
        raise ValueError("Patient mismatch between two reconstructed specifications")
    a = pivot[.6].to_numpy(dtype=int)
    b = pivot[1.0].to_numpy(dtype=int)
    changes, mat, _ = aligned_differences(a, b)
    return {
        "n": int(len(pivot)), "ari": float(adjusted_rand_score(a, b)),
        "changed_count": changes, "changed_percent": 100 * changes / len(pivot),
        "transition_matrix_unaligned": mat.tolist(),
    }


class Audit:
    def __init__(self, out: Path):
        self.out = out
        self.events = []
        self.source_files = {}
        self.output = {}
    def source(self, path: Path | None):
        if path is not None and path.is_file():
            self.source_files[str(path.resolve())] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    def event(self, ident, status, detail, origin=None, section="V"):
        if status not in ("PASS", "FAIL", "NOT_EVALUATED", "NOT_APPLICABLE", "OBSERVED"):
            raise ValueError(status)
        self.events.append({"evidence_id": ident, "status": status,
                            "details": str(detail), "source": str(origin) if origin else "",
                            "manuscript_section": section})
        print(f"[{status}] {ident}: {detail}")
    def save(self, args):
        self.out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.events).to_csv(self.out / "validation_evidence_register.csv", index=False)
        payload = {
            "script_version": SCRIPT_VERSION,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "environment": {"python": sys.version.split()[0], "platform": platform.platform()},
            "config": {"project": str(args.project), "primary_project": str(args.primary_project),
                       "mask_repetitions": args.mask_repetitions, "mask_rate": args.mask_rate,
                       "optional_new_experiment": bool(args.run_reconstruction_masking)},
            "source_files": self.source_files,
            "results": self.output,
            "checks": self.events,
        }
        (self.out / "validation_manifest.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        lines = ["JBHI VALIDATION SNAPSHOT", "Generated: " + payload["generated_at_utc"], "",
                 "Do not treat skipped tests as passed or combine primary/reconstruction records.", ""]
        lines.extend(f"[{e['status']}] {e['evidence_id']}: {e['details']}" for e in self.events)
        lines.append("\nSource hashes and run protocol are stored in validation_manifest.json.")
        (self.out / "validation_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def check_existing(a, audit):
    path = a.project / "w060_characterization" / "characterization_manifest.json"
    audit.source(path)
    if not path.exists():
        audit.event("RECON_CHARACTERIZATION_MANIFEST", "NOT_EVALUATED", "Manifest missing", path, "V-D")
        return {}
    m = json.loads(path.read_text(encoding="utf-8"))
    checks = m.get("saved_results_comparison", {})
    names = ["cross_weight_matches_saved", "fit_inertia_w0.6_matches_saved", "fit_inertia_w1_matches_saved"]
    for key in names:
        value = checks.get(key)
        state = "PASS" if value is True else ("FAIL" if value is False else "NOT_EVALUATED")
        audit.event("SAVED_RECON_" + key.upper(), state, f"saved_results_comparison.{key}={value!r}", path, "V-D")
    audit.output["reconstructed_characterization"] = {
        "declared_saved_result_checks": checks,
        "development_crossweight_ari": m.get("development_crossweight_ari"),
        "development_changed_pct": m.get("development_changed_pct"),
        "heldout": m.get("heldout"),
        "selected_seed_by_weight": m.get("selected_seed_by_weight", {}),
    }
    return m


def check_primary_masking(a, audit):
    p = a.primary_project / "results" / "step25f_harmonized_missingness_runs.csv"
    audit.source(p)
    if not p.exists():
        audit.event("PRIMARY_20PCT_MASKING_DISTRIBUTION", "NOT_EVALUATED", "Run-level CSV missing", p, "V-B")
        return
    df = pd.read_csv(p)
    req = {"solution", "missing_rate", "ARI"}
    if not req.issubset(df.columns):
        raise ValueError(f"Masking CSV missing columns {sorted(req - set(df.columns))}")
    selected = df[(df.solution.astype(str) == "R1_K2") & np.isclose(df.missing_rate, 0.2)].copy()
    s = pd.to_numeric(selected.ARI, errors="raise")
    if len(s) != 20:
        audit.event("PRIMARY_20PCT_MASKING_DISTRIBUTION", "FAIL", f"Expected 20, found {len(s)}", p, "V-B")
        return
    q = {"n": 20, "mean": float(s.mean()), "sd_sample": float(s.std(ddof=1)),
         "median": float(s.median()), "min": float(s.min()), "max": float(s.max()),
         "q25": float(s.quantile(0.25)), "q75": float(s.quantile(0.75)),
         "count_gt_0_9": int((s > 0.9).sum())}
    audit.output["primary_r1_k2_refit_masking_20pct"] = q
    audit.event("PRIMARY_20PCT_MASKING_DISTRIBUTION", "PASS",
                f"n=20 median={q['median']:.6f}; range={q['min']:.6f}–{q['max']:.6f}; >0.9={q['count_gt_0_9']}", p, "V-B")
    selected.to_csv(audit.out / "primary_R1K2_refit_masking_20pct_runs.csv", index=False)
    pd.DataFrame([q]).to_csv(audit.out / "primary_R1K2_refit_masking_distribution_summary.csv", index=False)


def check_primary_r1_indicator(a, audit):
    profile_path = a.primary_project / "results" / "step22_patient_cluster_profiles.parquet"
    dev_path = a.primary_project / "data" / "processed" / "development_R1_10k.parquet"
    audit.source(profile_path)
    audit.source(dev_path)
    if not profile_path.exists() or not dev_path.exists():
        audit.event("PRIMARY_R1_VS_SECONDARY_CIRCULATORY", "NOT_EVALUATED",
                    "Requires saved primary profile and development R1 parquet", "primary results", "V-C")
        return
    prof = pd.read_parquet(profile_path)
    dev = pd.read_parquet(dev_path)
    if not {"patient_nbr", "cluster_R1"}.issubset(prof) or not {"patient_nbr", "diag2_family"}.issubset(dev):
        raise ValueError("Primary patient-level files lack patient_nbr, cluster_R1, or diag2_family")
    p = prof[["patient_nbr", "cluster_R1"]].copy()
    d = dev[["patient_nbr", "diag2_family"]].copy()
    p.patient_nbr = p.patient_nbr.astype(str)
    d.patient_nbr = d.patient_nbr.astype(str)
    if p.patient_nbr.duplicated().any() or d.patient_nbr.duplicated().any():
        raise ValueError("Duplicate patient IDs in primary cohort")
    merged = p.merge(d, on="patient_nbr", how="inner", validate="one_to_one")
    if len(merged) != len(d) or len(merged) != len(p):
        raise ValueError("Saved primary labels and features are not patient aligned")
    cats = merged.diag2_family.fillna("Missing").astype(str)
    distinct = sorted(cats.unique())
    match = [v for v in distinct if "circulatory" in v.lower()]
    if len(match) != 1:
        raise ValueError(f"Expected one Circulatory diagnosis family, got {match}; vocabulary={distinct}")
    ind = (cats == match[0]).astype(int)
    labels = merged.cluster_R1.astype(int)
    if labels.nunique() != 2:
        raise ValueError("Primary R1 partition must contain K=2")
    changes, _, mapping = aligned_differences(ind.to_numpy(), labels.to_numpy())
    cont = pd.crosstab(pd.Series(ind, name="secondary_circulatory_indicator"),
                       pd.Series(labels.to_numpy(), name="primary_R1_K2_label"))
    cont.to_csv(audit.out / "primary_R1_vs_secondary_circulatory_contingency.csv")
    res = {"n": int(len(merged)), "indicator_n": int(ind.sum()),
           "ari": float(adjusted_rand_score(ind, labels)),
           "aligned_change_count": changes, "aligned_difference_pct": 100*changes/len(merged),
           "category_literal": match[0]}
    audit.output["primary_r1_vs_secondary_circulatory"] = res
    audit.event("PRIMARY_R1_VS_SECONDARY_CIRCULATORY", "PASS",
                f"n={res['n']}; ARI={res['ari']:.4f}; after-alignment difference={res['aligned_difference_pct']:.2f}% (not a cross-cohort ARI)",
                profile_path, "V-C")


def reimport(path):
    spec = importlib.util.spec_from_file_location("jbhi_existing_06c", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def new_reconstruction_position_masking(a, audit, characterization):
    """New experiment. Baseline fitted recon 06C selected seed; fixed-model masking.
    No historical primary inference and no preprocessing refit.
    """
    raw = a.raw_csv
    validator = a.project / "jbhi_06C_validation_v2.py"
    labels_path = a.project / "w060_characterization" / "patient_assignments_w060_w100.csv"
    fit_ref = a.project / "fit_level_seed_runs_ninit20_REFERENCE.csv"
    for p in (raw, validator, labels_path, fit_ref): audit.source(p)
    missing = [str(p) for p in (raw, validator, labels_path, fit_ref) if not p or not p.is_file()]
    if missing:
        audit.event("NEW_RECON_POSITION_MASKING", "NOT_EVALUATED", f"Missing {missing}", "Reconstruction", "V-D")
        return
    seeds = characterization.get("selected_seed_by_weight", {})
    if not all(str(w) in seeds for w in WEIGHTS):
        audit.event("NEW_RECON_POSITION_MASKING", "FAIL", "Selected model seeds missing from characterization manifest", "characterization_manifest.json", "V-D")
        return
    import sklearn
    from sklearn.cluster import KMeans
    from threadpoolctl import threadpool_limits
    mod = reimport(validator)
    opts = SimpleNamespace(raw_csv=raw, split_manifest=a.split_manifest, seed=a.seed, age_divisor=a.age_divisor)
    dev, held, dm, hm, source, notes = mod.load_raw(opts)
    if not len(dev) == 10000 or len(held) != 61518:
        raise ValueError("Reconstructed cohort size differs from saved 10,000/61,518")
    labels_df = read_reference_labels(labels_path)
    fits = pd.read_csv(fit_ref)
    util_ix = np.array([i for i, c in enumerate(dev.columns) if c.startswith("util__")])
    if len(util_ix) != 3:
        raise ValueError("Expected exactly 3 utilization coordinates")
    masking_rows = []
    encoding_info = []
    with threadpool_limits(limits=1):
        for weight in WEIGHTS:
            x, _ = mod.transforms(dev, held, "robust", weight, util_ix, "direct")
            seed = int(seeds[str(weight)])
            model = KMeans(n_clusters=2, n_init=20, random_state=seed, algorithm="lloyd").fit(x)
            baseline = model.labels_.astype(int)
            actual = pd.DataFrame({"patient_nbr": dm.patient_nbr.astype(str).to_numpy(), "cluster": baseline})
            reference = labels_df[(labels_df.cohort == "development") & np.isclose(labels_df.weight, weight)]
            if len(reference) != 10000 or reference.patient_nbr.duplicated().any():
                raise ValueError(f"Saved labels for weight {weight} incomplete")
            paired = actual.merge(reference[["patient_nbr", "cluster"]], on="patient_nbr", suffixes=("_new", "_saved"), validate="one_to_one")
            labels_ari = adjusted_rand_score(paired.cluster_new, paired.cluster_saved)
            if labels_ari < 1 - 1e-10:
                raise ValueError(f"Selected fit at w={weight} differs from saved patient-level labels; ARI={labels_ari}")
            subset = fits[(fits.scaler == "robust") & np.isclose(fits.weight, weight) & (fits.n_init == 20)]
            if subset.empty or abs(model.inertia_ - subset.inertia.min()) > 1e-5:
                raise ValueError(f"Selected model inertia mismatch at weight={weight}; verify preprocessing and seed")
            audit.event(f"RECON_SELECTED_FIT_{weight:g}", "PASS", f"saved-label ARI={labels_ari:.3f}, selected seed={seed}, inertia={model.inertia_:.6f}", labels_path, "V-D")
            for position in (1, 2, 3):
                prefix = f"clinical__diag_family_{position}_"
                block = [i for i, c in enumerate(dev.columns) if c.startswith(prefix)]
                missing_ix = [i for i, c in enumerate(dev.columns) if c == f"{prefix}Missing"]
                if len(block) < 2 or len(missing_ix) != 1:
                    raise ValueError(f"Cannot locate explicit Missing one-hot coordinate for diagnosis position {position}")
                m_idx = missing_ix[0]
                observed = np.flatnonzero(x[:, m_idx] < 0.5)
                selected_count = round(len(observed) * a.mask_rate)
                encoding_info.append({"weight":weight, "position":position, "eligible_observed":len(observed),
                                      "per_repetition_masked":selected_count})
                for rep in range(a.mask_repetitions):
                    rng = np.random.default_rng(a.seed + 900000 + 1000*position + rep)
                    chosen = rng.choice(observed, selected_count, replace=False)
                    masked = x.copy()
                    masked[np.ix_(chosen,block)] = 0.0
                    masked[chosen,m_idx] = 1.0
                    pred = model.predict(masked)
                    # Cluster IDs of a fixed model are stable: compare directly; ARI is label independent.
                    masking_rows.append({"cohort":"development", "source_stream":"separate_reconstruction",
                                         "weight":weight, "diagnosis_position":position, "rate":a.mask_rate,
                                         "rep":rep, "masked_count":selected_count,
                                         "ARI":adjusted_rand_score(baseline,pred),
                                         "fraction_changed":float(np.mean(baseline != pred)),
                                         "protocol":"fixed-model one-hot Missing replacement; no refit"})
    res = pd.DataFrame(masking_rows)
    res.to_csv(audit.out / "NEW_reconstruction_position_masking_runs.csv", index=False)
    summ = (res.groupby(["weight", "diagnosis_position", "rate"], as_index=False)
            .agg(repetitions=("ARI", "count"), mean_ari=("ARI", "mean"), sd_ari=("ARI", "std"),
                 median_ari=("ARI", "median"), min_ari=("ARI", "min"), max_ari=("ARI", "max"),
                 mean_changed_fraction=("fraction_changed", "mean")))
    summ.to_csv(audit.out / "NEW_reconstruction_position_masking_summary.csv", index=False)
    pd.DataFrame(encoding_info).to_csv(audit.out / "NEW_reconstruction_masking_eligible_counts.csv", index=False)
    audit.output["new_reconstruction_position_masking"] = json.loads(summ.to_json(orient="records"))
    audit.event("NEW_RECON_POSITION_MASKING", "PASS", f"{len(res)} runs, positions 1/2/3, weights 0.6/1.0; new independent measurement", raw, "V-D")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", type=Path, default=Path.cwd() / "local-reconstruction")
    p.add_argument("--primary-project", type=Path, default=Path.cwd() / "local-primary")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--raw-csv", type=Path, default=Path.cwd() / "data" / "diabetic_data.csv")
    p.add_argument("--split-manifest", type=Path, default=None)
    p.add_argument("--run-reconstruction-masking", action="store_true", help="Run NEW fixed-model, diagnosis-position masking for reconstructed w.6 and w1")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--age-divisor", type=float, default=100.0)
    p.add_argument("--mask-rate", type=float, default=0.2)
    p.add_argument("--mask-repetitions", type=int, default=20)
    args = p.parse_args()
    if args.out is None:
        args.out = args.project / "validation_tracking_v3"
    if not 0 < args.mask_rate < 1:
        p.error("--mask-rate must be between 0 and 1")
    if args.mask_repetitions <= 0:
        p.error("--mask-repetitions must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    audit = Audit(args.out)
    audit.source(Path(__file__))
    try:
        manifest = check_existing(args, audit)
        check_primary_masking(args, audit)
        check_primary_r1_indicator(args, audit)
        reference_labels = args.project / "w060_characterization" / "patient_assignments_w060_w100.csv"
        audit.source(reference_labels)
        if reference_labels.is_file():
            stats = verify_reconstruction_labels(read_reference_labels(reference_labels))
            audit.output["reconstructed_crossweight_patient_transition"] = stats
            match = (manifest and abs(float(manifest["development_crossweight_ari"])-stats["ari"]) < 1e-6
                     and abs(float(manifest["development_changed_pct"])-stats["changed_percent"]) < 1e-4)
            audit.event("RECON_DEVELOPMENT_TRANSITION", "PASS" if match else "FAIL",
                        f"n={stats['n']} ARI={stats['ari']:.6f}, changed={stats['changed_count']}/{stats['n']} ({stats['changed_percent']:.3f}%)", reference_labels, "V-D")
        else:
            audit.event("RECON_DEVELOPMENT_TRANSITION", "NOT_EVALUATED", "Saved patient assignments unavailable", reference_labels, "V-D")
        if args.run_reconstruction_masking:
            new_reconstruction_position_masking(args, audit, manifest)
        else:
            audit.event("NEW_RECON_POSITION_MASKING", "NOT_EVALUATED", "Optional experiment was not requested; use --run-reconstruction-masking", "new experiment", "V-D")
    except Exception as exc:
        audit.event("RUN_FATAL_ERROR", "FAIL", f"{type(exc).__name__}: {exc}", "execution")
        traceback.print_exc()
        audit.save(args)
        return 2
    audit.save(args)
    nfail = sum(e["status"] == "FAIL" for e in audit.events)
    print("\nReports:", args.out)
    print("FAILED checks:", nfail, "(NOT_EVALUATED is not PASS)")
    return int(bool(nfail))

if __name__ == "__main__":
    raise SystemExit(main())
