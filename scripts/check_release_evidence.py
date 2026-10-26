#!/usr/bin/env python3
"""Check the supplementary aggregate files before a software release.

The checks compare stored results with values reported in the manuscript.
They do not rerun the experiments or establish that Zenodo has archived them.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


REQUIRED_FILES = {
    "r1_multihot": Path("evidence/sensitivity/multihot_r1_sensitivity_summary.json"),
    "r3_domain": Path("evidence/primary/r3_heldout_domain_equivalence_summary.csv"),
    "figure3": Path("figures/figure3_sources.json"),
    "figure4": Path("figures/figure4_sources.json"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_close(value: object, expected: float, label: str, tolerance: float = 2e-5) -> None:
    if not math.isclose(float(value), expected, rel_tol=0, abs_tol=tolerance):
        raise ValueError(f"{label}: {value!r}; expected {expected}")


def check_r1_multihot(path: Path) -> None:
    record = json.loads(path.read_text(encoding="utf-8"))
    if (record["n_dev"], record["k"], record["n_init"]) != (10000, 2, 20):
        raise ValueError("Unexpected R1 cohort size or K-means settings")

    positional = record["original_positional_control"]
    multihot = record["new_multihot"]
    values = {
        "positional indicator ARI": (positional["diag2_indicator_ari"], 1.0),
        "positional seed ARI": (positional["seed_ari_pairwise_mean"], 0.976899),
        "positional masking ARI": (positional["refit_masking_20pct_ari_mean"], 0.164985),
        "multi-hot seed ARI": (multihot["seed_ari_pairwise_mean"], 1.0),
        "any-position Circulatory ARI": (multihot["any_position_circulatory_ari"], 1.0),
        "multi-hot masking ARI": (multihot["refit_masking_20pct_ari_mean"], 0.739794),
        "cross-encoding ARI": (multihot["vs_new_positional_ari"], 0.213334),
    }
    for label, (value, expected) in values.items():
        require_close(value, expected, label)

    for name, result in (("positional", positional), ("multi-hot", multihot)):
        if len(result["refit_masking_20pct_ari_runs"]) != 20:
            raise ValueError(f"{name}: expected 20 masking repetitions")
    print("R1 sensitivity: reported metrics match the saved aggregate JSON")


def check_r3_domain(path: Path) -> None:
    with path.open(newline="", encoding="utf-8-sig") as source:
        rows = list(csv.DictReader(source))
    if len(rows) != 1:
        raise ValueError("Expected exactly one R3 summary row")
    row = rows[0]
    for name, expected in (
        ("k", 2),
        ("development_patients", 10000),
        ("heldout_patients", 61518),
        ("development_ari", 1.0),
        ("heldout_ari", 1.0),
        ("seed", 42),
        ("n_init", 20),
    ):
        require_close(row[name], expected, name, tolerance=1e-10)
    if row["algorithm"].strip().lower() != "lloyd":
        raise ValueError("Unexpected R3 K-means algorithm")
    print("R3 domain comparison: both reported ARIs match the saved CSV")


def check_figure3(path: Path, root: Path) -> None:
    record = json.loads(path.read_text(encoding="utf-8"))
    if int(record["n_tested_weights"]) != 15:
        raise ValueError("Figure 3 must include all 15 tested weights")
    verify_source_hashes(record["inputs"], root / "evidence" / "primary")
    print("Figure 3: aggregate source-file hashes match")


def verify_source_hashes(sources: list[dict], folder: Path) -> None:
    for entry in sources:
        name = entry["file"]
        if "REFERENCE_REPORTED" in name:
            raise ValueError(f"Illustrative source is not release evidence: {name}")
        actual_file = folder / name
        if not actual_file.is_file():
            raise FileNotFoundError(f"Source file is not in the release: {actual_file}")
        if sha256(actual_file) != entry["sha256"]:
            raise ValueError(f"Source hash differs from plot record: {actual_file}")


def check_figure4(path: Path, root: Path) -> None:
    record = json.loads(path.read_text(encoding="utf-8"))
    if record["aligned_transition"] != [[6449, 435], [2931, 185]]:
        raise ValueError("Figure 4 transition counts differ from the manuscript")
    if record["changed_assignments"] != 3366:
        raise ValueError("Figure 4 changed-assignment count differs")
    require_close(record["diag2_share_at_w060_pct"], 88.423949, "Diagnosis-2 share")
    require_close(record["utilization_share_at_w100_pct"], 99.931092, "Utilization share")
    verify_source_hashes(record["source_files"], root / "evidence" / "reconstruction")
    print("Figure 4: counts and reconstruction source-file hashes match")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--write-checksums", action="store_true", help="Write hashes for the four checked records")
    args = parser.parse_args()
    root = args.repo_root.resolve()

    checks = {
        "r1_multihot": lambda p: check_r1_multihot(p),
        "r3_domain": lambda p: check_r3_domain(p),
        "figure3": lambda p: check_figure3(p, root),
        "figure4": lambda p: check_figure4(p, root),
    }
    missing = []
    errors = []
    checksums = []
    for role, relative_path in REQUIRED_FILES.items():
        path = root / relative_path
        if not path.is_file():
            missing.append(relative_path.as_posix())
            continue
        try:
            checks[role](path)
            checksums.append({"role": role, "file": relative_path.as_posix(), "sha256": sha256(path)})
        except (KeyError, ValueError, FileNotFoundError, TypeError) as exc:
            errors.append(f"{role}: {exc}")

    for entry in missing:
        print(f"MISSING: {entry}")
    for entry in errors:
        print(f"FAILED: {entry}")

    if errors:
        return 1
    if missing:
        return 2
    if args.write_checksums:
        output = root / "evidence" / "supplementary_file_sha256.csv"
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as dest:
            writer = csv.DictWriter(dest, fieldnames=["role", "file", "sha256"])
            writer.writeheader()
            writer.writerows(checksums)
        print(f"Wrote {output}")
    print("All four aggregate records agree with the supplied results and released source files.")
    print("This check does not reproduce the original patient-level experiments.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
