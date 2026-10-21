#!/usr/bin/env python3
"""Check the files shipped in this repository against the release manifest.

This verifies file identity. It does not rerun the clinical experiments or
confirm the scientific interpretation of any result.
"""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "evidence" / "release_file_sha256.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    rows = list(csv.DictReader(MANIFEST.open(encoding="utf-8", newline="")))
    errors = []
    for row in rows:
        path = ROOT / row["file"]
        if not path.is_file():
            errors.append(f"Missing: {row['file']}")
        elif sha256(path) != row["sha256"]:
            errors.append(f"Hash mismatch: {row['file']}")

    print(f"Checked {len(rows)} files in the release-candidate manifest.")
    if errors:
        for message in errors:
            print(message)
        return 1
    print("All packaged-file hashes match. This is not a full study reproduction.")
    missing = [
        "Original primary notebooks 01-03 and 05",
        "Primary refit-masking and held-out masking run exports",
        "Primary diagnosis-position indicator cross-tab export",
        "Executed 120-run diagnosis-position masking results",
        "Original 17-artifact inventory and per-stream run manifests",
    ]
    print("Outstanding original research sources:")
    for item in missing:
        print(f"  - {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
