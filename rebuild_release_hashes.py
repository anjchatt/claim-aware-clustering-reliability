#!/usr/bin/env python3
"""Build a release hash manifest from bytes stored in the current Git commit.

Run from the repository root after committing the code and evidence files.
Commit the resulting manifest separately, then run verify_release_files.py.
"""

import csv
import hashlib
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "evidence" / "release_file_sha256.csv"


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args])


def main():
    raw_paths = git("ls-files", "-z")
    files = sorted(p.decode("utf-8") for p in raw_paths.split(b"\0") if p)
    rows = []
    for name in files:
        if name == "evidence/release_file_sha256.csv":
            continue  # A checksum file cannot include its own checksum.
        contents = git("show", "HEAD:" + name)
        rows.append({
            "file": name,
            "sha256": hashlib.sha256(contents).hexdigest(),
            "bytes": len(contents),
        })

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file", "sha256", "bytes"],
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} Git-blob hashes to {MANIFEST}")
    print("Commit the manifest, then run scripts/verify_release_files.py.")


if __name__ == "__main__":
    main()
