#!/usr/bin/env python3
"""Plot reconstructed R2 feature shares and cross-weight cluster membership.

Both inputs must be aggregate outputs from the executed reconstruction.
The analysis does not fit K-means or read patient-level data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


BLOCKS = ["Age", "Diagnosis family 1", "Diagnosis family 2", "Diagnosis family 3", "Utilization"]
COLORS = {
    "Age": "#1B5377",
    "Diagnosis family 1": "#93B5BC",
    "Diagnosis family 2": "#BE683A",
    "Diagnosis family 3": "#D7AC7B",
    "Utilization": "#228978",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_concentration(path: Path) -> pd.DataFrame:
    shares = pd.read_csv(path)
    required = {"weight", "feature_block", "share_pct"}
    missing = required.difference(shares.columns)
    if missing:
        raise ValueError(f"{path} is missing {sorted(missing)}")

    shares["weight"] = pd.to_numeric(shares["weight"], errors="raise")
    shares["share_pct"] = pd.to_numeric(shares["share_pct"], errors="raise")
    shares = shares[shares["weight"].apply(lambda value: np.isclose(value, 0.6) or np.isclose(value, 1.0))]
    if shares.duplicated(["weight", "feature_block"]).any():
        raise ValueError("Duplicate weight/feature-block rows")
    if len(shares) != 10 or set(shares["feature_block"]) != set(BLOCKS):
        raise ValueError("Expected five feature blocks at each of weights 0.600 and 1.000")

    table = shares.pivot(index="weight", columns="feature_block", values="share_pct")
    table = table.reindex(index=[0.6, 1.0], columns=BLOCKS)
    if table.isna().any().any() or (table.to_numpy() < 0).any():
        raise ValueError("Feature-share table contains missing or negative values")
    if not np.allclose(table.sum(axis=1).to_numpy(), 100.0, atol=0.02):
        raise ValueError("Feature shares must add to 100% at each weight")
    for weight, block, expected in (
        (0.6, "Diagnosis family 2", 88.4),
        (1.0, "Utilization", 99.9),
    ):
        if abs(float(table.loc[weight, block]) - expected) > 0.15:
            raise ValueError(f"Unexpected {block} share at weight {weight}")
    return table


def load_aligned_counts(path: Path):
    counts = pd.read_csv(path, index_col=0)
    if counts.shape != (2, 2):
        raise ValueError(f"Expected a 2-by-2 table, got {counts.shape}")

    values = counts.apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Transition counts must be finite and nonnegative")
    if not np.all(np.isclose(values, np.round(values))):
        raise ValueError("Transition counts must be integers")

    # Cluster indices are arbitrary; align rows and columns by decreasing size.
    values = values[np.argsort(-values.sum(axis=1)), :]
    values = values[:, np.argsort(-values.sum(axis=0))].astype(int)
    if values.sum() != 10000:
        raise ValueError("Unexpected reconstruction development cohort size")
    if values.sum(axis=1).tolist() != [6884, 3116]:
        raise ValueError("Weight 0.600 cluster sizes differ from the saved reconstruction")
    if values.sum(axis=0).tolist() != [9380, 620]:
        raise ValueError("Weight 1.000 cluster sizes differ from the saved reconstruction")
    if values.tolist() != [[6449, 435], [2931, 185]]:
        raise ValueError("The aligned transition table differs from the reported counts")

    changed = int(values[0, 1] + values[1, 0])
    return values, changed, 100.0 * changed / values.sum()


def plot_figure(shares: pd.DataFrame, counts: np.ndarray, output_dir: Path) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 10.5,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    figure, (bar_axis, matrix_axis) = plt.subplots(
        1, 2, figsize=(9.15, 3.55), gridspec_kw={"width_ratios": [1.30, 1.0]}
    )

    for row, weight in enumerate((0.6, 1.0)):
        left = 0.0
        for block in BLOCKS:
            share = float(shares.loc[weight, block])
            bar_axis.barh(
                row, share, left=left, height=0.46, color=COLORS[block],
                edgecolor="white", linewidth=0.4, label=block if row == 0 else None,
            )
            if share >= 12:
                text_color = "white" if block in ("Diagnosis family 2", "Utilization") else "#18323d"
                bar_axis.text(left + share / 2, row, f"{share:.1f}%", ha="center",
                              va="center", fontsize=10, color=text_color, fontweight="bold")
            left += share

    bar_axis.set_yticks((0, 1), ("w = 0.600", "w = 1.000"))
    bar_axis.invert_yaxis()
    bar_axis.set_xlim(0, 100)
    bar_axis.set_xlabel("Between-cluster dispersion share (%)")
    bar_axis.set_title("(a) Selected partition geometry", loc="left", pad=10)
    handles, _ = bar_axis.get_legend_handles_labels()
    bar_axis.legend(
        handles, ["Age", "Diag. 1", "Diag. 2", "Diag. 3", "Utilization"],
        ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.21),
        frameon=False, fontsize=8.4, columnspacing=1.25, handlelength=1.3,
    )

    colors = LinearSegmentedColormap.from_list("counts", ["#F4F8FC", "#7FB9D8", "#083A76"])
    matrix_axis.imshow(counts, cmap=colors, aspect="auto", vmin=0, vmax=int(counts.max()))
    matrix_axis.set_xticks((0, 1), ("Other (9,380)", "High utilization (620)"))
    matrix_axis.set_yticks((0, 1), ("Other\n(6,884)", "Diag. 2 circulatory\n(3,116)"))
    matrix_axis.set_xlabel("w = 1.000 cluster", labelpad=8)
    matrix_axis.set_ylabel("w = 0.600 cluster", labelpad=1)
    matrix_axis.set_title("(b) Label-aligned cross-weight membership", loc="left", pad=10)
    matrix_axis.tick_params(axis="both", length=0)

    for row in range(2):
        for column in range(2):
            count = int(counts[row, column])
            color = "white" if count > counts.max() * 0.55 else "#172934"
            matrix_axis.text(column, row, f"{count:,}", ha="center", va="center",
                             fontsize=12.5, color=color, fontweight="bold")

    figure.subplots_adjust(left=0.085, right=0.985, bottom=0.28, top=0.86, wspace=1.05)
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        output = output_dir / f"figure4.{suffix}"
        figure.savefig(output, dpi=600 if suffix == "png" else None,
                       bbox_inches="tight", pad_inches=0.10)
        print(f"Saved {output}")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concentration", required=True, type=Path)
    parser.add_argument("--transition", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "figures")
    args = parser.parse_args()

    for source in (args.concentration, args.transition):
        if not source.is_file():
            parser.error(f"Missing source CSV: {source}")
        if "REFERENCE_REPORTED" in source.name:
            parser.error(f"Use the executed reconstruction files instead of {source.name}")

    shares = load_concentration(args.concentration)
    counts, changed, changed_pct = load_aligned_counts(args.transition)
    plot_figure(shares, counts, args.output_dir)

    record = {
        "data_status": "reconstruction aggregate CSV inputs; no model rerun",
        "source_files": [
            {"file": source.name, "sha256": sha256(source)}
            for source in (args.concentration, args.transition)
        ],
        "n_dev": int(counts.sum()),
        "weight_values": [0.6, 1.0],
        "diag2_share_at_w060_pct": float(shares.loc[0.6, "Diagnosis family 2"]),
        "utilization_share_at_w100_pct": float(shares.loc[1.0, "Utilization"]),
        "aligned_transition": counts.tolist(),
        "changed_assignments": changed,
        "changed_pct": changed_pct,
    }
    destination = args.output_dir / "figure4_sources.json"
    destination.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"Aligned counts: {counts.tolist()}; {changed:,} changed ({changed_pct:.2f}%).")


if __name__ == "__main__":
    main()
