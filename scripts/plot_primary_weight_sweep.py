#!/usr/bin/env python3
"""Plot the primary R2 utilization-weight sweep from saved aggregate CSVs.

The plot includes all 15 tested weights and the observed minimum/maximum
across seeded fits. No patient-level models are refitted.
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


REPOSITORY = Path(__file__).resolve().parents[1]
SWEEP_COLUMNS = {
    "representation", "weight", "utilization_ari_mean", "utilization_ari_min", "utilization_ari_max"
}
SEED_COLUMNS = {"representation", "weight", "seed_pairwise_ari_mean"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_results(sweep_path: Path, seeds_path: Path) -> pd.DataFrame:
    sweep = pd.read_csv(sweep_path)
    seeds = pd.read_csv(seeds_path)
    if missing := SWEEP_COLUMNS.difference(sweep.columns):
        raise ValueError(f"Missing weight-sweep columns: {sorted(missing)}")
    if missing := SEED_COLUMNS.difference(seeds.columns):
        raise ValueError(f"Missing seed-stability columns: {sorted(missing)}")

    sweep = sweep.loc[sweep["representation"] == "R2"].copy()
    seeds = seeds.loc[seeds["representation"] == "R2"].copy()
    if sweep["weight"].duplicated().any() or seeds["weight"].duplicated().any():
        raise ValueError("Duplicate R2 weight settings")
    joined = sweep.merge(
        seeds[["weight", "seed_pairwise_ari_mean"]],
        on="weight", how="inner", validate="one_to_one",
    ).sort_values("weight").reset_index(drop=True)
    if len(joined) != 15 or len(sweep) != 15 or len(seeds) != 15:
        raise ValueError("Expected 15 matching R2 weights in each input")

    values = [
        "weight", "utilization_ari_mean", "utilization_ari_min",
        "utilization_ari_max", "seed_pairwise_ari_mean",
    ]
    joined[values] = joined[values].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(joined[values].to_numpy()).all():
        raise ValueError("Non-finite plot value")
    outside_range = (
        (joined["utilization_ari_mean"] < joined["utilization_ari_min"] - 1e-10)
        | (joined["utilization_ari_mean"] > joined["utilization_ari_max"] + 1e-10)
    )
    if outside_range.any():
        raise ValueError("A mean ARI lies outside its observed seed range")

    for weight, expected in ((0.6, -0.007489), (0.675, 0.597004),
                             (0.7, 0.596851), (0.725, 1.0)):
        entry = joined.loc[np.isclose(joined["weight"], weight)]
        if len(entry) != 1 or not np.isclose(
            float(entry.iloc[0]["utilization_ari_mean"]), expected, atol=2e-5
        ):
            raise ValueError(f"Unexpected primary R2 agreement at weight {weight}")
    return joined


def draw_panel(axis, data: pd.DataFrame, expanded: bool) -> None:
    weights = data["weight"].to_numpy(dtype=float)
    blue, orange = "#21627D", "#B45C32"
    axis.fill_between(
        weights,
        data["utilization_ari_min"].to_numpy(dtype=float),
        data["utilization_ari_max"].to_numpy(dtype=float),
        alpha=0.16, color=blue, label="Across-seed range (min–max)", zorder=1,
    )
    axis.plot(
        weights, data["utilization_ari_mean"], "-o", lw=1.8, ms=4.1,
        c=blue, label="Full vs. utilization-only ARI", zorder=4,
    )
    axis.plot(
        weights, data["seed_pairwise_ari_mean"], "-s", lw=1.7, ms=3.7,
        c=orange, label="Mean pairwise seed ARI", zorder=5,
    )
    axis.set_ylim(-0.075, 1.08)
    axis.set_yticks(np.arange(0, 1.01, 0.2))
    axis.grid(axis="y", alpha=0.25, lw=0.55)
    axis.axhline(0, c="#888888", lw=0.7, ls="--")
    axis.set_xlabel("Utilization-coordinate weight")

    if expanded:
        axis.set_xlim(0.54, 0.76)
        ticks = [0.550, 0.600, 0.625, 0.650, 0.675, 0.700, 0.725, 0.750]
        axis.set_xticks(ticks, [f"{tick:.3f}" for tick in ticks], rotation=45, ha="right")
    else:
        axis.set_xlim(-0.05, 2.05)
        ticks = [0, 0.25, 0.5, 0.75, 1, 1.5, 2]
        axis.set_xticks(ticks, [f"{tick:.2f}" for tick in ticks])
        axis.axvspan(0.650, 0.725, alpha=0.07, color=blue, zorder=0)


def save_figure(data: pd.DataFrame, output_dir: Path) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    figure, (full_axis, zoom_axis) = plt.subplots(1, 2, figsize=(9.1, 3.8), sharey=True)
    draw_panel(full_axis, data, expanded=False)
    draw_panel(zoom_axis, data, expanded=True)
    full_axis.set_title("(a) Full evaluated weight grid", loc="left", pad=8)
    zoom_axis.set_title("(b) Expanded sensitivity region", loc="left", pad=8)
    full_axis.set_ylabel("Adjusted Rand index (ARI)")
    handles, labels = full_axis.get_legend_handles_labels()
    figure.legend(handles, labels, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.98),
                  frameon=False, fontsize=9, handlelength=2)
    figure.subplots_adjust(left=0.079, right=0.985, bottom=0.245, top=0.765, wspace=0.18)

    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        output = output_dir / f"figure3.{suffix}"
        figure.savefig(output, dpi=600 if suffix == "png" else None,
                       bbox_inches="tight", pad_inches=0.08)
        print(f"Saved {output}")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    primary = REPOSITORY / "evidence" / "primary"
    parser.add_argument("--sweep", type=Path, default=primary / "06B_weight_sweep.csv")
    parser.add_argument("--seeds", type=Path, default=primary / "06B_seed_stability.csv")
    parser.add_argument("--output-dir", type=Path, default=REPOSITORY / "figures")
    args = parser.parse_args()

    for source in (args.sweep, args.seeds):
        if not source.is_file():
            parser.error(f"Missing input CSV: {source}")
    results = load_results(args.sweep, args.seeds)
    save_figure(results, args.output_dir)
    record = {
        "description": "Primary R2 weight sweep: 15 weights, five seeds, 10 restarts",
        "inputs": [
            {"file": path.name, "sha256": sha256(path)}
            for path in (args.sweep, args.seeds)
        ],
        "n_tested_weights": len(results),
        "plot": "figure3.png, figure3.pdf, figure3.svg",
        "notes": "Lines join tested weights; shaded range is min–max across seeded fits.",
    }
    (args.output_dir / "figure3_sources.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    print("Figure 3 saved with decimal weight labels on the expanded panel.")


if __name__ == "__main__":
    main()
