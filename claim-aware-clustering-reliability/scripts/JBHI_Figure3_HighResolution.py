"""Reproduce JBHI Figure 3 from VERIFIED primary 06B weight-sweep CSVs.

Primary R2 (not the separate R2 reconstruction): 15 tested utilization
weights; 5 independently seeded fits per weight, K-means n_init=10.

Requirements
    pip install matplotlib pandas numpy

From the repository root
    python scripts/JBHI_Figure3_HighResolution.py

From a Jupyter notebook opened at the repository root
    %run scripts/JBHI_Figure3_HighResolution.py

Outputs in figures/generated
    JBHI_Fig3_Weight_Sweep_TwoPanel.png/.pdf/.svg
    JBHI_Fig3_Transition_Column.png/.pdf/.svg

Note: Lines connect discretely evaluated weight settings; the apparent change
is exploratory and is not an estimated universal transition threshold.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')   # Works from both notebook and standard Python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SWEEP_NAME = '06B_weight_sweep.csv'
STABILITY_NAME = '06B_seed_stability.csv'

BLUE = '#1E607D'
ORANGE = '#B65D2F'
GRID = '#D7DBDF'


def load_verified(sweep_path: Path, seed_path: Path, representation: str) -> pd.DataFrame:
    sweep = pd.read_csv(sweep_path)
    seeds = pd.read_csv(seed_path)
    sweep_cols = {'representation', 'weight', 'utilization_ari_mean',
                  'utilization_ari_min', 'utilization_ari_max'}
    seeds_cols = {'representation', 'weight', 'seed_pairwise_ari_mean'}
    for filename, table, required in [(sweep_path, sweep, sweep_cols),
                                       (seed_path, seeds, seeds_cols)]:
        missing = required - set(table.columns)
        if missing:
            raise ValueError(f'{filename}: required columns missing: {sorted(missing)}')
    sweep = sweep.loc[sweep['representation'] == representation].copy()
    seeds = seeds.loc[seeds['representation'] == representation].copy()
    if len(sweep) == 0 or len(seeds) == 0:
        raise ValueError(f'No {representation} data in both files.')
    if sweep.weight.duplicated().any() or seeds.weight.duplicated().any():
        raise ValueError('Weights must be unique within each representation.')
    data = sweep.merge(seeds[['weight', 'seed_pairwise_ari_mean']],
                       on='weight', how='inner', validate='one_to_one')
    if len(data) != len(sweep) or len(data) != len(seeds):
        raise ValueError('Weight grids in the two files do not match.')
    for c in ('weight', 'utilization_ari_mean', 'utilization_ari_min',
              'utilization_ari_max', 'seed_pairwise_ari_mean'):
        data[c] = pd.to_numeric(data[c], errors='raise')
    data = data.sort_values('weight').reset_index(drop=True)
    if not np.isfinite(data.select_dtypes('number').to_numpy()).all():
        raise ValueError('Input contains missing or nonfinite numeric results.')
    if ((data.utilization_ari_min > data.utilization_ari_mean) |
        (data.utilization_ari_mean > data.utilization_ari_max)).any():
        raise ValueError('Reported ARI mean is outside reported min-max.')
    return data


def setup_style():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 9.3,
        'axes.labelsize': 9.3,
        'axes.titlesize': 10.2,
        'xtick.labelsize': 8.5,
        'ytick.labelsize': 8.5,
        'legend.fontsize': 8.3,
        'axes.linewidth': 0.7,
        'pdf.fonttype': 42,   # Embed editable TrueType-like text where supported
        'ps.fonttype': 42,
        'savefig.facecolor': 'white',
        'figure.facecolor': 'white',
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


def draw_panel(ax, df: pd.DataFrame, zoom: bool = False):
    x = df.weight.to_numpy(float)
    y_full = df.utilization_ari_mean.to_numpy(float)
    y_seed = df.seed_pairwise_ari_mean.to_numpy(float)
    low = df.utilization_ari_min.to_numpy(float)
    high = df.utilization_ari_max.to_numpy(float)

    # Plot observed min/max among five fits. The area between endpoints is a
    # display aid for discrete tested weights, not a continuous confidence band.
    ax.fill_between(x, low, high, color=BLUE, alpha=0.13,
                    zorder=1, label='Across-seed range (min-max)')
    ax.plot(x, y_full, '-o', color=BLUE, linewidth=1.85,
            markersize=4.2, markeredgecolor='white', markeredgewidth=.5,
            label='Full vs utilization-only ARI', zorder=4)
    ax.plot(x, y_seed, '-s', color=ORANGE, linewidth=1.8,
            markersize=3.8, markeredgecolor='white', markeredgewidth=.4,
            label='Mean pairwise seed ARI', zorder=5)
    ax.axhline(0, color='#838383', linestyle=(0, (3, 3)), linewidth=.8, zorder=0)
    ax.grid(axis='y', color=GRID, linewidth=.6, alpha=.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(-.065, 1.07)
    ax.set_yticks([0, .2, .4, .6, .8, 1])
    ax.set_xlabel('Utilization-coordinate weight')

    if zoom:
        ax.set_xlim(.542, .756)
        ticks = [.55, .60, .625, .65, .675, .70, .725, .75]
        ax.set_xticks(ticks)
        ax.set_xticklabels(['.550', '.600', '.625', '.650',
                            '.675', '.700', '.725', '.750'], rotation=45,
                           ha='right', rotation_mode='anchor')
    else:
        ax.set_xlim(-.055, 2.055)
        ticks = [0, .25, .5, .75, 1, 1.5, 2]
        ax.set_xticks(ticks)
        ax.set_xticklabels(['0', '0.25', '0.50', '0.75',
                            '1.00', '1.50', '2.00'])
        ax.axvspan(.65, .725, alpha=.065, color=BLUE, zorder=0)


def save_triplet(fig, outdir: Path, stem: str):
    for ext in ('png', 'pdf', 'svg'):
        target = outdir / f'{stem}.{ext}'
        fig.savefig(target, dpi=600 if ext == 'png' else None,
                    bbox_inches='tight', pad_inches=.065)
        print(f'Saved: {target}')
    plt.close(fig)


def build_two_panel(df: pd.DataFrame, outdir: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.65, 3.50),
                                   gridspec_kw={'width_ratios': [1.05, 1.05]},
                                   sharey=True)
    draw_panel(ax1, df, zoom=False)
    draw_panel(ax2, df, zoom=True)
    ax1.set_title('(a) Full evaluated weight grid', loc='left', pad=9)
    ax2.set_title('(b) Expanded sensitivity region', loc='left', pad=9)
    ax1.set_ylabel('Adjusted Rand index (ARI)')
    fig.legend(handles=ax1.get_legend_handles_labels()[0],
               labels=ax1.get_legend_handles_labels()[1],
               loc='upper center', bbox_to_anchor=(.5, .99),
               ncol=3, columnspacing=1.25, handlelength=2.1,
               frameon=False, fontsize=8.3)
    fig.subplots_adjust(left=.075, right=.986, bottom=.235,
                        top=.76, wspace=.18)
    save_triplet(fig, outdir, 'JBHI_Fig3_Weight_Sweep_TwoPanel')


def build_one_column(df: pd.DataFrame, outdir: Path):
    fig, ax = plt.subplots(figsize=(3.55, 2.85))
    draw_panel(ax, df, zoom=True)
    ax.set_ylabel('Adjusted Rand index (ARI)')
    ax.set_title('Primary R2: tested weights 0.55-0.75', loc='left', pad=7)
    fig.legend(handles=ax.get_legend_handles_labels()[0],
               labels=ax.get_legend_handles_labels()[1],
               loc='lower center', bbox_to_anchor=(.52, .005),
               ncol=1, handlelength=2.0, frameon=False, fontsize=7.4)
    fig.subplots_adjust(left=.17, right=.98, bottom=.41, top=.88)
    save_triplet(fig, outdir, 'JBHI_Fig3_Transition_Column')


def main():
    p = argparse.ArgumentParser(description='Replot validated JBHI primary 06B weight sweep')
    p.add_argument('--sweep', type=Path, default=HERE.parent / 'evidence' / 'primary' / SWEEP_NAME)
    p.add_argument('--seeds', type=Path, default=HERE.parent / 'evidence' / 'primary' / STABILITY_NAME)
    p.add_argument('--output-dir', type=Path, default=HERE.parent / 'figures' / 'generated')
    p.add_argument('--representation', choices=['R2', 'R3'], default='R2')
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_verified(args.sweep, args.seeds, args.representation)
    setup_style()
    build_two_panel(data, args.output_dir)
    build_one_column(data, args.output_dir)
    print(f'Verified source records: {len(data)} tested weights for {args.representation}.')
    print('Primary 06B: 5 seeded fits/weight, n_init=10. Not the separate reconstruction.')

if __name__ == '__main__':
    main()
