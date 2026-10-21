#!/usr/bin/env python3
"""Examine utilization weighting and preprocessing choices in R2 clustering.

The public-data reconstruction is separate from the original study. Its
results should not be interpreted as a row-matched replication.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import scipy
from scipy.optimize import linear_sum_assignment
from scipy.stats import fisher_exact
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import RobustScaler, StandardScaler
from threadpoolctl import threadpool_limits

UTIL = ["number_outpatient", "number_emergency", "number_inpatient"]
DIAG = ["diag_1", "diag_2", "diag_3"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    inp = p.add_mutually_exclusive_group(required=True)
    inp.add_argument("--raw-csv", type=Path, help="Original UCI diabetic_data.csv; new extraction, not exact historical reproduction")
    inp.add_argument("--dev-matrix", type=Path, help="EXACT encoded 06B R2 development matrix exported as CSV")
    p.add_argument("--heldout-matrix", type=Path, help="Exact 06B R2 held-out matrix, same encoded columns as --dev-matrix")
    p.add_argument("--dev-meta", type=Path, help="Development metadata CSV with patient_nbr, race, raw utilization counts")
    p.add_argument("--heldout-meta", type=Path, help="Held-out metadata, optional")
    p.add_argument("--split-manifest", type=Path, help="CSV columns patient_nbr, split (development/heldout) for --raw-csv")
    p.add_argument("--historical-dev-labels", type=Path, help="CSV patient_nbr,label from original 06B development R2 baseline")
    p.add_argument("--historical-heldout-labels", type=Path, help="CSV patient_nbr,label from original held-out R2 baseline")
    p.add_argument("--out", type=Path, default=Path("weight_scaling_results"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seeds-per-weight", type=int, default=40,
                   help="Independent KMeans fits per weight")
    p.add_argument("--audit-n-init", type=int, default=1,
                   help="KMeans restarts PER independent seeded fit (1 for original 06C audit; 20 for manuscript-comparable multistart audit)")
    p.add_argument("--silhouette-sample", type=int, default=1200)
    p.add_argument("--pair-sample", type=int, default=20000)
    p.add_argument("--bootstrap-reps", type=int, default=1000)
    p.add_argument("--weights", type=str,
                   default="0.600,0.650,0.675,0.680,0.690,0.700,0.710,0.725,0.750,1.000")
    p.add_argument("--scalers", type=str, default="robust,standard,none")
    p.add_argument("--weight-transform", choices=["direct", "sqrt"], default="direct",
                   help="direct: multiply utilization coordinates by w; sqrt: by sqrt(w). Confirm 06B convention")
    p.add_argument("--age-divisor", type=float, default=100.0,
                   help="Raw UCI reconstruction ONLY; use exact matrix for manuscript comparability")
    p.add_argument("--skip-heldout", action="store_true")
    p.add_argument("--skip-negative-control", action="store_true")
    p.add_argument("--quick", action="store_true", help="Small non-publication smoke test; overrides seeds/weights")
    a = p.parse_args()
    if a.heldout_matrix and not a.dev_matrix:
        p.error("--heldout-matrix needs --dev-matrix")
    if a.dev_matrix and a.split_manifest:
        p.error("--split-manifest only applies to --raw-csv")
    if a.dev_matrix and not a.dev_meta:
        print("WARNING: --dev-meta absent; race analyses may be skipped", file=sys.stderr)
    if a.seeds_per_weight < 2:
        p.error("--seeds-per-weight must be >= 2")
    if a.audit_n_init < 1:
        p.error("--audit-n-init must be >= 1")
    if a.age_divisor <= 0:
        p.error("--age-divisor must be >0")
    if a.quick:
        a.weights = "0.600,0.700,0.725,1.000"
        a.scalers = "robust,standard"
        a.seeds_per_weight = 4
        a.silhouette_sample = 250
        a.pair_sample = 1000
        a.bootstrap_reps = 100
    a.weights = sorted(set(float(i.strip()) for i in a.weights.split(",") if i.strip()))
    a.scalers = [i.strip() for i in a.scalers.split(",") if i.strip()]
    if not a.weights or any(w <= 0 for w in a.weights):
        p.error("All weights must be > 0")
    if any(s not in ("robust", "standard", "none") for s in a.scalers):
        p.error("Allowed scalers: robust,standard,none")
    return a


def sha256(path):
    if path is None:
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def age_midpoint(value):
    try:
        s = str(value).replace("[", "").replace("]", "").replace(")", "")
        lo, hi = map(float, s.split("-"))
        return (lo + hi) / 2
    except (ValueError, TypeError):
        return np.nan


def icd9_family(value):
    """Explicit exploratory mapping. MUST be reconciled to notebooks before manuscript claims."""
    if pd.isna(value) or str(value).strip() in {"?", "", "nan", "None"}:
        return "Missing"
    s = str(value).strip().upper()
    if s.startswith("V"):
        return "V_codes"
    if s.startswith("E"):
        return "E_codes"
    try:
        n = float(s)
    except ValueError:
        return "Other"
    # ICD-9 broad chapter boundaries. Diabetes has separate grouping.
    if 250 <= n < 251:
        return "Diabetes"
    if 1 <= n < 140: return "Infectious"
    if 140 <= n < 240: return "Neoplasms"
    if 240 <= n < 280: return "Endocrine_metabolic"
    if 280 <= n < 290: return "Blood"
    if 290 <= n < 320: return "Mental"
    if 320 <= n < 390: return "Nervous_sense"
    if 390 <= n < 460: return "Circulatory"
    if 460 <= n < 520: return "Respiratory"
    if 520 <= n < 580: return "Digestive"
    if 580 <= n < 630: return "Genitourinary"
    if 630 <= n < 680: return "Pregnancy"
    if 680 <= n < 710: return "Skin"
    if 710 <= n < 740: return "Musculoskeletal"
    if 740 <= n < 760: return "Congenital"
    if 760 <= n < 780: return "Perinatal"
    if 780 <= n < 800: return "Symptoms"
    if 800 <= n < 1000: return "Injury"
    return "Other"


def load_raw(a):
    df = pd.read_csv(a.raw_csv, dtype={"patient_nbr": str, "encounter_id": str}, low_memory=False)
    req = {"patient_nbr", "encounter_id", "age", "race", *DIAG, *UTIL}
    missing = sorted(req - set(df.columns))
    if missing:
        raise ValueError(f"Raw UCI file missing columns: {missing}")
    # numeric encounter ordering, not lexical ordering and not guaranteed chronological
    df["_encnum"] = pd.to_numeric(df["encounter_id"], errors="raise")
    df = df.sort_values(["_encnum", "patient_nbr"], kind="stable")
    df = df.drop_duplicates("patient_nbr", keep="first").reset_index(drop=True)
    if df["patient_nbr"].nunique() != len(df):
        raise AssertionError("Deduplication failed")
    if a.split_manifest:
        split = pd.read_csv(a.split_manifest, dtype={"patient_nbr": str})
        if not {"patient_nbr", "split"}.issubset(split.columns):
            raise ValueError("Split manifest needs patient_nbr and split")
        if split["patient_nbr"].duplicated().any():
            raise ValueError("Duplicate patients in split manifest")
        split["split"] = split["split"].str.lower().str.strip().replace({"dev": "development", "test": "heldout", "held-out": "heldout"})
        if not set(split.split).issubset({"development", "heldout"}):
            raise ValueError("split values must be development or heldout")
        df = df.merge(split[["patient_nbr", "split"]], on="patient_nbr", how="left", validate="one_to_one")
        if df["split"].isna().any():
            raise ValueError("Split manifest is missing patients from deduplicated raw data")
        split_status = "HISTORICAL_SPLIT_SUPPLIED_RAW_RECONSTRUCTION_MAPPING_UNVERIFIED"
    else:
        if len(df) < 100:
            raise ValueError("Too few patient rows")
        n_dev = 10000 if len(df) >= 20000 else max(40, int(round(len(df) * 0.14)))
        rng = np.random.default_rng(a.seed)
        sel = rng.choice(len(df), size=n_dev, replace=False)
        df["split"] = "heldout"
        df.loc[sel, "split"] = "development"
        split_status = "NEW_RANDOM_SPLIT_NOT_HISTORICAL_REPLICATION"
    for c in DIAG:
        df[f"diag_family_{c[-1]}"] = df[c].map(icd9_family)
    df["age_midpoint"] = df["age"].map(age_midpoint)
    dev = df.loc[df.split == "development"].copy().reset_index(drop=True)
    held = df.loc[df.split == "heldout"].copy().reset_index(drop=True)
    if len(dev) < 40:
        raise ValueError("Too few development rows")
    age_median = float(dev.age_midpoint.median())
    if not math.isfinite(age_median):
        raise ValueError("Could not parse age bands")
    clinical = ["diag_family_1", "diag_family_2", "diag_family_3"]
    for subset in (dev, held):
        subset["clinical__age"] = subset.age_midpoint.fillna(age_median).astype(float) / a.age_divisor
    # Fit one-hot category universe only to development patients; force Missing on all three fields
    frame_dev = pd.DataFrame(index=dev.index)
    frame_hold = pd.DataFrame(index=held.index)
    for c in clinical:
        cats = sorted(set(dev[c].astype(str)) | {"Missing"})
        d = pd.Categorical(dev[c].fillna("Missing"), categories=cats)
        h = pd.Categorical(held[c].fillna("Missing"), categories=cats)
        x = pd.get_dummies(d, prefix=f"clinical__{c}", dtype=float)
        y = pd.get_dummies(h, prefix=f"clinical__{c}", dtype=float)
        x.index = dev.index
        y.index = held.index
        frame_dev = pd.concat([frame_dev, x], axis=1)
        frame_hold = pd.concat([frame_hold, y], axis=1)
    frame_dev.insert(0, "clinical__age", dev["clinical__age"].values)
    frame_hold.insert(0, "clinical__age", held["clinical__age"].values)
    for c in UTIL:
        fill = pd.to_numeric(dev[c], errors="coerce").median()
        if not np.isfinite(fill):
            fill = 0
        frame_dev[f"util__{c}"] = pd.to_numeric(dev[c], errors="coerce").fillna(fill).to_numpy(float)
        frame_hold[f"util__{c}"] = pd.to_numeric(held[c], errors="coerce").fillna(fill).to_numpy(float)
    dev_meta = dev[["patient_nbr", "race", *UTIL]].copy()
    hold_meta = held[["patient_nbr", "race", *UTIL]].copy()
    return (frame_dev, frame_hold, dev_meta, hold_meta, split_status,
            {"mapping": "Exploratory ICD9 chapter map in source", "age": f"midpoint/{a.age_divisor}",
             "one_hot": "development-fitted only; explicit Missing", "encounter": "minimum encounter_id (numeric)"})


def load_matrix(a):
    dev_raw = pd.read_csv(a.dev_matrix, dtype={"patient_nbr": str})
    held_raw = pd.read_csv(a.heldout_matrix, dtype={"patient_nbr": str}) if a.heldout_matrix else None
    def matrix(source):
        cols = [c for c in source if c.startswith("clinical__") or c.startswith("util__") or c.startswith("lab__")]
        if not any(c.startswith("clinical__") for c in cols) or len([c for c in cols if c.startswith("util__")]) != 3:
            raise ValueError("Matrix columns must contain clinical__... and exactly three util__... columns")
        x = source[cols].apply(pd.to_numeric, errors="raise")
        if not np.isfinite(x.to_numpy(float)).all():
            raise ValueError("Matrix has missing or infinite entries")
        return x
    dev = matrix(dev_raw)
    held = matrix(held_raw) if held_raw is not None else pd.DataFrame(columns=dev.columns)
    if list(dev.columns) != list(held.columns):
        raise ValueError("Held-out feature columns must be in same order as development feature columns")
    if len(dev) < 40:
        raise ValueError("Too few development rows")
    def meta(path, original):
        m = pd.read_csv(path, dtype={"patient_nbr": str}) if path else pd.DataFrame()
        if m.empty and original is not None:
            extra = [c for c in ["patient_nbr", "race", *UTIL] if c in original.columns]
            m = original[extra].copy()
        if not m.empty and len(m) != len(original):
            raise ValueError("Metadata MUST match matrix row count and row order exactly")
        if not m.empty and "patient_nbr" in m and "patient_nbr" in original:
            if not np.array_equal(m["patient_nbr"].astype(str).to_numpy(), original["patient_nbr"].astype(str).to_numpy()):
                raise ValueError("Metadata patient_nbr does not match feature matrix patient_nbr row-by-row")
        if not m.empty and "patient_nbr" in m and m["patient_nbr"].duplicated().any():
            raise ValueError("Duplicate patient_nbr in metadata")
        return m
    dm = meta(a.dev_meta, dev_raw)
    hm = meta(a.heldout_meta, held_raw)
    if not dm.empty and not hm.empty and "patient_nbr" in dm and "patient_nbr" in hm:
        overlap = set(dm.patient_nbr) & set(hm.patient_nbr)
        if overlap:
            raise ValueError(f"Development/held-out patient identifiers overlap (example: {list(overlap)[:3]})")
    return (dev, held, dm, hm, "ORIGINAL_FEATURE_MATRIX_USER_SUPPLIED_NO_AUTOMATIC_NOTEBOOK_VERIFICATION",
            {"mapping": "Provided encoded matrix; no code inference", "scaling": "Inputs assumed BASE/unscaled numerical coordinates unless user verifies otherwise"})


def transforms(dev, held, scaler, weight, util_ix, weight_transform):
    x = dev.to_numpy(dtype=np.float64, copy=True)
    z = held.to_numpy(dtype=np.float64, copy=True)
    if scaler != "none":
        sc = RobustScaler() if scaler == "robust" else StandardScaler()
        x[:, util_ix] = sc.fit_transform(x[:, util_ix])
        if z.size:
            z[:, util_ix] = sc.transform(z[:, util_ix])
    factor = weight if weight_transform == "direct" else math.sqrt(weight)
    x[:, util_ix] *= factor
    if z.size:
        z[:, util_ix] *= factor
    return x, z


def aligned_changed(a, b):
    aa, bb = np.unique(a, return_inverse=True)[1], np.unique(b, return_inverse=True)[1]
    k = max(int(aa.max()), int(bb.max())) + 1
    confusion = np.zeros((k, k), dtype=np.int64)
    np.add.at(confusion, (aa, bb), 1)
    _, cols = linear_sum_assignment(-confusion)
    matched = confusion[np.arange(k), cols].sum()
    return float(1.0 - matched / len(aa))


def seed_interval(labels, reps, seed):
    """Full pairwise ARI; CI uses disjoint pairs of independent fit runs only.

    CI describes random-initialization variation, not patient-sampling uncertainty.
    """
    n = len(labels)
    mat = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            mat[i, j] = mat[j, i] = adjusted_rand_score(labels[i], labels[j])
    full = float(mat[np.triu_indices(n, 1)].mean())
    pairs = np.array([mat[i, i+1] for i in range(0, n-1, 2)], dtype=float)
    if len(pairs) < 2:
        return full, float(pairs.mean()), np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(pairs), size=(reps, len(pairs)))
    means = pairs[idx].mean(axis=1)
    return full, float(pairs.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def representative_index(inertias):
    return int(np.argmin(inertias))


def sampled_silhouette(x, y, n, seed):
    if len(np.unique(y)) < 2:
        return np.nan
    if len(y) <= n:
        take = np.arange(len(y))
    else:
        rng = np.random.default_rng(seed)
        take = rng.choice(len(y), size=n, replace=False)
    if len(np.unique(y[take])) < 2:
        return np.nan
    return float(silhouette_score(x[take], y[take]))


def util_fit(dev, held, scaler, weight, transform, util_ix, seed):
    udev = dev.iloc[:, util_ix].copy()
    uhold = held.iloc[:, util_ix].copy()
    x, z = transforms(udev, uhold, scaler, weight, np.arange(udev.shape[1]), transform)
    model = KMeans(n_clusters=2, n_init=20, random_state=seed)
    yl = model.fit_predict(x)
    yh = model.predict(z) if len(z) else np.array([], dtype=int)
    return yl, yh


def group_rate_table(meta, y, weight, scaler, cohort):
    if meta is None or meta.empty or "race" not in meta.columns or len(meta) != len(y):
        return []
    # Define the target cluster by mean observed utilization rather than arbitrary KMeans labels.
    if not set(UTIL).issubset(meta.columns):
        return []
    nums = meta[UTIL].apply(pd.to_numeric, errors="coerce").fillna(0)
    util_sum = nums.sum(axis=1).to_numpy()
    cluster_means = {int(z): float(util_sum[y == z].mean()) for z in np.unique(y)}
    higher = max(cluster_means, key=cluster_means.get)
    is_high = np.asarray(y == higher)
    # Report all categories with >= 20 members, including ? and other unknowns, descriptively
    races = meta["race"].fillna("Missing").astype(str)
    records = []
    for label, idx in races.groupby(races).groups.items():
        if len(idx) == 0:
            continue
        hits = int(is_high[np.asarray(list(idx), dtype=int)].sum())
        records.append({"cohort": cohort, "scaler": scaler, "weight": weight,
                        "race": str(label), "n": int(len(idx)), "high_util_cluster_n": hits,
                        "high_util_cluster_rate": float(hits/len(idx)),
                        "high_util_cluster_size": int(is_high.sum()),
                        "high_util_cluster_mean_count": cluster_means[higher]})
    return records


def group_comparison(rate_rows):
    # Only compare recorded Caucasian/AfricanAmerican; self-reported and unadjusted
    r = {v["race"]: v for v in rate_rows}
    if not {"Caucasian", "AfricanAmerican"}.issubset(r):
        return None
    a = r["Caucasian"]; b = r["AfricanAmerican"]
    table = [[a["high_util_cluster_n"], a["n"] - a["high_util_cluster_n"]],
             [b["high_util_cluster_n"], b["n"] - b["high_util_cluster_n"]]]
    OR, p = fisher_exact(table)
    return {"cohort": a["cohort"], "scaler": a["scaler"], "weight": a["weight"],
            "group_a": "Caucasian", "group_b": "AfricanAmerican",
            "n_a": a["n"], "n_b": b["n"],
            "events_a": a["high_util_cluster_n"], "events_b": b["high_util_cluster_n"],
            "risk_diff_pp_a_minus_b": 100*(a["high_util_cluster_rate"]-b["high_util_cluster_rate"]),
            "fisher_OR": float(OR), "fisher_p": float(p),
            "interpretation": "unadjusted descriptive composition only; not fairness or causation"}


def pair_distance(dev, pair_sample, seed):
    clinical_ix = [i for i,c in enumerate(dev.columns) if c.startswith("clinical__")]
    util_ix = [i for i,c in enumerate(dev.columns) if c.startswith("util__")]
    rng = np.random.default_rng(seed)
    n = len(dev)
    i = rng.integers(0, n, pair_sample)
    j = rng.integers(0, n, pair_sample)
    valid = i != j
    i, j = i[valid], j[valid]
    if len(i) == 0:
        return {}
    x = dev.to_numpy(float)
    c = ((x[i][:, clinical_ix] - x[j][:, clinical_ix])**2).sum(axis=1)
    u = ((x[i][:, util_ix] - x[j][:, util_ix])**2).sum(axis=1)
    return {"clinical_mean_sq_distance":float(c.mean()),
            "util_mean_sq_distance":float(u.mean()),
            "clinical_share":float(c.mean()/(c.mean()+u.mean())) if (c.mean()+u.mean()) else np.nan,
            "valid_sampled_pairs":int(len(i))}


def historical_ari(labels_path, metadata, current, cohort):
    if labels_path is None:
        return {"historical_ari": np.nan, "historical_comparison": "NOT_SUPPLIED"}
    if metadata.empty or "patient_nbr" not in metadata:
        raise ValueError(f"{cohort}: comparing historical labels requires metadata with patient_nbr")
    hist=pd.read_csv(labels_path,dtype={"patient_nbr":str})
    if not {"patient_nbr","label"}.issubset(hist.columns):
        raise ValueError("Historical labels CSV needs patient_nbr,label")
    if hist.patient_nbr.duplicated().any():
        raise ValueError("Duplicate historical patient IDs")
    # Audit identical order and complete identity, not just same row count.
    ordered=metadata[["patient_nbr"]].merge(hist,how="left",on="patient_nbr",validate="one_to_one",indicator=True)
    if not (ordered["_merge"]=="both").all() or len(ordered)!=len(hist):
        raise ValueError(f"{cohort}: historical patient IDs do not match input cohort exactly")
    return {"historical_ari":float(adjusted_rand_score(ordered.label,current)),
            "historical_comparison":"ARI_OF_FRESH_N_INIT_20_VS_RECORDED_LABELS_NOT_FULL_PIPELINE_ATTESTATION"}


def write_csv(rows, filename):
    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
    return df


def plot_transition(df, folder):
    if df.empty:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for scaler, t in df.groupby("scaler", sort=False):
        t=t.sort_values("weight")
        fig, ax = plt.subplots(figsize=(8.3,4.6))
        ax.plot(t.weight, t.seed_ari_all_pairs, "o-", label="Mean seed ARI")
        ax.plot(t.weight, t.util_agreement_best_fit, "s-", label="Utilization-only ARI")
        ax.fill_between(t.weight.to_numpy(float), t.seed_ari_disjoint_ci_low.to_numpy(float),
                        t.seed_ari_disjoint_ci_high.to_numpy(float), alpha=.15,
                        label="Disjoint seed-pair CI")
        ax.set(title=f"R2 K=2 numerical-weight audit: {scaler}",
               xlabel="Applied numerical weight (coordinate multiplier or sqrt, see manifest)",
               ylabel="Adjusted Rand index", ylim=(-.10, 1.04))
        ax.axvline(1.0, linestyle=":", color="grey", linewidth=1)
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=.18)
        fig.tight_layout()
        fig.savefig(folder/f"fig_weight_transition_{scaler}.png", dpi=170)
        plt.close(fig)


def plot_inertia(df, folder):
    if df.empty: return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sub = df[(df.scaler=="robust") & (np.isclose(df.weight, .700))]
    if sub.empty: return
    fig, ax=plt.subplots(figsize=(8.3,4.3))
    scatter=ax.scatter(sub.util_ari, sub.inertia, c=sub.seed, cmap="viridis", s=28)
    ax.set(title="R2 K=2 at w=0.700: fitted solutions", xlabel="ARI to independently fitted utilization-only partition",
           ylabel="K-means inertia (same scaled feature specification)")
    fig.colorbar(scatter, label="random seed")
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(folder/"fig_competing_solutions_w0700.png", dpi=170)
    plt.close(fig)


def main():
    a=parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    np.random.seed(a.seed)
    if a.dev_matrix:
        dev, held, dm, hm, status, transform_note = load_matrix(a)
        input_files=[a.dev_matrix, a.heldout_matrix, a.dev_meta, a.heldout_meta,
                     a.historical_dev_labels,a.historical_heldout_labels]
    else:
        dev, held, dm, hm, status, transform_note = load_raw(a)
        input_files=[a.raw_csv, a.split_manifest,a.historical_dev_labels,a.historical_heldout_labels]
    util_ix=np.array([i for i,c in enumerate(dev.columns) if c.startswith("util__")], dtype=int)
    clin_ix=np.array([i for i,c in enumerate(dev.columns) if c.startswith("clinical__")], dtype=int)
    if len(util_ix)!=3 or not len(clin_ix):
        raise ValueError("Must contain exactly three util__ features and >=1 clinical__ features")
    if any(c.startswith("lab__") for c in dev.columns):
        raise ValueError("This is an R2-only audit. Remove lab__ columns to avoid conflating R3 weights")
    if len(dev)!=len(dev.drop_duplicates()):
        print("INFO: duplicate feature rows allowed; patient rows must be verified from metadata")
    if len(held) and set(held.columns)!=set(dev.columns):
        raise ValueError("Heldout columns differ")
    if a.skip_heldout:
        held=held.iloc[0:0].copy()
    if not dm.empty and len(dm)!=len(dev):
        raise ValueError("Dev metadata not aligned")
    if not hm.empty and len(hm)!=len(held) and not a.skip_heldout:
        raise ValueError("Heldout metadata not aligned")
    manifest={
        "title": "R2 K-means specification sensitivity audit",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": status, "quick_smoke_test": a.quick, "n_development":len(dev),"n_heldout":len(held),
        "reproduction_policy":"Reported v11/v12 values are NOT automatically established by running this script",
        "scalers":a.scalers, "weights":a.weights, "n_independent_seeds_per_weight":a.seeds_per_weight,
        "audit_fit_n_init":a.audit_n_init, "util_reference_n_init":20,
        "weight_transform":a.weight_transform,"transform_note":transform_note,
        "seed":a.seed,"clinical_feature_count":len(clin_ix),"util_feature_count":len(util_ix),
        "feature_names":list(dev.columns),
        "source_script_sha256":sha256(Path(__file__)),
        "input_sha256":{str(q):sha256(q) for q in input_files if q is not None},
        "python":sys.version,"platform":platform.platform(),
        "numpy":np.__version__,"pandas":pd.__version__,"scikit_learn":sklearn.__version__,"scipy":scipy.__version__,
        "limits":"Results are exploratory; developer must reconcile 06B mapping, cohorts, scaling, weights, and seeds."
    }
    (a.out/"run_manifest.json").write_text(json.dumps(manifest,indent=2,default=str),encoding="utf-8")
    print(f"STATUS: {status}; dev={len(dev):,}, heldout={len(held):,}",flush=True)
    all_rows=[]; summaries=[]; rate_rows=[]; compare_rows=[]; cross_rows=[]; dist_rows=[]; base_rows=[]
    reps_by_scaler={}
    with threadpool_limits(limits=1):
        # The historical baseline used n_init=20. Run it separately instead of
        # claiming that a single baseline run reproduces historical experiments.
        for base_scaler in a.scalers:
            xb,zb=transforms(dev,held,base_scaler,1.0,util_ix,a.weight_transform)
            fitted=KMeans(n_clusters=2,n_init=20,random_state=a.seed,algorithm="lloyd")
            yb=fitted.fit_predict(xb)
            hb=fitted.predict(zb) if len(zb) else np.array([],dtype=int)
            urb,_=util_fit(dev,held,base_scaler,1.0,a.weight_transform,util_ix,a.seed)
            base_rows.append({"scaler":base_scaler,"weight":1.0,"n_init":20,"seed":a.seed,
                              "development_cluster_sizes_sorted":str(sorted(np.bincount(yb).tolist())),
                              "heldout_cluster_sizes_sorted":str(sorted(np.bincount(hb,minlength=2).tolist())) if len(hb) else "",
                              "development_silhouette_sampled":sampled_silhouette(xb,yb,a.silhouette_sample,a.seed),
                              "development_utilization_ari":adjusted_rand_score(yb,urb),
                              "inertia":float(fitted.inertia_),
                              **historical_ari(a.historical_dev_labels,dm,yb,"development"),
                              "heldout_historical_ari":historical_ari(a.historical_heldout_labels,hm,hb,"heldout")["historical_ari"] if len(hb) else np.nan})
        for scaler in a.scalers:
            reps={}
            for w in a.weights:
                x,z=transforms(dev,held,scaler,w,util_ix,a.weight_transform)
                ref_dev,ref_held=util_fit(dev,held,scaler,w,a.weight_transform,util_ix,a.seed)
                labels=[]; hlabs=[]; inertias=[]; util_aris=[]
                for i in range(a.seeds_per_weight):
                    # Each fit has its own independently seeded n_init=a.audit_n_init restarts.
                    seed=a.seed+100003*i
                    km=KMeans(n_clusters=2,n_init=a.audit_n_init,random_state=seed,algorithm="lloyd")
                    y=km.fit_predict(x)
                    yh=km.predict(z) if len(z) else np.array([],dtype=int)
                    labels.append(y)
                    hlabs.append(yh)
                    inertias.append(float(km.inertia_))
                    utilari=adjusted_rand_score(y,ref_dev)
                    util_aris.append(utilari)
                    all_rows.append({"scaler":scaler,"weight":w,"seed":seed,"n_init":a.audit_n_init,
                                     "inertia":km.inertia_,"silhouette_sampled":sampled_silhouette(x,y,a.silhouette_sample,seed),
                                     "util_ari":utilari,"n_cluster_0":int((y==0).sum()),
                                     "n_cluster_1":int((y==1).sum()),
                                     "heldout_util_ari": adjusted_rand_score(yh,ref_held) if len(z) else np.nan})
                full,disjoint,ci_lo,ci_hi=seed_interval(labels,a.bootstrap_reps,a.seed)
                ri=representative_index(inertias)
                rep=labels[ri]; hrep=hlabs[ri]
                reps[w]={"dev":rep,"held":hrep,"inertia":inertias[ri],"seed":a.seed+100003*ri,
                         "util_ref_dev":ref_dev,"util_ref_held":ref_held}
                sm={"scaler":scaler,"weight":w,"seed_ari_all_pairs":full,
                    "seed_ari_disjoint_mean":disjoint,"seed_ari_disjoint_ci_low":ci_lo,
                    "seed_ari_disjoint_ci_high":ci_hi,
                    "util_agreement_best_fit":float(util_aris[ri]),
                    "inertia_best":float(min(inertias)),"inertia_max":float(max(inertias)),
                    "inertia_range":float(max(inertias)-min(inertias)),
                    "best_fit_cluster_0":int((rep==0).sum()), "best_fit_cluster_1":int((rep==1).sum()),
                    "heldout_seed_ari_all_pairs":seed_interval(hlabs,a.bootstrap_reps,a.seed)[0] if len(z) else np.nan,
                    "silhouette_best_fit":sampled_silhouette(x,rep,a.silhouette_sample,a.seed),
                    "n_independent_fits":len(labels)}
                summaries.append(sm)
                if not dm.empty:
                    rr=group_rate_table(dm,rep,w,scaler,"development")
                    rate_rows+=rr
                    cmp=group_comparison(rr)
                    if cmp: compare_rows.append(cmp)
                if len(z) and not hm.empty:
                    rr=group_rate_table(hm,hrep,w,scaler,"heldout")
                    rate_rows+=rr
                    cmp=group_comparison(rr)
                    if cmp: compare_rows.append(cmp)
                print(f"{scaler:<8} w={w:.3f}: seed_ARI={full:.3f}, util_ARI={util_aris[ri]:.3f}, inertia=[{min(inertias):.3f}, {max(inertias):.3f}]",flush=True)
            reps_by_scaler[scaler]=reps
            for ii, w1 in enumerate(a.weights):
                for w2 in a.weights[ii+1:]:
                    a1,a2=reps[w1],reps[w2]
                    cross_rows.append({"scaler":scaler,"weight_1":w1,"weight_2":w2,
                                       "development_ari":adjusted_rand_score(a1["dev"],a2["dev"]),
                                       "development_changed_pct_aligned":100*aligned_changed(a1["dev"],a2["dev"]),
                                       "heldout_ari_fixed_development_models":adjusted_rand_score(a1["held"],a2["held"]) if len(held) else np.nan,
                                       "heldout_changed_pct_aligned":100*aligned_changed(a1["held"],a2["held"]) if len(held) else np.nan})
            # Distance percentages evaluated at baseline w=1, distinct pairs; clinical features unchanged
            xbase,_=transforms(dev,held,scaler,1.0,util_ix,a.weight_transform)
            frame=pd.DataFrame(xbase,columns=dev.columns)
            dist_rows.append({"scaler":scaler,"weight":1.0,**pair_distance(frame,a.pair_sample,a.seed)})
        if not a.skip_negative_control:
            # Joint shuffle preserves the 3-dimensional utilization profile but breaks patient pairing.
            rng=np.random.default_rng(a.seed)
            shuffle=rng.permutation(len(dev))
            for scaler in a.scalers:
                wx=1.0
                x,_=transforms(dev,held.iloc[0:0],scaler,wx,util_ix,a.weight_transform)
                u=x[:,util_ix].copy(); x[:,util_ix]=u[shuffle]
                km=KMeans(n_clusters=2,n_init=20,random_state=a.seed)
                y=km.fit_predict(x)
                uy=KMeans(n_clusters=2,n_init=20,random_state=a.seed).fit_predict(x[:,util_ix])
                dist_rows.append({"scaler":scaler,"weight":1.0,"experiment":"joint_utilization_shuffle_control",
                                  "full_vs_shuffled_util_only_ari":adjusted_rand_score(y,uy),
                                  "interpretation":"Breaks clinical/util pairing while keeping util rows intact"})

    write_csv(base_rows,a.out/"baseline_n_init20_reconciliation.csv")
    seed_df=write_csv(all_rows,a.out/"fit_level_seed_runs.csv")
    summary_df=write_csv(summaries,a.out/"weight_transition_summary.csv")
    cross_df=write_csv(cross_rows,a.out/"cross_weight_partition_agreement.csv")
    write_csv(rate_rows,a.out/"race_membership_by_weight.csv")
    write_csv(compare_rows,a.out/"race_descriptive_comparisons.csv")
    write_csv(dist_rows,a.out/"distance_and_negative_control.csv")
    plot_transition(summary_df,a.out)
    plot_inertia(seed_df,a.out)
    # Hard-coded previously published statements are REFERENCE CLAIMS only, never counted as confirmation.
    checks=[
        ("R2 full-versus-utilization-only ARI at robust w=1",1.0,1.0, "util_agreement_best_fit"),
        ("R2 mean seed ARI at robust w=0.600",.600,1.0,"seed_ari_all_pairs"),
        ("R2 mean seed ARI at robust w=0.700",.700,.363,"seed_ari_all_pairs"),
        ("R2 mean seed ARI at robust w=0.725",.725,1.0,"seed_ari_all_pairs"),
    ]
    validation=[]
    for label,w,ref,col in checks:
        candidate=summary_df[(summary_df.scaler=="robust") & (np.isclose(summary_df.weight,w))]
        observed=float(candidate.iloc[0][col]) if not candidate.empty else np.nan
        validation.append({"reported_manuscript_claim":label,"historical_reported_value":ref,
                           "current_run_value":observed,"absolute_difference":abs(observed-ref) if np.isfinite(observed) else np.nan,
                           "status":"NOT_COMPARABLE_UNTIL_PIPELINE_AND_SPLIT_RECONCILED",
                           "note":"A numerical match alone cannot prove same data/methodology; divergence may reflect changed specification."})
    write_csv(validation,a.out/"MANUSCRIPT_CLAIM_COMPARISON_NOT_VERIFICATION.csv")
    md=["# Reconstruction: reproducibility and evidence report", "",
        f"**Status:** `{status}`",f"**Execution type:** {'quick test, not publishable' if a.quick else 'extended audit, subject to 06B reconciliation'}",
        f"**Sample:** {len(dev):,} development; {len(held):,} held-out.",
        f"**Weights:** {', '.join(f'{w:g}' for w in a.weights)}. **Independent seeds:** {a.seeds_per_weight} per specification (n_init={a.audit_n_init} each).",
        f"**Numerical-weight convention:** `{a.weight_transform}`. **Scaler(s):** {', '.join(a.scalers)}.","",
        "## Data and interpretation", "",
        "This script does not establish identity with original notebook preprocessing or sampling. "
        "Do not replace manuscript figures/numbers until the 06B artifacts are reconciled, all runs complete, "
        "and findings checked for alternate mechanisms.","",
        "## Outputs", "",
        "- `baseline_n_init20_reconciliation.csv`: historical baseline-style KMeans with n_init=20; comparison with original notebook labels if supplied.",
        "- `fit_level_seed_runs.csv`: fit-level inertia, silhouette, clusters, utilization agreement.",
        "- `weight_transition_summary.csv`: full mean pairwise seed ARI, independent disjoint-pair CI, inertia range.",
        "- `cross_weight_partition_agreement.csv`: comparisons between selected lowest-inertia solutions at distinct weights, on development and held-out patients.",
        "- `race_membership_by_weight.csv`: descriptive cluster composition with target defined by *higher observed prior-use count*.",
        "- `race_descriptive_comparisons.csv`: unadjusted Fisher comparisons; no causal or fairness conclusions.",
        "- `distance_and_negative_control.csv`: pairwise squared distance decomposition and shuffled-utilization control.",
        "- `MANUSCRIPT_CLAIM_COMPARISON_NOT_VERIFICATION.csv`: numerical comparison against cited historical values, never automatic verification.",
        "- `fig_weight_transition_*.png`, `fig_competing_solutions_w0700.png`: exploratory illustrations.",
        "- `run_manifest.json`: input hashes, source mapping, package versions, seeds, and settings.", "",
        "## Important measurement distinctions", "",
        f"1. Seed ARI compares independently seeded fits with `n_init={a.audit_n_init}`. Historical comparability still depends on identical patient split, features, weighting, and scaling.",
        "2. The disjoint-seed-pair bootstrap CI represents algorithmic initialization variation, **not** patient sampling uncertainty.",
        "3. Cross-weight patient changes use optimal label alignment; inertia values should be compared **only within the same weight/scaling specification**.",
        "4. Held-out comparisons apply development-fitted models; they do not show independently refitted held-out transitions.",
        "5. Cluster labels are arbitrary. Demographic comparisons use the cluster with higher *observed utilization counts* at each weight; group identity may still change.",
        "6. Numerical weight multiplies input coordinates directly unless `--weight-transform sqrt` is chosen; the 06B convention must be verified.",
        "7. The original ICD-9 mapping, age treatment, selected patient IDs, and preprocessing were **not** provided. Raw-mode output is therefore a **new sensitivity analysis**, not a reproduction.", "",
        "## Checks required before citation", "",
        "- Compare development patient IDs, encounter selection, mapped diagnosis families, age preprocessing, encoded feature columns, utilization scaling, and 06B weight convention.",
        "- Compare baseline R2 K=2 cluster sizes and ARI against saved 06B labels. Supply `--historical-dev-labels` if original assignments are available.",
        "- Confirm patient-disjoint held-out cohort and whether patient IDs reflect the published 10,000/61,518 split.",
        "- Examine inertia/partition scatter at w=0.700; do not declare near-equal local minima without fit-level evidence.",
        "- Report negative findings and any material differences rather than editing them away.",
    ]
    (a.out/"EVIDENCE_REPORT.md").write_text("\n".join(md)+"\n",encoding="utf-8")
    print("\nWrote results to",a.out.resolve(), "\nReview EVIDENCE_REPORT.md and run_manifest.json before citing anything.",flush=True)


if __name__=="__main__":
    main()
