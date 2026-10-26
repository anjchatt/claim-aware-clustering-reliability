"""R3 full-feature versus utilization-only clustering.

Fit models on the saved development cohort and compare assignments on the
61,518 held-out patients. The laboratory-observation flags are checked
against the saved development features before any models are fitted.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

CATEGORICAL_COLUMNS = ["age", "diag1_family", "diag2_family", "diag3_family"]
UTILIZATION_COLUMNS = ["number_outpatient", "number_emergency", "number_inpatient"]
LAB_COLUMNS = ["A1Cresult_observed", "max_glu_serum_observed"]
RAW_LAB_COLUMNS = ["A1Cresult", "max_glu_serum"]
PATIENT_ID = "patient_nbr"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def lab_test_recorded(series: pd.Series) -> pd.Series:
    # UCI lab results use 'None' for tests that were not performed.
    value = series.astype("string").str.strip().str.lower()
    absent = value.isna() | value.isin(["", "none", "?", "nan", "null", "na"])
    return (~absent).astype("int8")


def normalize_patient_id(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip()


def derive_lab_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for dest, source in zip(LAB_COLUMNS, RAW_LAB_COLUMNS):
        out[dest] = lab_test_recorded(df[source])
    return out


def raw_data_locations(project: Path) -> list[Path]:
    return [project / "data" / "raw" / "diabetic_data.csv",
            project / "data" / "diabetic_data.csv",
            project / "diabetic_data.csv"]


def attach_lab_flags(cohort: pd.DataFrame, project: Path, raw_path: Path | None):
    """Add flags to the index-patient table without changing its rows or order."""
    cohort = cohort.copy()
    if set(LAB_COLUMNS).issubset(cohort.columns):
        return cohort, "existing cohort binary flags", None
    if set(RAW_LAB_COLUMNS).issubset(cohort.columns):
        for key, value in derive_lab_flags(cohort).items():
            cohort[key] = value
        return cohort, "cohort raw test categories", None

    if raw_path is None:
        raw_path = next((p for p in raw_data_locations(project) if p.is_file()), None)
    if raw_path is None or not raw_path.is_file():
        print("Available cohort fields:", list(cohort.columns))
        raise FileNotFoundError(
            "The index-cohort parquet has neither raw lab-result categories nor R3 flags. "
            "Supply the ORIGINAL UCI diabetic_data.csv with --raw-csv. "
            "Search your computer: Get-ChildItem C:\\Projects -Recurse -Filter diabetic_data.csv"
        )

    source = pd.read_csv(raw_path, low_memory=False)
    required = {PATIENT_ID, "encounter_id", *RAW_LAB_COLUMNS}
    missing = required - set(source.columns)
    if missing:
        raise ValueError(f"Raw UCI file missing required columns: {sorted(missing)}")

    # Select the minimum encounter PATIENT_ID for each patient, as in cohort construction.
    source[PATIENT_ID] = normalize_patient_id(source[PATIENT_ID])
    source["encounter_id"] = pd.to_numeric(source["encounter_id"], errors="raise").astype("int64")
    if source["encounter_id"].duplicated().any():
        raise ValueError("The raw source has duplicate encounter_id values")
    first = (source.sort_values("encounter_id", kind="stable")
                   .drop_duplicates(PATIENT_ID, keep="first")
                   [[PATIENT_ID, "encounter_id", *RAW_LAB_COLUMNS]])

    # Check encounter selection when the index cohort retains encounter_id.
    cohort[PATIENT_ID] = normalize_patient_id(cohort[PATIENT_ID])
    if cohort[PATIENT_ID].duplicated().any():
        raise ValueError("Duplicate patient_nbr in index cohort")
    lookup = first.set_index(PATIENT_ID)
    aligned = lookup.reindex(pd.Index(cohort[PATIENT_ID]))
    if aligned["encounter_id"].isna().any():
        count = int(aligned["encounter_id"].isna().sum())
        raise ValueError(f"Cannot locate {count} index patients in the original UCI CSV")
    if "encounter_id" in cohort.columns:
        source_enc = aligned["encounter_id"].to_numpy(dtype="int64")
        cohort_enc = pd.to_numeric(cohort["encounter_id"], errors="raise").to_numpy(dtype="int64")
        if not np.array_equal(source_enc, cohort_enc):
            raise ValueError("Index cohort encounter_id does not match raw minimum encounter_id")
    aligned.index = cohort.index  # Restore the index cohort's row order
    for key, value in derive_lab_flags(aligned).items():
        cohort[key] = value
    return cohort, "index encounter from original UCI CSV", raw_path


def r3_preprocessor() -> ColumnTransformer:
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
    ])
    return ColumnTransformer([("cat", categorical, CATEGORICAL_COLUMNS),
                              ("num", numeric, UTILIZATION_COLUMNS + LAB_COLUMNS)])


def utilization_preprocessor() -> Pipeline:
    return Pipeline([("imputer", SimpleImputer(strategy="median")),
                     ("scaler", RobustScaler())])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--raw-csv", type=Path, default=None,
                        help="Original UCI diabetic_data.csv; used if index cohort has no lab columns")
    parser.add_argument("--output", type=Path, default=None)
    a = parser.parse_args()
    project = a.project.resolve()
    folder = project / "data" / "processed"
    dev_path = folder / "development_R3_10k.parquet"
    cohort_path = folder / "uci_diabetes_index_patient_features_v2.parquet"
    if not dev_path.is_file() or not cohort_path.is_file():
        raise FileNotFoundError(f"Expected original files:\n  {dev_path}\n  {cohort_path}")
    dev = pd.read_parquet(dev_path)
    cohort = pd.read_parquet(cohort_path)
    print("Development rows:", len(dev), "Index cohort rows:", len(cohort))
    print("Available cohort fields:", list(cohort.columns))
    cohort, source_desc, raw_path = attach_lab_flags(cohort, project, a.raw_csv)
    print("Laboratory flags created from:", source_desc)

    required = {PATIENT_ID, *CATEGORICAL_COLUMNS, *UTILIZATION_COLUMNS, *LAB_COLUMNS}
    for name, frame in [("Development R3", dev), ("Index cohort", cohort)]:
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} missing columns: {sorted(missing)}")
        if frame[PATIENT_ID].isna().any() or normalize_patient_id(frame[PATIENT_ID]).duplicated().any():
            raise ValueError(f"{name} has missing or duplicated patient IDs")

    # The saved R3 development cohort is the reference for the lab flags.
    dev = dev.copy()
    dev[PATIENT_ID] = normalize_patient_id(dev[PATIENT_ID])
    cohort[PATIENT_ID] = normalize_patient_id(cohort[PATIENT_ID])
    indexed = cohort.set_index(PATIENT_ID, verify_integrity=True)
    matched = indexed.reindex(pd.Index(dev[PATIENT_ID]))
    if matched[LAB_COLUMNS].isna().any().any():
        raise ValueError("Development patient IDs do not all appear in the index cohort")
    # Development and index-cohort data must use identical
    # clinical and utilization features for the same patients.
    for feature in CATEGORICAL_COLUMNS + UTILIZATION_COLUMNS:
        if feature in CATEGORICAL_COLUMNS:
            left = dev[feature].astype("string").fillna("<missing>").str.strip().to_numpy()
            right = matched[feature].astype("string").fillna("<missing>").str.strip().to_numpy()
            mismatch = int(np.count_nonzero(left != right))
        else:
            left = pd.to_numeric(dev[feature], errors="coerce").to_numpy(dtype=float)
            right = pd.to_numeric(matched[feature], errors="coerce").to_numpy(dtype=float)
            mismatch = int(np.count_nonzero(~np.isclose(left, right, rtol=0, atol=1e-10, equal_nan=True)))
        if mismatch:
            raise ValueError(f"Feature {feature} differs between development R3 and the "
                             f"index cohort for {mismatch} development patients. "
                             "Check cohort version or feature definitions before comparing models.")
    print("Development clinical/utilization feature validation: PASS")
    for feature in LAB_COLUMNS:
        originally_saved = pd.to_numeric(dev[feature], errors="raise").to_numpy()
        reconstructed = pd.to_numeric(matched[feature], errors="raise").to_numpy()
        mismatch = int(np.count_nonzero(originally_saved != reconstructed))
        print(f"Development {feature} validation: {mismatch} mismatches / {len(dev)}")
        if mismatch:
            raise ValueError("Recreated laboratory flags differ from saved development_R3_10k. "
                             "Do not report a held-out ARI until the source mapping is reconciled.")

    dev_ids = set(dev[PATIENT_ID])
    heldout = cohort.loc[~cohort[PATIENT_ID].isin(dev_ids)].copy()
    if len(dev) != 10000 or len(heldout) != 61518 or len(cohort) != 71518:
        raise ValueError(f"Expected 10000/61518/71518; got {len(dev)}/{len(heldout)}/{len(cohort)}")
    print("Patient split PASS: development=10000 heldout=61518, no overlap")

    prep = r3_preprocessor()
    x_dev = prep.fit_transform(dev)
    x_held = prep.transform(heldout)
    zmodel = KMeans(n_clusters=2, random_state=42, n_init=20, algorithm="lloyd")
    zd = zmodel.fit_predict(x_dev)
    zh = zmodel.predict(x_held)

    utilprep = utilization_preprocessor()
    u_dev = utilprep.fit_transform(dev[UTILIZATION_COLUMNS])
    u_held = utilprep.transform(heldout[UTILIZATION_COLUMNS])
    umodel = KMeans(n_clusters=2, random_state=42, n_init=20, algorithm="lloyd")
    ud = umodel.fit_predict(u_dev)
    uh = umodel.predict(u_held)

    dev_ari = float(adjusted_rand_score(zd, ud))
    held_ari = float(adjusted_rand_score(zh, uh))
    print(f"R3 development full-vs-utilization ARI: {dev_ari:.6f}")
    print(f"R3 held-out full-vs-utilization ARI: {held_ari:.6f}")
    if not np.isclose(dev_ari, 1.0, atol=1e-9):
        raise ValueError("Development ARI differs from the reported 1.000. Check the original "
                         "preprocessing and fit settings; results were NOT saved.")

    out = (a.output or project / "results" / "r3_heldout_domain_equivalence_summary.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "representation": "R3", "k": 2,
        "development_patients": len(dev), "heldout_patients": len(heldout),
        "development_ari": dev_ari, "heldout_ari": held_ari,
        "seed": 42, "n_init": 20, "algorithm": "lloyd",
        "lab_flag_source": source_desc,
        "development_r3_sha256": sha256(dev_path),
        "index_cohort_sha256": sha256(cohort_path),
        "raw_csv_sha256": sha256(raw_path) if raw_path is not None else "",
        "scikit_learn_version": sklearn.__version__,
    }]).to_csv(out, index=False)
    print("Aggregate verification saved:", out)
    print("No patient-level assignments were exported.")


if __name__ == "__main__":
    main()
