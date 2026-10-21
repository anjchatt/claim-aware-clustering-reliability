# Reproducibility map

This table connects manuscript claims with the files currently available. A saved aggregate CSV can verify a plotted number, but it is not the same as reproducing an experiment from patient-level inputs.

| Manuscript item | Available source | Runnable entry point | Status and outstanding source |
|---|---|---|---|
| Fig. 1 and Table I: audit architecture | Manuscript definitions and `evidence/claim_aware_evidence_records.csv` | `scripts/build_audit_report.py` | Framework example, not clinically calibrated. |
| Table II: five primary candidates | `evidence/primary_candidates_MANUSCRIPT_TABLE_II_ONLY.csv` | Audit reporting script loads reported values | **Reported**, not recomputed. Original development fit summaries and source notebooks needed. |
| Fig. 2: held-out fixed-model masking | Manuscript summary only | Original primary notebook 05 not included | **Missing** run-level masking and manifest. |
| Primary R1 K=2 indicator equivalence | Manuscript summary only | `scripts/validate_experiment_evidence.py` with local source Parquet files | **Missing** source profile, original labels, and verification export. |
| Primary 20-run refit masking | Manuscript summary only | `scripts/validate_experiment_evidence.py` with local results | **Missing** `step25f_harmonized_missingness_runs.csv`. |
| Fig. 3: primary R2 weighting sweep | `evidence/primary/06B_weight_sweep.csv`, `06B_seed_stability.csv` | `python scripts/plot_weight_sensitivity.py` | **Replots saved aggregates**; original feature matrices and notebook environment needed for refit. |
| Table III: feature concentration / scaling mechanism | `evidence/primary/06B_scaler_audit.csv`, `06B_squared_distance.csv`, `06B_scaler_K.csv` | Primary 06B notebook | **Saved aggregate results**; exact source input not bundled. |
| Table IV and Figs. 4-5: reconstruction | `evidence/reconstruction/fit_level_seed_runs_ninit1.csv`, `fit_level_seed_runs_ninit20_REFERENCE.csv`, `cross_weight_partition_agreement_ninit20_REFERENCE.csv`, `distance_and_negative_control.csv` | `scripts/analyze_weight_scaling.py` | **Fit-level and summary evidence present**; exact original split/manifest not bundled. Reconstruction is not the primary cohort. |
| 120-run diagnosis-position masking | Manuscript numerical summary only | `scripts/validate_experiment_evidence.py --run-reconstruction-masking` with local input | **Missing** executed summary and 120-run detail CSVs, selected model metadata. |
| Table V: 30-day readmission | Event counts reported in manuscript and register | `python -m unittest discover -s tests` checks Wald arithmetic | **Arithmetic check only**; no outcome-level source export bundled. Unadjusted descriptive result. |
| Table VI: evidence-to-review record | `evidence/claim_aware_evidence_records.csv`, audit reporting script | `scripts/build_audit_report.py` | **Reporting demonstration**; not a validated clinical decision system. |

Original-file names in the historical SHA-256 tables are retained intentionally. The original-file hash history lives in `evidence/source_hashes.csv`. Package hashes live in `evidence/release_file_sha256.csv` and are checked using `scripts/verify_release_files.py`. These are distinct records; the package does not assert that its file list is the 17-item core artifact inventory described in the manuscript.

### Commands from the repository root

```powershell
# Quick mechanical tests, no patient data required.
python -m unittest discover -s tests -v

# Validate file hashes and see which source evidence remains outstanding.
python scripts/verify_release_files.py

# Regenerate Figure 3 using the supplied primary aggregate CSVs.
python scripts/plot_weight_sensitivity.py

# Optional inspection of saved reconstruction values; do not describe as a full rerun.
python scripts/build_audit_report.py --help
```

### Historical environment

`requirements.txt` pins the dependencies used to check this *release candidate* with Python 3.13. It does not establish the package versions used during the original primary and reconstruction executions. Those must be recorded from the corresponding historical manifests before claiming exact reproducibility.
