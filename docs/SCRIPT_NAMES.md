# Script and notebook names

Names describe their responsibility. The analysis remains in the original scripts;
we did not split it into new modules or change the clustering experiments.

| Old name | Current name | Responsibility |
|---|---|---|
| `scripts/JBHI_Figure3_HighResolution.py` | `scripts/plot_weight_sensitivity.py` | Existing code or notebook, renamed. |
| `scripts/JBHI_characterize_weight060.py` | `scripts/characterize_weighted_clusters.py` | Existing code or notebook, renamed. |
| `scripts/JBHI_sectionV_audit_visuals.py` | `scripts/build_audit_report.py` | Existing code or notebook, renamed. |
| `scripts/JBHI_validation_tracker_v3.py` | `scripts/validate_experiment_evidence.py` | Existing code or notebook, renamed. |
| `scripts/jbhi_06C_validation_v2.py` | `scripts/analyze_weight_scaling.py` | Existing code or notebook, renamed. |
| `scripts/jbhi_06D_comparators.py` | `scripts/compare_clustering_stability.py` | Existing code or notebook, renamed. |
| `scripts/check_release_evidence.py` | `scripts/verify_release_files.py` | Existing code or notebook, renamed. |
| `notebooks/local-runners/Run_JBHI_SectionV_Visualizations.ipynb` | `notebooks/local-runners/run_audit_report.ipynb` | Existing code or notebook, renamed. |
| `notebooks/local-runners/Run_JBHI_Validation_Tracker_v3.ipynb` | `notebooks/local-runners/run_evidence_validation.ipynb` | Existing code or notebook, renamed. |
| `notebooks/local-runners/Run_JBHI_Weight060_Characterization.ipynb` | `notebooks/local-runners/run_weight_characterization.ipynb` | Existing code or notebook, renamed. |
| `evidence/reconstruction/JBHI_06C_ninit1_vs_ninit20_comparison.csv` | `evidence/reconstruction/restart_budget_comparison.csv` | Saved result or figure; content unchanged. |
| `figures/example/JBHI_Fig3_Weight_Sweep_TwoPanel.png` | `figures/example/weight_sensitivity_overview.png` | Saved result or figure; content unchanged. |
| `figures/example/JBHI_Fig3_Weight_Sweep_TwoPanel.svg` | `figures/example/weight_sensitivity_overview.svg` | Saved result or figure; content unchanged. |

Historical file names inside `evidence/earlier_sha256_snapshot.csv` and
`evidence/source_hashes.csv` are retained as records of their original sources.
The current-file hashes are generated in `evidence/release_file_sha256.csv`.

The update uses a normal Git commit and push. Previously published commits still
retain earlier names in repository history; no force push or history rewrite is needed.
