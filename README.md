# Claim-aware reliability audit for patient clustering

This repository accompanies a study of reliability assessment in healthcare clustering using the UCI Diabetes 130-US Hospitals dataset. We examine five aspects of clustering evidence: internal quality, initialization and subsample stability, missing-data robustness, feature-domain dependence, and sensitivity to analytical specifications.

K-means is held fixed as an experimental control. Its centroid geometry makes feature weighting, masking and partition changes directly inspectable. The work does not introduce a new clustering algorithm or establish clinically validated phenotypes.

## What the experiments show

The primary analysis and the reconstruction are separate studies of the same source dataset, with different patient splits and diagnosis encodings. They must not be compared as matched-patient replications.

* **Primary study:** Some clusters are reproducible across random seeds yet sensitive to missing diagnosis information. The primary clinical-only R1 K=2 partition matches a position-specific diagnosis-family indicator; a clinical-plus-utilization partition matches independently fitted utilization-only clustering.
* **Reconstruction:** Under robust scaling, changing the utilization-coordinate weight from 0.600 to 1.000 selects substantially different partitions. The selected development partitions have an adjusted Rand index of about -0.002 and differ for 33.7% of records after label alignment.
* **Audit record:** Each result is linked to its evidence source, intended interpretation and outstanding validation work. The framework does not collapse all diagnostics into one score.

## Repository layout

| Folder | Purpose |
|---|---|
| `scripts/` | Clustering analysis, diagnostic validation, figure generation, and release checks. |
| `notebooks/` | Available primary 06B notebook, historical notebook 04, and local runners. |
| `evidence/primary/` | Saved aggregate results from the primary 06B study. |
| `evidence/reconstruction/` | 06C reconstruction fit-level and aggregate results. |
| `evidence/` | Evidence register, original-source hash history, and a manifest of files in this package. |
| `figures/example/` | Published-form Figure 3 example from the saved 06B tables. |
| `tests/` | Small checks of partition alignment, scaling behavior, and evidence arithmetic. |
| `docs/` | Study-to-source mapping, execution notes, and GitHub/Zenodo release instructions. |

## Run the included reporting code

From PowerShell in the repository folder:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts\plot_weight_sensitivity.py
.\.venv\Scripts\python.exe scripts\verify_release_files.py
```

These commands check the packaged evidence and regenerate Figure 3. They do not reproduce the complete cohort creation or all patient-level experiments. Source data must be downloaded independently and the original run configurations restored to reproduce the patient-level results.

The optional 06D comparison script implements bootstrap-refit Jaccard and prediction strength. It has **not** been established as manuscript evidence and should not be used to add Table II numbers until executed and reviewed under the primary preprocessing specification.

## Evidence and reproducibility

Start with [the reproducibility map](docs/REPRODUCIBILITY.md) and [the review change log](docs/REVIEW_RESPONSE.md). It identifies the input and output for each main paper table or figure and distinguishes provided aggregate results from missing original source records. This is a partial public-code package, **not** yet an end-to-end replication of all manuscript claims.

The original notebooks 01-03 and 05, primary 20-run and held-out masking outputs, the raw 120-run masking exports, and the execution manifest inventory are not yet included. Copies cannot be reconstructed from rounded manuscript numbers. See [release blockers](docs/PREPUBLICATION_GAPS.md) before publishing a release under a DOI.

We do not distribute the raw UCI download, MIMIC-IV records, patient-level extracts or private credentials. Files in `evidence/primary/` and `evidence/reconstruction/` are aggregate research outputs. The manuscript's clinical readmission comparisons are descriptive and unadjusted.

## Release and citation

The repository has an MIT software license and a `CITATION.cff` template populated with the currently documented contributor. Before any public release, confirm that the final author list and licensing permissions are correct, and that third-party material is not covered by the code license inadvertently.

For a durable citation, publish a versioned GitHub release and archive it with Zenodo. [Zenodo instructions](docs/ZENODO_RELEASE_GUIDE.md) cover both GitHub integration and a direct-upload draft. A DOI should be added to the manuscript only after publication of a verified archive.

## Script names

Script filenames describe their purpose and use lowercase `snake_case`.
Experiment labels such as R1, R2, 06B and 06C remain in the scientific evidence
because they distinguish the original analysis from subsequent reconstructions.
See [the filename map](docs/SCRIPT_NAMES.md) for renamed files and notebook runners.

If this repository was already pushed to GitHub, use `update_existing_repo.ps1`
from this download to apply the rename as a new commit. Git history is preserved.
