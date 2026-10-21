# Items to resolve before the citable release

The sources below were not present in the supplied package. Do not recreate them from manuscript numbers.

1. Recover original executable notebooks `01_environment_and_data_check.ipynb`, `02_diabetes_cohort.ipynb`, `03_clustering_feature_preparation.ipynb`, and `05_differentiated_reliability_analyses_FIXED.ipynb` from the Windows project directory. Preserve original SHA-256 hashes. The available HTML rendering of notebook 05 is not an executable replacement.
2. Add the executed primary `step25f_harmonized_missingness_runs.csv`, held-out diagnosis-masking exports, R1/diagnosis-position-2 indicator check, and primary candidate Table II evidence. Keep any patient-level raw material private unless redistribution is permitted.
3. Add the **executed** `NEW_reconstruction_position_masking_summary.csv`, the 120-run detailed outputs and model/run manifest. The existence of a script that can conduct the experiment is not proof that the saved run is included.
4. Collect original preprocessing, cohort split and software environment manifests for both streams. Do not use the release-candidate environment as the original experiment environment.
5. Resolve the paper's statement about *17 core SHA-256 artifacts*: either publish the original 17-artifact inventory or revise Section IV-G to describe only the hashes actually deposited.
6. Confirm whether the Section V/Table V R2 Wald lower confidence bound should be reported as **0.27 pp** (direct rounding of 0.2749798 pp) rather than **0.28 pp** (rounding after an intermediate 0.275 approximation). The register uses 0.27 pp. Document the selected precision in the manuscript.
7. Do not add 06D prediction-strength/Jaccard values to the manuscript until 06D runs against the intended primary representation, with a reviewed method and reproducible output. Its presence in `scripts/` is an option, not completed evidence.
8. Confirm the research author list, rights to license the code, third-party data licenses, and ethics wording before making the repository public.

A GitHub repository can be private while these items are collected. A Zenodo **draft** can also be started. Avoid publishing the current package as a complete replication archive.
