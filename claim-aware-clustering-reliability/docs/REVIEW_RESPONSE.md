# Repository review changes, October 2026

This revision addresses the repository review without changing the numerical experiments or splitting the original scripts into a new framework package.

| Review point | Change | Remaining work |
|---|---|---|
| Missing source for several main findings | Added a manuscript-to-source map (`REPRODUCIBILITY.md`) and a source-recovery list (`PREPUBLICATION_GAPS.md`). | The primary masking results, original notebooks 01-03/05, executed 120-run exports, and original manifests still need to be supplied. |
| Unclear provenance and missing hashes | Kept original source hashes, labeled prior and current packaged script hashes separately, and generated a fresh release-candidate manifest. | Reconcile the manuscript's 17-artifact claim with the actual original inventory. |
| No tests | Added five small arithmetic and scaler controls. | These are not a replacement for end-to-end cohort reproduction. |
| Version-free dependencies | Pinned the versions used in release-candidate checks. | Restore historical packages from the original experiment manifests. |
| License/citation | Added MIT `LICENSE` and `CITATION.cff`. | Confirm ownership, final author list and rights before publishing. |
| Notebook paths and project structure | Kept the notebooks intact as research history; made the tracker defaults repository-relative while preserving explicit path flags. | Original notebook 04 includes historical local-path code and must not be treated as a single canonical production pipeline. |
| Documentation voice and code names | Rewrote README in a consistent research-author voice and corrected the selected-weight script's introductory documentation. Kept existing script filenames because local notebooks invoke them. | Broad automated formatting and version suffix renaming were not performed; neither is needed to reproduce the included aggregate reporting. |
| Unreported 06D comparators | Labeled 06D as optional and not yet evaluated in the paper. | Run against the actual primary features, verify the metric implementation and only then consider manuscript changes. |
| Evidence register mismatches | Corrected E11 because `fit_level_seed_runs_ninit1.csv` is already included; corrected E13's rounded lower Wald bound from .28 to .27 based on direct rounding of 0.2749798. | Reconcile the manuscript if its printed interval differs. |
| Zenodo publication | Added a step-by-step signup, GitHub integration, and direct draft guide. | Do not publish a complete-reproduction claim until the missing files have been verified. |

The original 06C, 06D, visualization, primary 06B, and historical notebook code paths were not algorithmically rewritten. One existing validation tracker receives repository-relative default source locations, which can still be overridden using the same flags.
