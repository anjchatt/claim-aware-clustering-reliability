# GitHub/Zenodo release checklist

- [ ] Review `docs/PREPUBLICATION_GAPS.md`, recover missing executed source and masks, and reconcile the 17-artifact claim.
- [ ] Check the primary and reconstruction experiment identities separately. Record original software versions and source data license.
- [ ] Run `python -m unittest discover -s tests -v` and `python scripts/verify_release_files.py`.
- [ ] Confirm all included research outputs have the appropriate evidence status; manuscript-reported CSV entries are not raw reruns.
- [ ] Confirm authorship in `CITATION.cff` and that the MIT license is appropriate for the original research code.
- [ ] Review notebook code, outputs, metadata and paths for private information. Do not include raw UCI/MIMIC clinical records or patient-level files by default.
- [ ] Check Table V CI rounding, Section IV-G hash inventory, and agreement between data-availability statement and actual published files.
- [ ] Push the checked revision to a private GitHub repository first.
- [ ] When ready, make a tagged public GitHub release, enable Zenodo integration, and inspect the Zenodo DOI record.
- [ ] Update the manuscript to cite the **actual published DOI**, and record the corresponding Git tag and Git commit hash.

Releasing reproducibility material is separate from obtaining IRB determinations and clinical validation. No software release by itself establishes clinical utility.
