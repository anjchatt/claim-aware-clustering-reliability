# Publishing this study's code with Zenodo

## Create an account (no existing Zenodo account needed)

1. Open https://zenodo.org and select **Sign up**.
2. Select **Sign up with GitHub** if the GitHub account holding this repository is already available. Authorize the Zenodo application. Alternatively, sign up using ORCID, or create a local account and confirm the email address.
3. Sign in and open your profile. Under **Linked accounts**, connect GitHub if it was not linked during sign-up.
4. Use https://sandbox.zenodo.org for a practice deposit, but note that it is a separate test service and does not issue the production archive you will cite.

## Recommended: GitHub release archived by Zenodo

Complete `docs/PREPUBLICATION_GAPS.md` and check the exact release content before making it public.

1. Push the reviewed repository to GitHub, initially as **Private**. Check for private datasets, patient IDs, keys, or local artifacts. The GitHub README is already included.
2. Confirm the `CITATION.cff` author list and `LICENSE` selection. Run `python -m unittest discover -s tests -v` and `python scripts/check_release_evidence.py`.
3. When ready for a public software release, make the reviewed repository public. In Zenodo, select the profile menu and **GitHub**, click **Sync now**, locate the repository, and switch its integration on.
4. On GitHub, create a release from a reviewed commit: **Releases > Draft a new release**. Use a tag such as `v1.0.0`, enter a short description of the included experiments and remaining limitations, and publish the release. Zenodo processes enabled GitHub releases and creates the archived software record.
5. Inspect the Zenodo record, including the authors, license, files, version, and DOI. Add that versioned DOI to the manuscript after the Zenodo record is live. Do not invent a DOI in `CITATION.cff` beforehand.

The integration is designed for published software releases. A GitHub push or a draft release is not itself a Zenodo publication.

## Alternative: create a direct Zenodo draft now

1. Sign in at https://zenodo.org; select the **+** menu, then **New upload**.
2. Upload the ZIP only when its contents are reviewed, or create the metadata first. Set **Resource type** to **Software**; use a research code title consistent with `CITATION.cff`.
3. Add the actual author(s), a description, keywords, software version, and a suitable license. The code may be MIT-licensed if the author confirms they hold the relevant rights. Set visibility explicitly.
4. Click **Save draft**. You may reserve a DOI in the draft with **Get a DOI now**, but it is registered only on publication.
5. Do not select **Publish** until the release check and evidence reconciliation are complete. After publishing, use new versions for substantive changes rather than silently replacing the cited source archive.

**Suggested description for a future deposit:**

> Code and supporting evidence for a claim-aware clustering reliability audit using the UCI Diabetes 130-US Hospitals dataset. The archive separates the primary patient clustering study from a reconstruction examining diagnosis missingness, feature-domain dependence, utilization weighting, and optimization stability. The accompanying evidence map identifies the provenance and validation status of each manuscript result. The archive does not contain restricted clinical data or claim independent clinical phenotype validation.

After completing the missing sources, update the description to match exactly what the final upload contains.

Official documentation: https://help.zenodo.org/docs/get-started/create-an-account/ ; https://help.zenodo.org/docs/github/enable-repository/ ; https://help.zenodo.org/docs/github/archive-software/github-upload/ ; https://help.zenodo.org/docs/deposit/create-new-upload/
