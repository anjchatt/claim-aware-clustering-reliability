# ICD-9 diagnosis-family mapping used in the primary cohort

The primary cohort-construction notebook (`02_diabetes_cohort.ipynb`) defines `icd9_family`. It compares the **full parsed numeric value** with inclusive interval limits. It does not truncate to the first three digits. The independently reconstructed R2 experiment uses a different mapping, including a separate diabetes family; keep the two specifications separate.

| Family | Input code |
| --- | --- |
| infectious | 1–139 |
| neoplasm | 140–239 |
| endocrine_metabolic | 240–279, including 250.x diabetes |
| blood | 280–289 |
| mental | 290–319 |
| nervous_sensory | 320–389 |
| circulatory | 390–459 |
| respiratory | 460–519 |
| digestive | 520–579 |
| genitourinary | 580–629 |
| pregnancy | 630–679 |
| skin | 680–709 |
| musculoskeletal | 710–739 |
| congenital | 740–759 |
| perinatal | 760–779 |
| symptoms_signs | 780–799 |
| injury_poisoning | 800–999 |
| supplementary_V | Prefix `V` |
| external_E | Prefix `E` |
| missing | Null, `?`, empty string, literal `nan` |
| other | Failed numeric conversion or no matching interval |

The researcher's source-data audit covered 101,766 UCI encounters and 305,298 entries across three diagnosis positions. It reported 1,802 missing values, no values mapped to `other`, and no fractional values immediately above an upper interval boundary. These are **source-encounter counts**, not separately recomputed results for the 71,518-patient index cohort.

The original notebook and its source hash should remain in the provenance inventory. Do not silently change its diagnosis grouping when reproducing the manuscript's original R1/R2/R3 results.
