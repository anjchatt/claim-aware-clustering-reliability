# Results and provenance notes

These measurements came from the researcher's local executions and manuscript review. The ZIP contains code, not copies of the researcher's local result files.

## R1 position-independent sensitivity

Saved original local output: `results/multihot_r1_sensitivity/multihot_r1_sensitivity_summary.json`.

| Metric | Positional control (new fit) | Multi-hot (new fit) |
| --- | ---: | ---: |
| Seed agreement ARI | 0.976899 | 1.000000 |
| Indicator agreement ARI | 1.000000, position-2 Circulatory | 1.000000, Circulatory in any position |
| Refitted 20% masking mean ARI | 0.164985 | 0.739794 |
| Cross-encoding ARI | — | 0.213334 |

The **historical** R1 positional masking mean (0.178) used another masking schedule. Do not replace it with 0.164985 or interpret the two values as identical-protocol repeats. ARI 0.213334 is adjusted partition agreement, not a percent of unchanged patients.

## R3 domain comparison

Saved local output: `results/r3_heldout_domain_equivalence_summary.csv`.

The independently fitted R3 and utilization-only partitions agreed in development (n=10,000, ARI 1.000) and in the patient-disjoint held-out cohort (n=61,518, ARI 1.000). Both models were fit on development patients. Parameters were K=2, Lloyd algorithm, 20 internal initializations, and seed 42. This supplementary fit has not been shown to have identical labels to every original stored R3 fit.

## Plots

Figure 3 uses **primary R2** weight-sweep aggregates in `evidence/primary/06B_*`.
Figure 4 uses the **independently reconstructed R2** feature concentration and aligned transition table. The reported aligned matrix is `[[6449, 435], [2931, 185]]`, with 3,366 changed assignments after alignment. It is not a primary-pipeline replication.

The plotting programs write the hashes of their input aggregate CSVs. Those hashes tie a plot to a particular set of aggregate files; they do not demonstrate the source raw-data calculations or independent clinical validation.

## Updating the manuscript

- Section IV-B and IV-C: describe the new development-only multi-hot analysis and its masking procedure.
- Section V-B and Table VI: report the new multi-hot metrics alongside, but separate from, historical R1 masking.
- Section V-C and Table VI: cite R3 development and held-out ARI 1.000.
- Section IV-G: link validated release files to the evidence manifest.
- Figure 3 and Figure 4: use the plots generated from the saved aggregate CSVs.
- Data Availability and software reference: change the version and DOI **only after** a later release is actually archived.
