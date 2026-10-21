# Package-level checks

The following were executed on the packaged sources on October 21, 2026:

- All six Python scripts compiled without a syntax error.
- The primary Figure 3 script read the 15 tested R2 weights from the included 06B CSVs and generated PNG, PDF and SVG files.
- The audit reporting script read the included primary candidate summary and reconstruction CSVs, produced six separated evidence records, and generated figures. A patient-level transition figure was correctly skipped because assignment-level CSVs were not included.
- The 06C script and validation tracker accepted `--help` and parsed their documented CLI options.

These checks exercise source access, syntax and the plot/report paths. They **do not** rerun primary K-means experiments, replicate the 480-fit 06C analysis, or verify the 120-run masking result. The optional 06D comparator was not run as part of the manuscript.
