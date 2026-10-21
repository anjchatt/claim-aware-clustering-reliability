# Python environment provenance

The package dependency pins refer to the environment used for release-candidate smoke checks, **not** the original clinical clustering experiments.

* Release-candidate checks: Python 3.13.5, NumPy 2.3.5, pandas 2.2.3, SciPy 1.17.0, scikit-learn 1.8.0, Matplotlib 3.10.8, threadpoolctl 3.6.0.
* Primary and reconstruction historical environments: restore directly from saved run manifests or the original Jupyter kernels. Versions have not been recovered in this package.
* `pyarrow` is needed for optional historical Parquet inputs but is not needed for the included aggregate CSV verification. Install using `requirements-notebooks.txt` if you intend to open the original Parquet source files.

A `pip freeze` from a different computer does not establish the historical experiment environment. Archive the original `sys.version`, relevant package versions, and input-source hashes with each executed experiment.
