## Molecular simulation data for hydrogen adsorption in nanoporous material.

Each subdirectory contains the molecular simulation data of a type of nanoporous materials. Within the directory the files below are present (with `X=iza|pcod|hcp|mof`):
* `X_hydrogen.npy`: NumPy array files containing normalized hydrogen loading and the corresponding temperature and pressure. For each material, 64 state points are available.
* `X_hydrogen_[random8|random8_1|diagonal|edge]`: NumPy array files containing normalized hydrogen loading and the corresponding temperature and pressure. For each material, 8 out of 64 state points are available (see Figure 2 in the manuscript).
* `names.csv`: List of names of the nanoporous materials within its class. It contains only 1 column without headers.
* `norms.csv`: List of maximum observed hydrogen loading (in mol/L) of the nanoporous materials within its class. It contains only 1 column without headers.

For IZA zeolites, high-throughput simulation data and additional validation simulation data are also available in the CSV format (`iza/iza_hydrogen.csv` and `iza/validated.csv`).

For cation-exchanged zeolites, hydrogen loading files are named `LTA-sim.npy` and `LTA_sim.csv`. The results for LTA-Li is placed in a separate directory `Li-FF-5`.
