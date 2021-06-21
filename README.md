# Manuscript data for: Fingerprinting diverse nanoporous materials for optimal hydrogen storage conditions using meta-learning

This repository branch contains the complete data collection for the manuscript "Fingerprinting diverse nanoporous materials for optimal hydrogen storage conditions using meta-learning", *Journal* **Vol.**, pages (year).

## Contents
Below is a description of each subdirectory and its contents:
* `data/`: Hydrogen adsorption simulation datasets of different types of nanoporous materials, including IZA all-silica zeolites (`data/iza`), PCOD-syn all-silica zeolites (`data/pcod`), cation-exchanged zeolites (`data/cation-exchange`), metal-organic frameworks (`data/mof`), and hyper-cross-linked polymers (`data/hcp`).  
* `manuscript/`: IPython notebook which reproduces all non-descriptive figures and tables in the main text and Supplementary Materials for tha manuscript and numerical results reported. See [manuscript/readme.md](`manuscript/readme.md`) for details.
* `models/`: Trained SorbMetaML models to predict hydrogen storage of nanoporous materials.
* `sorbmetaml/`: The training and evaluation programs for SorbMetaML.
* `supp_info/`: miscellaneous files for calculating the compressibility factors and chemical potentials of H2.

## Acknowledgement
The development of SorbMetaML is supported by the U.S. Department of Energy, Office of Basic Energy Sciences, Division of Chemical Sciences, Geosciences and Biosciences through the Nanoporous Materials Genome Center under award DE-FG02-17ER16362.
