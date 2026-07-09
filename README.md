# Exact-Descriptor Active Screening

Code and source data for the manuscript:

**Exact-descriptor active screening for finite-pool molecular prioritization**

This repository supports the reported finite-pool molecular active-screening
experiments. The workflow separates exactly computable molecular descriptors
from expensive oracle readouts: QED and synthetic-accessibility score are used
as exact candidate-state coordinates, while docking score is represented by a
probabilistic surrogate posterior and selected with hypervolume-improvement
acquisition rules.

## Highlights

- Exact-descriptor active screening for finite candidate pools.
- Segmented EHVI and independent-marginal PoMHI acquisition code for the
  one-uncertain-objective setting.
- Matched Dockstring acquisition controls across five targets and five seeds.
- KRAS-G13D 8UN4 computational docking-prioritization source tables.
- Processed source data for the main figures, supplementary figures and summary
  tables.

## Repository layout

```text
.
|-- method_code/              # Acquisition, surrogate, generator and workflow code
|   |-- config/               # Experiment configurations
|   |-- data/                 # Dataset adapters
|   |-- mobo/                 # Finite-pool acquisition and screening utilities
|   `-- scripts/              # Benchmarking and workflow entrypoints
`-- source_data/              # Processed source data used in the manuscript
    |-- fig2_controlled_validation/
    |-- fig3_dockstring_screening/
    |-- fig4_domain_prior_and_generators/
    |-- fig6_8un4_closed_loop/
    |-- summary_tables/
    `-- supplementary/
```

See [`method_code/README.md`](method_code/README.md) and
[`source_data/README.md`](source_data/README.md) for directory-level details.

## Installation

The analysis code is Python-based. A typical environment uses PyTorch, RDKit,
BoTorch/GPyTorch, NumPy, SciPy, pandas, scikit-learn, PyYAML and tqdm.

```bash
git clone https://github.com/Fabien916/information-structured-active-screening.git
cd information-structured-active-screening

conda create -n exact-descriptor-screening python=3.10
conda activate exact-descriptor-screening

# Install packages according to the CUDA/CPU setup of the target machine.
# Core packages include torch, rdkit, botorch, gpytorch, numpy, scipy,
# pandas, scikit-learn, pyyaml and tqdm.
```

Docking workflows use the docking backends and receptor assets described in the
manuscript. The processed tables in `source_data/` can be inspected without
rerunning docking.

## Source data

The `source_data/` directory contains processed tables used to generate the
reported figures and summary values. The main blocks are:

- `summary_tables/`: manuscript-level summary tables.
- `fig2_controlled_validation/`: posterior-controlled acquisition and timing
  comparisons.
- `fig3_dockstring_screening/`: matched Dockstring active-screening results.
- `fig4_domain_prior_and_generators/`: natural-product prior and candidate-pool
  analyses.
- `fig6_8un4_closed_loop/`: 8UN4 docking-prioritization source tables.
- `supplementary/`: supplementary surrogate, Dockstring, 8UN4 and cell-line
  profiling tables.

## Code entrypoints

Representative scripts are:

```text
method_code/scripts/benchmark_analytic_vs_mc.py
method_code/scripts/benchmark_initial_surrogate_ssl.py
method_code/scripts/run_dockstring_experiments.py
method_code/scripts/run_mobo_main_experiment.py
method_code/scripts/run_population_baseline.py
method_code/scripts/run_vae_bo_baseline.py
method_code/scripts/validate_acq_replay.py
```

Experiment configurations are stored under `method_code/config/`. The main
finite-pool screening configurations are in `method_code/config/mobo/`; surrogate
benchmark configurations are in `method_code/config/surrogate/`; generator
configurations are in `method_code/config/generative/`.

## Citation

If this repository is useful for your work, please cite the manuscript once the
preprint or journal version is available.

```bibtex
@article{exactDescriptorActiveScreening,
  title  = {Exact-descriptor active screening for finite-pool molecular prioritization},
  author = {Author list to be confirmed},
  year   = {2026},
  note   = {Manuscript in preparation}
}
```

## Contact

For questions about the manuscript code or source data, please open an issue in
this repository.
