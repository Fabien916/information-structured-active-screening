# Method code

This directory contains the method code supporting the manuscript analyses.

## Contents

- `mobo/`: finite-pool acquisition functions, candidate-pool utilities,
  surrogate interfaces and molecular-screening helpers.
- `scripts/`: analysis entrypoints for Dockstring controls, 8UN4 workflows,
  surrogate benchmarks and acquisition replay diagnostics.
- `config/`: configuration files for the reported generator, surrogate and
  multi-objective screening experiments.
- `data/`: dataset adapters used by the screening and surrogate workflows.

Configuration names that contain `qpmhi` correspond to the manuscript PoMHI /
independent-marginal PoMHI implementation unless explicitly described as the
external sampling reference.
