# Restart Lanczos MPS

Python implementations of restarted Lanczos methods for matrix product states (MPS), with example calculations for the Fermi–Hubbard model.

**MTBL = Modified Thick-Block Lanczos.** The residual-enriched restart routine used in the examples is `trl_variance_solver`. Its historical name is retained for compatibility. The method reference is [An accurate Lanczos method for the matrix product state representation](https://doi.org/10.1038/s42005-026-02896-x), *Communications Physics* **9**, 326 (2026). Read the [algorithm notes](docs/algorithms.md) for the relationship between the supplied code and the published method.

## Install

Use Python 3.10 or later in a fresh environment. From this repository's root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[examples]"
python -m ipykernel install --sys-prefix --name restart-lanczos-mps --display-name "Restart Lanczos MPS"
jupyter lab
```

On Windows, activate the environment with `.venv\Scripts\activate` instead. Select the **Restart Lanczos MPS** kernel in Jupyter. For solver use without plotting or Jupyter, install with `python -m pip install .`.

The modified `pytenet` source from the supplied archive is bundled. Use an isolated environment to avoid conflicting with a separately installed `pytenet` package. The core dependencies are NumPy, SciPy, and gmpy2; plotting and notebooks are optional.

## Quick start

```python
from restart_lanczos_mps import trl_variance_solver
from restart_lanczos_mps.examples import make_hubbard_problem

hamiltonian, initial_state = make_hubbard_problem(nsites=3, seed=42)
eigenvalues, eigenvectors, history = trl_variance_solver(
    hamiltonian, k=1, max_bond=16, v0=initial_state,
    initial_lanczos_steps=5, num_restarts=2, lanczos_steps=1,
    tol=1e-8, trunc_tol=1e-10,
)
```

This is an execution example, not a converged research benchmark. The model setup disables quantum numbers, matching the original notebooks, and does not impose a half-filling constraint. In `trl_variance_solver`, `tol` is compared with the **squared compressed residual norm**. See the [API reference](docs/api.md) before interpreting convergence or switching solver variants.

## Examples

| Notebook | Purpose | Parameters |
| --- | --- | --- |
| [Quick start](examples/quickstart.ipynb) | Small runnable example | 3 sites, M = 16, 1 state, 2 restart cycles |
| [Fermi–Hubbard test](examples/Fermi_hubbard_test.ipynb) | Original small research configuration | 8 sites, M = 32, 4 states, up to 100 cycles |
| [Fermi–Hubbard L30](examples/Fermi_hubbard_L30_M256.ipynb) | Original larger research configuration | 30 sites, M = 256, 4 states, up to 1000 cycles |

All notebooks contain example calls only: no function or class definitions, embedded solver implementations, saved outputs, or execution counts. The two original research configurations are preserved. They may take substantial time and are not run in continuous integration.

## Repository layout

```text
src/
  restart_lanczos_mps/
    lanczos.py           # Plain, thick-restart, and band basis construction
    ritz.py              # Projection and Ritz reconstruction
    solvers.py           # Residual-enriched restart solvers / MTBL example
    inverse_iteration.py # Local projected systems and sweep solvers
    shift_invert.py      # Shift-and-invert targeting and state tracking
    examples.py          # Model setup and convergence plotting
  pytenet/               # Tensor-network backend supplied with the archive
examples/                # Clean notebooks containing example calls only
docs/                    # API, algorithms, development, and validation notes
tests/                   # Numerical and notebook checks
pyproject.toml           # Installation metadata and dependencies
.github/workflows/       # Automated checks for GitHub
```

## Documentation

- [Functions and API](docs/api.md): signatures, parameters, return values, and module roles.
- [Algorithms and MTBL](docs/algorithms.md): method naming, convergence diagnostics, and implementation limits.
- [Development](docs/development.md): tests, builds, and notebook conventions.
- [Repository migration](docs/repository-migration.md): historical notebooks and changes from the older GitHub tree.
- [Refactoring notes](docs/refactoring.md): source provenance and intentional changes.
- [Validation](docs/validation.md): checks actually performed and their scope.

## Development

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check src/restart_lanczos_mps tests
python -m build
```

The layout follows the [Python Packaging User Guide](https://packaging.python.org/en/latest/tutorials/packaging-projects/). This is research software; small-system tests do not establish convergence, accuracy, or equivalence to every benchmark in the paper.

## Citation and attribution

For the scientific method, cite the [MTBL paper](https://doi.org/10.1038/s42005-026-02896-x). The original notebook names and callable names are mapped in [refactoring notes](docs/refactoring.md).

The input archive did not contain a license or copyright attribution file. No license or authorship has been invented for this refactor. Before an open-source release, the maintainer needs to supply the intended project license and confirm the license/attribution for the bundled, modified `pytenet` source. See [third-party notices](THIRD_PARTY_NOTICES.md).
