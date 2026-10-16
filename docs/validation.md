# Validation record

Validation date: 2026-10-16.

## Scope and environment

Local checks used CPython 3.12.14 on macOS with NumPy 2.5.4, SciPy 1.18.1, gmpy2 2.3.2, Matplotlib 3.11.2, pytest 9.1.1, and nbclient 0.11.0. GitHub CI is configured for Python 3.10 and 3.12 on Linux; those hosted runs have not been executed as part of this local delivery.

## Source preservation

The two original notebooks and exported scripts contain the same 21 top-level function definitions. All 21 extracted function bodies were compared using Python syntax trees, excluding docstrings, comments, and formatting, and matched the original definitions. The `pytenet` backend is retained from the supplied source, with only comment translation and docstring escaping changes.

## Numerical regression

The original and refactored `trl_variance_solver` were run independently with:

- 3-site Hubbard model, U = 8, t = 1, mu = 0, quantum numbers disabled.
- Random seed 42, one target state, nominal bond parameter 16.
- 5 initial Lanczos steps, 2 restart cycles, 1 expansion step.
- Squared-residual threshold 1e-8 and truncation tolerance 1e-10.

Both runs returned the Ritz value `-6.448356766932857`, with diagnostic history `0.960329259661433` and `0.3354418207234244`. The maximum differences in Ritz value and history were zero in this environment. Returned dense MPS vectors also agreed within 1e-11. These two cycles do not establish convergence to an eigenstate.

## Automated checks

**Result: all 16 tests passed locally.** The suite includes these checks, with parameterized cases:

- Repeatable seeded model construction, normalization, and Hamiltonian Hermiticity.
- Input checks for the new model-setup helper.
- Gram and projected-Hamiltonian agreement with independent dense contractions.
- Reorthogonalization of a positive-definite Gram matrix.
- Complex MPS linear combinations, zero combinations, and input preservation.
- Plain-Lanczos projection-prefix consistency.
- Ritz reconstruction agreement with a projected dense eigensolve.
- The small main-solver regression above.
- Dense residual diagnostics and identity local contractions.
- Correct plotting coordinates and masking of locked-slot placeholders.
- Format and cleanliness of all three notebooks, including no function/class/lambda definitions.
- Full execution of the quick-start notebook in a temporary environment without changing the clean source.

The local sandbox blocks Jupyter's localhost sockets, so the notebook execution check was run with that restriction lifted. This is an execution-environment requirement, not a change to the notebook.

## Packaging

The package is checked by editable installation, undefined-name linting of the extracted package and tests, source/wheel construction, and installation/import of the built wheel outside the repository. The source archive is checked for unwanted bytecode, caches, macOS metadata, saved simulation data, and notebook output.

## Not established by these checks

The full 8-site and 30-site experiment configurations were not executed. No performance claim, full-paper benchmark reproduction, complete solver convergence guarantee, or comprehensive validation of the legacy backend is made. Partial locking, complex band-Lanczos initialization, singular-basis handling, and shift-invert convergence retain the limitations documented in [algorithm notes](algorithms.md).
