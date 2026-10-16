# Refactoring notes

## Source

Both supplied notebooks contained identical definitions for 21 top-level functions. The accompanying `.py` files duplicated those definitions and ran experiments immediately when imported. The common definitions now have one location in `src/restart_lanczos_mps/`.

| Original content | New location |
| --- | --- |
| Plain, thick-restart, and band Lanczos | `lanczos.py` |
| Ritz reconstruction and projection helpers | `ritz.py` |
| Residual-enriched restart solvers | `solvers.py` |
| Inverse-iteration sweeps and local systems | `inverse_iteration.py` |
| Shift-invert restart | `shift_invert.py` |
| Model construction and plots | `examples.py` |
| `Fermi_hubbard_test.ipynb` | `examples/Fermi_hubbard_test.ipynb` |
| `Fermi_hubbard_L30_M256.ipynb` | `examples/Fermi_hubbard_L30_M256.ipynb` |
| Supplied `pytenet/` source | `src/pytenet/` |

## Changes

- Extracted all 21 top-level functions into five modules and exposed the 20 public functions through `restart_lanczos_mps`; the private local solver remains internal.
- Replaced broad and duplicate notebook imports with explicit module dependencies. Removed unused dependencies such as h5py from the notebook interface.
- Added the missing explicit `numpy.linalg.norm` import for `norm_test`.
- Preserved numerical function bodies, original callable names, default values, restart logic, and experiment parameters. Corrected docstrings and comments where they contradicted the return values or convergence comparison.
- Added two reusable example helpers for Hubbard setup and convergence plotting.
- Corrected plot semantics: use restart cycles and unscaled squared compressed residuals, masking zero placeholders.
- Added English documentation, package metadata, automated tests, and a GitHub Actions workflow.
- Marked mathematical/diagram docstrings as raw strings to avoid invalid-escape warnings.
- Translated two Chinese comments in the supplied backend to English. No backend algorithm was replaced with an external package version.
- Removed notebook outputs, empty cells, duplicated exported scripts, Python bytecode, macOS metadata, and saved Krylov pickle files from the deliverable. The original archive remains unchanged.

## Scope

This is a structural packaging refactor, with documented inherited numerical limitations. It does not claim to repair or validate every research solver. The small regression run checks preservation of the supplied implementation rather than agreement with every published benchmark. See `algorithms.md` and `validation.md` before scientific use.
