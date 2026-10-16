# Repository migration

This package update starts from repository commit `04768cc8a663d073e991dcc5ada81e5b404b2a57` and uses the newer source supplied in `Restart-Lanczos-MPS-main.zip`.

## Historical notebooks

The older repository contains exploratory notebooks with additional experiments, saved output, external data paths, and inline function definitions. They are not equivalent to the two Fermi-Hubbard notebooks supplied in the newer ZIP. They have been removed from the current package layout; their complete sources and outputs remain available in Git history:

- [On_Restart_Lanczos_MPS.ipynb](https://github.com/yangz15/Restart-Lanczos-MPS/blob/04768cc8a663d073e991dcc5ada81e5b404b2a57/On_Restart_Lanczos_MPS.ipynb)
- [Restart_MPS_Lanczos_drafted.ipynb](https://github.com/yangz15/Restart-Lanczos-MPS/blob/04768cc8a663d073e991dcc5ada81e5b404b2a57/Restart_MPS_Lanczos_drafted.ipynb)

The new `examples/` directory contains a quick start and the two supplied Fermi-Hubbard configurations. Additional exploratory routines from the old notebooks, including Chebyshev filtering and alternative shift-invert experiments, are not part of the new public API. They have not been revalidated against the newer backend.

## Backend changes relative to the old repository

The top-level `pytenet/` package is replaced by `src/pytenet/`, preventing an old local package from shadowing the installed implementation. Most backend source files in the ZIP match the old repository byte-for-byte. The ZIP supplies newer versions of:

- `mps.py`
- `bond_ops_max_bond.py`
- `operation.py`
- `minimization.py`

It also adds `zipup_var.py`. Those upstream-in-the-supplied-ZIP changes are part of this repository update; they were not introduced by extracting the notebook functions. The 21-function preservation and numerical regression checks in `validation.md` refer to the ZIP's function bodies, not the older exploratory notebooks.

Tracked `__pycache__` files are removed. Install the package in a fresh environment with `python -m pip install -e ".[examples]"` before running the new notebooks.
