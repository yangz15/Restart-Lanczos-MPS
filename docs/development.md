# Development

## Environment

Use an isolated Python environment. Install the package from the repository root with `python -m pip install -e ".[dev]"`. The `src` layout deliberately requires installation; notebooks do not modify `sys.path` or depend on a particular working directory.

For Jupyter, register the environment using `python -m ipykernel install --sys-prefix --name restart-lanczos-mps --display-name "Restart Lanczos MPS"`. Select it in the notebook interface.

## Checks

```bash
python -m pytest
python -m ruff check src/restart_lanczos_mps tests
python -m build
```

Tests check projected matrices against dense contractions, reconstruction helpers, repeatable model setup, a small restarted run, plotting semantics, and notebook structure. The quick-start notebook is executed in a temporary copy; execution does not save output into the distributed notebook.

GitHub Actions installs the project and runs these checks. The two research configurations are excluded from routine execution because of their cost. They are still checked for valid notebook format, clean outputs, and the absence of function/class/lambda definitions.

## Notebook convention

Place all reusable computation and plotting functions in `src/restart_lanczos_mps/`. Example notebooks should contain explanatory Markdown, imports, parameter values, and function calls. Before committing a notebook, clear all outputs and execution counts. Do not add local paths, installation cells, cached figures, serialized states, or copied solver code.

## Distribution

`python -m build` creates a source distribution and a wheel under `dist/`. The source distribution includes documentation and notebooks; the wheel contains the importable packages. Build artifacts, caches, macOS metadata, and saved simulation data are ignored by Git.

The source is prepared for repository upload. A public release still needs maintainer-supplied license/attribution information as described in `THIRD_PARTY_NOTICES.md`. The repository migration and links to historical notebooks are recorded in `repository-migration.md`.
