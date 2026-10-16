"""Ensure notebooks stay clean examples, and execute the small example."""

import ast
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted((ROOT / "examples").glob("*.ipynb"))


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_is_clean_example(path):
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    assert "Modified Thick-Block Lanczos" in notebook.cells[0].source
    for cell in notebook.cells:
        if cell.cell_type == "code":
            assert cell.execution_count is None
            assert not cell.outputs
            tree = ast.parse(cell.source)
            assert not any(
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda))
                for node in ast.walk(tree)
            )


def test_quickstart_executes(tmp_path):
    notebook = nbformat.read(ROOT / "examples/quickstart.ipynb", as_version=4)
    # Keep execution in memory: never rewrite the clean source notebook.
    executed = NotebookClient(
        notebook,
        timeout=180,
        kernel_name="python3",
        resources={"metadata": {"path": str(tmp_path)}},
    ).execute()
    assert all(
        cell.execution_count is not None for cell in executed.cells if cell.cell_type == "code"
    )
    assert not any(
        output.output_type == "error"
        for cell in executed.cells
        if cell.cell_type == "code"
        for output in cell.outputs
    )
