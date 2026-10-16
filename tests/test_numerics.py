"""Small-system tests that do not run the full research configurations."""

import copy
import numpy as np
import pytest

from restart_lanczos_mps import (
    generate_reduced_H,
    get_W1,
    get_S0,
    form_linear_combination,
    lanczos_lr_py,
    trl_variance_solver,
    solve_ritz_vecs_from_list,
    norm_test,
    apply_local_mps,
)
from restart_lanczos_mps.examples import make_hubbard_problem, plot_variance_history


def test_seeded_problem_and_hermiticity():
    h, psi = make_hubbard_problem(nsites=3, seed=42)
    h2, psi2 = make_hubbard_problem(nsites=3, seed=42)
    np.testing.assert_allclose(h.as_matrix(), h2.as_matrix())
    np.testing.assert_allclose(psi.as_vector(), psi2.as_vector())
    np.testing.assert_allclose(np.linalg.norm(psi.as_vector()), 1.0)
    np.testing.assert_allclose(h.as_matrix(), h.as_matrix().conj().T)


@pytest.mark.parametrize("nsites", [0, 1, 2.5, True])
def test_invalid_model_size(nsites):
    with pytest.raises(ValueError):
        make_hubbard_problem(nsites=nsites)


def test_projection_and_reorthogonalization_against_dense():
    h, a = make_hubbard_problem(nsites=2, seed=3)
    _, b = make_hubbard_problem(nsites=2, seed=4)
    basis = [a, b]
    vectors = np.column_stack([v.as_vector() for v in basis])
    gram = get_W1(2, basis)
    projected = generate_reduced_H(2, basis, h)
    np.testing.assert_allclose(gram, vectors.conj().T @ vectors, atol=1e-12)
    np.testing.assert_allclose(projected, vectors.conj().T @ h.as_matrix() @ vectors, atol=1e-12)
    s = get_S0(gram)
    np.testing.assert_allclose(s.conj().T @ gram @ s, np.eye(2), atol=1e-12)


def test_linear_combination_preserves_inputs():
    _, a = make_hubbard_problem(nsites=2, seed=7)
    _, b = make_hubbard_problem(nsites=2, seed=9)
    before = [a.as_vector().copy(), b.as_vector().copy()]
    coeffs = np.array([1.0, 0.5j])
    actual = form_linear_combination([a, b], coeffs, max_bond=16, trunc_tol=0)
    expected = coeffs[0] * before[0] + coeffs[1] * before[1]
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(actual.as_vector(), expected, atol=1e-10)
    np.testing.assert_allclose(a.as_vector(), before[0])
    np.testing.assert_allclose(b.as_vector(), before[1])
    zero = form_linear_combination([a, b], np.zeros(2), max_bond=16, trunc_tol=0)
    np.testing.assert_allclose(zero.as_vector(), 0, atol=1e-12)


def test_lanczos_projection_prefix():
    h, psi = make_hubbard_problem(nsites=3, seed=42)
    basis, projected, gram, history = lanczos_lr_py(h, psi, maxiter=4, tol=1e-10, rmax=16)
    n = gram.shape[0]
    dense = np.column_stack([v.as_vector() for v in basis[:n]])
    np.testing.assert_allclose(gram, dense.conj().T @ dense, atol=1e-10)
    np.testing.assert_allclose(projected, dense.conj().T @ h.as_matrix() @ dense, atol=1e-10)
    assert len(basis) == n + 1
    assert len(history) == n


def test_ritz_reconstruction_matches_projected_problem():
    import scipy.linalg

    h, a = make_hubbard_problem(nsites=2, seed=12)
    _, b = make_hubbard_problem(nsites=2, seed=15)
    gram = get_W1(2, [a, b])
    projected = generate_reduced_H(2, [a, b], h)
    energies, states = solve_ritz_vecs_from_list([a, b], projected, gram, 2, 16, 2, h)
    expected = scipy.linalg.eigvalsh(projected, gram)
    np.testing.assert_allclose(energies, expected, atol=1e-10)
    for energy, state in zip(energies, states):
        vec = state.as_vector()
        np.testing.assert_allclose(np.linalg.norm(vec), 1, atol=1e-10)
        np.testing.assert_allclose(np.vdot(vec, h.as_matrix() @ vec), energy, atol=1e-9)


def test_main_solver_small_regression(capsys):
    # These values are independently checked against the original notebook body.
    h, psi = make_hubbard_problem(nsites=3, seed=42)
    before = psi.as_vector().copy()
    options = np.get_printoptions()
    try:
        energies, states, history = trl_variance_solver(
            h,
            1,
            16,
            psi,
            initial_lanczos_steps=5,
            num_restarts=2,
            lanczos_steps=1,
            tol=1e-8,
            trunc_tol=1e-10,
        )
    finally:
        np.set_printoptions(**options)
    assert len(energies) == len(states) == 1
    assert history.shape == (2, 1)
    assert np.isfinite(history).all()
    np.testing.assert_allclose(energies, [-6.448357], rtol=2e-6)
    np.testing.assert_allclose(history[:, 0], [0.9603293, 0.3354418], rtol=2e-6)
    np.testing.assert_allclose(psi.as_vector(), before, atol=1e-12)
    assert energies[0] >= np.linalg.eigvalsh(h.as_matrix())[0] - 1e-8
    np.testing.assert_allclose(np.linalg.norm(states[0].as_vector()), 1.0, atol=1e-10)


def test_dense_residual_diagnostic_and_local_identity():
    h, psi = make_hubbard_problem(nsites=2, seed=10)
    ref = copy.deepcopy(psi)
    sigma = 0.7
    expected = np.linalg.norm(h.as_matrix() @ psi.as_vector() - (sigma + 1) * psi.as_vector())
    np.testing.assert_allclose(norm_test(h, psi, ref, sigma), expected, atol=1e-10)
    tensor = np.arange(24).reshape(2, 3, 4)
    np.testing.assert_allclose(apply_local_mps(np.eye(3), np.eye(4), tensor), tensor)


def test_plot_cycle_axis_and_locked_placeholders():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ax = plot_variance_history([[1.0, 0.1], [0.5, 0.0], [0.25, 0.0]])
    np.testing.assert_array_equal(ax.lines[0].get_xdata(), [1, 2, 3])
    np.testing.assert_allclose(ax.lines[0].get_ydata(), [1.0, 0.5, 0.25])
    assert np.isnan(ax.lines[1].get_ydata()[1:]).all()
    assert ax.get_xlabel() == "Restart cycle"
    plt.close(ax.figure)
