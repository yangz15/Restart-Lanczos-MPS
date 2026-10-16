"""Reusable model setup and plotting helpers for the example notebooks."""

import numpy as np
from pytenet.hamiltonian import fermi_hubbard_mpo
from pytenet.mps import create_random_mps_4


def make_hubbard_problem(nsites=3, interaction=8.0, hopping=1.0, chemical_potential=0.0, seed=42):
    """Return a Hubbard MPO and normalized random product MPS.

    Parameters are the site count, on-site interaction U, hopping t, chemical
    potential mu, and NumPy random seed. Quantum numbers are disabled, matching
    the supplied notebooks. This does not restrict the state to half filling.
    The local physical dimension is four; all initial virtual bonds have size one.
    """
    if not isinstance(nsites, (int, np.integer)) or isinstance(nsites, bool) or nsites < 2:
        raise ValueError("nsites must be an integer of at least two.")
    hamiltonian = fermi_hubbard_mpo(
        L=nsites, U=interaction, t=hopping, mu=chemical_potential
    ).zero_qnumbers()
    initial_state = create_random_mps_4(
        [1] * (nsites + 1), "right", rng=np.random.default_rng(seed)
    )
    return hamiltonian, initial_state


def plot_variance_history(variance_history, *, ax=None):
    """Plot squared compressed residuals against one-based restart cycles.

    Parameters
    ----------
    variance_history : array-like, shape (cycles, states)
        History returned by a variance solver. Zeros from locked slots are
        masked because they are placeholders, not measured exact zeros.
    ax : matplotlib.axes.Axes, optional
        Existing axes; otherwise a new figure is created.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the convergence curves. Call ``plt.show()`` to display.
    """
    import matplotlib.pyplot as plt

    history = np.asarray(variance_history)
    if history.ndim != 2 or not history.size:
        raise ValueError("variance_history must be a nonempty (cycles, states) array.")
    history = np.real_if_close(history)
    if np.iscomplexobj(history):
        raise ValueError("variance_history must be real up to numerical roundoff.")
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    cycles = np.arange(1, history.shape[0] + 1)
    for state, values in enumerate(history.T):
        displayed = np.where(np.isfinite(values) & (values > 0), values, np.nan)
        ax.semilogy(cycles, displayed, marker="o", label=f"State {state}")
    ax.set_xlabel("Restart cycle")
    ax.set_ylabel("Squared compressed residual norm")
    ax.set_title("MTBL convergence")
    ax.legend()
    ax.grid(True, which="both", alpha=0.25)
    return ax
