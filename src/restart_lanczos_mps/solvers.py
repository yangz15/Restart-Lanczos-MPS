"""Residual-enriched restart solvers, including the MTBL example routine."""

import copy
import numpy as np
import scipy.linalg
from pytenet.mps import apply_mpo_to_mps_variational_only
from pytenet.operation import vdot, add_mps_and_compress, operator_inner_product
from .lanczos import lanczos_lr_py
from .ritz import form_linear_combination


def trl_plus_r_solver(
    H_mpo,
    k,
    max_bond,
    initial_guess_block,
    realeig,
    num_restarts=20,
    lanczos_steps=10,  # Number of new vectors to add per restart
    tol=1e-12,
    trunc_tol=1e-12,
):
    """
    Finds the 'k' lowest eigenpairs of a Hamiltonian MPO using a
    Thick Restart Lanczos with Residual Enrichment (TRL+R) algorithm.

    This implementation follows the correct TRL procedure where the basis
    is augmented by a new Lanczos run seeded by the residuals.

    Args:
        H_mpo (MPO): The Hamiltonian operator.
        k (int): The number of lowest eigenvalues to find.
        max_bond (int): The maximum bond dimension for the MPS.
        initial_guess_block (list): A list of 'k' MPS objects to start with.
        realeig (np.ndarray): The true eigenvalues for error tracking.
        num_restarts (int): The number of macro-restart iterations to perform.
        lanczos_steps (int): The number of new vectors to generate in each restart cycle.
        tol (float): Convergence tolerance for the residual norm.
        trunc_tol (float): Truncation tolerance for MPS operations.

    Returns:
        final_eigs (np.ndarray): The k lowest computed eigenvalues.
        final_vecs (list): A list of k MPS objects for the eigenvectors.
        error_history (np.ndarray): A (num_restarts x k) array of absolute errors.
    """

    # Start with the initial guess as our first basis
    V_basis = copy.deepcopy(initial_guess_block)
    error_history = []

    # --- Main Restart Loop ---
    for i_restart in range(num_restarts):
        print(f"\n{'=' * 20} TRL+R Macro-Iteration {i_restart + 1}/{num_restarts} {'=' * 20}")
        print(f"  Current basis size: {len(V_basis)}")

        # --- Phase 1: Rayleigh-Ritz on the Current Basis ---
        print(f"  Phase 1: Solving Rayleigh-Ritz problem...")
        dim_basis = len(V_basis)
        W = np.zeros((dim_basis, dim_basis), dtype=np.complex128)
        P = np.zeros((dim_basis, dim_basis), dtype=np.complex128)
        for i in range(dim_basis):
            for j in range(i, dim_basis):
                W[i, j] = vdot(V_basis[i], V_basis[j])
                W[j, i] = np.conj(W[i, j])
                P[i, j] = operator_inner_product(V_basis[i], H_mpo, V_basis[j])
                P[j, i] = np.conj(P[i, j])

        try:
            ritz_values, ritz_coeffs = scipy.linalg.eigh(P, W)
        except scipy.linalg.LinAlgError:
            print("  FATAL: Eigenvalue solve failed on current basis. Stopping.")
            break

        X_block = [
            form_linear_combination(V_basis, ritz_coeffs[:, j], max_bond, trunc_tol)
            for j in range(k)
        ]

        # --- Phase 2: Calculate Residuals & Check Convergence ---
        print("  Phase 2: Calculating residuals...")
        R_block = []
        for j_k in range(k):
            ritz_val = ritz_values[j_k]
            ritz_vec = X_block[j_k]
            # H_phi = apply_operator_and_compress(H_mpo, ritz_vec, trunc_tol, max_bond)
            H_phi = apply_mpo_to_mps_variational_only(
                H_mpo, ritz_vec, max_bond=max_bond, nsweeps=3, initial_guess=ritz_vec
            )
            lambda_phi = copy.deepcopy(ritz_vec)
            lambda_phi.A[0] *= -ritz_val
            residual_vec = add_mps_and_compress(H_phi, lambda_phi, trunc_tol, max_bond)
            R_block.append(residual_vec)

        residual_norms = [r.orthonormalize("right") for r in R_block]
        print(f"\n  Convergence Check: Residual norms = {np.array(residual_norms)}")

        errors = np.abs(np.sort(ritz_values[:k]) - realeig[:k])
        error_history.append(errors)
        print(f"  Current Errors = {errors}")

        if np.all(np.array(residual_norms) < tol):
            print("\nConvergence criteria met. Stopping.")
            break

        if i_restart == num_restarts - 1:
            print("\nMax restarts reached.")
            break

        # --- Phase 3: Augment the Basis (Corrected TRL Step) ---
        print("  Phase 3: Augmenting basis with new Lanczos run from residuals...")

        # Start the next basis with the k best Ritz vectors. This is the "Thick" part.
        V_next_basis = copy.deepcopy(X_block)

        # Orthonormalize the residual block against the new Ritz vectors and itself
        # to create a set of clean, new directions to start the Lanczos run.
        R_ortho_block = []
        for r_vec in R_block:
            # Orthogonalize against all vectors currently in the next basis
            for v_prev in V_next_basis:
                overlap = vdot(r_vec, v_prev)
                term = copy.deepcopy(v_prev)
                term.A[0] *= -overlap
                r_vec = add_mps_and_compress(r_vec, term, trunc_tol, max_bond)

            norm = r_vec.orthonormalize("right")
            if norm > 1e-12:
                # Add the new, clean search direction to the basis
                V_next_basis.append(r_vec)

        # If all residuals collapsed, the subspace is invariant, so we stop.
        if len(V_next_basis) == k:
            print("  Residuals collapsed into the existing Ritz space. Stopping.")
            break

        # Orthogonalization

        # ortho_window_size = 2 * k
        ortho_window_size = k

        num_generated_so_far = len(V_next_basis) - k
        for j_lan in range(num_generated_so_far, lanczos_steps):
            seed_vec = V_next_basis[-1]
            # new_vec = apply_operator_and_compress(H_mpo, seed_vec, trunc_tol, max_bond)
            new_vec = apply_mpo_to_mps_variational_only(
                H_mpo, seed_vec, max_bond=max_bond, nsweeps=3, initial_guess=seed_vec
            )
            # Define the window of recent vectors to orthogonalize against
            start_index = max(0, len(V_next_basis) - ortho_window_size)
            ortho_basis = V_next_basis[start_index:]

            # Orthogonalize against the vectors in the window
            for v_prev in ortho_basis:
                overlap = vdot(new_vec, v_prev)
                term = copy.deepcopy(v_prev)
                term.A[0] *= -overlap
                new_vec = add_mps_and_compress(new_vec, term, trunc_tol, max_bond)

            norm = new_vec.orthonormalize("right")
            if norm > 1e-12:
                V_next_basis.append(new_vec)
            else:
                print(f"  Lanczos vector {j_lan + 1} collapsed. Ending augmentation early.")
                break

        V_basis = V_next_basis

    return np.sort(ritz_values[:k]), X_block, np.array(error_history)


def trl_plus_r_solver_variance(
    H_mpo,
    k,
    max_bond,
    initial_guess_block,
    num_restarts,
    lanczos_steps,  # Number of new vectors to add per restart
    tol,
    trunc_tol,
):
    """
    Finds the 'k' lowest eigenpairs of a Hamiltonian MPO using a
    Thick Restart Lanczos with Residual Enrichment (TRL+R) algorithm.

    This version uses the energy variance (sigma^2) as the error metric,
    removing the dependency on known exact eigenvalues.

    Args:
        H_mpo (MPO): The Hamiltonian operator.
        k (int): The number of lowest eigenvalues to find.
        max_bond (int): The maximum bond dimension for the MPS.
        initial_guess_block (list): A list of 'k' MPS objects to start with.
        num_restarts (int): The number of macro-restart iterations to perform.
        lanczos_steps (int): The number of new vectors to generate in each restart cycle.
        tol (float): Convergence tolerance for the residual norm.
        trunc_tol (float): Truncation tolerance for MPS operations.

    Returns:
        final_eigs (np.ndarray): The k lowest computed eigenvalues.
        final_vecs (list): A list of k MPS objects for the eigenvectors.
        variance_history (np.ndarray): A (num_restarts x k) array of energy variances.
    """

    # Start with the initial guess as our first basis
    V_basis = copy.deepcopy(initial_guess_block)
    variance_history = []

    # --- Main Restart Loop ---
    for i_restart in range(num_restarts):
        print(f"\n{'=' * 20} TRL+R Macro-Iteration {i_restart + 1}/{num_restarts} {'=' * 20}")
        print(f"  Current basis size: {len(V_basis)}")

        # --- Phase 1: Rayleigh-Ritz on the Current Basis ---
        print(f"  Phase 1: Solving Rayleigh-Ritz problem...")
        dim_basis = len(V_basis)
        W = np.zeros((dim_basis, dim_basis), dtype=np.complex128)
        P = np.zeros((dim_basis, dim_basis), dtype=np.complex128)
        for i in range(dim_basis):
            for j in range(i, dim_basis):
                W[i, j] = vdot(V_basis[i], V_basis[j])
                W[j, i] = np.conj(W[i, j])
                P[i, j] = operator_inner_product(V_basis[i], H_mpo, V_basis[j])
                P[j, i] = np.conj(P[i, j])

        try:
            ritz_values, ritz_coeffs = scipy.linalg.eigh(P, W)
        except scipy.linalg.LinAlgError:
            print("  FATAL: Eigenvalue solve failed on current basis. Stopping.")
            break

        X_block = [
            form_linear_combination(V_basis, ritz_coeffs[:, j], max_bond, trunc_tol)
            for j in range(k)
        ]

        # --- Phase 2: Calculate Residuals & Check Convergence ---
        print("  Phase 2: Calculating residuals...")
        R_block = []
        for j_k in range(k):
            ritz_val = ritz_values[j_k]
            ritz_vec = X_block[j_k]
            # H_phi = apply_operator_and_compress(H_mpo, ritz_vec, trunc_tol, max_bond)
            H_phi = apply_mpo_to_mps_variational_only(
                H_mpo, ritz_vec, max_bond=max_bond, nsweeps=3, initial_guess=ritz_vec
            )
            lambda_phi = copy.deepcopy(ritz_vec)
            # This creates the -theta * |phi> term
            lambda_phi.A[0] *= -ritz_val
            # This calculates the residual R = H|phi> - theta|phi>
            residual_vec = add_mps_and_compress(H_phi, lambda_phi, trunc_tol, max_bond)
            R_block.append(residual_vec)
        variances = [vdot(r0, r0) for r0 in R_block]
        variance_history.append(variances)
        print(f"  Current Variances = {variances}")
        # The norm of the residual is sqrt(variance). This is used for convergence.
        residual_norms = [r.orthonormalize("right") for r in R_block]
        print(f"\n  Convergence Check: Residual norms = {np.array(residual_norms)}")

        if np.all(np.array(residual_norms) < tol):
            print("\nConvergence criteria met. Stopping.")
            break

        if i_restart == num_restarts - 1:
            print("\nMax restarts reached.")
            break

        # --- Phase 3: Augment the Basis (Corrected TRL Step) ---
        print("  Phase 3: Augmenting basis with new Lanczos run from residuals...")

        # Start the next basis with the k best Ritz vectors. This is the "Thick" part.
        V_next_basis = copy.deepcopy(X_block)

        # Orthonormalize the residual block against the new Ritz vectors and itself
        # to create a set of clean, new directions to start the Lanczos run.
        for r_vec in R_block:
            # Orthogonalize against all vectors currently in the next basis
            for v_prev in V_next_basis:
                overlap = vdot(r_vec, v_prev)
                term = copy.deepcopy(v_prev)
                term.A[0] *= -overlap
                r_vec = add_mps_and_compress(r_vec, term, trunc_tol, max_bond)

            norm = r_vec.orthonormalize("right")
            if norm > 1e-12:
                # Add the new, clean search direction to the basis
                V_next_basis.append(r_vec)

        # If all residuals collapsed, the subspace is invariant, so we stop.
        if len(V_next_basis) == k:
            print("  Residuals collapsed into the existing Ritz space. Stopping.")
            break

        # Orthogonalization window size for Lanczos steps
        ortho_window_size = k

        num_generated_so_far = len(V_next_basis) - k
        for j_lan in range(num_generated_so_far, lanczos_steps):
            seed_vec = V_next_basis[-1]
            new_vec = apply_mpo_to_mps_variational_only(
                H_mpo, seed_vec, max_bond=max_bond, nsweeps=3, initial_guess=seed_vec
            )

            # Define the window of recent vectors to orthogonalize against
            start_index = max(0, len(V_next_basis) - ortho_window_size)
            ortho_basis = V_next_basis[start_index:]

            # Orthogonalize against the vectors in the window
            for v_prev in ortho_basis:
                overlap = vdot(new_vec, v_prev)
                term = copy.deepcopy(v_prev)
                term.A[0] *= -overlap
                new_vec = add_mps_and_compress(new_vec, term, trunc_tol, max_bond)

            norm = new_vec.orthonormalize("right")
            if norm > 1e-12:
                V_next_basis.append(new_vec)
            else:
                print(f"  Lanczos vector {j_lan + 1} collapsed. Ending augmentation early.")
                break

        V_basis = V_next_basis

    return np.sort(ritz_values[:k]), X_block, np.array(variance_history)


def trl_solver(
    H_mpo,
    k,
    max_bond,
    v0,  # Changed from initial_guess_block
    realeig,
    initial_lanczos_steps=20,  # New parameter for the initial run
    num_restarts=20,
    lanczos_steps=10,
    tol=1e-12,
    trunc_tol=1e-12,
):
    """
    Finds the 'k' lowest eigenpairs using TRL+R, but initializes the
    first basis with a standard Lanczos run.

    Args:
        H_mpo (MPO): The Hamiltonian operator.
        k (int): The number of lowest eigenvalues to find.
        max_bond (int): The maximum bond dimension for the MPS.
        v0 (MPS): A single, normalized MPS to seed the initial Lanczos run.
        realeig (np.ndarray): The true eigenvalues for error tracking.
        initial_lanczos_steps (int): The size of the initial Krylov space.
        num_restarts (int): The number of macro-restart iterations to perform.
        lanczos_steps (int): The number of new vectors to add per restart.
        tol (float): Convergence tolerance for the residual norm.
        trunc_tol (float): Truncation tolerance for MPS operations.

    Returns:
        final_eigs (np.ndarray): The k lowest computed eigenvalues.
        final_vecs (list): A list of k MPS objects for the eigenvectors.
        error_history (np.ndarray): A (num_restarts x k) array of absolute errors.
    """
    error_history = []

    # --- NEW: Phase 0: Initial Lanczos Run ---
    print(f"{'=' * 20} TRL+R Phase 0: Initial Lanczos Run {'=' * 20}")
    print(f" Generating an initial basis of size {initial_lanczos_steps}...")
    # Use the Lanczos code to generate the first Krylov basis
    V_basis, _, _, _ = lanczos_lr_py(
        H=H_mpo, v0_mps=v0, maxiter=initial_lanczos_steps, tol=trunc_tol, rmax=max_bond
    )
    print(" Initial basis generated.")

    # --- Main Restart Loop ---
    for i_restart in range(num_restarts):
        print(f"\n{'=' * 20} TRL+R Macro-Iteration {i_restart + 1}/{num_restarts} {'=' * 20}")
        print(f"   Current basis size: {len(V_basis)}")

        # --- Phase 1: Rayleigh-Ritz on the Current Basis ---
        print(f"   Phase 1: Solving Rayleigh-Ritz problem...")
        dim_basis = len(V_basis)
        W = np.zeros((dim_basis, dim_basis), dtype=np.complex128)
        P = np.zeros((dim_basis, dim_basis), dtype=np.complex128)
        for i in range(dim_basis):
            for j in range(i, dim_basis):
                W[i, j] = vdot(V_basis[i], V_basis[j])
                W[j, i] = np.conj(W[i, j])
                P[i, j] = operator_inner_product(V_basis[i], H_mpo, V_basis[j])
                P[j, i] = np.conj(P[i, j])

        try:
            ritz_values, ritz_coeffs = scipy.linalg.eigh(P, W)
        except scipy.linalg.LinAlgError:
            print("   FATAL: Eigenvalue solve failed on current basis. Stopping.")
            break

        X_block = [
            form_linear_combination(V_basis, ritz_coeffs[:, j], max_bond, trunc_tol)
            for j in range(k)
        ]

        # --- Phase 2: Calculate Residuals & Check Convergence ---
        print("   Phase 2: Calculating residuals...")
        R_block = []
        for j_k in range(k):
            ritz_val = ritz_values[j_k]
            ritz_vec = X_block[j_k]
            # H_phi = apply_operator_and_compress(H_mpo, ritz_vec, trunc_tol, max_bond)
            H_phi = apply_mpo_to_mps_variational_only(
                H_mpo, ritz_vec, max_bond=max_bond, nsweeps=3, initial_guess=ritz_vec
            )

            lambda_phi = copy.deepcopy(ritz_vec)

            # --- FIX #1: Avoid in-place multiplication ---
            lambda_phi.A[0] = lambda_phi.A[0] * -ritz_val

            residual_vec = add_mps_and_compress(H_phi, lambda_phi, trunc_tol, max_bond)
            R_block.append(residual_vec)

        residual_norms = [r.orthonormalize("right") for r in R_block]
        print(f"\n   Convergence Check: Residual norms = {np.array(residual_norms)}")

        errors = np.abs(np.sort(ritz_values[:k]) - realeig[:k])
        error_history.append(errors)
        print(f"   Current Errors = {errors}")

        if np.all(np.array(residual_norms) < tol):
            print("\nConvergence criteria met. Stopping.")
            break

        if i_restart == num_restarts - 1:
            print("\nMax restarts reached.")
            break

        # --- Phase 3: Augment the Basis (Corrected TRL Step) ---
        print("   Phase 3: Augmenting basis with new Lanczos run from residuals...")

        V_next_basis = copy.deepcopy(X_block)
        R_ortho_block = []
        for r_vec in R_block:
            for v_prev in V_next_basis:
                overlap = vdot(r_vec, v_prev)
                term = copy.deepcopy(v_prev)

                # --- FIX #2: Avoid in-place multiplication ---
                term.A[0] = term.A[0] * -overlap

                r_vec = add_mps_and_compress(r_vec, term, trunc_tol, max_bond)

            norm = r_vec.orthonormalize("right")
            if norm > 1e-12:
                V_next_basis.append(r_vec)

        if len(V_next_basis) == k:
            print("   Residuals collapsed into the existing Ritz space. Stopping.")
            break

        ortho_window_size = k
        num_generated_so_far = len(V_next_basis) - k
        for j_lan in range(num_generated_so_far, lanczos_steps):
            seed_vec = V_next_basis[-1]
            # new_vec = apply_operator_and_compress(H_mpo, seed_vec, trunc_tol, max_bond)
            new_vec = apply_mpo_to_mps_variational_only(
                H_mpo, seed_vec, max_bond=max_bond, nsweeps=3, initial_guess=seed_vec
            )

            start_index = max(0, len(V_next_basis) - ortho_window_size)
            ortho_basis = V_next_basis[start_index:]

            for v_prev in ortho_basis:
                overlap = vdot(new_vec, v_prev)
                term = copy.deepcopy(v_prev)

                # --- FIX #3: Avoid in-place multiplication ---
                term.A[0] = term.A[0] * -overlap

                new_vec = add_mps_and_compress(new_vec, term, trunc_tol, max_bond)

            norm = new_vec.orthonormalize("right")
            if norm > 1e-12:
                V_next_basis.append(new_vec)
            else:
                print(f"   Lanczos vector {j_lan + 1} collapsed. Ending augmentation early.")
                break

        V_basis = V_next_basis

    return np.sort(ritz_values[:k]), X_block, np.array(error_history)


def trl_variance_solver(
    H_mpo,
    k,
    max_bond,
    v0,  # single initial MPS
    initial_lanczos_steps=20,
    num_restarts=200,
    lanczos_steps=10,  # EXACT number of H applications per restart PER BLOCK SLOT (active ones only)
    tol=1e-12,  # locking threshold on the squared residual norm
    trunc_tol=1e-16,
):
    """MTBL (Modified Thick-Block Lanczos), with a single initial MPS.

    This is the MTBL-related residual-enriched restart routine used by the
    supplied Fermi-Hubbard examples; the historical function name is retained.
    Reference: https://doi.org/10.1038/s42005-026-02896-x

    The restart retains Ritz vectors and adds their residual directions.
    The implementation uses a recent-vector orthogonalization window and
    slot-based locking; see docs/algorithms.md for implementation limits.

    Parameters
    ----------
    H_mpo : pytenet.mpo.MPO
        Hamiltonian operator.
    k : int
        Number of requested low-energy states.
    max_bond : int
        Bond cap passed to variational MPS operations.
    v0 : pytenet.mps.MPS
        Normalized initial state.
    initial_lanczos_steps : int, default=20
        Initial Lanczos iteration count; the basis includes a continuation vector.
    num_restarts : int, default=200
        Maximum number of Rayleigh-Ritz restart cycles.
    lanczos_steps : int, default=10
        Expansion budget per active slot, excluding residual construction.
    tol : float, default=1e-12
        Locking threshold on the squared compressed residual norm, not its norm.
    trunc_tol : float, default=1e-16
        Tolerance forwarded to MPS addition/compression.

    Returns
    -------
    eigenvalues : numpy.ndarray
        Returned Ritz values; legacy partial-lock ordering is described in docs.
    eigenvectors : list of pytenet.mps.MPS
        Corresponding returned MPS vectors.
    variance_history : numpy.ndarray
        Squared compressed residuals by cycle and slot. Locked slots are
        recorded as zero placeholders, not newly measured exact variances.
    """
    np.set_printoptions(precision=10, suppress=False, formatter={"float_kind": "{:.6e}".format})

    variance_history = []

    # Track which block positions are locked; we also store their accepted eigenvalues/vectors
    locked_mask = np.zeros(k, dtype=bool)  # True if that slot is locked
    locked_eigs_accum = [None] * k  # final outputs by rank position
    locked_vecs_accum = [None] * k

    # ---- Phase 0: Initial Lanczos from single seed v0 ----
    print(f"{'=' * 20} TRL+R Phase 0: Initial Lanczos Run {'=' * 20}")
    print(f" Generating an initial basis of size {initial_lanczos_steps} from a single seed...")
    V_basis, _, _, _ = lanczos_lr_py(
        H=H_mpo, v0_mps=v0, maxiter=initial_lanczos_steps, tol=trunc_tol, rmax=max_bond
    )
    print(" Initial basis generated.")

    # ---- Main Restart Loop ----
    for i_restart in range(num_restarts):
        n_locked = int(locked_mask.sum())
        print(f"\n{'=' * 20} TRL+R Macro-Iteration {i_restart + 1}/{num_restarts} {'=' * 20}")
        print(f"   Current basis size: {len(V_basis)}  |  Locked: {n_locked}/{k}")

        if n_locked >= k:
            print(" All requested eigenpairs locked. Stopping.")
            break

        # ---- Phase 1: Rayleigh–Ritz on current basis (unchanged structure) ----
        print("   Phase 1: Solving Rayleigh–Ritz problem on the projected block...")
        dim_basis = len(V_basis)
        W = np.zeros((dim_basis, dim_basis), dtype=np.complex128)
        P = np.zeros((dim_basis, dim_basis), dtype=np.complex128)
        for i in range(dim_basis):
            for j in range(i, dim_basis):
                Wij = vdot(V_basis[i], V_basis[j])
                W[i, j] = Wij
                W[j, i] = np.conj(Wij)
                Pij = operator_inner_product(V_basis[i], H_mpo, V_basis[j])
                P[i, j] = Pij
                P[j, i] = np.conj(Pij)

        try:
            ritz_values_all, ritz_coeffs_all = scipy.linalg.eigh(P, W)
        except scipy.linalg.LinAlgError:
            print("   FATAL: Generalized eigen-solve failed on current basis. Stopping.")
            break

        # Build the k lowest Ritz vectors as MPS linear combinations of V_basis
        take = min(k, ritz_coeffs_all.shape[1])
        ritz_values = ritz_values_all[:take]
        X_block = [
            form_linear_combination(V_basis, ritz_coeffs_all[:, j], max_bond, trunc_tol)
            for j in range(take)
        ]
        for j in range(take):
            _ = X_block[j].orthonormalize("right")  # keep them normalized for variance=r^2

        # ---- Phase 2: Residuals & variance-based convergence (skip locked slots) ----
        print("   Phase 2: Form residuals and compute variances...")
        R_block = [None] * take
        variances = np.full(take, np.nan, dtype=float)
        res_norms = np.full(take, np.nan, dtype=float)

        for j_k in range(take):
            if locked_mask[j_k]:
                # This slot is frozen; report 0 variance to keep the curve flat
                variances[j_k] = 0.0
                res_norms[j_k] = 0.0
                continue

            theta = float(ritz_values[j_k])
            phi = X_block[j_k]

            # H_phi = apply_operator_and_compress(H_mpo, phi, trunc_tol, max_bond)
            H_phi = apply_mpo_to_mps_variational_only(
                H_mpo, phi, max_bond=max_bond, nsweeps=3, initial_guess=phi
            )
            minus_theta_phi = copy.deepcopy(phi)
            minus_theta_phi.A[0] = minus_theta_phi.A[0] * (-theta)

            r_vec = add_mps_and_compress(
                H_phi, minus_theta_phi, trunc_tol, max_bond
            )  # r = H|phi> - theta|phi>
            # DO NOT normalize residual before measuring
            var = float(np.real(vdot(r_vec, r_vec)))
            rn = float(np.sqrt(max(var, 0.0)))

            R_block[j_k] = r_vec
            variances[j_k] = var
            res_norms[j_k] = rn

        variance_history.append(variances.copy())
        print(f"   Variances = {variances}")
        print(f"   Residual norms (sqrt(variance)) = {res_norms}")

        # ---- Lock newly converged slots (freeze them in-place) ----
        newly_locked = []
        for j in range(take):
            if not locked_mask[j] and np.isfinite(variances[j]) and variances[j] < tol:
                locked_mask[j] = True
                locked_eigs_accum[j] = float(ritz_values[j])
                v = copy.deepcopy(X_block[j])
                _ = v.orthonormalize("right")
                locked_vecs_accum[j] = v
                newly_locked.append(j)

        if len(newly_locked) > 0:
            print(
                f"   Locked this cycle at slots {newly_locked}  (total locked: {int(locked_mask.sum())}/{k})"
            )

        if int(locked_mask.sum()) >= k:
            print("\n Convergence criteria met via locking. Stopping.")
            break

        if i_restart == num_restarts - 1:
            print("\n Max restarts reached.")
            break

        # ---- Phase 3: Build next basis
        # Start with ALL k Ritz vectors (including the frozen/locked ones)
        print("   Phase 3: Augment basis via residual-enriched (block) Lanczos...")
        V_next_basis = [copy.deepcopy(X_block[j]) for j in range(take)]

        # Append residuals only for the UNLOCKED slots, orthogonalizing vs current V_next_basis
        for j in range(take):
            if locked_mask[j]:
                continue
            r_vec = R_block[j]
            if r_vec is None:
                continue

            # GS vs a small window of current V_next_basis
            start_idx = max(0, len(V_next_basis) - k)  # window ~ block width
            for v_prev in V_next_basis[start_idx:]:
                overlap = vdot(r_vec, v_prev)
                if np.abs(overlap) > 0.0:
                    corr = copy.deepcopy(v_prev)
                    corr.A[0] = corr.A[0] * (-overlap)
                    r_vec = add_mps_and_compress(r_vec, corr, trunc_tol, max_bond)

            rn = r_vec.orthonormalize("right")
            if rn > 1e-12:
                V_next_basis.append(r_vec)

        # EXACT H expansion:
        # total allowed H applications this restart = (#UNLOCKED slots) * lanczos_steps
        n_active_slots = int((~locked_mask[:take]).sum())
        allowed_H = n_active_slots * lanczos_steps
        used_H = 0
        target_dim = k * (lanczos_steps + 2)

        while len(V_next_basis) < target_dim and used_H < allowed_H:
            seed = V_next_basis[-1]
            # new_vec = apply_operator_and_compress(H_mpo, seed, trunc_tol, max_bond)  # one H application
            new_vec = apply_mpo_to_mps_variational_only(
                H_mpo, seed, max_bond=max_bond, nsweeps=3, initial_guess=seed
            )
            used_H += 1

            # GS vs a small window of current V_next_basis
            start_idx = max(0, len(V_next_basis) - k)
            for v_prev in V_next_basis[start_idx:]:
                overlap = vdot(new_vec, v_prev)
                if np.abs(overlap) > 0.0:
                    corr = copy.deepcopy(v_prev)
                    corr.A[0] = corr.A[0] * (-overlap)
                    new_vec = add_mps_and_compress(new_vec, corr, trunc_tol, max_bond)

            nn = new_vec.orthonormalize("right")
            if nn > 1e-12:
                V_next_basis.append(new_vec)
            else:
                print("   New Lanczos vector collapsed; continuing (H op counted).")

        # If we overshot (rare), trim tail to exact target_dim
        if len(V_next_basis) > target_dim:
            V_next_basis = V_next_basis[:target_dim]

        print(
            f"   Next basis size: {len(V_next_basis)}  (target={target_dim}, used_H={used_H}/{allowed_H})"
        )
        V_basis = V_next_basis

    # ---------------- Final return ----------------
    # Gather outputs in block order (locked slots first if available; fill with last X_block if needed)
    out_eigs = []
    out_vecs = []

    for j in range(k):
        if locked_mask[j] and locked_eigs_accum[j] is not None and locked_vecs_accum[j] is not None:
            out_eigs.append(float(locked_eigs_accum[j]))
            out_vecs.append(locked_vecs_accum[j])

    # If some slots never locked, try to fill from the last RR block
    try:
        for j in range(k):
            if len(out_eigs) >= k:
                break
            out_eigs.append(float(ritz_values[j]))
            out_vecs.append(X_block[j])
    except Exception:
        pass

    out_eigs = np.array(out_eigs[:k], dtype=float)
    out_vecs = out_vecs[:k]
    return out_eigs, out_vecs, np.array(variance_history)
