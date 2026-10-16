"""Shift-and-invert residual-enriched restart with overlap-based state tracking."""

import copy
import numpy as np
import scipy.linalg
from scipy.optimize import linear_sum_assignment
from pytenet.mps import apply_mpo_to_mps_variational_only
from pytenet.operation import vdot, add_mps_and_compress, operator_inner_product
from .ritz import form_linear_combination
from .inverse_iteration import robust_inverse_iteration


def shift_invert_trl_solver(
    H_mpo,
    k,
    max_bond,
    initial_guess_block,
    sigma,
    num_restarts=50,
    lanczos_steps=20,
    tol=1e-12,
    trunc_tol=1e-12,
):
    """
    TRL+R Solver with State Tracking and Robust Ghost Handling.

    Features:
      1. Tracks states via Overlap (Linear Assignment) to fix plot jumps.
      2. Fixes 'NoneType' error in inverse iteration.
      3. Uses full re-orthogonalization for stability.
    """

    # --- Initialize Basis ---
    V_basis = [copy.deepcopy(psi) for psi in initial_guess_block]

    # Storage for history
    variance_history = []
    energy_history = []

    # We keep track of the PREVIOUS Ritz vectors to align the new ones
    prev_X_block = None

    # Safety: Ensure initial guess is orthonormal
    for i in range(len(V_basis)):
        for j in range(i):
            overlap = vdot(V_basis[j], V_basis[i])
            term = copy.deepcopy(V_basis[j])
            term.A[0] *= -overlap
            V_basis[i] = add_mps_and_compress(V_basis[i], term, trunc_tol, max_bond)
        V_basis[i].orthonormalize("right")

    # --- Main TRL Loop ---
    for i_restart in range(num_restarts):
        print(f"\n{'=' * 20} Iteration {i_restart + 1}/{num_restarts} {'=' * 20}")
        print(f"  Basis size: {len(V_basis)}")

        # ---------------------------------------------------------
        # Phase 1: Rayleigh-Ritz Projection
        # ---------------------------------------------------------
        print(f"  Phase 1: Solving Projected Eigenproblem...")
        dim = len(V_basis)
        W = np.zeros((dim, dim), dtype=np.complex128)  # Overlap
        P = np.zeros((dim, dim), dtype=np.complex128)  # Hamiltonian

        for i in range(dim):
            for j in range(i, dim):
                # vdot conjugates the first argument: <Vi | Vj>
                W[i, j] = vdot(V_basis[i], V_basis[j])
                W[j, i] = np.conj(W[i, j])

                P[i, j] = operator_inner_product(V_basis[i], H_mpo, V_basis[j])
                P[j, i] = np.conj(P[i, j])

        try:
            # Solve P * c = val * W * c
            # Returns eigenvalues sorted lowest -> highest (Algebraic)
            all_vals, all_vecs = scipy.linalg.eigh(P, W)
        except scipy.linalg.LinAlgError:
            print("  FATAL: Generalized eigenvalue solve failed.")
            break

        # 1. Selection: Pick 'k' states closest to sigma
        distances = np.abs(all_vals - sigma)
        target_idx = np.argsort(distances)[:k]

        selected_vals = all_vals[target_idx]
        selected_coeffs = all_vecs[:, target_idx]

        # Build the 'k' candidate Ritz vectors (MPS)
        candidates_X = [
            form_linear_combination(V_basis, selected_coeffs[:, j], max_bond, trunc_tol)
            for j in range(k)
        ]

        # 2. Tracking: Reorder candidates to match previous iteration
        # If this is the first step, we just sort by Energy.
        # If not, we sort by Overlap with prev_X_block.

        final_permutation = np.arange(k)  # Default: no change

        if prev_X_block is None:
            # First run: Sort by Energy (Real part)
            final_permutation = np.argsort(selected_vals.real)
        else:
            # Subsequent runs: Match via Overlap (The Hungarian Algorithm)
            print("  ... Tracking states via Overlap matching ...")
            cost_matrix = np.zeros((k, k))

            for i_new in range(k):
                for j_old in range(k):
                    # Compute overlap magnitude
                    ovlp = abs(vdot(prev_X_block[j_old], candidates_X[i_new]))
                    # Cost is 1 - overlap (we want to minimize cost / maximize overlap)
                    cost_matrix[j_old, i_new] = 1.0 - ovlp

            # linear_sum_assignment finds indices that minimize total cost
            row_inds, col_inds = linear_sum_assignment(cost_matrix)

            # col_inds contains the new indices sorted to match the old rows (0..k)
            final_permutation = col_inds

        # Apply the permutation
        ritz_values = selected_vals[final_permutation]
        X_block = [candidates_X[i] for i in final_permutation]

        # Update History
        prev_X_block = X_block  # Store for next matching
        energy_history.append(ritz_values.real)

        # ---------------------------------------------------------
        # Phase 2: Residuals & Convergence Check
        # ---------------------------------------------------------
        print("  Phase 2: Checking Convergence...")
        R_block = []
        current_variances = []
        si_residuals = []

        for j in range(k):
            # (A) Energy Variance (H - E)
            # H_psi = apply_operator_and_compress(H_mpo, X_block[j], trunc_tol, max_bond)
            H_psi = apply_mpo_to_mps_variational_only(
                H_mpo, X_block[j], max_bond=max_bond, nsweeps=3, initial_guess=X_block[j]
            )
            E_psi = copy.deepcopy(X_block[j])
            E_psi.A[0] *= -ritz_values[j]

            res_vec_phys = add_mps_and_compress(H_psi, E_psi, trunc_tol, max_bond)
            phys_err = res_vec_phys.orthonormalize("right")
            current_variances.append(phys_err**2)

            # (B) Shift-Invert Residual (H - sigma)^-1
            # FIX: Used the robust wrapper here to prevent NoneType error
            H_inv_psi = robust_inverse_iteration(H_mpo, X_block[j], sigma, numsweeps=2)

            denom = ritz_values[j] - sigma
            inv_val = 1.0 / denom if abs(denom) > 1e-15 else 0.0

            scale_psi = copy.deepcopy(X_block[j])
            scale_psi.A[0] *= -inv_val

            res_vec_si = add_mps_and_compress(H_inv_psi, scale_psi, trunc_tol, max_bond)
            norm_si = res_vec_si.orthonormalize("right")

            si_residuals.append(norm_si)
            R_block.append(res_vec_si)

        variance_history.append(current_variances)
        current_variances = np.array(current_variances)

        print(f"    Energies:  {ritz_values.real}")
        print(f"    Variances: {current_variances}")

        # Check Convergence
        if np.all(np.array(si_residuals) < tol):
            print(f"\nCONVERGED at iteration {i_restart + 1}!")
            return (ritz_values, X_block, np.array(variance_history), np.array(energy_history))

        if i_restart == num_restarts - 1:
            print("\nMax restarts reached without full convergence.")
            break

        # ---------------------------------------------------------
        # Phase 3: Basis Enrichment (TRL)
        # ---------------------------------------------------------
        print("  Phase 3: Augmenting Basis...")

        V_next_basis = copy.deepcopy(X_block)

        # Helper for Safe Orthogonalization
        def robust_orthogonalize_and_add(candidate, basis_list):
            for v_existing in basis_list:
                overlap = vdot(v_existing, candidate)
                term = copy.deepcopy(v_existing)
                term.A[0] *= -overlap
                candidate = add_mps_and_compress(candidate, term, trunc_tol, max_bond)

            norm = candidate.orthonormalize("right")
            # 1e-10 threshold prevents adding numerical noise
            if norm > 1e-10:
                basis_list.append(candidate)
                return True
            return False

        # 1. Add Residuals
        for r_vec in R_block:
            robust_orthogonalize_and_add(r_vec, V_next_basis)

        if len(V_next_basis) == k:
            print("    Residuals collapsed. Subspace is invariant.")
            break

        # 2. Lanczos Extension
        target_size = len(V_next_basis) + lanczos_steps

        while len(V_next_basis) < target_size:
            seed = V_next_basis[-1]

            # FIX: Robust wrapper here too
            new_vec = robust_inverse_iteration(H_mpo, seed, sigma, numsweeps=2)

            added = robust_orthogonalize_and_add(new_vec, V_next_basis)
            if not added:
                print(f"    Lanczos breakdown at size {len(V_next_basis)}.")
                break

        V_basis = V_next_basis

    return (ritz_values, X_block, np.array(variance_history), np.array(energy_history))
