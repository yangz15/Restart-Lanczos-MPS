"""Plain and conventional thick-restart Lanczos basis construction."""

import copy
import numpy as np
import scipy.linalg
from pytenet.mps import apply_mpo_to_mps_variational_only
from pytenet.operation import vdot, add_mps_and_compress, operator_inner_product
from pytenet.Lanczos import ortho_to_list


def lanczos_lr_py(H, v0_mps, maxiter, tol, rmax):
    """
    Python implementation of the rank-truncated Lanczos algorithm, mirroring the
    logic of the Dektor et al. MATLAB script.

    This function generates a non-orthogonal basis and solves the generalized
    eigenvalue problem <v|H|v> = lambda * <v|v> at each step.

    Args:
        H (MPO): The Hamiltonian operator.
        v0_mps (MPS): The initial, normalized MPS vector.
        maxiter (int): The maximum number of Lanczos iterations.
        tol (float): The truncation tolerance for SVD.
        rmax (int): The maximum allowed bond dimension.

    Returns:
        V (list): Krylov vectors, including the final unnormalized continuation vector.
        P (np.ndarray): Projected Hamiltonian for the normalized basis prefix.
        W (np.ndarray): Gram matrix for the same prefix.
        eigs (list): A list where each element contains the eigenvalues at that iteration.
    """
    # --- Initialization ---
    alpha = []
    # In MATLAB, beta starts with 1 element, then grows.
    # We will append the norm before normalization.
    beta = [0.0]

    eigs = []
    V = [copy.deepcopy(v0_mps)]

    # --- First Step (j=1) ---
    # Corresponds to MATLAB lines 14-22

    # 1. Apply operator and truncate
    # v_next corresponds to V{2} in MATLAB before orthogonalization
    # v_next = apply_operator_and_compress(H, V[0], tol, rmax)
    v_next = apply_mpo_to_mps_variational_only(
        H, V[0], max_bond=rmax, nsweeps=3, initial_guess=V[0]
    )

    # 2. Calculate alpha_0
    alpha.append(vdot(v_next, V[0]))

    # 3. Orthogonalize against v_0 and truncate
    temp = copy.deepcopy(V[0])
    # Scale MPS by multiplying the first tensor
    temp.A[0] = -alpha[0] * temp.A[0]
    v_next = add_mps_and_compress(v_next, temp, tol, rmax)

    # V[1] is now the un-normalized result of the first recurrence
    V.append(v_next)

    # 4. Initialize the W and P matrices
    W = np.array([[vdot(V[0], V[0])]], dtype=np.complex128)
    P = np.array([[operator_inner_product(V[0], H, V[0])]], dtype=np.complex128)

    print(f"Lanczos Iteration 1: Eigenvalues = {np.linalg.eigvalsh(P)}")
    eigs.append(np.sort(np.real(np.linalg.eigvalsh(P))))

    # --- Main Loop ---
    # Corresponds to MATLAB lines 23-40
    for j in range(1, maxiter):
        # 5. Calculate beta_j and normalize V[j]
        # V[j] is the un-normalized vector from the previous step
        current_beta = V[j].orthonormalize("right")
        if current_beta < 1e-14:
            print("Encountered zero-norm vector. Stopping.")
            break
        beta.append(current_beta)

        # Normalize V[j] in-place
        # V[j].A[0] /= current_beta

        # 6. Apply operator to the new normalized vector
        # v_next = apply_operator_and_compress(H, V[j], tol, rmax)
        v_next = apply_mpo_to_mps_variational_only(
            H, V[j], max_bond=rmax, nsweeps=3, initial_guess=V[j]
        )  # --- MODIFICATION: Use the current vector as an initial guess for the variational apply ---

        # 7. Calculate alpha_j
        alpha.append(vdot(v_next, V[j]))

        # 8. Three-term recurrence with sequential truncation
        # Subtract alpha_j * v_j term
        temp1 = copy.deepcopy(V[j])
        temp1.A[0] = -alpha[j] * temp1.A[0]
        v_next = add_mps_and_compress(v_next, temp1, tol, rmax)

        # Subtract beta_j * v_{j-1} term
        temp2 = copy.deepcopy(V[j - 1])
        temp2.A[0] = -beta[j] * temp2.A[0]
        v_next = add_mps_and_compress(v_next, temp2, tol, rmax)

        # Store the new un-normalized vector
        V.append(v_next)

        # 9. Update W and P matrices by adding a new row/column
        # This is more efficient than rebuilding the matrices every time
        new_dim = j + 1
        W_new = np.zeros((new_dim, new_dim), dtype=np.complex128)
        P_new = np.zeros((new_dim, new_dim), dtype=np.complex128)
        W_new[:j, :j] = W
        P_new[:j, :j] = P

        for k in range(new_dim):
            W_new[j, k] = vdot(V[j], V[k])
            W_new[k, j] = np.conj(W_new[j, k])

            P_new[j, k] = operator_inner_product(V[j], H, V[k])
            P_new[k, j] = np.conj(P_new[j, k])

        W, P = W_new, P_new

        # 10. Solve the Generalized Eigenvalue Problem
        try:
            current_eigs, _ = scipy.linalg.eigh(P, W)
            eigs.append(np.sort(np.real(current_eigs)))
        except np.linalg.LinAlgError:
            print(
                f"Warning: Generalized eigenvalue solve failed at iteration {j + 1}. W may be singular."
            )
            eigs.append(eigs[-1])  # Reuse previous eigenvalues

        # 11. Diagnostic print
        if (j + 1) % 10 == 0:
            print(f"Lanczos Iteration {j + 1}: \t Condition # of basis: {np.linalg.cond(W):.3e}")
            print(f"Lowest eigenvalue: {eigs[-1][0]}")

    return V, P, W, eigs


def trl_lr_compact(
    H,
    v0,
    m=30,
    k=10,
    nrestart=20,
    tol=1e-10,
    rmax=32,
    epsW=1e-12,
    orthog_last_to_ritz=False,
    orthog_new_to_ritz=False,
):
    """
    Traditional Thick-Restart Lanczos (TRL), in-memory (no folder).
    Restart block: k Ritz vectors + last Krylov vector (like your old TRL code).
    """

    def normalize(psi):
        n = psi.orthonormalize("right")
        if n < 1e-14:
            return n
        psi.A[0] /= n
        return n

    def axpy(out, a, x):
        tmp = copy.deepcopy(x)
        tmp.A[0] *= a
        return add_mps_and_compress(out, tmp, tol, rmax)

    def proj_out(vec, basis):
        out = copy.deepcopy(vec)
        for b in basis:
            bb = vdot(b, b)
            if abs(bb) < 1e-14:
                continue
            out = axpy(out, -(vdot(b, out) / bb), b)
        return out

    e_hist = []
    U = []  # thick part (k Ritz vectors)
    q0 = copy.deepcopy(v0)
    normalize(q0)

    for it in range(nrestart):
        # ---- build V_total of length (m+1): [U..., q0, q1, ..., q_{m-len(U)}] ----
        V = [copy.deepcopy(u) for u in U]
        for u in V:
            normalize(u)

        q = copy.deepcopy(q0)
        normalize(q)
        if orthog_new_to_ritz and len(V) > 0:
            q = proj_out(q, V)
            normalize(q)

        V.append(q)
        q_prev, beta_prev = None, 0.0

        while len(V) < (m + 1):
            qj = V[-1]
            # w = apply_operator_and_compress(H, qj, tol, rmax)
            w = apply_mpo_to_mps_variational_only(H, qj, max_bond=rmax, nsweeps=3, initial_guess=qj)

            alpha = vdot(w, qj)
            w = axpy(w, -alpha, qj)
            if q_prev is not None:
                w = axpy(w, -beta_prev, q_prev)

            if orthog_new_to_ritz and len(U) > 0:
                w = proj_out(w, U)

            beta = w.orthonormalize("right")
            if beta < 1e-14:
                break
            w.A[0] /= beta

            q_prev, beta_prev = qj, beta
            V.append(w)

        # ensure we have m+1 (for comparison runs)
        if len(V) < (m + 1):
            V += [copy.deepcopy(V[-1]) for _ in range((m + 1) - len(V))]

        Vproj = V[:m]
        v_last = V[m]

        # ---- build W,P on Vproj ----
        W = np.zeros((m, m), dtype=np.complex128)
        P = np.zeros((m, m), dtype=np.complex128)
        for i in range(m):
            for j in range(i, m):
                wij = vdot(Vproj[i], Vproj[j])
                pij = operator_inner_product(Vproj[i], H, Vproj[j])
                W[i, j] = wij
                W[j, i] = np.conj(wij)
                P[i, j] = pij
                P[j, i] = np.conj(pij)
        W = 0.5 * (W + W.conj().T)
        P = 0.5 * (P + P.conj().T)

        # ---- traditional RR metric step: S^H W S ≈ I (stable) ----
        s, Ue = np.linalg.eigh(W)
        smax = np.max(np.abs(s)) if s.size else 0.0
        floor = epsW * smax if smax > 0 else 1e-30
        s = np.maximum(s, floor)
        S = Ue @ np.diag(1.0 / np.sqrt(s)) @ Ue.conj().T

        Hred = 0.5 * (S.conj().T @ P @ S + (S.conj().T @ P @ S).conj().T)
        evals, Z = np.linalg.eigh(Hred)
        idx = np.argsort(np.real(evals))
        evals = np.real(evals[idx])
        Z = Z[:, idx]

        e_hist.append(evals[: min(5, len(evals))])

        # ---- build k Ritz MPS: y = S z, psi = sum_i y_i V_i ----
        Y = S @ Z[:, :k]
        ritz = []
        for j in range(min(k, Y.shape[1])):
            psi = None
            for i in range(m):
                c = Y[i, j]
                if abs(c) < 1e-14:
                    continue
                if psi is None:
                    psi = copy.deepcopy(Vproj[i])
                    psi.A[0] *= c
                else:
                    psi = axpy(psi, c, Vproj[i])
            normalize(psi)
            ritz.append(psi)

        # ---- TRL restart block: k Ritz + last Krylov vector ----
        U = ritz
        q0 = copy.deepcopy(v_last)
        normalize(q0)
        if orthog_last_to_ritz and len(U) > 0:
            q0 = proj_out(q0, U)
            normalize(q0)

        print(f"[restart {it}] E0={evals[0]:.12f}, cond(W)={np.linalg.cond(W):.3e}")

    return e_hist


def band_lanczos(H, initial_block, num_new_vecs, tol, rmax, lock_idx=0):
    """
    Python implementation of a Band Lanczos / MRL algorithm with
    SELECTIVE RE-ORTHOGONALIZATION.

    Each new vector is only orthogonalized against up to '2*l - 1' recent vectors
    in the basis, where 'l' is the size of the initial block.

    Args:
        H (MPO): The Hamiltonian operator.
        initial_block (list): A list of starting, normalized MPS vectors.
        num_new_vecs (int): The number of new Krylov vectors to generate.
        tol (float): The truncation tolerance for SVD.
        rmax (int): The maximum allowed bond dimension.
        lock_idx (int): The number of initial vectors to "lock" (not use for generation).

    Returns:
        V (list): A list of the generated non-orthogonal MPS Krylov vectors.
        P (np.ndarray): The final projected Hamiltonian matrix in the non-orthogonal basis.
        W (np.ndarray): The final Gram (overlap) matrix.
        eigs (list): A list where each element contains the eigenvalues at that iteration.
    """
    if not initial_block:
        raise ValueError("Initial block cannot be empty.")

    # 'l' is the width of the band (the size of the initial block)
    l = len(initial_block)
    V = copy.deepcopy(initial_block)
    eigs = []

    # Initialize the W and P matrices from the starting block
    W = np.zeros((l, l))
    P = np.zeros((l, l))
    for i in range(l):
        for k in range(i, l):
            W[i, k] = vdot(V[i], V[k])
            W[k, i] = np.conj(W[i, k])
            P[i, k] = operator_inner_product(V[i], H, V[k])
            P[k, i] = np.conj(P[i, k])

    try:
        current_eigs, _ = scipy.linalg.eigh(P, W)
        eigs.append(np.sort(np.real(current_eigs)))
        print(f"Initial block (size {l}): Lowest eigenvalue = {eigs[-1][0]}")
    except np.linalg.LinAlgError:
        print("Error: Initial block is linearly dependent.")
        return V, None, None, []

    # --- Main Loop to generate new vectors ---
    for i in range(num_new_vecs):
        # 1. Select a vector to expand from
        seed_idx = lock_idx + (i % (l - lock_idx))
        seed_vector = V[seed_idx]
        print(f"\n--- Iteration {i + 1}/{num_new_vecs} (expanding from vector {seed_idx}) ---")

        # 2. Generate new candidate vector
        # v_next = apply_operator_and_compress(H, seed_vector, tol, rmax)
        v_next = apply_mpo_to_mps_variational_only(
            H, seed_vector, max_bond=rmax, nsweeps=3, initial_guess=seed_vector
        )  # --- MODIFICATION: Use the seed vector as an initial guess for the variational apply ---

        # 3. --- MODIFICATION HERE: SELECTIVE ORTHOGONALIZATION ---
        # Orthogonalize the new vector against only the last '2l' vectors in the basis.
        # This is a fixed-window Gram-Schmidt procedure.

        # Determine the slice of the basis to orthogonalize against
        ortho_basis = V[-2 * l + 1 :]

        v_next = ortho_to_list(ortho_basis, v_next, rmax, tol)

        # 4. Normalize the new vector
        current_norm = v_next.orthonormalize("right")
        if current_norm < 1e-14:
            print("Warning: New vector has nearly zero norm. Subspace may have converged.")
            continue

        V.append(v_next)

        # 5. Update W and P matrices (still done on the FULL basis for accuracy)
        current_basis_size = len(V)
        new_dim = current_basis_size
        W_new = np.zeros((new_dim, new_dim), dtype=np.complex128)
        P_new = np.zeros((new_dim, new_dim), dtype=np.complex128)
        W_new[: new_dim - 1, : new_dim - 1] = W
        P_new[: new_dim - 1, : new_dim - 1] = P

        for k in range(new_dim):
            W_new[new_dim - 1, k] = vdot(V[-1], V[k])
            W_new[k, new_dim - 1] = np.conj(W_new[new_dim - 1, k])
            P_new[new_dim - 1, k] = operator_inner_product(V[-1], H, V[k])
            P_new[k, new_dim - 1] = np.conj(P_new[new_dim - 1, k])

        W, P = W_new, P_new

        # 6. Solve the Generalized Eigenvalue Problem
        try:
            current_eigs, _ = scipy.linalg.eigh(P, W)
            eigs.append(np.sort(np.real(current_eigs)))
        except np.linalg.LinAlgError:
            print(
                f"Warning: Generalized eigenvalue solve failed at iteration {i + 1}. W may be singular."
            )
            eigs.append(eigs[-1])

        # 7. Diagnostic print
        print(f"Basis size {new_dim}: \t Condition # of W: {np.linalg.cond(W):.3e}")
        print(f"Lowest eigenvalue: {eigs[-1][0]}")

    return V, P, W, eigs
