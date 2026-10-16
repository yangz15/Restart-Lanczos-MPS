"""Projection, reorthogonalization, and Ritz-vector reconstruction."""

import copy
import pickle
import numpy as np
import scipy.linalg
from pytenet.operation import vdot, add_mps_and_compress, operator_average, operator_inner_product


def solve_ritz_vecs_generalized(folder_containing_Krylov, P, W, n, max_bond, num_ritz, mpo_ref):
    """
    Constructs the Ritz vector MPSs using the stable generalized eigenvalue method.

    Args:
        folder_containing_Krylov (str): Foldername which contains Krylov vectors.
        P (np.ndarray): The projected Hamiltonian in the non-orthogonal basis (<v_i|H|v_j>).
        W (np.ndarray): The Gram matrix of the non-orthogonal basis (<v_i|v_j>).
        n (int): The current size of the Krylov subspace to use.
        max_bond (int): The max_bond used to represent the Ritz vector.
        num_ritz (int): The number of Ritz vectors to calculate.
        mpo_ref (MPO): MPO used to calculate the final energy expectation value.

    Returns:
        test_energy (list): The list of calculated Ritz values (energies).
        krylov_vec (list): A list of the explicitly constructed Ritz vector MPS objects.
    """
    # Step 1: Solve the generalized eigenvalue problem. This is the core change.
    # We solve P*v = e*W*v. The columns of 'v' are the coefficients we need.
    try:
        # We only need the first 'n' vectors from the basis
        e, v = scipy.linalg.eigh(P[:n, :n], W[:n, :n])
    except np.linalg.LinAlgError:
        print("Error: Generalized eigenvalue problem did not converge. W may be singular.")
        return [], []

    ritz_vecs = []
    ritz_vals = []

    print(f"\n--- Constructing {num_ritz} Ritz Vectors from a basis of size {n} ---")

    # Step 2: Loop to construct each of the desired Ritz vectors
    for count in range(num_ritz):
        # The k-th column of v gives the coefficients for the k-th Ritz vector.
        # There is no need to multiply by a 'coeff' matrix.
        C_ritz = v[:, count]

        # Step 3: Explicitly construct the Ritz vector MPS by summing the basis vectors.
        # NOTE: This part is still computationally expensive and introduces truncation error,
        # but it is necessary if you need the explicit MPS object.

        # Load the first basis vector to start the sum
        filename = folder_containing_Krylov + f"/Krylov_vec{0}.pkl"
        with open(filename, "rb") as file:
            # Use deepcopy to avoid modifying the stored vectors
            ritz_vec = copy.deepcopy(pickle.load(file))
        ritz_vec.A[0] = C_ritz[0] * ritz_vec.A[0]

        # Add the remaining weighted basis vectors
        for i in range(1, n):
            filename = folder_containing_Krylov + f"/Krylov_vec{i}.pkl"
            with open(filename, "rb") as file:
                temp = copy.deepcopy(pickle.load(file))

            temp.A[0] = C_ritz[i] * temp.A[0]

            # This is the expensive summation and compression step
            ritz_vec = add_mps_and_compress(ritz_vec, temp, 0, 10 * max_bond)

        # Final compression to the target bond dimension
        ritz_vec.compress_direct_svd_right_max_bond(0, max_bond)
        ritz_vec.orthonormalize("right")

        # Calculate the expectation value to verify the energy
        e_ritz = operator_average(ritz_vec, mpo_ref)

        ritz_vecs.append(ritz_vec)
        # The eigenvalue from the small matrix problem is the more accurate one
        ritz_vals.append(e[count])

        print(f"Constructed Ritz vector {count}. Ritz value: {e[count]:.12f}, <H>: {e_ritz:.12f}")

    return ritz_vals, ritz_vecs


def solve_ritz_vecs_from_list(V_basis, P, W, n, max_bond, num_ritz, mpo_ref):
    """
    Constructs the Ritz vector MPSs from a list of basis vectors using the
    stable generalized eigenvalue method.

    Args:
        V_basis (list): A list containing the non-orthogonal MPS basis vectors.
        P (np.ndarray): The projected Hamiltonian in the non-orthogonal basis (<v_i|H|v_j>).
        W (np.ndarray): The Gram matrix of the non-orthogonal basis (<v_i|v_j>).
        n (int): The current size of the Krylov subspace to use.
        max_bond (int): The max_bond used to represent the Ritz vector.
        num_ritz (int): The number of Ritz vectors to calculate.
        mpo_ref (MPO): MPO used to calculate the final energy expectation value.

    Returns:
        ritz_vals (list): The list of calculated Ritz values (energies).
        ritz_vecs (list): A list of the explicitly constructed Ritz vector MPS objects.
    """
    # Step 1: Solve the generalized eigenvalue problem.
    # The columns of 'v' are the coefficients for the non-orthogonal V_basis.
    try:
        e, v = scipy.linalg.eigh(P[:n, :n], W[:n, :n])
    except np.linalg.LinAlgError:
        print("Error: Generalized eigenvalue problem did not converge. W may be singular.")
        return [], []

    ritz_vecs = []
    ritz_vals = []

    print(f"\n--- Constructing {num_ritz} Ritz Vectors from a basis of size {n} ---")

    # Step 2: Loop to construct each of the desired Ritz vectors
    for count in range(num_ritz):
        # The k-th column of v gives the coefficients for the k-th Ritz vector.
        C_ritz = v[:, count]

        # Step 3: Explicitly construct the Ritz vector MPS by summing the basis vectors.

        # --- MODIFICATION START ---
        # Load the first basis vector from the list instead of a file
        # Use deepcopy to avoid modifying the original basis vectors in the list
        ritz_vec = copy.deepcopy(V_basis[0])
        # --- MODIFICATION END ---

        ritz_vec.A[0] = C_ritz[0] * ritz_vec.A[0]

        # Add the remaining weighted basis vectors
        for i in range(1, n):
            # --- MODIFICATION START ---
            # Load the i-th basis vector from the list
            temp = copy.deepcopy(V_basis[i])
            # --- MODIFICATION END ---

            temp.A[0] = C_ritz[i] * temp.A[0]

            # This is the expensive summation and compression step
            ritz_vec = add_mps_and_compress(ritz_vec, temp, 0, 10 * max_bond)

        ritz_vec.orthonormalize("right")

        # Calculate the expectation value to verify the energy
        e_ritz = operator_average(ritz_vec, mpo_ref)

        ritz_vecs.append(ritz_vec)
        # The eigenvalue from the small matrix problem is the more accurate one
        ritz_vals.append(e[count])

        print(f"Constructed Ritz vector {count}. Ritz value: {e[count]:.12f}, <H>: {e_ritz:.12f}")

    return ritz_vals, ritz_vecs


def get_S0(W):
    """
    Correctly implements the recursive reorthogonalization from the appendix of
    Phys. Rev. B 85, 205119, adapted for complex arithmetic.

    Args:
        W (np.ndarray): The complex, Hermitian overlap matrix W_ij = <f_i|f_j>.

    Returns:
        np.ndarray: The final transformation matrix S.
    """
    W = np.asarray(W, dtype=np.complex128)
    n_krylov = W.shape[0]

    # Initialize S and S_tilde as empty matrices. We will build them column by column.
    S = np.zeros_like(W, dtype=np.complex128)
    S_tilde = np.zeros_like(W, dtype=np.complex128)

    # Base case for n=0
    S_tilde[0, 0] = 1.0
    # The norm N_0 is sqrt(S_tilde_0 * W * S_tilde_0) = sqrt(1*W_00*1).
    # Assuming the first vector is normalized, W_00 = 1.
    N0 = np.sqrt(np.real(W[0, 0]))
    S[:, 0] = S_tilde[:, 0] / N0

    # Main recursion loop to build columns n=1, 2, ...
    for n in range(1, n_krylov):
        # This loop implements Eq. (A2) from the paper's appendix.

        # 1. Calculate the intermediate overlap terms k_in = <ψ_i|f_n>
        # k_in = Σ_k S_ki* W_kn. This uses the final, normalized S columns.
        k = np.zeros(n, dtype=np.complex128)
        for i in range(n):
            # Correct complex inner product: S_i^dagger @ W_n
            k[i] = np.dot(np.conj(S[:, i]), W[:, n])

        # 2. Build the un-normalized column S̃_n
        S_tilde[n, n] = 1.0  # For p = n
        for p in range(n):  # For p < n
            # Recursion: S̃_pn = -Σ_{q=p}^{n-1} k_qn * S_pq
            # S_pq is the p-th component of the q-th S vector
            term = 0.0
            for q in range(p, n):
                term += k[q] * S[p, q]
            S_tilde[p, n] = -term

        # 3. Calculate the norm N_n of the new un-normalized vector S̃_n
        # N_n^2 = S̃_n^dagger * W * S̃_n. This is the corrected norm formula.
        norm_sq = np.dot(np.conj(S_tilde[:, n]), W @ S_tilde[:, n])
        Nn = np.sqrt(np.real(norm_sq))

        # 4. Normalize the column to get the final S_n
        if Nn > 1e-14:
            S[:, n] = S_tilde[:, n] / Nn
        # else: S[:, n] remains zeros, indicating linear dependence.

    return S


def get_W1(N_use, V0):
    # Initialize the matrix with the complex data type
    """Return the Hermitian Gram matrix for the first N_use MPS vectors."""
    W = np.zeros([N_use, N_use], dtype=np.complex128)

    for i in range(N_use):
        for j in range(i, N_use):  # Loop only over the upper triangle (j >= i)
            # Calculate the upper-triangle element
            overlap = vdot(V0[i], V0[j])
            W[i, j] = overlap

            # Enforce the Hermitian property for the lower-triangle element
            if i != j:
                W[j, i] = np.conj(overlap)

    return W


def generate_reduced_H(N, V0, H_mpo):
    # Initialize the matrix with the complex data type
    """Return the projected Hamiltonian for the first N basis vectors."""
    H_reduced = np.zeros([N, N], dtype=np.complex128)

    for i in range(N):
        for j in range(i, N):  # Loop only over the upper triangle (j >= i)
            # Calculate the upper-triangle element
            element = operator_inner_product(V0[i], H_mpo, V0[j])
            H_reduced[i, j] = element

            # Enforce the Hermitian property for the lower-triangle element
            if i != j:
                H_reduced[j, i] = np.conj(element)

    return H_reduced


def form_linear_combination(basis, coeffs_vec, max_bond, trunc_tol):
    """Form and normalize a compressed MPS combination; a zero combination stays zero."""
    active_indices = np.where(np.abs(coeffs_vec) > 1e-14)[0]
    if len(active_indices) == 0:
        new_mps = copy.deepcopy(basis[0])
        new_mps.A[0] *= 0
        return new_mps
    start_idx = active_indices[0]
    new_mps = copy.deepcopy(basis[start_idx])
    new_mps.A[0] *= coeffs_vec[start_idx]
    for i in active_indices[1:]:
        term = copy.deepcopy(basis[i])
        term.A[0] *= coeffs_vec[i]
        new_mps = add_mps_and_compress(new_mps, term, trunc_tol, max_bond)
    new_mps.orthonormalize("right")
    return new_mps
