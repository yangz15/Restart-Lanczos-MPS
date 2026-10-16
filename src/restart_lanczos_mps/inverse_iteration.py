"""Local projected linear solves and inverse-iteration sweeps."""

import copy
import numpy as np
from numpy.linalg import norm
from scipy.sparse.linalg import LinearOperator, minres, gmres
from pytenet.mps import (
    MPS,
    local_orthonormalize_left_qr,
    local_orthonormalize_right_qr,
    merge_mps_tensor_pair,
    split_mps_tensor,
)
from pytenet.mpo import MPO, merge_mpo_tensor_pair
from pytenet.operation import (
    contraction_step_right,
    contraction_step_left,
    contraction_operator_step_right,
    contraction_operator_step_left,
    compute_right_operator_blocks,
    apply_local_hamiltonian,
    apply_operator,
)


def inverse_iteration(
    H: MPO,
    X: MPS,
    Xk_ref: MPS,
    sigma: float,
    numsweeps: int,
    local_solver: str = "gmres",  # "gmres" or "minres"
    local_maxit: int = 3,
    local_tol: float = 1e-10,
):
    # number of lattice sites
    """Solve a shifted system by single-site sweeps, modifying X and normalizing Xk_ref."""
    L = H.nsites
    assert L == L == X.nsites == Xk_ref.nsites
    # right-normalize input matrix product state
    X.orthonormalize(mode="right")
    Xk_ref.orthonormalize(mode="right")

    # right-normalize input matrix product state
    # left and right operator blocks
    # initialize leftmost block by 1x1x1 identity
    BR = compute_right_operator_blocks(X, H)
    BL = [None for _ in range(L)]
    BL[0] = np.array([[[1]]], dtype=BR[0].dtype)

    IBR = compute_right_mixed_blocks(X, X)
    IBL = [None for _ in range(L)]
    IBL[0] = np.array([[1]], dtype=IBR[0].dtype)

    WR = compute_right_mixed_blocks(X, Xk_ref)
    WL = [None for _ in range(L)]
    WL[0] = np.array([[1]], dtype=WR[0].dtype)

    for sweep in range(numsweeps):
        for i in range(L - 1):
            X.A[i] = _solve_local_projected_system(
                BL[i],
                BR[i],
                IBL[i],
                IBR[i],
                WL[i],
                WR[i],
                H_local=H.A[i],
                A_local=X.A[i],
                B_local=Xk_ref.A[i],
                sigma=sigma,
                solver=local_solver,
                maxit=local_maxit,
                tol=local_tol,
            )

            X.A[i], X.A[i + 1], X.qD[i + 1] = local_orthonormalize_left_qr(
                X.A[i], X.A[i + 1], X.qd, X.qD[i : i + 2]
            )

            BL[i + 1] = contraction_operator_step_left(X.A[i], X.A[i], H.A[i], BL[i])
            IBL[i + 1] = contraction_step_left(X.A[i], X.A[i], IBL[i])
            WL[i + 1] = contraction_step_left(Xk_ref.A[i], X.A[i], WL[i])

        for i in reversed(range(1, L)):
            X.A[i] = _solve_local_projected_system(
                BL[i],
                BR[i],
                IBL[i],
                IBR[i],
                WL[i],
                WR[i],
                H_local=H.A[i],
                A_local=X.A[i],
                B_local=Xk_ref.A[i],
                sigma=sigma,
                solver=local_solver,
                maxit=local_maxit,
                tol=local_tol,
            )

            X.A[i], X.A[i - 1], X.qD[i] = local_orthonormalize_right_qr(
                X.A[i], X.A[i - 1], X.qd, X.qD[i : i + 2]
            )

            BR[i - 1] = contraction_operator_step_right(X.A[i], X.A[i], H.A[i], BR[i])
            IBR[i - 1] = contraction_step_right(X.A[i], X.A[i], IBR[i])
            WR[i - 1] = contraction_step_right(Xk_ref.A[i], X.A[i], WR[i])

        X.A[0], _, X.qD[0] = local_orthonormalize_right_qr(
            X.A[0], np.array([[[1]]]), X.qd, X.qD[:2]
        )
        # res_hist[sweep] = norm_test(H, X, Xk_ref, sigma)

    return X


def inverse_iteration_twosite(
    H: MPO,
    X: MPS,
    Xk_ref: MPS,
    sigma: float,
    numsweeps: int,
    tol_split: float = 0,
    local_solver: str = "gmres",  # "gmres" or "minres"
    local_maxit: int = 2,
    local_tol: float = 1e-8,
):
    # number of lattice sites
    """Solve a shifted system by two-site sweeps, modifying X and normalizing Xk_ref."""
    L = H.nsites
    assert L == L == X.nsites == Xk_ref.nsites
    # right-normalize input matrix product state
    X.orthonormalize(mode="right")
    Xk_ref.orthonormalize(mode="right")

    # right-normalize input matrix product state
    # left and right operator blocks
    # initialize leftmost block by 1x1x1 identity
    BR = compute_right_operator_blocks(X, H)
    BL = [None for _ in range(L)]
    BL[0] = np.array([[[1]]], dtype=BR[0].dtype)

    IBR = compute_right_mixed_blocks(X, X)
    IBL = [None for _ in range(L)]
    IBL[0] = np.array([[1]], dtype=IBR[0].dtype)

    WR = compute_right_mixed_blocks(X, Xk_ref)
    WL = [None for _ in range(L)]
    WL[0] = np.array([[1]], dtype=WR[0].dtype)

    for sweep in range(numsweeps):
        for i in range(L - 2):
            Am = merge_mps_tensor_pair(X.A[i], X.A[i + 1])
            Hm = merge_mpo_tensor_pair(H.A[i], H.A[i + 1])
            Bm = merge_mps_tensor_pair(Xk_ref.A[i], Xk_ref.A[i + 1])

            Am = _solve_local_projected_system(
                BL[i],
                BR[i + 1],
                IBL[i],
                IBR[i + 1],
                WL[i],
                WR[i + 1],
                H_local=Hm,
                A_local=Am,
                B_local=Bm,
                sigma=sigma,
                solver=local_solver,
                maxit=local_maxit,
                tol=local_tol,
            )

            X.A[i], X.A[i + 1], X.qD[i + 1] = split_mps_tensor(
                Am, X.qd, X.qd, [X.qD[i], X.qD[i + 2]], "right", tol=tol_split
            )

            BL[i + 1] = contraction_operator_step_left(X.A[i], X.A[i], H.A[i], BL[i])
            IBL[i + 1] = contraction_step_left(X.A[i], X.A[i], IBL[i])
            WL[i + 1] = contraction_step_left(Xk_ref.A[i], X.A[i], WL[i])

        for i in reversed(range(L - 1)):
            Am = merge_mps_tensor_pair(X.A[i], X.A[i + 1])
            Hm = merge_mpo_tensor_pair(H.A[i], H.A[i + 1])
            Bm = merge_mps_tensor_pair(Xk_ref.A[i], Xk_ref.A[i + 1])

            Am = _solve_local_projected_system(
                BL[i],
                BR[i + 1],
                IBL[i],
                IBR[i + 1],
                WL[i],
                WR[i + 1],
                H_local=Hm,
                A_local=Am,
                B_local=Bm,
                sigma=sigma,
                solver=local_solver,
                maxit=local_maxit,
                tol=local_tol,
            )

            X.A[i], X.A[i + 1], X.qD[i + 1] = split_mps_tensor(
                Am, X.qd, X.qd, [X.qD[i], X.qD[i + 2]], "left", tol=tol_split
            )

            BR[i] = contraction_operator_step_right(X.A[i + 1], X.A[i + 1], H.A[i + 1], BR[i + 1])
            IBR[i] = contraction_step_right(X.A[i + 1], X.A[i + 1], IBR[i + 1])
            WR[i] = contraction_step_right(Xk_ref.A[i + 1], X.A[i + 1], WR[i + 1])

        X.A[0], _, X.qD[0] = local_orthonormalize_right_qr(
            X.A[0], np.array([[[1]]]), X.qd, X.qD[:2]
        )
        # res_hist[sweep] = norm_test(H, X, Xk_ref, sigma)

    return X


def norm_test(H: MPO, X: MPS, Xk_ref: MPS, sigma: float) -> float:
    """Return the relative dense residual norm; use only for small systems."""
    Y = apply_operator(H, X)
    R = Y.as_vector() - sigma * X.as_vector() - Xk_ref.as_vector()
    res = norm(R) / norm(Xk_ref.as_vector())
    return res


def compute_right_mixed_blocks(X: MPS, Xk_ref: MPS):
    """
    Compute all partial contractions from the right.
    """
    L = X.nsites
    assert L == Xk_ref.nsites
    WR = [None for _ in range(L)]
    # initialize rightmost dummy block
    WR[L - 1] = np.array([[1]], dtype=complex)
    for i in reversed(range(L - 1)):
        WR[i] = contraction_step_right(Xk_ref.A[i + 1], X.A[i + 1], WR[i + 1])
    return WR


def apply_local_mps(L: np.ndarray, R: np.ndarray, A: np.ndarray):
    r"""
    ______                              ______
           \                          /
      L   1|--- 1               2 ---|1   R
           |                         |
           |            0            |
           |            |            |
           |          __|__          |
           |         /  0  \         |
          0|---   ---|1 A 2|---   ---|0
     ______/         \_____/         \______

    """
    assert L.ndim == 2
    assert R.ndim == 2
    assert A.ndim == 3
    # multiply A with R tensor and store result in T
    T = np.tensordot(A, R, 1)
    # multiply T with L tensor
    T = np.tensordot(L, T, axes=(0, 1))
    T = T.transpose(1, 0, 2)
    return T


def _solve_local_projected_system(
    BL_i,
    BR_i,
    IBL_i,
    IBR_i,
    WL_i,
    WR_i,
    H_local,
    A_local,
    B_local,
    sigma: float,
    solver: str = "gmres",
    maxit: int = 15,
    tol: float = 1e-6,
):
    """
    Solve at site i the projected linear system:
        X_(≠i)^T (H - omega I) X_(≠i) vec(a) = X_(≠i)^T vec(B)
    where `a` is the unknown local tensor (updated core),
    and `B` is the reference core taken from Xk_ref.

    - `matvec` applies the effective operator (H - omega I) using (BL_i, H_local, BR_i)
    - The right-hand side is constructed by contracting the mixed blocks
      (WL_i, WR_i) with B_local
    - Returns the updated core tensor reshaped back to the shape of A_local
    """
    shape = A_local.shape
    dim = np.prod(shape)

    def matvec(v):
        # reshape vector back to local tensor form
        A_try = v.reshape(shape)
        # Apply the effective operator (H - σ I) on the single site:
        #    Heff(A_try) ≈ contraction of (BL_i) — H_local — (BR_i),
        # then subtract σ times the identity channel
        HAv = apply_local_hamiltonian(BL_i, BR_i, H_local, A_try)
        Hv_shift = HAv - sigma * apply_local_mps(IBL_i, IBR_i, A_try)
        return Hv_shift.reshape(-1)

    # Right-hand side:  X_(≠i)^T vec(B)
    # obtained by contracting WL_i, WR_i with B_local
    rhs = apply_local_mps(WL_i, WR_i, B_local).reshape(-1)

    Lop = LinearOperator((dim, dim), matvec=matvec, dtype=A_local.dtype)

    if solver == "gmres":
        sol, info = gmres(Lop, rhs, x0=None, rtol=tol, atol=0.0, restart=None, maxiter=maxit)
    else:
        # default: GMRES (more robust; H-σI does not need to be SPD)
        sol, info = sol, info = minres(Lop, rhs, rtol=tol, maxiter=maxit)

    # info == 0 means convergence; >0 means the solver did not reach tolerance
    # reshape solution back to tensor core form
    A_upd = sol.reshape(shape)
    return A_upd


def robust_inverse_iteration(H, mps_input, sigma, numsweeps):
    """
    Wraps the inverse iteration solver to ensure it returns an MPS.
    Handles cases where the external solver modifies in-place and returns None.
    """
    # Create a copy to be safe (unless your memory is very tight)
    mps_target = copy.deepcopy(mps_input)

    # Call the user's solver
    result = inverse_iteration_twosite(H, mps_input, mps_target, sigma=sigma, numsweeps=numsweeps)

    # If solver returns the MPS, use it.
    # If it returns None (in-place mod), use the modified target.
    if result is not None:
        return result
    return mps_target
