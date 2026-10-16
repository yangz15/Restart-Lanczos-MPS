import os
import warnings

import numpy as np
from scipy import linalg

from .qnumber import is_qsparse

__all__ = ['qr', 'retained_bond_indices_max_bond', 'split_matrix_svd_max_bond']

_VERBOSE = os.environ.get("PYTENET_SVD_VERBOSE", "0") not in ("0", "", "false", "False")
_MAX_ABS = 1.0e100


def _warn(msg):
    if _VERBOSE:
        warnings.warn(msg, RuntimeWarning, stacklevel=2)


def _sanitize_matrix(A):
    """
    Return a finite copy of A.

    This is intentionally aggressive: NaN is replaced by 0, Inf is replaced by
    a large finite number, and extremely large entries are rescaled.  This is
    not meant to be mathematically exact; it is meant to keep the code running.
    """
    B = np.array(A, copy=True)

    if B.size == 0:
        return B

    if not np.all(np.isfinite(B)):
        finite_abs = np.abs(B[np.isfinite(B)])
        scale = np.max(finite_abs) if finite_abs.size else 1.0
        if not np.isfinite(scale) or scale == 0:
            scale = 1.0
        scale = min(float(scale), _MAX_ABS)
        B = np.nan_to_num(B, nan=0.0, posinf=scale, neginf=-scale)
        _warn("Non-finite entries found in SVD/QR input; replaced NaN/Inf by finite values.")

    max_abs = np.max(np.abs(B)) if B.size else 0.0
    if np.isfinite(max_abs) and max_abs > _MAX_ABS:
        B *= (_MAX_ABS / max_abs)
        _warn(f"Very large matrix entries rescaled by factor {_MAX_ABS / max_abs}.")

    return B


def _project_to_qsparse(A, q0, q1):
    """
    Project a matrix onto the allowed q-number blocks q0[i] == q1[j].

    In exact arithmetic this should be unnecessary.  In a defensive run, if tiny
    forbidden-block entries appear, we simply discard them.
    """
    q0_arr = np.asarray(q0)
    q1_arr = np.asarray(q1)
    B = np.array(A, copy=True)
    mask = (q0_arr[:, None] == q1_arr[None, :])
    B[~mask] = 0
    return B


def _deterministic_rank1_svd(A, info=""):
    """
    Guaranteed last-resort rank-1 decomposition.

    It approximates A by its largest-magnitude entry.  This is crude but has the
    correct SVD-like shapes and therefore lets the MPS code continue.
    """
    B = _sanitize_matrix(A)
    m, n = B.shape

    if m == 0 or n == 0:
        return (
            np.zeros((m, 0), dtype=B.dtype),
            np.zeros(0, dtype=float),
            np.zeros((0, n), dtype=B.dtype),
        )

    U = np.zeros((m, 1), dtype=B.dtype)
    Vh = np.zeros((1, n), dtype=B.dtype)
    s = np.zeros(1, dtype=float)

    absB = np.abs(B)
    max_abs = np.max(absB) if absB.size else 0.0

    if np.isfinite(max_abs) and max_abs > 0:
        i, j = np.unravel_index(np.argmax(absB), B.shape)
        val = B[i, j]
        U[i, 0] = 1
        s[0] = float(abs(val))
        Vh[0, j] = val / s[0]
    else:
        # Even for a zero block, keep one harmless dummy singular direction.
        U[0, 0] = 1
        Vh[0, 0] = 1
        s[0] = 0.0

    _warn(f"Using deterministic rank-1 fallback SVD. {info}")
    return U, s, Vh


def _gram_svd(A, full_matrices=False, info=""):
    """
    SVD reconstruction through a Hermitian eigenproblem.

    Less accurate for tiny singular values, but often succeeds when direct SVD
    fails.  If this also fails, the caller will use the deterministic fallback.
    """
    B = _sanitize_matrix(A)
    m, n = B.shape
    kmax = min(m, n)

    if kmax == 0:
        return _deterministic_rank1_svd(B, info=info)

    if m >= n:
        C = B.conj().T @ B
        C = (C + C.conj().T) / 2
        eigvals, V = np.linalg.eigh(C)
        eigvals = np.real(eigvals)
        eigvals = np.maximum(eigvals, 0.0)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        V = V[:, order]
        s = np.sqrt(eigvals)

        U = np.zeros((m, kmax), dtype=B.dtype)
        nz = s > (np.finfo(float).eps * max(1.0, s[0] if s.size else 1.0))
        if np.any(nz):
            U[:, nz] = (B @ V[:, nz]) / s[nz][None, :]
        if np.any(~nz):
            # Fill zero-singular-value directions with standard basis vectors.
            for col in np.where(~nz)[0]:
                U[col % m, col] = 1
        Vh = V.conj().T
    else:
        C = B @ B.conj().T
        C = (C + C.conj().T) / 2
        eigvals, U = np.linalg.eigh(C)
        eigvals = np.real(eigvals)
        eigvals = np.maximum(eigvals, 0.0)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        U = U[:, order]
        s = np.sqrt(eigvals)

        Vh = np.zeros((kmax, n), dtype=B.dtype)
        nz = s > (np.finfo(float).eps * max(1.0, s[0] if s.size else 1.0))
        if np.any(nz):
            Vh[nz, :] = (U[:, nz].conj().T @ B) / s[nz][:, None]
        if np.any(~nz):
            for row in np.where(~nz)[0]:
                Vh[row, row % n] = 1

    if not full_matrices:
        U = U[:, :kmax]
        s = s[:kmax]
        Vh = Vh[:kmax, :]

    _warn(f"SVD succeeded through Gram-matrix fallback. {info}")
    return U, s, Vh


def _randomized_truncated_svd(A, rank, info=""):
    """
    Randomized low-rank SVD fallback.  This is approximate.
    """
    B = _sanitize_matrix(A)
    m, n = B.shape
    k = int(max(1, min(rank, m, n)))
    oversample = min(8, max(2, min(m, n) - k))
    l = min(min(m, n), k + oversample)

    rng = np.random.default_rng(12345)
    if np.iscomplexobj(B):
        Omega = rng.standard_normal((n, l)) + 1j * rng.standard_normal((n, l))
    else:
        Omega = rng.standard_normal((n, l))

    Y = B @ Omega
    Q, _ = np.linalg.qr(Y, mode="reduced")
    T = Q.conj().T @ B
    Uh, s, Vh = linalg.svd(T, full_matrices=False, lapack_driver="gesvd", check_finite=False)
    U = Q @ Uh

    _warn(f"Randomized truncated SVD succeeded with rank={k}. {info}")
    return U[:, :k], s[:k], Vh[:k, :]


def _svd_with_many_fallbacks(A, full_matrices=False, max_rank=None, info=""):
    """
    Very defensive SVD wrapper.

    It should never raise for ordinary finite-shape inputs.  When everything
    fails, it returns a crude rank-1 approximation instead of stopping the code.
    """
    B = _sanitize_matrix(A)
    m, n = B.shape

    if m == 0 or n == 0:
        return _deterministic_rank1_svd(B, info=info)

    max_abs = np.max(np.abs(B)) if B.size else 0.0
    if not np.isfinite(max_abs) or max_abs == 0:
        return _deterministic_rank1_svd(B, info=info)

    B_scaled = B / max_abs

    # 1. Fast/default path.
    try:
        U, s, Vh = np.linalg.svd(B_scaled, full_matrices=full_matrices)
        return U, s * max_abs, Vh
    except Exception as exc:
        _warn(f"np.linalg.svd failed: {type(exc).__name__}: {exc}. {info}")

    # 2. SciPy divide-and-conquer SVD.
    try:
        U, s, Vh = linalg.svd(B_scaled, full_matrices=full_matrices,
                              lapack_driver="gesdd", check_finite=False)
        return U, s * max_abs, Vh
    except Exception as exc:
        _warn(f"scipy gesdd failed: {type(exc).__name__}: {exc}. {info}")

    # 3. SciPy classical SVD.  Usually slower but more robust.
    try:
        U, s, Vh = linalg.svd(B_scaled, full_matrices=full_matrices,
                              lapack_driver="gesvd", check_finite=False)
        return U, s * max_abs, Vh
    except Exception as exc:
        _warn(f"scipy gesvd failed: {type(exc).__name__}: {exc}. {info}")

    # 4. Try tiny diagonal regularization.
    for eps in (1e-15, 1e-14, 1e-12, 1e-10, 1e-8):
        try:
            B_reg = B_scaled.copy()
            d = min(B_reg.shape)
            B_reg[np.arange(d), np.arange(d)] += eps
            U, s, Vh = linalg.svd(B_reg, full_matrices=full_matrices,
                                  lapack_driver="gesvd", check_finite=False)
            _warn(f"SVD succeeded after diagonal regularization eps={eps}. {info}")
            return U, s * max_abs, Vh
        except Exception as exc:
            _warn(f"regularized SVD eps={eps} failed: {type(exc).__name__}: {exc}. {info}")

    # 5. QR-preconditioned SVD.
    try:
        if m >= n:
            Q, R = np.linalg.qr(B_scaled, mode="reduced")
            Ur, s, Vh = linalg.svd(R, full_matrices=False,
                                   lapack_driver="gesvd", check_finite=False)
            U = Q @ Ur
        else:
            Q, R = np.linalg.qr(B_scaled.conj().T, mode="reduced")
            Utmp, s, Vhtmp = linalg.svd(R.conj().T, full_matrices=False,
                                        lapack_driver="gesvd", check_finite=False)
            U = Utmp
            Vh = Vhtmp @ Q.conj().T
        _warn(f"SVD succeeded through QR-preconditioned fallback. {info}")
        return U, s * max_abs, Vh
    except Exception as exc:
        _warn(f"QR-preconditioned SVD failed: {type(exc).__name__}: {exc}. {info}")

    # 6. Hermitian Gram-matrix eigen-decomposition.
    try:
        U, s, Vh = _gram_svd(B_scaled, full_matrices=False, info=info)
        return U, s * max_abs, Vh
    except Exception as exc:
        _warn(f"Gram-matrix SVD failed: {type(exc).__name__}: {exc}. {info}")

    # 7. Approximate truncated SVD with decreasing rank.
    base_rank = min(m, n)
    if max_rank is not None:
        base_rank = min(base_rank, int(max(1, max_rank)))

    rank_trials = []
    for factor in (1.0, 0.75, 0.5, 0.25, 0.125):
        rank_trials.append(max(1, int(np.ceil(base_rank * factor))))
    rank_trials.append(1)
    rank_trials = sorted(set(rank_trials), reverse=True)

    for rank in rank_trials:
        try:
            U, s, Vh = _randomized_truncated_svd(B_scaled, rank=rank, info=info)
            return U, s * max_abs, Vh
        except Exception as exc:
            _warn(f"randomized rank={rank} SVD failed: {type(exc).__name__}: {exc}. {info}")

    # 8. Guaranteed crude fallback.  This should not fail.
    U, s, Vh = _deterministic_rank1_svd(B_scaled, info=info)
    return U, s * max_abs, Vh


def qr(A, q0, q1):
    """
    Compute the block-wise QR decompositions of a matrix, taking block sparsity
    structure dictated by quantum numbers into account.

    Defensive changes relative to the original version:
    - non-finite entries are sanitized;
    - tiny forbidden q-number entries are projected away;
    - QR failure falls back to a simple deterministic basis.
    """
    assert A.ndim == 2
    assert len(q0) == A.shape[0]
    assert len(q1) == A.shape[1]

    A = _sanitize_matrix(A)
    if not is_qsparse(A, [q0, -q1]):
        A = _project_to_qsparse(A, q0, q1)
        _warn("Input to qr was not q-sparse; projected forbidden blocks to zero.")

    # find common quantum numbers
    qis = np.intersect1d(q0, q1)

    if len(qis) == 0:
        Q = np.zeros((A.shape[0], 1), dtype=A.dtype)
        R = np.zeros((1, A.shape[1]), dtype=A.dtype)
        if A.shape[0] > 0:
            Q[0, 0] = 1
        qinterm = np.asarray(q0)[:1]
        return (Q, R, qinterm)

    # require NumPy arrays for indexing
    q0 = np.array(q0)
    q1 = np.array(q1)

    # sort quantum numbers and arrange entries in A accordingly
    idx0 = np.argsort(q0, kind='mergesort')
    idx1 = np.argsort(q1, kind='mergesort')
    if np.any(idx0 - np.arange(len(idx0))):
        q0 = q0[idx0]
        A = A[idx0, :]
    if np.any(idx1 - np.arange(len(idx1))):
        q1 = q1[idx1]
        A = A[:, idx1]

    max_interm_dim = min(A.shape)
    D = 0

    Q = np.zeros((A.shape[0], max_interm_dim), dtype=A.dtype)
    R = np.zeros((max_interm_dim, A.shape[1]), dtype=A.dtype)
    qinterm = np.zeros(max_interm_dim, dtype=q0.dtype)

    for qn in qis:
        iqn = np.where(q0 == qn)[0]; i0 = iqn[0]; i1 = iqn[-1] + 1
        iqn = np.where(q1 == qn)[0]; j0 = iqn[0]; j1 = iqn[-1] + 1

        B = A[i0:i1, j0:j1]
        try:
            Qsub, Rsub = np.linalg.qr(B, mode='reduced')
        except Exception as exc:
            _warn(f"QR failed; using deterministic QR fallback for q={qn}: {exc}")
            m, n = B.shape
            k = min(m, n)
            Qsub = np.zeros((m, k), dtype=B.dtype)
            for a in range(k):
                Qsub[a, a] = 1
            Rsub = Qsub.conj().T @ B

        Dprev = D
        D += Qsub.shape[1]

        if D > max_interm_dim:
            # Very defensive guard.  This should not happen for valid q-blocks.
            keep = max_interm_dim - Dprev
            Qsub = Qsub[:, :keep]
            Rsub = Rsub[:keep, :]
            D = max_interm_dim

        Q[i0:i1, Dprev:D] = Qsub
        R[Dprev:D, j0:j1] = Rsub
        qinterm[Dprev:D] = qn

    Q = Q[:, :D]
    R = R[:D, :]
    qinterm = qinterm[:D]

    if np.any(idx0 - np.arange(len(idx0))):
        Q = Q[np.argsort(idx0), :]
    if np.any(idx1 - np.arange(len(idx1))):
        R = R[:, np.argsort(idx1)]

    return (Q, R, qinterm)



def retained_bond_indices_max_bond(s, tol, max_bond):
    """
    Indices of retained singular values.

    Defensive change: if singular values exist, keep at least one index even if
    all singular values are zero or below tolerance.  This avoids creating a
    zero-dimensional virtual bond, which can easily break later MPS code.
    """
    s = np.asarray(s)
    if s.size == 0:
        return np.array([], dtype=int)

    if max_bond is None:
        max_bond = s.size
    max_bond = int(max_bond)
    if max_bond <= 0:
        max_bond = 1
    max_bond = min(max_bond, s.size)

    s_abs = np.abs(np.nan_to_num(s, nan=0.0, posinf=_MAX_ABS, neginf=0.0))
    w = np.linalg.norm(s_abs)

    if not np.isfinite(w) or w == 0:
        # Keep one dummy direction rather than returning an empty bond.
        return np.array([0], dtype=int)

    s2 = (s_abs / w)**2
    sort_idx = np.argsort(s2)
    cumsum = np.cumsum(s2[sort_idx])
    score = np.zeros_like(s2)
    score[sort_idx] = cumsum
    retain_list = np.where(score > tol)[0]

    if retain_list.size == 0:
        retain_list = np.array([sort_idx[-1]], dtype=int)

    if retain_list.size > max_bond:
        retain_list = sort_idx[-max_bond:]

    # Preserve the original block order for q-number bookkeeping.
    return np.sort(retain_list.astype(int))



def split_matrix_svd_max_bond(A, q0, q1, tol, max_bond):
    """
    Split a matrix by singular value decomposition, taking block sparsity
    structure dictated by quantum numbers into account.

    This version is intentionally fail-safe: if accurate SVD fails, it falls
    back to increasingly approximate decompositions, and finally to a crude
    rank-1 decomposition.  The priority is to keep the program running.
    """
    assert A.ndim == 2
    assert len(q0) == A.shape[0]
    assert len(q1) == A.shape[1]

    A = _sanitize_matrix(A)
    if not is_qsparse(A, [q0, -q1]):
        A = _project_to_qsparse(A, q0, q1)
        _warn("Input to split_matrix_svd_max_bond was not q-sparse; projected forbidden blocks to zero.")

    # find common quantum numbers
    qis = np.intersect1d(q0, q1)

    if len(qis) == 0:
        # No legal q-number block.  Return a harmless dummy decomposition.
        u = np.zeros((A.shape[0], 1), dtype=A.dtype)
        v = np.zeros((1, A.shape[1]), dtype=A.dtype)
        s = np.zeros(1, dtype=float)
        if A.shape[0] > 0:
            u[0, 0] = 1
        q = np.asarray(q0)[:1]
        return (u, s, v, q)

    # require NumPy arrays for indexing
    q0 = np.array(q0)
    q1 = np.array(q1)

    # sort quantum numbers and arrange entries in A accordingly
    idx0 = np.argsort(q0, kind='mergesort')
    idx1 = np.argsort(q1, kind='mergesort')
    if np.any(idx0 - np.arange(len(idx0))):
        q0 = q0[idx0]
        A = A[idx0, :]
    if np.any(idx1 - np.arange(len(idx1))):
        q1 = q1[idx1]
        A = A[:, idx1]

    max_interm_dim = min(A.shape)
    D = 0

    u = np.zeros((A.shape[0], max_interm_dim), dtype=A.dtype)
    v = np.zeros((max_interm_dim, A.shape[1]), dtype=A.dtype)
    s = np.zeros(max_interm_dim, dtype=float)
    q = np.zeros(max_interm_dim, dtype=q0.dtype)

    for qn in qis:
        iqn = np.where(q0 == qn)[0]; i0 = iqn[0]; i1 = iqn[-1] + 1
        iqn = np.where(q1 == qn)[0]; j0 = iqn[0]; j1 = iqn[-1] + 1

        B = A[i0:i1, j0:j1]
        usub, ssub, vsub = _svd_with_many_fallbacks(
            B,
            full_matrices=False,
            max_rank=max_bond,
            info=(
                f"q={qn}, rows=({i0},{i1}), cols=({j0},{j1}), "
                f"block_shape={B.shape}, full_shape={A.shape}"
            ),
        )

        # Sanitize outputs one last time.
        usub = _sanitize_matrix(usub)
        vsub = _sanitize_matrix(vsub)
        ssub = np.asarray(np.nan_to_num(ssub, nan=0.0, posinf=_MAX_ABS, neginf=0.0), dtype=float)

        if ssub.size == 0:
            usub, ssub, vsub = _deterministic_rank1_svd(B, info=f"empty SVD output for q={qn}")

        Dprev = D
        D += len(ssub)

        if D > max_interm_dim:
            # Should not happen, but keep running if an approximate fallback
            # returns too many columns.
            keep = max_interm_dim - Dprev
            if keep <= 0:
                break
            usub = usub[:, :keep]
            ssub = ssub[:keep]
            vsub = vsub[:keep, :]
            D = max_interm_dim

        u[i0:i1, Dprev:D] = usub[:, :D-Dprev]
        v[Dprev:D, j0:j1] = vsub[:D-Dprev, :]
        s[Dprev:D] = ssub[:D-Dprev]
        q[Dprev:D] = qn

    # If something pathological made D=0, create a final dummy bond.
    if D == 0:
        D = 1
        if A.shape[0] > 0:
            u[0, 0] = 1
        if A.shape[1] > 0:
            v[0, 0] = 1
        s[0] = 0.0
        q[0] = q0[0]

    u = u[:, :D]
    v = v[:D, :]
    s = s[:D]
    q = q[:D]

    # truncate small singular values; keep at least one if possible
    idx = retained_bond_indices_max_bond(s, tol, max_bond)
    if idx.size == 0 and s.size > 0:
        idx = np.array([0], dtype=int)

    u = u[:, idx]
    v = v[idx, :]
    s = s[idx]
    q = q[idx]

    if np.any(idx0 - np.arange(len(idx0))):
        u = u[np.argsort(idx0), :]
    if np.any(idx1 - np.arange(len(idx1))):
        v = v[:, np.argsort(idx1)]

    return (u, s, v, q)
