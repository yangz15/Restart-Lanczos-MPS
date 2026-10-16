import numpy as np
from .mps import MPS, add_mps
from .mpo import MPO
from .qnumber import qnumber_flatten, is_qsparse
from .mps import add_mps_compress_variational_only_fast

__all__ = ['vdot', 'norm', 'operator_average', 'operator_inner_product', 'operator_density_average',
           'apply_operator', 'compute_right_operator_blocks', 'apply_local_hamiltonian', 'apply_local_bond_contraction',
           'outer_product']


def vdot(chi: MPS, psi: MPS):
    """
    Compute the dot (scalar) product `<chi | psi>`, complex conjugating `chi`.

    Args:
        chi: wavefunction represented as MPS
        psi: wavefunction represented as MPS

    Returns:
        complex: `<chi | psi>`
    """
    assert psi.nsites == chi.nsites
    if psi.nsites == 0:
        return 0
    # initialize T by identity matrix
    T = np.identity(psi.A[-1].shape[2], dtype=psi.A[-1].dtype)
    for i in reversed(range(psi.nsites)):
        T = contraction_step_right(psi.A[i], chi.A[i], T)
    # T should now be a 1x1 tensor
    assert T.shape == (1, 1)
    return T[0, 0]


def norm(psi: MPS):
    """
    Compute the standard L2 norm of a matrix product state.
    """
    return np.sqrt(vdot(psi, psi).real)


def contraction_step_right(A: np.ndarray, B: np.ndarray, R: np.ndarray):
    r"""
    Contraction step from right to left, for example to compute the
    inner product of two matrix product states.

    To-be contracted tensor network::

          _____           ______
         /     \         /
      ---|1 B*2|---   ---|1
         \__0__/         |
            |            |
                         |    R
          __|__          |
         /  0  \         |
      ---|1 A 2|---   ---|0
         \_____/         \______
    """

    assert A.ndim == 3
    assert B.ndim == 3
    assert R.ndim == 2
    # multiply with A tensor
    T = np.tensordot(A, R, 1)
    # multiply with conjugated B tensor
    Rnext = np.tensordot(T, B.conj(), axes=((0, 2), (0, 2)))
    return Rnext


def contraction_step_left(A: np.ndarray, B: np.ndarray, L: np.ndarray):
    r"""
    Contraction step from left to right, for example to compute the
    inner product of two matrix product states.

    To-be contracted tensor network::

     ______           _____
           \         /     \
          1|---   ---|1 B*2|---
           |         \__0__/
           |            |
      L    |
           |          __|__
           |         /  0  \
          0|---   ---|1 A 2|---
     ______/         \_____/
    """
    assert A.ndim == 3
    assert B.ndim == 3
    assert L.ndim == 2
    # multiply with conjugated B tensor
    T = np.tensordot(L, B.conj(), axes=(1, 1))
    # multiply with A tensor
    Lnext = np.tensordot(A, T, axes=((0, 1), (1, 0)))
    return Lnext


def operator_average(psi: MPS, op: MPO):
    """
    Compute the expectation value `<psi | op | psi>`.

    Args:
        psi: wavefunction represented as MPS
        op:  operator represented as MPO

    Returns:
        complex: `<psi | op | psi>`
    """
    assert psi.nsites == op.nsites
    if psi.nsites == 0:
        return 0
    # initialize T by identity matrix
    T = np.identity(psi.A[-1].shape[2], dtype=psi.A[-1].dtype)
    T = T.reshape((psi.A[-1].shape[2], 1, psi.A[-1].shape[2]))
    for i in reversed(range(psi.nsites)):
        T = contraction_operator_step_right(psi.A[i], psi.A[i], op.A[i], T)
    # T should now be a 1x1x1 tensor
    assert T.shape == (1, 1, 1)
    return T[0, 0, 0]


def operator_inner_product(chi: MPS, op: MPO, psi: MPS):
    """
    Compute the inner product `<chi | op | psi>`.

    Args:
        chi: wavefunction represented as MPS
        op:  operator represented as MPO
        psi: wavefunction represented as MPS

    Returns:
        complex: `<chi | op | psi>`
    """
    assert chi.nsites == op.nsites
    assert psi.nsites == op.nsites
    if psi.nsites == 0:
        return 0
    # initialize T by identity matrix
    assert chi.A[-1].shape[2] == psi.A[-1].shape[2]
    T = np.identity(psi.A[-1].shape[2], dtype=psi.A[-1].dtype)
    T = T.reshape((psi.A[-1].shape[2], 1, psi.A[-1].shape[2]))
    for i in reversed(range(psi.nsites)):
        T = contraction_operator_step_right(psi.A[i], chi.A[i], op.A[i], T)
    # T should now be a 1x1x1 tensor
    assert T.shape == (1, 1, 1)
    return T[0, 0, 0]


def operator_density_average(rho: MPO, op: MPO):
    """
    Compute the expectation value `tr[op rho]`.

    Args:
        rho: density matrix represented as MPO
        op:  operator represented as MPO

    Returns:
        complex: `tr[op rho]`
    """
    assert rho.nsites == op.nsites
    if rho.nsites == 0:
        return 0
    # initialize T as 1x1 matrix
    T = np.identity(1, dtype=rho.A[-1].dtype)
    for i in reversed(range(rho.nsites)):
        T = contraction_operator_density_step_right(rho.A[i], op.A[i], T)
    # T should now be a 1x1 matrix
    assert T.shape == (1, 1)
    return T[0, 0]


def apply_operator(op: MPO, psi: MPS) -> MPS:
    """
    Apply an operator represented as MPO to a state in MPS form.
    """
    # quantum numbers on physical sites must match
    assert np.array_equal(psi.qd, op.qd)
    assert psi.nsites == op.nsites
    # bond quantum numbers
    qD = [qnumber_flatten((op.qD[i], psi.qD[i])) for i in range(psi.nsites + 1)]
    op_psi = MPS(psi.qd, qD, fill='postpone')
    for i in range(psi.nsites):
        A = np.tensordot(op.A[i], psi.A[i], axes=(1, 0))
        A = A.transpose((0, 1, 3, 2, 4))
        # group virtual bonds
        s = A.shape
        A = A.reshape((s[0], s[1]*s[2], s[3]*s[4]))
        op_psi.A[i] = A
        assert is_qsparse(op_psi.A[i], [op_psi.qd, op_psi.qD[i], -op_psi.qD[i+1]]), \
            'sparsity pattern of MPS tensor does not match quantum numbers'
    return op_psi


def contraction_operator_step_right(A: np.ndarray, B: np.ndarray, W: np.ndarray, R: np.ndarray):
    r"""
    Contraction step from right to left, with a matrix product operator
    sandwiched in between.

    To-be contracted tensor network::

          _____           ______
         /     \         /
      ---|1 B*2|---   ---|2
         \__0__/         |
            |            |
                         |
          __|__          |
         /  0  \         |
      ---|2 W 3|---   ---|1   R
         \__1__/         |
            |            |
                         |
          __|__          |
         /  0  \         |
      ---|1 A 2|---   ---|0
         \_____/         \______
    """
    assert A.ndim == 3
    assert B.ndim == 3
    assert W.ndim == 4
    assert R.ndim == 3
    # multiply with A tensor
    T = np.tensordot(A, R, 1)
    # multiply with W tensor
    T = np.tensordot(W, T, axes=((1, 3), (0, 2)))
    # interchange levels 0 <-> 2 in T
    T = T.transpose((2, 1, 0, 3))
    # multiply with conjugated B tensor
    Rnext = np.tensordot(T, B.conj(), axes=((2, 3), (0, 2)))
    return Rnext


def contraction_operator_step_left(A: np.ndarray, B: np.ndarray, W: np.ndarray, L: np.ndarray):
    r"""
    Contraction step from left to right, with a matrix product operator
    sandwiched in between.

    To-be contracted tensor network::

     ______           _____
           \         /     \
          2|---   ---|1 B*2|---
           |         \__0__/
           |            |
           |
           |          __|__
           |         /  0  \
      L   1|---   ---|2 W 3|---
           |         \__1__/
           |            |
           |
           |          __|__
           |         /  0  \
          0|---   ---|1 A 2|---
     ______/         \_____/
    """
    assert A.ndim == 3
    assert B.ndim == 3
    assert W.ndim == 4
    assert L.ndim == 3
    # multiply with conjugated B tensor
    T = np.tensordot(L, B.conj(), axes=(2, 1))
    # multiply with W tensor
    T = np.tensordot(W, T, axes=((0, 2), (2, 1)))
    # multiply with A tensor
    Lnext = np.tensordot(A, T, axes=((0, 1), (0, 2)))
    return Lnext


def contraction_operator_density_step_right(A: np.ndarray, W: np.ndarray, R: np.ndarray):
    r"""
    Contraction step between two matrix product operators
    (typically density matrix and Hamiltonian).

    To-be contracted tensor network (with = denoting a connected loop)::

            =
          __|__           ______
         /  0  \         /
      ---|2 W 3|---   ---|1
         \__1__/         |
            |            |
                         |    R
          __|__          |
         /  0  \         |
      ---|2 A 3|---   ---|0
         \__1__/         \______
            |
            =
    """
    assert A.ndim == 4
    assert W.ndim == 4
    assert R.ndim == 2
    # multiply with A tensor
    T = np.tensordot(A, R, 1)
    # multiply with W tensor
    T = np.tensordot(T, W, axes=((1, 0, 3), (0, 1, 3)))
    return T


def compute_right_operator_blocks(psi: MPS, op: MPO):
    """
    Compute all partial contractions from the right.
    """
    L = psi.nsites
    assert L == op.nsites
    BR = [None for _ in range(L)]
    # initialize rightmost dummy block
    BR[L-1] = np.array([[[1]]], dtype=complex)
    for i in reversed(range(L-1)):
        BR[i] = contraction_operator_step_right(psi.A[i+1], psi.A[i+1], op.A[i+1], BR[i+1])
    return BR


def apply_local_hamiltonian(L: np.ndarray, R: np.ndarray, W: np.ndarray, A: np.ndarray):
    r"""
    Apply a local Hamiltonian operator.

    To-be contracted tensor network (the indices at the open legs
    show the ordering for the output tensor)::

     ______                           ______
           \                         /
          2|---1                 2---|2
           |                         |
           |                         |
           |            0            |
           |          __|__          |
           |         /  0  \         |
      L   1|---   ---|2 W 3|---   ---|1   R
           |         \__1__/         |
           |            |            |
           |                         |
           |          __|__          |
           |         /  0  \         |
          0|---   ---|1 A 2|---   ---|0
     ______/         \_____/         \______
    """
    assert L.ndim == 3
    assert R.ndim == 3
    assert W.ndim == 4
    assert A.ndim == 3
    # multiply A with R tensor and store result in T
    T = np.tensordot(A, R, 1)
    # multiply T with W tensor
    T = np.tensordot(W, T, axes=((1, 3), (0, 2)))
    # multiply T with L tensor
    T = np.tensordot(T, L, axes=((2, 1), (0, 1)))
    # interchange levels 1 <-> 2 in T
    T = T.transpose((0, 2, 1))
    return T


def apply_local_bond_contraction(L, R, C):
    r"""
    Apply "zero-site" bond contraction.

    To-be contracted tensor network::

     ______                           ______
           \                         /
          2|---                   ---|2
           |                         |
           |                         |
           |                         |
           |                         |
           |                         |
      L   1|-----------   -----------|1   R
           |                         |
           |                         |
           |                         |
           |          _____          |
           |         /     \         |
          0|---   ---|0 C 1|---   ---|0
     ______/         \_____/         \______
    """
    assert L.ndim == 3
    assert R.ndim == 3
    assert C.ndim == 2
    # multiply C with R tensor and store result in T
    T = np.tensordot(C, R, 1)
    # multiply L with T tensor
    T = np.tensordot(L, T, axes=((0, 1), (0, 1)))
    return T


#added by Yu:

def apply_operator_and_compress(op: MPO, psi: MPS, tol, max_bond_apply):
    """
    Apply an operator represented as MPO to a state in MPS form.
    
    And then compress it, with tol and maximum bond dims.
    
    See ptn.mps.local_orthonormalize_right_svd_max_bond for compression details.
    """
    # quantum numbers on physical sites must match
    op_psi = apply_operator(op, psi)
    #by default, mode = 'left' for the following line
    op_psi.compress_no_normalization_max_bond(tol, max_bond = max_bond_apply)
            
    return op_psi

def add_mps_and_compress(psi1: MPS, psi2: MPS, tol, max_bond_apply):
    """
    Add two MPS and compress the result.

    Primary route:
        psi = add_mps(psi1, psi2)
        psi.compress_no_normalization_max_bond(tol, max_bond=max_bond_apply)

    Fallback route:
        If the direct SVD-based compression fails, fall back to
        add_mps_compress_variational_only_fast(psi1, psi2, max_bond_apply).

    This version prioritizes robustness: if the SVD compression fails,
    the program continues with the variational-only compression.
    """
    try:
        psi = add_mps(psi1, psi2)
        psi.compress_no_normalization_max_bond(tol, max_bond=max_bond_apply)
        return psi

    except Exception as err:
        print(
            "[WARNING] add_mps_and_compress: direct SVD compression failed. "
            "Falling back to add_mps_compress_variational_only_fast.\n"
            f"Original error: {type(err).__name__}: {err}"
        )

        return add_mps_compress_variational_only_fast(
            psi1,
            psi2,
            max_bond_apply,
        )

def add_mps_and_compress_direct_SVD(psi1: MPS, psi2: MPS, tol, max_bond_apply):
    """
    Add two MPS, and then compress it, with tol and maximum bond dims.
    Start from left-canonical form, bring to right-canonical form
    """
    psi = add_mps(psi1, psi2)
    psi.compress_direct_svd_left_max_bond(tol, max_bond = max_bond_apply)

    return psi


def outer_product(psi: MPS, phi: MPS) -> MPO:
    r"""
    Build the matrix‑product operator that corresponds to the outer product
    :math:`|\Psi\rangle\langle\Phi|`.

    The resulting MPO has physical dimensions *(σ_out , σ_in)* and virtual
    bond dimensions that are the Kronecker products of the bonds of *psi* and
    *phi*.  Quantum numbers are combined consistently so that the standard
    sparsity checks (`is_qsparse`) continue to hold.

    Parameters
    ----------
    psi : MPS
        Ket state :math:`|\Psi\rangle`.
    phi : MPS
        Ket state :math:`|\Phi\rangle` († is taken inside this routine).

    Returns
    -------
    MPO
        An MPO `O` with

        .. math::
            O[\sigma_1\!\ldots\sigma_L,\ \sigma'_1\!\ldots\sigma'_L]
            = \Psi_{\sigma_1\!\ldots\sigma_L}\;
              \Phi^*_{\sigma'_1\!\ldots\sigma'_L}.
    """
    # --- sanity checks ----------------------------------------------------
    assert psi.nsites == phi.nsites,  "number of sites must agree"
    assert np.array_equal(psi.qd,  phi.qd), "local basis (qd) must match"

    L   = psi.nsites
    d   = len(psi.qd)

    # ----------------------------------------------------------------------
    # 1. build combined virtual quantum numbers  Qᵢ = qD_ψᵢ  ⊕ (−qD_φᵢ)
    #    ⊕   = element‑wise outer sum produced by `qnumber_flatten`
    # ----------------------------------------------------------------------
    qD_mpo = [qnumber_flatten([ psi.qD[i] , -phi.qD[i] ]) for i in range(L+1)]

    # 2. allocate empty MPO container
    mpo = MPO(psi.qd, qD_mpo, fill='postpone')

    # 3. create every local MPO tensor
    for i in range(L):
        A = psi.A[i]            # shape (d , Dψ_L , Dψ_R)
        B = phi.A[i].conj()     # shape (d , Dφ_L , Dφ_R)   (bra part)

        # |Ψ⟩⟨Φ| → W[σ,σ', l , r] = A[σ, lψ , rψ] * B[σ', lφ , rφ]^*
        W = np.tensordot(A, B, axes=0)                 # (d, DψL, DψR, d, DφL, DφR)
        W = W.transpose((0, 3, 1, 4, 2, 5))            # (σ, σ', DψL, DφL, DψR, DφR)
        W = W.reshape((d, d,
                       A.shape[1] * B.shape[1],        # left  bond  dim
                       A.shape[2] * B.shape[2]))       # right bond dim

        mpo.A[i] = W

        # consistency (debug switch off in production if desired)
        assert is_qsparse(
            mpo.A[i],
            [mpo.qd, -mpo.qd, mpo.qD[i], -mpo.qD[i+1]]
        ), "sparsity pattern mismatch at site %d" % i

    return mpo
