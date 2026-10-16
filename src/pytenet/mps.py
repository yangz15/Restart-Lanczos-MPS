from collections.abc import Sequence
import numpy as np
from .qnumber import qnumber_outer_sum, qnumber_flatten, is_qsparse
from .bond_ops import qr, retained_bond_indices, split_matrix_svd
from .bond_ops_max_bond import split_matrix_svd_max_bond
from .util import crandn
from .mpo import MPO

#__all__ = ['MPS', 'merge_mps_tensor_pair', 'split_mps_tensor']


class MPS:
    """
    Matrix product state (MPS) class.

    The i-th MPS tensor has dimension `[d, D[i], D[i+1]]` with `d` the physical
    dimension at each site and `D` the list of virtual bond dimensions.

    Quantum numbers are assumed to be additive and stored as integers.
    `qd` stores the list of physical quantum numbers at each site,
    and `qD` the virtual bond quantum numbers.
    The sum of physical and left virtual bond quantum number of each non-zero
    tensor entry must be equal to the right virtual bond quantum number.
    """

    def __init__(self, qd: Sequence[int], qD: Sequence[Sequence[int]], fill=0.0, rng: np.random.Generator=None):
        """
        Create a matrix product state.

        Args:
            qd: physical quantum numbers at each site (same for all sites)
            qD: virtual bond quantum numbers (list of quantum number lists)
            fill: explicit scalar number to fill MPS tensors with, or
                  'random' to initialize tensors with random complex entries, or
                  'postpone' to leave MPS tensors unallocated
            rng: (optional) random number generator for drawing entries
        """
        # require NumPy arrays
        self.qd = np.array(qd)
        self.qD = [np.array(qDi) for qDi in qD]
        # create list of MPS tensors
        d = len(qd)
        D = [len(qb) for qb in qD]
        # leading and trailing bond dimensions must be 1
        assert D[0] == 1 and D[-1] == 1
        if isinstance(fill, (int, float, complex)):
            self.A = [np.full((d, D[i], D[i+1]), fill) for i in range(len(D)-1)]
        elif fill == 'random':
            # random complex entries
            if rng is None:
                rng = np.random.default_rng()
            self.A = [crandn((d, D[i], D[i+1]), rng) / np.sqrt(d*D[i]*D[i+1]) for i in range(len(D)-1)]
        elif fill == 'postpone':
            self.A = (len(D) - 1) * [None]
        else:
            raise ValueError(f'fill = {fill} invalid; must be a number, "random" or "postpone".')
        if fill != 'postpone':
            # enforce block sparsity structure dictated by quantum numbers
            for i in range(len(self.A)):
                mask = qnumber_outer_sum([self.qd, self.qD[i], -self.qD[i+1]])
                self.A[i] = np.where(mask == 0, self.A[i], 0)

    @property
    def nsites(self) -> int:
        """
        Number of lattice sites.
        """
        return len(self.A)

    @property
    def bond_dims(self) -> list:
        """
        Virtual bond dimensions.
        """
        if len(self.A) == 0:
            return []
        D = [self.A[i].shape[1] for i in range(len(self.A))]
        D.append(self.A[-1].shape[2])
        return D

    def zero_qnumbers(self):
        """
        Set all quantum numbers to zero (effectively disabling them).
        """
        self.qd.fill(0)
        for i in range(len(self.qD)):
            self.qD[i].fill(0)
        # enable chaining
        return self

    def orthonormalize(self, mode='left'):
        """
        Left- or right-orthonormalize the MPS using QR decompositions.
        """
        if len(self.A) == 0:
            return 1

        if mode == 'left':
            for i in range(len(self.A) - 1):
                self.A[i], self.A[i+1], self.qD[i+1] = local_orthonormalize_left_qr(self.A[i], self.A[i+1], self.qd, self.qD[i:i+2])
            # last tensor
            self.A[-1], T, self.qD[-1] = local_orthonormalize_left_qr(self.A[-1], np.array([[[1]]]), self.qd, self.qD[-2:])
            # normalization factor (real-valued since diagonal of R matrix is real)
            assert T.shape == (1, 1, 1)
            nrm = T[0, 0, 0].real
            if nrm < 0:
                # flip sign such that normalization factor is always non-negative
                self.A[-1] = -self.A[-1]
                nrm = -nrm
            return nrm
        if mode == 'right':
            for i in reversed(range(1, len(self.A))):
                self.A[i], self.A[i-1], self.qD[i] = local_orthonormalize_right_qr(self.A[i], self.A[i-1], self.qd, self.qD[i:i+2])
            # first tensor
            self.A[0], T, self.qD[0] = local_orthonormalize_right_qr(self.A[0], np.array([[[1]]]), self.qd, self.qD[:2])
            # normalization factor (real-valued since diagonal of R matrix is real)
            assert T.shape == (1, 1, 1)
            nrm = T[0, 0, 0].real
            if nrm < 0:
                # flip sign such that normalization factor is always non-negative
                self.A[0] = -self.A[0]
                nrm = -nrm
            return nrm
        raise ValueError(f'mode = {mode} invalid; must be "left" or "right".')

    def compress(self, tol: float, mode='left'):
        """
        Compress and orthonormalize a MPS by site-local SVDs and singular value truncations.

        Returns original norm and scaling factor due to compression.
        """
        if mode == 'left':
            # transform to right-canonical form first
            nrm = self.orthonormalize(mode='right')
            for i in range(len(self.A) - 1):
                self.A[i], self.A[i+1], self.qD[i+1] = local_orthonormalize_left_svd(self.A[i], self.A[i+1], self.qd, self.qD[i:i+2], tol)
                assert is_qsparse(self.A[i], [self.qd, self.qD[i], -self.qD[i+1]]), \
                    'sparsity pattern of MPS tensor does not match quantum numbers'
            # last tensor
            self.A[-1], T, self.qD[-1] = local_orthonormalize_left_svd(self.A[-1], np.array([[[1]]]), self.qd, self.qD[-2:], tol)
            assert is_qsparse(self.A[-1], [self.qd, self.qD[-2], -self.qD[-1]]), \
                'sparsity pattern of MPS tensor does not match quantum numbers'
            assert T.shape == (1, 1, 1)
            # absorb potential phase factor into MPS tensor
            self.A[-1] *= T[0, 0, 0] / abs(T[0, 0, 0])
            return (nrm, abs(T[0, 0, 0]))
        #return: left-cano MPS
        
        if mode == 'right':
            # transform to left-canonical form first
            nrm = self.orthonormalize(mode='left')
            for i in reversed(range(1, len(self.A))):
                self.A[i], self.A[i-1], self.qD[i] = local_orthonormalize_right_svd(self.A[i], self.A[i-1], self.qd, self.qD[i:i+2], tol)
                assert is_qsparse(self.A[i], [self.qd, self.qD[i], -self.qD[i+1]]), \
                    'sparsity pattern of MPS tensor does not match quantum numbers'
            # first tensor
            self.A[0], T, self.qD[0] = local_orthonormalize_right_svd(self.A[0], np.array([[[1]]]), self.qd, self.qD[:2], tol)
            assert is_qsparse(self.A[0], [self.qd, self.qD[0], -self.qD[1]]), \
                'sparsity pattern of MPS tensor does not match quantum numbers'
            assert T.shape == (1, 1, 1)
            # absorb potential phase factor into MPS tensor
            self.A[0] *= T[0, 0, 0] / abs(T[0, 0, 0])
            return (nrm, abs(T[0, 0, 0]))
        raise ValueError(f'mode = {mode} invalid; must be "left" or "right".')
        #return: lright-cano MPS
    
    def compress_no_normalization_max_bond(self, tol: float, mode='left', max_bond = 10):
        r"""
        This compression doesn't work with normalized MPS. Mainly used for H|\psi> calculation, which is not normalzied.
        
        Note that we don't return the norm in this function, while previous self.compress() returns it.
        
        Note that, in compress considering normalization, one need to absorb T to make sure the whole MPS normalized, but
        one don't have to do it for this version. 
        
        Compress and orthonormalize a MPS by site-local SVDs and singular value truncations.
        
        Attention: the tol here works on normalized matrix, thus the final error will be scaled by nrm!
        """
        if mode == 'left':
            # transform to right-canonical form first
            nrm = self.orthonormalize(mode='right')
            # cancel the normalization in this step:
            self.A[0] = nrm* self.A[0]
            for i in range(len(self.A) - 1):
                self.A[i], self.A[i+1], self.qD[i+1] = local_orthonormalize_left_svd_max_bond(self.A[i], self.A[i+1], self.qd, self.qD[i:i+2], tol, max_bond)
                assert is_qsparse(self.A[i], [self.qd, self.qD[i], -self.qD[i+1]]), \
                    'sparsity pattern of MPS tensor does not match quantum numbers'
            # last tensor
            self.A[-1], T, self.qD[-1] = local_orthonormalize_left_svd_max_bond(self.A[-1], np.array([[[1]]]), self.qd, self.qD[-2:], tol, max_bond)
            assert is_qsparse(self.A[-1], [self.qd, self.qD[-2], -self.qD[-1]]), \
                'sparsity pattern of MPS tensor does not match quantum numbers'
    
            self.A[-1] *= T[0, 0, 0]
            #in the case tol is too high, the mps will be truncated to 0 
            if T.shape != (1, 1, 1):
                self = empty_mps(self.nsites)
            #return: left-cano MPS
            
        if mode == 'right':
            # transform to left-canonical form first
            nrm = self.orthonormalize(mode='left')
            #cancel the normalization in this step:
            self.A[0] = nrm* self.A[0]
            for i in reversed(range(1, len(self.A))):
                self.A[i], self.A[i-1], self.qD[i] = local_orthonormalize_right_svd_max_bond(self.A[i], self.A[i-1], self.qd, self.qD[i:i+2], tol, max_bond)
                assert is_qsparse(self.A[i], [self.qd, self.qD[i], -self.qD[i+1]]), \
                    'sparsity pattern of MPS tensor does not match quantum numbers'
            # first tensor
            self.A[0], T, self.qD[0] = local_orthonormalize_right_svd_max_bond(self.A[0], np.array([[[1]]]), self.qd, self.qD[:2], tol, max_bond)
            assert is_qsparse(self.A[0], [self.qd, self.qD[0], -self.qD[1]]), \
                'sparsity pattern of MPS tensor does not match quantum numbers'
                
            self.A[0] *= T[0, 0, 0]
            #in the case tol is too high, the mps will be truncated to 0 
            if T.shape != (1, 1, 1):
                self = empty_mps(self.nsites)
            #return: right-cano MPS
            
        #raise ValueError(f'mode = {mode} invalid; must be "left" or "right".')
        
    def compress_direct_svd_left_max_bond(self, tol: float, max_bond = 10):
        """
        Simply use SVD to truncate a left-canonical form MPS. 
        
        For accuracy, the MPS here should be in canonical form. Thus this function is suitable for 
        compressing addition of two canonical MPS.
        
        Attention: please start from left-canonical form.
        
        Return: truncated MPS in right-canonical form
        """
        #start from left side:
        for i in range(len(self.A) - 1):
            self.A[i], self.A[i+1], self.qD[i+1] = local_orthonormalize_left_svd_max_bond(self.A[i], self.A[i+1], self.qd, self.qD[i:i+2], tol, max_bond)
            assert is_qsparse(self.A[i], [self.qd, self.qD[i], -self.qD[i+1]]), \
                'sparsity pattern of MPS tensor does not match quantum numbers'
        # last tensor
        self.A[-1], T, self.qD[-1] = local_orthonormalize_left_svd_max_bond(self.A[-1], np.array([[[1]]]), self.qd, self.qD[-2:], tol, max_bond)
        assert is_qsparse(self.A[-1], [self.qd, self.qD[-2], -self.qD[-1]]), \
            'sparsity pattern of MPS tensor does not match quantum numbers'

        self.A[-1] *= T[0, 0, 0]
        #in the case tol is too high, the mps will be truncated to 0 
        if T.shape != (1, 1, 1):
            self = empty_mps(self.nsites)


    
    def compress_direct_svd_left_max_bond1(self, tol: float, max_bond: int = 10):

    # 1) sweep
        for i in range(len(self.A) - 1):
            self.A[i], self.A[i+1], self.qD[i+1] = \
                local_orthonormalize_left_svd_max_bond(
                    self.A[i], self.A[i+1],
                    self.qd, self.qD[i:i+2],
                    tol, max_bond)

    # 2) tail tensor
        self.A[-1], T, self.qD[-1] = \
            local_orthonormalize_left_svd_max_bond(
                self.A[-1], np.array([[[1]]]),
                self.qd, self.qD[-2:], tol, max_bond)
        self.A[-1] *= T[0, 0, 0] / abs(T[0, 0, 0])   # absorb magnitude, fix phase

    # 3) OPTIONAL: right sweep to tighten the gauge & recover norm
        norm = self.orthonormalize('right')           # ~1 if little was truncated
        self.A[0] *= norm

        return self







            
            
    def compress_direct_svd_right_max_bond(self, tol: float, max_bond = 10):
        """
        Simply use SVD to truncate a right-canonical form MPS. 
        
        For accuracy, the MPS here should be in canonical form. Thus this function is suitable for 
        compressing addition of two canonical MPS.
        
        Attention: please start from left-canonical form.
        
        Return: truncated MPS in left-canonical form
        """
        #start from right side
        for i in reversed(range(1, len(self.A))):
            self.A[i], self.A[i-1], self.qD[i] = local_orthonormalize_right_svd_max_bond(self.A[i], self.A[i-1], self.qd, self.qD[i:i+2], tol, max_bond)
            assert is_qsparse(self.A[i], [self.qd, self.qD[i], -self.qD[i+1]]), \
                'sparsity pattern of MPS tensor does not match quantum numbers'
        # first tensor
        self.A[0], T, self.qD[0] = local_orthonormalize_right_svd_max_bond(self.A[0], np.array([[[1]]]), self.qd, self.qD[:2], tol, max_bond)
        assert is_qsparse(self.A[0], [self.qd, self.qD[0], -self.qD[1]]), \
            'sparsity pattern of MPS tensor does not match quantum numbers'
            
        self.A[0] *= T[0, 0, 0]
        #in the case tol is too high, the mps will be truncated to 0 
        if T.shape != (1, 1, 1):
            self = empty_mps(self.nsites)
    #raise ValueError(f'mode = {mode} invalid; must be "left" or "right".')
    
    
            
      

    def as_vector(self) -> np.ndarray:
        """
        Merge all tensors to obtain the vector representation on the full Hilbert space.
        """
        psi = self.A[0]
        for i in range(1, len(self.A)):
            psi = merge_mps_tensor_pair(psi, self.A[i])
        assert psi.ndim == 3
        assert psi.shape[1] == 1 and psi.shape[2] == 1
        return psi.reshape(-1)

    @classmethod
    def from_vector(cls, d: int, nsites: int, v: np.ndarray, tol: float = 0):
        """
        Construct the MPS representation of the vector `v` via the TT-SVD algorithm,
        for local dimension `d`.

        All quantum numbers are set to zero for simplicity.
        """
        v = np.array(v)
        assert v.ndim == 1 and len(v) == d**nsites, f"`v` has length {len(v)}, expecting {d**nsites}."
        # allocate a MPS with dummy virtual bonds of dimension 1
        mps = cls(d*[0], [[0] for _ in range(nsites + 1)], fill='postpone')
        # endow `v` with a dummy left virtual bond dimension
        v = np.reshape(v, (1, len(v)))
        for i in range(nsites):
            assert v.shape[1] == d**(nsites - i)
            Dleft = v.shape[0]
            # partition physical dimension into local dimension and the rest,
            # split by SVD and assign the right SVD matrix to `v`
            u, s, v = np.linalg.svd(v.reshape((Dleft*d, d**(nsites-i-1))), full_matrices=False)
            # truncate small singular values
            idx = retained_bond_indices(s, tol)
            u = u[:, idx]
            v = v[idx, :]
            s = s[idx]
            # use broadcasting to distribute singular values to the right
            v = v * s[:, None]
            # assign MPS tensor at current site i, transposing physical dimension to the front
            mps.A[i] = u.reshape((Dleft, d, len(s))).transpose((1, 0, 2))
            #mps.qD[i + 1] = len(s) * [0]
            mps.qD[i + 1] = np.array(len(s) * [0])
        assert v.shape == (1, 1)
        # include scalar factor in last MPS tensor
        mps.A[-1] *= v[0, 0]
        return mps
    

    def __add__(self, other):
        """
        Add MPS to another.
        """
        return add_mps(self, other)

    def __sub__(self, other):
        """
        Subtract another MPS.
        """
        return add_mps(self, other, alpha=-1)


def local_orthonormalize_left_qr(A: np.ndarray, Anext: np.ndarray, qd: Sequence[int], qD: Sequence[Sequence[int]]):
    """
    Left-orthonormalize the local site tensor `A` by a QR decomposition,
    and update the tensor at the next site.
    """
    # perform QR decomposition and replace A by reshaped Q matrix
    s = A.shape
    assert len(s) == 3
    q0 = qnumber_flatten([qd, qD[0]])
    Q, R, qbond = qr(A.reshape((s[0]*s[1], s[2])), q0, qD[1])
    A = Q.reshape((s[0], s[1], Q.shape[1]))
    # update Anext tensor: multiply with R from left
    Anext = np.tensordot(R, Anext, (1, 1)).transpose((1, 0, 2))
    return (A, Anext, qbond)


def local_orthonormalize_right_qr(A: np.ndarray, Aprev: np.ndarray, qd: Sequence[int], qD: Sequence[Sequence[int]]):
    """
    Right-orthonormalize the local site tensor `A` by a QR decomposition,
    and update the tensor at the previous site.
    """
    # flip left and right virtual bond dimensions
    A = A.transpose((0, 2, 1))
    # perform QR decomposition and replace A by reshaped Q matrix
    s = A.shape
    assert len(s) == 3
    q0 = qnumber_flatten([qd, -qD[1]])
    Q, R, qbond = qr(A.reshape((s[0]*s[1], s[2])), q0, -qD[0])
    A = Q.reshape((s[0], s[1], Q.shape[1])).transpose((0, 2, 1))
    # update Aprev tensor: multiply with R from right
    Aprev = np.tensordot(Aprev, R, (2, 1))
    return (A, Aprev, -qbond)


def local_orthonormalize_left_svd(A, Anext, qd: Sequence[int], qD: Sequence[Sequence[int]], tol: float):
    """
    Left-orthonormalize the local site tensor `A` by a SVD,
    and update the tensor at the next site.
    """
    # perform SVD and replace A by reshaped U matrix
    s = A.shape
    assert len(s) == 3
    q0 = qnumber_flatten([qd, qD[0]])
    U, sigma, V, qbond = split_matrix_svd(A.reshape((s[0]*s[1], s[2])), q0, qD[1], tol)
    A = U.reshape((s[0], s[1], U.shape[1]))
    # update Anext tensor: multiply with (sigma @ V) from left
    Anext = np.tensordot(sigma[:, None] * V, Anext, (1, 1)).transpose((1, 0, 2))
    return (A, Anext, qbond)


def local_orthonormalize_right_svd(A, Aprev, qd: Sequence[int], qD: Sequence[Sequence[int]], tol: float):
    """
    Right-orthonormalize the local site tensor `A` by a SVD,
    and update the tensor at the previous site.
    """
    # flip physical and left virtual bond dimension
    A = A.transpose((1, 0, 2))
    # perform SVD and replace A by reshaped V matrix
    s = A.shape
    assert len(s) == 3
    q1 = qnumber_flatten([-qd, qD[1]])
    U, sigma, V, qbond = split_matrix_svd(A.reshape((s[0], s[1]*s[2])), qD[0], q1, tol)
    A = V.reshape((V.shape[0], s[1], s[2]))
    # undo flip of physical and left virtual bond dimension
    A = A.transpose((1, 0, 2))
    # update Aprev tensor: multiply with (U @ sigma) from right
    Aprev = np.tensordot(Aprev, U * sigma, (2, 0))
    return (A, Aprev, qbond)

def merge_mps_tensor_pair(A0: np.ndarray, A1: np.ndarray) -> np.ndarray:
    """
    Merge two neighboring MPS tensors.
    """
    A = np.einsum(A0, (0, 2, 3), A1, (1, 3, 4), (0, 1, 2, 4), optimize=True)
    # combine original physical dimensions
    A = A.reshape((A.shape[0]*A.shape[1], A.shape[2], A.shape[3]))
    return A


def split_mps_tensor(A: np.ndarray, qd0: Sequence[int], qd1: Sequence[int], qD: Sequence[Sequence[int]], svd_distr: str, tol=0):
    """
    Split a MPS tensor with dimension `d0*d1 x D0 x D2` into two MPS tensors
    with dimensions `d0 x D0 x D1` and `d1 x D1 x D2`, respectively.
    """
    assert A.ndim == 3
    d0 = len(qd0)
    d1 = len(qd1)
    assert d0 * d1 == A.shape[0], 'physical dimension of MPS tensor must be equal to d0 * d1'
    # reshape as matrix and split by SVD
    A = A.reshape((d0, d1, A.shape[1], A.shape[2])).transpose((0, 2, 1, 3))
    s = A.shape
    q0 = qnumber_flatten([ qd0, qD[0]])
    q1 = qnumber_flatten([-qd1, qD[1]])
    A0, sigma, A1, qbond = split_matrix_svd(A.reshape((s[0]*s[1], s[2]*s[3])), q0, q1, tol)
    A0.shape = (s[0], s[1], len(sigma))
    A1.shape = (len(sigma), s[2], s[3])
    # use broadcasting to distribute singular values
    if svd_distr == 'left':
        A0 = A0 * sigma
    elif svd_distr == 'right':
        A1 = A1 * sigma[:, None, None]
    elif svd_distr == 'sqrt':
        s = np.sqrt(sigma)
        A0 = A0 * s
        A1 = A1 * s[:, None, None]
    else:
        raise ValueError('svd_distr parameter must be "left", "right" or "sqrt".')
    # move physical dimension to the front
    A1 = A1.transpose((1, 0, 2))
    return (A0, A1, qbond)


def add_mps(mps0: MPS, mps1: MPS, alpha=1) -> MPS:
    """
    Logical addition of two matrix product states (effectively sum virtual bond dimensions),
    with the second MPS scaled by 'alpha'.
    """
    # number of lattice sites must agree
    assert mps0.nsites == mps1.nsites
    L = mps0.nsites
    # physical quantum numbers must agree
    assert np.array_equal(mps0.qd, mps1.qd)
    d = len(mps0.qd)

    # initialize with dummy tensors and bond quantum numbers
    mps = MPS(mps0.qd, (L+1)*[[0]])

    if L == 1:
        # single site
        # dummy bond quantum numbers must agree
        assert np.array_equal(mps0.qD[0], mps1.qD[0])
        assert np.array_equal(mps0.qD[1], mps1.qD[1])
        mps.qD[0] = mps0.qD[0].copy()
        mps.qD[1] = mps0.qD[1].copy()
        # simply add MPS tensors
        mps.A[0] = mps0.A[0] + alpha*mps1.A[0]
        # consistency check
        assert is_qsparse(mps.A[0], [mps.qd, mps.qD[0], -mps.qD[1]]), \
            'sparsity pattern of MPS tensor does not match quantum numbers'
    elif L > 1:
        # combine virtual bond quantum numbers
        # leading and trailing (dummy) bond quantum numbers must agree
        assert np.array_equal(mps0.qD[ 0], mps1.qD[ 0])
        assert np.array_equal(mps0.qD[-1], mps1.qD[-1])
        mps.qD[ 0] = mps0.qD[ 0].copy()
        mps.qD[-1] = mps0.qD[-1].copy()
        # intermediate bond quantum numbers
        for i in range(1, L):
            mps.qD[i] = np.concatenate((mps0.qD[i], mps1.qD[i]))

        # leftmost tensor
        mps.A[0] = np.block([mps0.A[0], alpha*mps1.A[0]])
        # intermediate tensors
        for i in range(1, L - 1):
            s0 = mps0.A[i].shape
            s1 = mps1.A[i].shape
            # form block-diagonal tensor
            mps.A[i] = np.block([[mps0.A[i], np.zeros((d, s0[1], s1[2]))], [np.zeros((d, s1[1], s0[2])), mps1.A[i]]])
        # rightmost tensor
        mps.A[-1] = np.block([[mps0.A[-1]], [mps1.A[-1]]])

        # consistency check
        for i in range(1, L):
            assert is_qsparse(mps.A[i], [mps.qd, mps.qD[i], -mps.qD[i+1]]), \
                'sparsity pattern of MPS tensor does not match quantum numbers'
    return mps

#added by Yu:
def local_orthonormalize_left_svd_max_bond(A, Anext, qd: Sequence[int], qD: Sequence[Sequence[int]], tol: float, max_bond):
    """
    Left-orthonormalize the local site tensor `A` by a SVD,
    and update the tensor at the next site.
    """
    # perform SVD and replace A by reshaped U matrix
    s = A.shape
    assert len(s) == 3
    q0 = qnumber_flatten([qd, qD[0]])
    U, sigma, V, qbond = split_matrix_svd_max_bond(A.reshape((s[0]*s[1], s[2])), q0, qD[1], tol, max_bond)
    A = U.reshape((s[0], s[1], U.shape[1]))
    # update Anext tensor: multiply with (sigma @ V) from left
    Anext = np.tensordot(sigma[:, None] * V, Anext, (1, 1)).transpose((1, 0, 2))
    return (A, Anext, qbond)


def local_orthonormalize_right_svd_max_bond(A, Aprev, qd: Sequence[int], qD: Sequence[Sequence[int]], tol: float, max_bond):
    """
    Right-orthonormalize the local site tensor `A` by a SVD,
    and update the tensor at the previous site.
    """
    # flip physical and left virtual bond dimension
    A = A.transpose((1, 0, 2))
    # perform SVD and replace A by reshaped V matrix
    s = A.shape
    assert len(s) == 3
    q1 = qnumber_flatten([-qd, qD[1]])
    U, sigma, V, qbond = split_matrix_svd_max_bond(A.reshape((s[0], s[1]*s[2])), qD[0], q1, tol, max_bond)
    A = V.reshape((V.shape[0], s[1], s[2]))
    # undo flip of physical and left virtual bond dimension
    A = A.transpose((1, 0, 2))
    # update Aprev tensor: multiply with (U @ sigma) from right
    Aprev = np.tensordot(Aprev, U * sigma, (2, 0))
    return (A, Aprev, qbond)


    

def empty_mps(L):
    #return MPS whose entires are all zero. 
    #L determines the system size.
    #define mps dimension using D_empty:
    D_empty = []
    for i in range (L+1):
        D_empty.append (1)
        qD_empty = [np.zeros(Di, dtype=int) for Di in D_empty]
        mps = MPS(np.array([0, 1]), qD_empty, fill= 0)  

    mps.qD = [np.zeros(Di, dtype=int) for Di in mps.bond_dims]   
    
    return mps

import numpy as np
# Assuming the MPS class is defined elsewhere
# from .mps import MPS

def create_random_mps(bond_set, dir, rng):
    """
    Creates a random, orthonormalized MPS using a provided generator.
    """
    # physical dimension (of each lattice site);
    d = 2
    D = bond_set
    qd = np.zeros(d, dtype=int)
    qD = [np.zeros(Di, dtype=int) for Di in D]

    # Use the generator that was passed in, DO NOT create a new one here.
    A = MPS(qd, qD, fill='random', rng=rng)

    A.orthonormalize(mode = dir)
 
    return A


def create_random_mps_4(bond_set, dir, rng):
    """
    Creates a random, orthonormalized MPS using a provided generator.
    """
    # physical dimension (of each lattice site);
    d = 4
    D = bond_set
    qd = np.zeros(d, dtype=int)
    qD = [np.zeros(Di, dtype=int) for Di in D]

    # Use the generator that was passed in, DO NOT create a new one here.
    A = MPS(qd, qD, fill='random', rng=rng)

    A.orthonormalize(mode = dir)
 
    return A


def create_random_mps_3(bond_set, dir, rng):
    """
    Creates a random, orthonormalized MPS using a provided generator.
    """
    # physical dimension (of each lattice site);
    d = 3
    D = bond_set
    qd = np.zeros(d, dtype=int)
    qD = [np.zeros(Di, dtype=int) for Di in D]

    # Use the generator that was passed in, DO NOT create a new one here.
    A = MPS(qd, qD, fill='random', rng=rng)

    A.orthonormalize(mode = dir)
 
    return A

# -----------------------------------------------------------------------------
# MPO--MPS multiplication-compressions based on the zip-up sweep.
# Added for H|psi> without first materializing the full product MPS.
# -----------------------------------------------------------------------------

def _contract_mpo_mps_tensor(W: np.ndarray, A: np.ndarray) -> np.ndarray:
    """
    Contract one MPO tensor with one MPS tensor.

    Args:
        W: MPO tensor with shape (d_out, d_in, W_left, W_right).
        A: MPS tensor with shape (d_in, D_left, D_right).

    Returns:
        Tensor with shape (d_out, W_left*D_left, W_right*D_right), using the
        virtual-bond ordering (W, D) on both sides.
    """
    if W.ndim != 4:
        raise ValueError(f'expected an MPO tensor with ndim=4, got shape {W.shape}')
    if A.ndim != 3:
        raise ValueError(f'expected an MPS tensor with ndim=3, got shape {A.shape}')
    if W.shape[1] != A.shape[0]:
        raise ValueError(
            f'incompatible physical dimensions: MPO input dim {W.shape[1]} '
            f'but MPS physical dim {A.shape[0]}')

    T = np.einsum('osab,slr->oalbr', W, A, optimize=True)
    return T.reshape((W.shape[0], W.shape[2] * A.shape[1], W.shape[3] * A.shape[2]))


def _split_zipup_matrix(theta: np.ndarray, qleft_rows: np.ndarray, qright_cols: np.ndarray,
                        tol: float, max_bond=None):
    """
    SVD split used by the zip-up sweep, with a dense fallback for zero matrices.
    """
    if max_bond is None:
        U, sigma, V, qbond = split_matrix_svd(theta, qleft_rows, qright_cols, tol)
    else:
        if max_bond < 1:
            raise ValueError('max_bond must be at least 1')
        U, sigma, V, qbond = split_matrix_svd_max_bond(theta, qleft_rows, qright_cols, tol, max_bond)

    # Avoid producing an invalid zero-dimensional MPS bond for the exactly-zero case.
    if len(sigma) == 0:
        U = np.zeros((theta.shape[0], 1), dtype=theta.dtype)
        sigma = np.zeros(1, dtype=float)
        V = np.zeros((1, theta.shape[1]), dtype=theta.dtype)
        qbond = np.zeros(1, dtype=int)
    return U, sigma, V, qbond


def apply_mpo_to_mps_naive(op, psi: MPS) -> MPS:
    """
    Form the exact, uncompressed product MPS ``op @ psi``.

    The output virtual bond at bond i is the tensor-product bond
    ``op.qD[i] + psi.qD[i]``. This routine is mainly useful for debugging and
    for comparison with ``apply_mpo_to_mps_zipup``; its bond dimensions are the
    products of the MPO and MPS bond dimensions.
    """
    if op.nsites != psi.nsites:
        raise ValueError('MPO and MPS must have the same number of sites')
    if not np.array_equal(op.qd, psi.qd):
        raise ValueError('MPO and MPS physical quantum numbers must agree')

    L = psi.nsites
    qD = [qnumber_flatten([op.qD[i], psi.qD[i]]) for i in range(L + 1)]
    out = MPS(op.qd, qD, fill='postpone')
    for i in range(L):
        out.A[i] = _contract_mpo_mps_tensor(op.A[i], psi.A[i])
        assert is_qsparse(out.A[i], [out.qd, out.qD[i], -out.qD[i + 1]]), \
            'sparsity pattern of product MPS tensor does not match quantum numbers'
    return out


def apply_mpo_to_mps_zipup(op, psi: MPS, tol: float = 0.0, max_bond=None) -> MPS:
    """
    Compute and compress ``op @ psi`` by a left-to-right zip-up sweep.

    At each site, the local MPO tensor is contracted with the local MPS tensor,
    the residual factor from the previous SVD is absorbed into the left bond,
    and a local SVD truncates the newly formed right bond. Thus the full product
    MPS with bond dimensions ``op.bond_dims[i] * psi.bond_dims[i]`` is never
    materialized as a separate object.

    Args:
        op: MPO with tensors shaped ``(d_out, d_in, W_left, W_right)``.
        psi: input MPS with tensors shaped ``(d_in, D_left, D_right)``.
        tol: relative discarded-weight tolerance passed to the SVD truncator.
        max_bond: optional maximum output MPS bond dimension. Use ``None`` for
            no explicit cap beyond ``tol``.

    Returns:
        MPS representing the compressed product ``op @ psi``. The result is in
        left-canonical zip-up gauge, except that the final tensor carries the
        remaining norm/scale of ``op @ psi``.
    """
    if op.nsites != psi.nsites:
        raise ValueError('MPO and MPS must have the same number of sites')
    if not np.array_equal(op.qd, psi.qd):
        raise ValueError('MPO and MPS physical quantum numbers must agree')
    if tol < 0:
        raise ValueError('tol must be non-negative')

    L = psi.nsites
    qD_prod = [qnumber_flatten([op.qD[i], psi.qD[i]]) for i in range(L + 1)]

    out_qD = [qD_prod[0].copy()]
    out_tensors = []

    # ``carry`` maps the exact product left bond of the current site to the
    # already truncated output left bond. Shape: (D_out_left, D_product_left).
    carry = np.ones((1, len(qD_prod[0])), dtype=np.result_type(op.A[0].dtype, psi.A[0].dtype))

    for i in range(L - 1):
        B = _contract_mpo_mps_tensor(op.A[i], psi.A[i])
        if carry.shape[1] != B.shape[1]:
            raise ValueError(
                f'internal zip-up carry mismatch at site {i}: carry right dim '
                f'{carry.shape[1]} but local product left dim {B.shape[1]}')

        # Absorb the previous right SVD factor into the local product tensor.
        theta = np.tensordot(carry, B, axes=(1, 1)).transpose((1, 0, 2))
        d, Dleft, Dright_prod = theta.shape
        qrows = qnumber_flatten([op.qd, out_qD[-1]])

        U, sigma, V, qbond = _split_zipup_matrix(
            theta.reshape((d * Dleft, Dright_prod)), qrows, qD_prod[i + 1], tol, max_bond)

        out_tensors.append(U.reshape((d, Dleft, U.shape[1])))
        out_qD.append(np.asarray(qbond, dtype=int))
        carry = sigma[:, None] * V

    # Last site: absorb the final carry and keep the remaining scale in the last tensor.
    B = _contract_mpo_mps_tensor(op.A[-1], psi.A[-1])
    if carry.shape[1] != B.shape[1]:
        raise ValueError(
            f'internal zip-up carry mismatch at last site: carry right dim '
            f'{carry.shape[1]} but local product left dim {B.shape[1]}')
    last = np.tensordot(carry, B, axes=(1, 1)).transpose((1, 0, 2))
    if last.shape[2] != len(qD_prod[-1]):
        raise ValueError('unexpected final right bond dimension in zip-up result')

    out_tensors.append(last)
    out_qD.append(qD_prod[-1].copy())

    out = MPS(op.qd, out_qD, fill='postpone')
    out.A = out_tensors
    for i in range(L):
        assert is_qsparse(out.A[i], [out.qd, out.qD[i], -out.qD[i + 1]]), \
            'sparsity pattern of zip-up MPS tensor does not match quantum numbers'
    return out


# Backward/alternative name that reads like the operation being performed.
multiply_mpo_mps_zipup = apply_mpo_to_mps_zipup
# -----------------------------------------------------------------------------
# Variational refinement for zip-up MPO--MPS multiplication-compression.
# -----------------------------------------------------------------------------

def _validate_mpo_mps_variational_inputs(op, psi: MPS, approx: MPS):
    """Common validation for variational MPO--MPS compression routines."""
    if op.nsites != psi.nsites or psi.nsites != approx.nsites:
        raise ValueError('MPO, input MPS, and approximate MPS must have the same number of sites')
    if not np.array_equal(op.qd, psi.qd):
        raise ValueError('MPO and input MPS physical quantum numbers must agree')
    if not np.array_equal(op.qd, approx.qd):
        raise ValueError('MPO output and approximate MPS physical quantum numbers must agree')



def _update_left_mpo_mps_overlap_env(Lenv: np.ndarray, B: np.ndarray, W: np.ndarray, A: np.ndarray) -> np.ndarray:
    """
    Extend a left mixed MPO-MPS overlap block by one site.

    The block represents the partial contraction of ``<chi|op|psi>`` and has
    index order ``(D_psi_left, W_left, D_chi_left)``.  The returned block has
    order ``(D_psi_right, W_right, D_chi_right)``.

    Important:
        The operation.py convention is ``contraction_operator_step_left(A, B, W, L)``,
        where ``A`` is the ket tensor and ``B`` is the bra tensor.  For the mixed
        compression problem, ``A`` must therefore be ``psi.A[i]`` and ``B`` must
        be ``chi.A[i]``.
    """
    from .operation import contraction_operator_step_left
    return contraction_operator_step_left(A, B, W, Lenv)


def _update_right_mpo_mps_overlap_env(Renv: np.ndarray, B: np.ndarray, W: np.ndarray, A: np.ndarray) -> np.ndarray:
    """
    Extend a right mixed MPO-MPS overlap block by one site.

    The block represents the partial contraction of ``<chi|op|psi>`` and has
    index order ``(D_psi_right, W_right, D_chi_right)``.  The returned block has
    order ``(D_psi_left, W_left, D_chi_left)``.

    Important:
        The operation.py convention is ``contraction_operator_step_right(A, B, W, R)``,
        where ``A`` is the ket tensor and ``B`` is the bra tensor.  For the mixed
        compression problem, ``A`` must therefore be ``psi.A[i]`` and ``B`` must
        be ``chi.A[i]``.
    """
    from .operation import contraction_operator_step_right
    return contraction_operator_step_right(A, B, W, Renv)


def _compute_left_mpo_mps_overlap_envs(op, psi: MPS, approx: MPS) -> list:
    """Return all prefix environments for ``<approx|op|psi>``."""
    _validate_mpo_mps_variational_inputs(op, psi, approx)
    envs = [np.ones((1, 1, 1), dtype=np.result_type(op.A[0].dtype, psi.A[0].dtype, approx.A[0].dtype))]
    for i in range(approx.nsites):
        envs.append(_update_left_mpo_mps_overlap_env(envs[-1], approx.A[i], op.A[i], psi.A[i]))
    return envs


def _compute_right_mpo_mps_overlap_envs(op, psi: MPS, approx: MPS) -> list:
    """Return all suffix environments for ``<approx|op|psi>``."""
    _validate_mpo_mps_variational_inputs(op, psi, approx)
    envs = [None] * (approx.nsites + 1)
    envs[-1] = np.ones((1, 1, 1), dtype=np.result_type(op.A[0].dtype, psi.A[0].dtype, approx.A[0].dtype))
    for i in reversed(range(approx.nsites)):
        envs[i] = _update_right_mpo_mps_overlap_env(envs[i + 1], approx.A[i], op.A[i], psi.A[i])
    return envs



def _local_mpo_mps_projection_tensor(Lenv: np.ndarray, W: np.ndarray, A: np.ndarray, Renv: np.ndarray) -> np.ndarray:
    """
    Project the exact target ``op|psi>`` onto the current left/right basis of
    the approximation ``chi``.

    The block orders are chosen to match ``operation.apply_local_hamiltonian``:

        Lenv: (D_psi_left,  W_left,  D_chi_left)
        Renv: (D_psi_right, W_right, D_chi_right)
        A:    (d_in, D_psi_left, D_psi_right)
        W:    (d_out, d_in, W_left, W_right)

    The output has shape ``(d_out, D_chi_left, D_chi_right)``, i.e. the shape of
    the local tensor of ``chi``.  Although the helper is named
    ``apply_local_hamiltonian`` in operation.py, its contraction order also works
    for this mixed case as long as the blocks use the order above.
    """
    from .operation import apply_local_hamiltonian
    return apply_local_hamiltonian(Lenv, Renv, W, A)


def _enforce_mps_tensor_qsparsity(A: np.ndarray, qd, qD_left, qD_right) -> np.ndarray:
    """Project tiny round-off leakage back to the allowed quantum-number sectors."""
    mask = qnumber_outer_sum([qd, qD_left, -qD_right])
    return np.where(mask == 0, A, 0)



def _compute_right_mixed_operator_blocks(op, psi: MPS, chi: MPS) -> list:
    """
    Compute DMRG-style right blocks for the mixed overlap ``<chi|op|psi>``.

    The returned list ``BR`` follows the same storage convention as
    ``operation.compute_right_operator_blocks`` used by DMRG: ``BR[i]`` contains
    the contraction of all sites strictly to the right of site ``i``.  Each block
    has index order ``(D_psi_right, W_right, D_chi_right)``.
    """
    _validate_mpo_mps_variational_inputs(op, psi, chi)
    L = chi.nsites
    BR = [None for _ in range(L)]
    BR[L - 1] = np.ones((1, 1, 1), dtype=np.result_type(op.A[0].dtype, psi.A[0].dtype, chi.A[0].dtype))
    for i in reversed(range(L - 1)):
        BR[i] = _update_right_mpo_mps_overlap_env(BR[i + 1], chi.A[i + 1], op.A[i + 1], psi.A[i + 1])
    return BR


def mpo_mps_variational_compression_sweep(op, psi: MPS, approx: MPS, nsweeps: int = 2,
                                          inplace: bool = False, return_history: bool = False):
    """
    Refine a capped approximation to ``op @ psi`` by variational MPS compression.

    This implementation follows the same sweep/block-storage pattern as the
    single-site DMRG code:

    1. right-normalize the current approximation ``chi``;
    2. precompute right mixed blocks ``BR`` for ``<chi|op|psi>``;
    3. maintain left mixed blocks ``BL`` incrementally during the left-to-right
       pass;
    4. maintain right mixed blocks ``BR`` incrementally during the right-to-left
       pass.

    The only difference from DMRG is the local update.  DMRG solves a local
    eigenproblem.  Here the local least-squares optimum is obtained directly by
    projecting ``op|psi>`` onto the current left and right bases of ``chi``.
    """
    if nsweeps < 0:
        raise ValueError('nsweeps must be non-negative')
    _validate_mpo_mps_variational_inputs(op, psi, approx)

    import copy as _copy
    chi = approx if inplace else _copy.deepcopy(approx)
    L = chi.nsites
    if L == 0:
        return (chi, []) if return_history else chi

    history = []

    # Same gauge setup as DMRG: start from a right-canonical MPS.
    # The removed scalar is not restored; the first local projection sets the
    # correct center-tensor scale.
    chi.orthonormalize(mode='right')

    # DMRG-style right blocks. BR[i] contracts sites i+1, ..., L-1.
    BR = _compute_right_mixed_operator_blocks(op, psi, chi)

    # DMRG-style left blocks. BL[i] contracts sites 0, ..., i-1.
    BL = [None for _ in range(L)]
    BL[0] = np.ones((1, 1, 1), dtype=np.result_type(op.A[0].dtype, psi.A[0].dtype, chi.A[0].dtype))

    for _ in range(nsweeps):
        # Sweep from left to right.  Site L-1 is handled in the subsequent
        # right-to-left pass, matching the single-site DMRG sweep structure.
        for i in range(L - 1):
            chi.A[i] = _local_mpo_mps_projection_tensor(BL[i], op.A[i], psi.A[i], BR[i])
            chi.A[i] = _enforce_mps_tensor_qsparsity(chi.A[i], chi.qd, chi.qD[i], chi.qD[i + 1])

            # Move the orthogonality center one site to the right.
            chi.A[i], chi.A[i + 1], chi.qD[i + 1] = local_orthonormalize_left_qr(
                chi.A[i], chi.A[i + 1], chi.qd, chi.qD[i:i + 2])

            # Update the left block with the new left-canonical chi.A[i].
            BL[i + 1] = _update_left_mpo_mps_overlap_env(BL[i], chi.A[i], op.A[i], psi.A[i])

        if return_history:
            # Approximate overlap diagnostic after the left-to-right pass.
            # Contract the last site with BL[L-1] and a dummy right block.
            Rlast = np.ones((1, 1, 1), dtype=BL[L - 1].dtype)
            Tlast = _local_mpo_mps_projection_tensor(BL[L - 1], op.A[L - 1], psi.A[L - 1], Rlast)
            history.append(np.vdot(chi.A[L - 1].reshape(-1), Tlast.reshape(-1)))

        # Sweep from right to left.  Site 0 is gauge-normalized at the end,
        # matching the single-site DMRG code.
        for i in reversed(range(1, L)):
            chi.A[i] = _local_mpo_mps_projection_tensor(BL[i], op.A[i], psi.A[i], BR[i])
            chi.A[i] = _enforce_mps_tensor_qsparsity(chi.A[i], chi.qd, chi.qD[i], chi.qD[i + 1])

            # Move the orthogonality center one site to the left.
            chi.A[i], chi.A[i - 1], chi.qD[i] = local_orthonormalize_right_qr(
                chi.A[i], chi.A[i - 1], chi.qd, chi.qD[i:i + 2])

            # Update the right block with the new right-canonical chi.A[i].
            BR[i - 1] = _update_right_mpo_mps_overlap_env(BR[i], chi.A[i], op.A[i], psi.A[i])

        # Do NOT normalize the leftmost tensor here.
        #
        # In DMRG, the optimized wavefunction is conventionally normalized, so
        # it is fine to right-normalize the leftmost tensor and discard the final
        # scalar.  In MPO-MPS variational compression, however, chi approximates
        # op|psi>, whose norm is generally not one.  Discarding the final scalar
        # changes the scale of chi and leads to a large relative error.
        #
        # After the right-to-left pass, sites 1, ..., L-1 are right-canonical
        # and site 0 is the orthogonality center carrying the correct norm.

        if return_history:
            Lfirst = np.ones((1, 1, 1), dtype=BR[0].dtype)
            Tfirst = _local_mpo_mps_projection_tensor(Lfirst, op.A[0], psi.A[0], BR[0])
            history.append(np.vdot(chi.A[0].reshape(-1), Tfirst.reshape(-1)))

        # After the final right-normalization of site 0, the right blocks remain
        # valid for the next left-to-right pass.  BL will be overwritten from
        # the left during the next pass.

    return (chi, history) if return_history else chi


def apply_mpo_to_mps_zipup_variational(op, psi: MPS, tol: float = 0.0, max_bond=None, return_history: bool = False):
    """
    Compute ``op @ psi`` by capped zip-up, then refine it variationally.

    The zip-up step fixes the compressed MPS bond dimensions, bounded by
    ``max_bond``. The subsequent variational sweeps improve the tensors for
    those fixed bond dimensions without ever forming the dense Hilbert-space
    vector. For small tests, the result can be compared with
    ``op.as_matrix() @ psi.as_vector()``.
    """
    guess = apply_mpo_to_mps_zipup(op, psi, tol=tol, max_bond=max_bond)
    return mpo_mps_variational_compression_sweep(
        op, psi, guess, nsweeps=1, inplace=True, return_history=return_history)


# Alternative name emphasizing the post-zip-up refinement step.
refine_mpo_mps_zipup_variational = apply_mpo_to_mps_zipup_variational


# ---------------------------------------------------------------------------
# Convenience helpers and pure variational MPO-MPS compression
# ---------------------------------------------------------------------------

def default_capped_bond_dims(nsites: int, local_dim: int, max_bond: int):
    """
    Construct default capped MPS bond dimensions.

    The i-th virtual bond dimension is bounded by both the Hilbert-space
    dimension across the cut and ``max_bond``:

        D[i] = min(max_bond, local_dim**min(i, nsites-i)).

    Thus D[0] = D[-1] = 1.
    """
    if nsites < 0:
        raise ValueError("nsites must be non-negative.")
    if local_dim < 1:
        raise ValueError("local_dim must be positive.")
    if max_bond < 1:
        raise ValueError("max_bond must be positive.")

    return [
        min(int(max_bond), int(local_dim) ** min(i, nsites - i))
        for i in range(nsites + 1)
    ]


def random_mps_variational_guess(
    qd,
    nsites: int,
    max_bond: int,
    rng: np.random.Generator = None,
    dtype=complex,
):
    """
    Create a random MPS with capped bond dimensions for pure variational
    MPO-MPS compression.

    This helper is intended for the quantum-number-free testing mode. It sets
    all virtual quantum numbers to zero.
    """
    if rng is None:
        rng = np.random.default_rng()

    qd = np.asarray(qd)
    d = len(qd)
    bond_dims = default_capped_bond_dims(nsites, d, max_bond)
    qD = [np.zeros(Di, dtype=int) for Di in bond_dims]

    guess = MPS(np.zeros(d, dtype=int), qD, fill="random", rng=rng)
    if dtype is not None:
        guess.A = [A.astype(dtype, copy=False) for A in guess.A]
    guess.orthonormalize(mode="right")
    return guess


def apply_mpo_to_mps_variational_only(
      op,
    psi: MPS,
    max_bond: int,
    nsweeps: int = 2,
    rng: np.random.Generator = None,
    zero_qnumbers: bool = True,
    initial_guess: MPS = None,
):
    """
    Compute a capped variational approximation to ``op @ psi`` without using
    zip-up initialization.

    Always returns only the compressed MPS.
    """
    import copy as _copy

    if op.nsites != psi.nsites:
        raise ValueError("MPO and MPS must have the same number of sites.")
    if len(op.qd) != len(psi.qd):
        raise ValueError("MPO and MPS physical dimensions do not match.")
    if max_bond < 1:
        raise ValueError("max_bond must be positive.")

    if zero_qnumbers:
        op_work = _copy.deepcopy(op).zero_qnumbers()
        psi_work = _copy.deepcopy(psi).zero_qnumbers()
        qd = np.zeros(len(psi_work.qd), dtype=int)
    else:
        op_work = op
        psi_work = psi
        qd = psi.qd

    if initial_guess is None:
        guess = random_mps_variational_guess(
            qd=qd,
            nsites=psi_work.nsites,
            max_bond=max_bond,
            rng=rng,
            dtype=np.result_type(op_work.A[0].dtype, psi_work.A[0].dtype),
        )
    else:
        guess = _copy.deepcopy(initial_guess)
        if zero_qnumbers:
            guess.zero_qnumbers()

    compressed = mpo_mps_variational_compression_sweep(
        op_work,
        psi_work,
        guess,
        nsweeps=nsweeps,
        inplace=True,
        return_history=False,
    )

    return compressed


# ---------------------------------------------------------------------------
# Variational compression for the sum of two MPSs
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Fast variational compression for the sum of two MPSs
# Uses a two-layer MPS overlap network, not an identity MPO.
# ---------------------------------------------------------------------------

def _canonicalize_mps_preserve_norm(mps: MPS, modes=("left", "right")):
    """
    Canonicalize an MPS while preserving the represented vector.

    In this implementation, mps.orthonormalize(mode) returns a scalar norm
    factor and removes it from the MPS. We multiply it back into a boundary
    tensor so that the represented vector is unchanged.
    """
    for mode in modes:
        nrm = mps.orthonormalize(mode=mode)

        if mode == "left":
            mps.A[-1] *= nrm
        elif mode == "right":
            mps.A[0] *= nrm
        else:
            raise ValueError('mode must be "left" or "right".')

    return mps


def _mps_overlap_step_left(Lenv: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Extend a left overlap environment for <B|A>.

    Contract order follows operation.contraction_step_left(A, B, L):
        A = ket tensor, here target.A[i]
        B = bra tensor, here chi.A[i]

    Lenv shape:
        (D_A_left, D_B_left)

    return shape:
        (D_A_right, D_B_right)
    """
    from .operation import contraction_step_left
    return contraction_step_left(A, B, Lenv)


def _mps_overlap_step_right(Renv: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Extend a right overlap environment for <B|A>.

    Contract order follows operation.contraction_step_right(A, B, R):
        A = ket tensor, here target.A[i]
        B = bra tensor, here chi.A[i]

    Renv shape:
        (D_A_right, D_B_right)

    return shape:
        (D_A_left, D_B_left)
    """
    from .operation import contraction_step_right
    return contraction_step_right(A, B, Renv)


def _compute_right_mps_overlap_blocks(target: MPS, chi: MPS) -> list:
    """
    DMRG-style right blocks for the overlap <chi|target>.

    BR[i] contains the contraction of all sites strictly to the right of i.
    BR[i] has shape:
        (D_target_right_at_i, D_chi_right_at_i)
    """
    if target.nsites != chi.nsites:
        raise ValueError("target and chi must have the same number of sites.")
    if not np.array_equal(target.qd, chi.qd):
        raise ValueError("target and chi must have the same physical quantum numbers.")

    L = target.nsites
    BR = [None for _ in range(L)]

    BR[L - 1] = np.ones(
        (target.A[-1].shape[2], chi.A[-1].shape[2]),
        dtype=np.result_type(target.A[-1].dtype, chi.A[-1].dtype),
    )

    for i in reversed(range(L - 1)):
        BR[i] = _mps_overlap_step_right(
            BR[i + 1],
            target.A[i + 1],
            chi.A[i + 1],
        )

    return BR


def _local_mps_projection_tensor(Lenv: np.ndarray, A: np.ndarray, Renv: np.ndarray) -> np.ndarray:
    """
    Project target MPS tensor A onto the current left/right basis of chi.

    Lenv shape:
        (D_target_left, D_chi_left)

    A shape:
        (d, D_target_left, D_target_right)

    Renv shape:
        (D_target_right, D_chi_right)

    return shape:
        (d, D_chi_left, D_chi_right)

    This is the local least-squares optimum for one-site variational MPS
    compression when chi is in mixed canonical form.
    """
    # First contract the target right bond with the right environment:
    # A[d, aL, aR] R[aR, bR] -> T[d, aL, bR]
    T = np.tensordot(A, Renv, axes=(2, 0))

    # Then contract the target left bond with the left environment:
    # T[d, aL, bR] L[aL, bL] -> P[d, bR, bL]
    P = np.tensordot(T, Lenv, axes=(1, 0))

    # Reorder to MPS tensor convention: (d, D_chi_left, D_chi_right)
    return P.transpose(0, 2, 1)


def _enforce_mps_tensor_qsparsity(A: np.ndarray, qd, qD_left, qD_right) -> np.ndarray:
    """
    Project tiny round-off leakage back to allowed quantum-number sectors.
    """
    mask = qnumber_outer_sum([qd, qD_left, -qD_right])
    return np.where(mask == 0, A, 0)


def variational_compress_mps_only_fast(
    target: MPS,
    max_bond: int,
    nsweeps: int = 3,
    rng: np.random.Generator = None,
    zero_qnumbers: bool = True,
    initial_guess: MPS = None,
):
    """
    Variationally compress a target MPS using a two-layer MPS overlap network.

    This is faster than using an identity MPO because it contracts

        <chi | target>

    directly, instead of contracting

        <chi | I | target>.

    Contract order:
        target = ket tensor A
        chi    = bra tensor B

    This matches operation.contraction_step_left/right(A, B, env).
    """
    import copy as _copy

    if max_bond < 1:
        raise ValueError("max_bond must be positive.")

    target_work = _copy.deepcopy(target)

    if zero_qnumbers:
        target_work.zero_qnumbers()

    # Reduce exact redundant direct-sum bonds if QR can do so,
    # but preserve the represented vector norm.
    target_work = _canonicalize_mps_preserve_norm(
        target_work,
        modes=("left", "right"),
    )

    L = target_work.nsites
    d = len(target_work.qd)

    if initial_guess is None:
        if rng is None:
            rng = np.random.default_rng()

        bond_dims = default_capped_bond_dims(L, d, max_bond)
        qD = [np.zeros(Di, dtype=int) for Di in bond_dims]

        if zero_qnumbers:
            qd = np.zeros(d, dtype=int)
        else:
            qd = target_work.qd.copy()

        chi = MPS(qd, qD, fill="random", rng=rng)
    else:
        chi = _copy.deepcopy(initial_guess)
        if zero_qnumbers:
            chi.zero_qnumbers()

    if target_work.nsites != chi.nsites:
        raise ValueError("target and initial guess must have the same number of sites.")
    if len(target_work.qd) != len(chi.qd):
        raise ValueError("target and initial guess physical dimensions do not match.")

    # Start in the same gauge style as DMRG.
    chi.orthonormalize(mode="right")

    # Right blocks BR[i] contract sites i+1,...,L-1.
    BR = _compute_right_mps_overlap_blocks(target_work, chi)

    # Left blocks BL[i] contract sites 0,...,i-1.
    BL = [None for _ in range(L)]
    BL[0] = np.ones(
        (target_work.A[0].shape[1], chi.A[0].shape[1]),
        dtype=np.result_type(target_work.A[0].dtype, chi.A[0].dtype),
    )

    for _ in range(nsweeps):
        # Left-to-right pass.
        for i in range(L - 1):
            chi.A[i] = _local_mps_projection_tensor(
                BL[i],
                target_work.A[i],
                BR[i],
            )
            chi.A[i] = _enforce_mps_tensor_qsparsity(
                chi.A[i],
                chi.qd,
                chi.qD[i],
                chi.qD[i + 1],
            )

            chi.A[i], chi.A[i + 1], chi.qD[i + 1] = local_orthonormalize_left_qr(
                chi.A[i],
                chi.A[i + 1],
                chi.qd,
                chi.qD[i:i + 2],
            )

            # Important order: target is ket A, chi is bra B.
            BL[i + 1] = _mps_overlap_step_left(
                BL[i],
                target_work.A[i],
                chi.A[i],
            )

        # Right-to-left pass.
        for i in reversed(range(1, L)):
            chi.A[i] = _local_mps_projection_tensor(
                BL[i],
                target_work.A[i],
                BR[i],
            )
            chi.A[i] = _enforce_mps_tensor_qsparsity(
                chi.A[i],
                chi.qd,
                chi.qD[i],
                chi.qD[i + 1],
            )

            chi.A[i], chi.A[i - 1], chi.qD[i] = local_orthonormalize_right_qr(
                chi.A[i],
                chi.A[i - 1],
                chi.qd,
                chi.qD[i:i + 2],
            )

            # Important order: target is ket A, chi is bra B.
            BR[i - 1] = _mps_overlap_step_right(
                BR[i],
                target_work.A[i],
                chi.A[i],
            )

        # Do not normalize away chi.A[0]; it carries the norm/scale
        # of the compressed target.

    return chi


def add_mps_compress_variational_only_fast(
    psi1: MPS,
    psi2: MPS,
    max_bond: int,
    nsweeps: int = 3,
    alpha=1,
    rng: np.random.Generator = None,
    zero_qnumbers: bool = True,
    initial_guess: MPS = None,
):
    """
    Add two MPSs by direct sum and then variationally compress the result.

    Steps:
        1. target = psi1 + alpha * psi2 using direct-sum add_mps.
        2. target is canonicalized left and right once, preserving norm.
        3. target is compressed by two-layer variational MPS compression.

    This avoids the identity-MPO overhead and uses the correct contract order:
        target = ket
        chi    = bra
    """
    import copy as _copy

    if psi1.nsites != psi2.nsites:
        raise ValueError("The two MPSs must have the same number of sites.")
    if not np.array_equal(psi1.qd, psi2.qd):
        raise ValueError("The two MPSs must have the same physical quantum numbers.")
    if max_bond < 1:
        raise ValueError("max_bond must be positive.")

    target = add_mps(
        _copy.deepcopy(psi1),
        _copy.deepcopy(psi2),
        alpha=alpha,
    )

    return variational_compress_mps_only_fast(
        target,
        max_bond=max_bond,
        nsweeps=nsweeps,
        rng=rng,
        zero_qnumbers=zero_qnumbers,
        initial_guess=initial_guess,
    )


# Compatible wrapper name for replacing old SVD-based add_mps_and_compress.
def add_mps_compress_variational_only(psi1: MPS, psi2: MPS, max_bond: int):
    return add_mps_compress_variational_only_fast(
        psi1,
        psi2,
        max_bond=max_bond,
        nsweeps=3,
        alpha=1,
        zero_qnumbers=True,
    )
