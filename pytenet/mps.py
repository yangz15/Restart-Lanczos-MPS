from collections.abc import Sequence
import numpy as np
from .qnumber import qnumber_outer_sum, qnumber_flatten, is_qsparse
from .bond_ops import qr, retained_bond_indices, split_matrix_svd
from .bond_ops_max_bond import split_matrix_svd_max_bond
from .util import crandn

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
        """
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