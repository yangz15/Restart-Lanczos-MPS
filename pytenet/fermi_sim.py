#Most of the code in this file are written by Christian Mendl

import numpy as np
from scipy.sparse import csr_matrix
from gmpy2 import popcount


def annihil_sign(n, a):
    """
    Sign factor of annihilating modes encoded in `a` as 1-bits
    applied to state with occupied modes represented by `n`.
    """
    if n & a == a:
        na = n - a
        counter = 0
        while a:
            # current least significant bit
            lsb = (a & -a)
            counter += popcount(na & (lsb - 1))
            a -= lsb
        return 1 - 2*(counter % 2)
    else:
        # applying annihilation operator yields zero
        return 0


def create_sign(n, c):
    """
    Sign factor of creating modes encoded in `c` as 1-bits
    applied to state with occupied modes represented by `n`.
    """
    if n & c == 0:
        counter = 0
        while c:
            # current least significant bit
            lsb = (c & -c)
            counter += popcount(n & (lsb - 1))
            c -= lsb
        return 1 - 2*(counter % 2)
    else:
        # applying creation operator yields zero
        return 0


def annihil_op(nmodes, a):
    """
    Fermionic annihilation operator on full Fock space.
    """
    data = np.array([annihil_sign(n, a) for n in range(2**nmodes)], dtype=float)
    row_ind = np.arange(2**nmodes) - a
    col_ind = np.arange(2**nmodes)
    nzi = np.nonzero(data)[0]
    return csr_matrix((data[nzi], (row_ind[nzi], col_ind[nzi])), shape=(2**nmodes, 2**nmodes))


def create_op(nmodes, c):
    """
    Fermionic creation operator on full Fock space.
    """
    data = np.array([create_sign(n, c) for n in range(2**nmodes)], dtype=float)
    row_ind = np.arange(2**nmodes) + c
    col_ind = np.arange(2**nmodes)
    nzi = np.nonzero(data)[0]
    return csr_matrix((data[nzi], (row_ind[nzi], col_ind[nzi])), shape=(2**nmodes, 2**nmodes))


def number_op(nmodes, f):
    """
    Fermionic number operator on full Fock space.
    """
    data = np.array([1 if (n & f == f) else 0 for n in range(2**nmodes)], dtype=float)
    ind = np.arange(2**nmodes)
    nzi = np.nonzero(data)[0]
    return csr_matrix((data[nzi], (ind[nzi], ind[nzi])), shape=(2**nmodes, 2**nmodes))


def total_number_op(nmodes):
    """
    Total number operator on full Fock space.
    """
    data = np.array([popcount(n) for n in range(2**nmodes)], dtype=float)
    ind = np.arange(2**nmodes)
    return csr_matrix((data, (ind, ind)), shape=(2**nmodes, 2**nmodes))


def parity_op(nmodes):
    """
    Parity operator on full Fock space.
    """
    data = np.array([(-1)**popcount(n) for n in range(2**nmodes)], dtype=float)
    ind = np.arange(2**nmodes)
    return csr_matrix((data, (ind, ind)), shape=(2**nmodes, 2**nmodes))


def gamma_re(nmodes, f):
    """
    First Majorana operator corresponding to "real" part of fermionic operators.
    """
    return create_op(nmodes, f) + annihil_op(nmodes, f)


def gamma_im(nmodes, f):
    """
    Second Majorana operator corresponding to "imaginary" part of fermionic operators.
    (Sign convention is not universal.)
    """
    return -1j*(create_op(nmodes, f) - annihil_op(nmodes, f))