import numpy as np
from .mps import MPS
from .mpo import MPO
from .krylov import eigh_krylov
from .hamiltonian import molecular_hamiltonian_mpo
import copy
from scipy import sparse
import pickle

#this file is old codes, which is used a reference of new codes, and will be deleted in the future.


def get_smallest_MPO(upper_index, op_value,  X_mo, L):
    
    #set size for "simple" MPO
    d = 2
    qd = np.zeros(d, dtype=int)
    
    DO = [1] + [2] * (L - 1) + [1]
    qDO = [np.zeros(Di, dtype=int) for Di in DO]
    
    #Pauli matrices:
    I2 = np.identity(2)
    bose_c = np.array([[0, 0.],[1, 0]])
    bose_a = np.array([[0, 1.],[0, 0]])
    #bose_n = np.array([[0, 0.],[0, 1]])
    pauli_Z = np.array([[1, 0.],[0, -1]])
    
    #mpo = ptn.MPO(qd, qDO, fill='random')
    mpo = MPO(qd, qDO, fill= 0.)

    #annihilation
    if op_value == 0:

        mpo.A[0][:, :, 0, 0] = pauli_Z
        mpo.A[0][:, :, 0, 1] = bose_a * X_mo[upper_index, 0]

        mpo.A[L-1][:, :, 0, 0] = bose_a * X_mo[upper_index, L-1]
        mpo.A[L-1][:, :, 1, 0] = I2 

        for i in range (1,L-1):
            mpo.A[i][:, :, 0, 0] = pauli_Z
            mpo.A[i][:, :, 0, 1] = bose_a * X_mo[upper_index, i]
            mpo.A[i][:, :, 1, 0] = 0
            mpo.A[i][:, :, 1, 1] = I2

    #creation
    if op_value == 1:

        mpo.A[0][:, :, 0, 0] = pauli_Z
        mpo.A[0][:, :, 0, 1] = bose_c * X_mo[upper_index, 0]

        mpo.A[L-1][:, :, 0, 0] = bose_c * X_mo[upper_index, L-1]
        mpo.A[L-1][:, :, 1, 0] = I2 

        for i in range (1,L-1):
            mpo.A[i][:, :, 0, 0] = pauli_Z
            mpo.A[i][:, :, 0, 1] = bose_c * X_mo[upper_index, i]
            mpo.A[i][:, :, 1, 0] = 0
            mpo.A[i][:, :, 1, 1] = I2

    return mpo

def get_four_layers_spin(mu, nu, X_mo_spin_up, X_mo_spin_down, Z_mo, L, spin1, spin2):
    
    if spin1 == 0:
        H1 = get_smallest_MPO(nu, 0, X_mo_spin_up, L)
        H2 = get_smallest_MPO(nu, 1, X_mo_spin_up, L)
    if spin1 == 1:
        H1 = get_smallest_MPO(nu, 0, X_mo_spin_down, L)
        H2 = get_smallest_MPO(nu, 1, X_mo_spin_down, L)

    if spin2 == 0:
        H3 = get_smallest_MPO(mu, 0, X_mo_spin_up, L)
        H4 = get_smallest_MPO(mu, 1, X_mo_spin_up, L)
    if spin2 == 1:
        H3 = get_smallest_MPO(mu, 0, X_mo_spin_down, L)
        H4 = get_smallest_MPO(mu, 1, X_mo_spin_down, L)
        
    H4.A[0] = Z_mo[mu, nu] * H4.A[0]
        
    return [H1, H2, H3, H4] #H_mu_nu = H4@H3@H2@H1