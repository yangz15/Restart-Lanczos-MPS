import numpy as np
from .mps import MPS
from .mpo import MPO
from .krylov import eigh_krylov
from .hamiltonian import spin_molecular_hamiltonian_mpo,  encode_quantum_number_pair
import copy
from scipy import sparse
import pickle
from .fermi_sim import *

# def get_elementary_mpo_qn(upper_index, op_value,  X_mo, L):
    
#     '''  
#     get elementary thc-mpo, in the form of mpo
#     the qnumber for elementary thc-mpo consisting of creation op has qD= [0, 1]
#     the qnumber for elementary thc-mpo consisting of anihilation op has qD= [0, -1]
#     op value: 0 for annihilation
#     op value: 1 for creation
#     upper_index: X_mo has two indices, one for site, one for thc-rank. Upper_index stands for mu/nu in thc-rank.
#     for spinor system, the X_mo here should be one of X_mo_up or X_mo_down
#     qn stands for quantum number, which means this func considers the qn
#     '''
#     qd = np.array([0, 1])

#     I2 = np.identity(2)
#     bose_c = np.array([[0, 0.],[1, 0]])
#     bose_a = np.array([[0, 1.],[0, 0]])
#     pauli_Z = np.array([[1, 0.],[0, -1]])
    
#     #annihilation
#     if op_value == 0:
        
#         qD = [np.array([0])] + [np.array([0, -1]) for i in range(L-1)] + [np.array([-1])]
#         mpo = MPO(qd, qD, fill= 0.)

#         mpo.A[0][:, :, 0, 0] = pauli_Z
#         mpo.A[0][:, :, 0, 1] = bose_a * X_mo[upper_index, 0]

#         mpo.A[L-1][:, :, 0, 0] = bose_a * X_mo[upper_index, L-1]
#         mpo.A[L-1][:, :, 1, 0] = I2 

#         for i in range (1,L-1):
#             mpo.A[i][:, :, 0, 0] = pauli_Z
#             mpo.A[i][:, :, 0, 1] = bose_a * X_mo[upper_index, i]
#             mpo.A[i][:, :, 1, 0] = 0
#             mpo.A[i][:, :, 1, 1] = I2

#     #creation
#     if op_value == 1:
        
#         qD = [np.array([0])] + [np.array([0, 1]) for i in range(L-1)] + [np.array([1])]
#         mpo = MPO(qd, qD, fill= 0.)
        
#         mpo.A[0][:, :, 0, 0] = pauli_Z
#         mpo.A[0][:, :, 0, 1] = bose_c * X_mo[upper_index, 0]

#         mpo.A[L-1][:, :, 0, 0] = bose_c * X_mo[upper_index, L-1]
#         mpo.A[L-1][:, :, 1, 0] = I2 

#         for i in range (1,L-1):
#             mpo.A[i][:, :, 0, 0] = pauli_Z
#             mpo.A[i][:, :, 0, 1] = bose_c * X_mo[upper_index, i]
#             mpo.A[i][:, :, 1, 0] = 0
#             mpo.A[i][:, :, 1, 1] = I2

#     return mpo

# def get_elementary_mpo_qn(upper_index, op_value,  X_mo, L, spin):
    
#     '''  
#     get elementary thc-mpo, in the form of mpo
#     the qnumber for elementary thc-mpo consisting of creation op has qD= [0, 1]
#     the qnumber for elementary thc-mpo consisting of anihilation op has qD= [0, -1]
#     op value: 0 for annihilation
#     op value: 1 for creation
#     upper_index: X_mo has two indices, one for site, one for thc-rank. Upper_index stands for mu/nu in thc-rank.
#     for spinor system, the X_mo here should be one of X_mo_up or X_mo_down
#     qn stands for quantum number, which means this func considers the qn
#     spin: 0 for up, 1 for down 
#     '''
#     qd = np.array([0, 1, 2**16, 2**16 +1])

#     I2 = np.identity(2)
#     I4 = np.identity(4)
#     bose_c = np.array([[0, 0.],[1, 0]])
#     bose_a = np.array([[0, 1.],[0, 0]])
#     pauli_Z = np.array([[1, 0.],[0, -1]])
    
#     #print(L)
    
#     #annihilation
#     if op_value == 0:
#         if spin == 0: #up
#             qD = [np.array([0])] + [np.array([0, -2**16]) for i in range(L-1)] + [np.array([-2**16])]
#             mpo = MPO(qd, qD, fill= 0.)
            
#             mpo.A[0][:, :, 0, 0] = np.kron(pauli_Z, pauli_Z)
#             mpo.A[0][:, :, 0, 1] = np.kron(bose_a, I2) * X_mo[upper_index, 0]

#             mpo.A[L-1][:, :, 0, 0] = np.kron(bose_a, I2) * X_mo[upper_index, L-1]
#             mpo.A[L-1][:, :, 1, 0] = I4 

#             for i in range (1, L-1):
#                 mpo.A[i][:, :, 0, 0] = np.kron(pauli_Z, pauli_Z)
#                 mpo.A[i][:, :, 0, 1] = np.kron(bose_a, I2) * X_mo[upper_index, i]
#                 mpo.A[i][:, :, 1, 0] = 0
#                 mpo.A[i][:, :, 1, 1] = I4
                
#         if spin == 1: #down
#             qD = [np.array([0])] + [np.array([0, -1]) for i in range(L-1)] + [np.array([-1])]
#             mpo = MPO(qd, qD, fill= 0.)
#             mpo.A[0][:, :, 0, 0] = np.kron(pauli_Z, pauli_Z)
#             mpo.A[0][:, :, 0, 1] = np.kron(pauli_Z, bose_a) * X_mo[upper_index, 0]

#             mpo.A[L-1][:, :, 0, 0] = np.kron(pauli_Z, bose_a) * X_mo[upper_index, L-1]
#             mpo.A[L-1][:, :, 1, 0] = I4 

#             for i in range (1,L-1):
#                 mpo.A[i][:, :, 0, 0] = np.kron(pauli_Z, pauli_Z)
#                 mpo.A[i][:, :, 0, 1] = np.kron(pauli_Z, bose_a) * X_mo[upper_index, i]
#                 mpo.A[i][:, :, 1, 0] = 0
#                 mpo.A[i][:, :, 1, 1] = I4

#     #creation
#     if op_value == 1:
#         if spin == 0: #up
#             qD = [np.array([0])] + [np.array([0, 2**16]) for i in range(L-1)] + [np.array([2**16])]
#             mpo = MPO(qd, qD, fill= 0.)
            
#             mpo.A[0][:, :, 0, 0] = np.kron(pauli_Z, pauli_Z)
#             mpo.A[0][:, :, 0, 1] = np.kron(bose_c, I2) * X_mo[upper_index, 0]

#             mpo.A[L-1][:, :, 0, 0] = np.kron(bose_c, I2) * X_mo[upper_index, L-1]
#             mpo.A[L-1][:, :, 1, 0] = I4

#             for i in range (1,L-1):
#                 mpo.A[i][:, :, 0, 0] = np.kron(pauli_Z, pauli_Z)
#                 mpo.A[i][:, :, 0, 1] = np.kron(bose_c, I2) * X_mo[upper_index, i]
#                 mpo.A[i][:, :, 1, 0] = 0
#                 mpo.A[i][:, :, 1, 1] = I4
        
#         if spin == 1: #down
#             qD = [np.array([0])] + [np.array([0, 1]) for i in range(L-1)] + [np.array([1])]
#             mpo = MPO(qd, qD, fill= 0.)
#             mpo.A[0][:, :, 0, 0] = np.kron(pauli_Z, pauli_Z)
#             mpo.A[0][:, :, 0, 1] = np.kron(pauli_Z, bose_c) * X_mo[upper_index, 0]

#             mpo.A[L-1][:, :, 0, 0] = np.kron(pauli_Z, bose_c) * X_mo[upper_index, L-1]
#             mpo.A[L-1][:, :, 1, 0] = I4

#             for i in range (1,L-1):
#                 mpo.A[i][:, :, 0, 0] = np.kron(pauli_Z, pauli_Z)
#                 mpo.A[i][:, :, 0, 1] = np.kron(pauli_Z, bose_c) * X_mo[upper_index, i]
#                 mpo.A[i][:, :, 1, 0] = 0
#                 mpo.A[i][:, :, 1, 1] = I4     

#     return mpo


def get_elementary_mpo_qn(upper_index, op_value,  X_mo, L, spin):
    
    '''  
    get elementary thc-mpo, in the form of mpo
    the qnumber for elementary thc-mpo consisting of creation op has qD= [0, 1]
    the qnumber for elementary thc-mpo consisting of anihilation op has qD= [0, -1]
    op value: 0 for annihilation
    op value: 1 for creation
    upper_index: X_mo has two indices, one for site, one for thc-rank. Upper_index stands for mu/nu in thc-rank.
    for spinor system, the X_mo here should be one of X_mo_up or X_mo_down
    qn stands for quantum number, which means this func considers the qn
    spin: 0 for up, 1 for down 
    '''
    qN = [0,  1,  1,  2]
    qS = [0, -1,  1,  0]
    qd = [encode_quantum_number_pair(q[0], q[1]) for q in zip(qN, qS)]

    I2 = np.identity(2)
    I4 = np.identity(4)
    bose_c = np.array([[0, 0.],[1, 0]])
    bose_a = np.array([[0, 1.],[0, 0]])
    pauli_Z = np.array([[1, 0.],[0, -1]])
    
    #print(L)
    
    #annihilation
    if op_value == 0:
        if spin == 0: #up
            qD = [np.array([0])] + [np.array([0, -qd[2]]) for i in range(L-1)] + [np.array([-qd[2]])]
            mpo = MPO(qd, qD, fill= 0.)
            
            mpo.A[0][:, :, 0, 0] = np.kron(pauli_Z, pauli_Z)
            mpo.A[0][:, :, 0, 1] = np.kron(bose_a, I2) * X_mo[upper_index, 0]

            mpo.A[L-1][:, :, 0, 0] = np.kron(bose_a, I2) * X_mo[upper_index, L-1]
            mpo.A[L-1][:, :, 1, 0] = I4 

            for i in range (1, L-1):
                mpo.A[i][:, :, 0, 0] = np.kron(pauli_Z, pauli_Z)
                mpo.A[i][:, :, 0, 1] = np.kron(bose_a, I2) * X_mo[upper_index, i]
                mpo.A[i][:, :, 1, 0] = 0
                mpo.A[i][:, :, 1, 1] = I4
                
        if spin == 1: #down
            qD = [np.array([0])] + [np.array([0, -qd[1]]) for i in range(L-1)] + [np.array([-qd[1]])]
            mpo = MPO(qd, qD, fill= 0.)
            mpo.A[0][:, :, 0, 0] = np.kron(pauli_Z, pauli_Z)
            mpo.A[0][:, :, 0, 1] = np.kron(pauli_Z, bose_a) * X_mo[upper_index, 0]

            mpo.A[L-1][:, :, 0, 0] = np.kron(pauli_Z, bose_a) * X_mo[upper_index, L-1]
            mpo.A[L-1][:, :, 1, 0] = I4 

            for i in range (1,L-1):
                mpo.A[i][:, :, 0, 0] = np.kron(pauli_Z, pauli_Z)
                mpo.A[i][:, :, 0, 1] = np.kron(pauli_Z, bose_a) * X_mo[upper_index, i]
                mpo.A[i][:, :, 1, 0] = 0
                mpo.A[i][:, :, 1, 1] = I4

    #creation
    if op_value == 1:
        if spin == 0: #up
            qD = [np.array([0])] + [np.array([0, qd[2]]) for i in range(L-1)] + [np.array([qd[2]])]
            mpo = MPO(qd, qD, fill= 0.)
            
            mpo.A[0][:, :, 0, 0] = np.kron(pauli_Z, pauli_Z)
            mpo.A[0][:, :, 0, 1] = np.kron(bose_c, I2) * X_mo[upper_index, 0]

            mpo.A[L-1][:, :, 0, 0] = np.kron(bose_c, I2) * X_mo[upper_index, L-1]
            mpo.A[L-1][:, :, 1, 0] = I4

            for i in range (1,L-1):
                mpo.A[i][:, :, 0, 0] = np.kron(pauli_Z, pauli_Z)
                mpo.A[i][:, :, 0, 1] = np.kron(bose_c, I2) * X_mo[upper_index, i]
                mpo.A[i][:, :, 1, 0] = 0
                mpo.A[i][:, :, 1, 1] = I4
        
        if spin == 1: #down
            qD = [np.array([0])] + [np.array([0, qd[1]]) for i in range(L-1)] + [np.array([qd[1]])]
            mpo = MPO(qd, qD, fill= 0.)
            mpo.A[0][:, :, 0, 0] = np.kron(pauli_Z, pauli_Z)
            mpo.A[0][:, :, 0, 1] = np.kron(pauli_Z, bose_c) * X_mo[upper_index, 0]

            mpo.A[L-1][:, :, 0, 0] = np.kron(pauli_Z, bose_c) * X_mo[upper_index, L-1]
            mpo.A[L-1][:, :, 1, 0] = I4

            for i in range (1,L-1):
                mpo.A[i][:, :, 0, 0] = np.kron(pauli_Z, pauli_Z)
                mpo.A[i][:, :, 0, 1] = np.kron(pauli_Z, bose_c) * X_mo[upper_index, i]
                mpo.A[i][:, :, 1, 0] = 0
                mpo.A[i][:, :, 1, 1] = I4     

    return mpo

# def get_four_layers_spin_qn(mu, nu, X_mo_spin_up, X_mo_spin_down, Z_mo, L, spin1, spin2):
#     ''' 
#     generate all four mpo layers for a sub-Hamiltonian
#     the scaler \zeta_{\mu,\nu} is absorbed into the top layer
#     the H1 in this func is the mpo on the bottom
#     '''
#     if spin1 == 0:
#         H1 = get_elementary_mpo_qn(nu, 0, X_mo_spin_up, L)
#         H2 = get_elementary_mpo_qn(nu, 1, X_mo_spin_up, L)
#     if spin1 == 1:
#         H1 = get_elementary_mpo_qn(nu, 0, X_mo_spin_down, L)
#         H2 = get_elementary_mpo_qn(nu, 1, X_mo_spin_down, L)

#     if spin2 == 0:
#         H3 = get_elementary_mpo_qn(mu, 0, X_mo_spin_up, L)
#         H4 = get_elementary_mpo_qn(mu, 1, X_mo_spin_up, L)
#     if spin2 == 1:
#         H3 = get_elementary_mpo_qn(mu, 0, X_mo_spin_down, L)
#         H4 = get_elementary_mpo_qn(mu, 1, X_mo_spin_down, L)
        
#     H4.A[0] = Z_mo[mu, nu] * H4.A[0]
        
#     return [H1, H2, H3, H4] #H_mu_nu = H4@H3@H2@H1

def get_four_layers_spin_qn(mu, nu, X_mo, Z_mo, L, spin1, spin2):
    ''' 
    generate all four mpo layers for a sub-Hamiltonian
    the scaler \zeta_{\mu,\nu} is absorbed into the top layer
    the H1 in this func is the mpo on the bottom
    '''
   
    H1 = get_elementary_mpo_qn(nu, 0, X_mo, L, spin1)
    H2 = get_elementary_mpo_qn(nu, 1, X_mo, L, spin1)

    H3 = get_elementary_mpo_qn(mu, 0, X_mo, L, spin2)
    H4 = get_elementary_mpo_qn(mu, 1, X_mo, L, spin2)

    H4.A[0] = Z_mo[mu, nu] * H4.A[0]
        
    return [H1, H2, H3, H4] #H_mu_nu = H4@H3@H2@H1



def get_h1_spin(h1):
    #note: h1 from pyscf uses chemist's notation
    #input: the kinetic term h1, as a 2d matrix of shape (nmo, nmo)
    #return: corresponding matrix considering the spins, of shape (2nmo, 2nmo)
    nmo = h1.shape[0]
    
    h1_upup = np.zeros([2*h1.shape[0], 2*h1.shape[1]], dtype = h1.dtype)
    h1_dd = np.zeros([2*h1.shape[0], 2*h1.shape[1]], dtype = h1.dtype)

    for i in range (nmo):
        for j in range (nmo):
            h1_upup[2*i, 2*j] = h1[i, j]

    for i in range (nmo):
        for j in range (nmo):
            h1_dd[2*i+1, 2*j+1] = h1[i, j]

    h1_spin = h1_upup + h1_dd
    
    return(h1_spin)

def get_g_spin(g_mo):
    #note: g from pyscf uses chemist's notation
    #input: the Coulomb term g, as a 4-d tensor of shape (nmo, nmo, nmo, nmo)
    #return: corresponding tensor considering the spins, of shape (2nmo, 2nmo, 2nmo, 2nmo)
    #physicist's notation:h1_spin and g_spin.transpose(0, 2, 1, 3)
    #chemist's notation: h1_spin and g_spin
    #<even more...> paper notation: t1_spin (later) and g_spin
     
    nmo = g_mo.shape[0]
    
    g_spin = np.zeros([2*nmo, 2*nmo, 2*nmo, 2*nmo], dtype = g_mo.dtype)

    for p in range(2*nmo):
        for q in range(2*nmo):
            for r in range(2*nmo):
                for s in range(2*nmo):
                    if p%2 == q%2:
                        if r%2 == s%2:
                            g_spin[p,q,r,s] = g_mo[p//2, q//2, r//2, s//2]
                            
    return(g_spin) 
    
    
def get_t_spin(h1, g_mo):
    
    #<even more...> paper notation: t1_spin (later) and g_spin
    
    nmo = g_mo.shape[0]
    
    g_trace_temp = np.zeros([nmo, nmo])
    for p in range (nmo):
        for s in range (nmo):
            for i in range (nmo):
                g_trace_temp[p, s] += g_mo[p, i, i, s] 
                
    t = h1 - 0.5 * g_trace_temp

    t_upup = np.zeros([2*t.shape[0], 2*t.shape[1]], dtype = t.dtype)
    t_dd = np.zeros([2*t.shape[0], 2*t.shape[1]], dtype = t.dtype)

    for i in range (nmo):
        for j in range (nmo):
            t_upup[2*i, 2*j] = t[i, j]

    for i in range (nmo):
        for j in range (nmo):
            t_dd[2*i+1, 2*j+1] = t[i, j]

    t_spin = t_upup + t_dd
    
    return(t_spin)

def get_t(h1, g_mo):
    
    #<even more...> paper notation: t1_spin (later) and g_spin
    
    nmo = g_mo.shape[0]
    
    g_trace_temp = np.zeros([nmo, nmo])
    for p in range (nmo):
        for s in range (nmo):
            for i in range (nmo):
                g_trace_temp[p, s] += g_mo[p, i, i, s] 
                
    t = h1 - 0.5 * g_trace_temp
    
    return(t)


def get_X_up(X_mo):
    ''' 
    Used when considering each spinor orbital as an MPS site.
    
    Input: X_mo, e.g, \chi tensor in THC
    
    Output: \chi which helps to implement \sum_s^L \chi a_{s, spin_up}
    
    We set sites with even indices for up spins.
    '''
    r_THC = X_mo.shape[0]
    nmo = X_mo.shape[1]
    X_mo_up = np.zeros([X_mo.shape[0], 2*X_mo.shape[1]], dtype = X_mo.dtype)
    for i in range (r_THC):
        for j in range (nmo):
            X_mo_up[i, 2*j] = X_mo[i, j]
            
    return X_mo_up


def get_X_down(X_mo):
    ''' 
    Used when considering each spinor orbital as an MPS site.
    
    Input: X_mo, e.g, \chi tensor in THC
    
    Output: \chi which helps to implement \sum_s^L \chi a_{s, spin_down}
    
    We set sites with odd indices for down spins.
    '''
    r_THC = X_mo.shape[0]
    nmo = X_mo.shape[1]
    X_mo_down = np.zeros([X_mo.shape[0], 2*X_mo.shape[1]], dtype = X_mo.dtype)
    for i in range (r_THC):
        for j in range (nmo):
            X_mo_down[i, 2*j + 1] = X_mo[i, j]
    
    return X_mo_down

def generate_thc_mpos_by_layer_qn(X_mo, Z_mo, L, t_spin):
    
    '''  
    Given: THC tensors, kinetic term t_spin, and system size L (actually unnecessary, since we can read it from t_spin).
    
    Output: a list containning MPOs for all sub-Hamiltonian H_mu_nu, as well as MPO for the kinectic term.
    
    For each sub-Hamiltonian H_mu_nu, it consists of four elementary MPOs, i.e.,  H_mu_nu_list_spin_layer[i] is also a list which contains four MPOs.
    '''
    
    # X_mo_up = get_X_up(X_mo)
    # X_mo_down = get_X_down(X_mo)
    r_THC = X_mo.shape[0]
    L = X_mo.shape[1]
    
    H_mu_nu_list_spin_layer = []
    for mu in range(r_THC):
        for nu in range(r_THC):
            for s1 in range(2):
                for s2 in range(2):
                    #H_mu_nu_list_spin_layer.append(get_four_layers_spin_qn(mu, nu, X_mo_up, X_mo_down, 0.5*Z_mo, L, s1, s2))
                    H_mu_nu_list_spin_layer.append(get_four_layers_spin_qn(mu, nu, X_mo, 0.5*Z_mo, L, s1, s2))
                    
    H_mu_nu_list_spin_layer.append([spin_molecular_hamiltonian_mpo(t_spin, np.zeros([L, L, L, L]))])
    
    
    return(H_mu_nu_list_spin_layer)


def conjmat(M):
    return np.conjugate(M.T)

def eval_func(a, eri, hkin, hnuc, rcond = 1e-13):
    
    
    no,m = a.shape
    dic = {}
    print(no)
    print(a.shape)

    MV = eri.reshape(no*no,no*no)
    B = np.einsum('ix,jx->ijx', a, a)
    MB = B.reshape(no**2,m)
    Binv = np.linalg.pinv(MB,rcond=rcond)
    W = Binv@MV@conjmat(Binv)
    D = Binv@(hkin.flatten())
    H = Binv@(hnuc.flatten())    
    print(W.shape)


    V_thc = MB @ W @ conjmat(MB)
    errV = np.linalg.norm(MB @ W @ conjmat(MB) - MV)/np.linalg.norm(MV)
    print("rl errV:", errV)
    errV = np.linalg.norm(MB @ W @ conjmat(MB) - MV)
    print("abs errV:", errV)

    
    errt = np.linalg.norm(a * D @ a.T - hkin)/np.linalg.norm(hkin)
    print("errt:", errt)
    errh = np.linalg.norm(a * H @ a.T - hnuc)/np.linalg.norm(hnuc)
    print("errh:", errh)
    errht = np.linalg.norm(a * (H+D) @ a.T - hnuc-hkin)/np.linalg.norm(hnuc+hkin)
    print("errht:", errht)
    dic["errV"] = errV
    dic["errt"] = errt
    dic["errh"] = errh
    dic["errht"] = errht
    
    return V_thc, W

def get_molecular_Hamiltonian_as_sparse_matrix(h1_spin, g_spin):
    
    ''' 
    get reference Hamiltonian, as a sparse matrix
    
    input: one-body and two-body integrals
    
    output: reference Hamiltonian, as a sparse matrix
    
    '''
    L = (g_spin.shape)[0]
    h1_ref = sparse.csr_matrix((2**L, 2**L), dtype=float)
    for p in range (L):
        for q in range (L):
            h1_ref += create_op(L, 2**(L-1-p))@annihil_op(L, 2**(L-1-q)) * h1_spin[p,q]

    V_correct = sparse.csr_matrix((2**L, 2**L), dtype=float)
    for p in range (L):
        for q in range (L):
            for r in range (L):
                for s in range (L):
                    V_correct += 0.5*g_spin[p,q,r,s] * create_op(L, 2**(L-1-p))@(create_op(L, 2**(L-1-r))@(annihil_op(L, 2**(L-1-s))@annihil_op(L, 2**(L-1-q))))

    H_correct = h1_ref + V_correct
    
    return(H_correct)