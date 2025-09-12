import numpy as np
from .operation import apply_operator, add_mps_and_compress, apply_operator_and_compress, add_mps_and_compress_direct_SVD
import copy
from .mpo import MPO

__all__ = ['H_on_mps_compress_by_layer', 'apply_thc_mpo_and_compress', 'find_indices_spin']


def H_on_mps_compress_by_layer(H_mu_nu_by_layer, psi, tol, max_bond_layer):
    ''' 
    H_mu_nu_by_layer: the four MPOs of sub-Hamiltonian H_mu_nu.
    Apply the four elementary MPOs to psi and compress, sequentially.
    return: compressed H_mu_nu|\psi> in left-canonical form.
    '''
    temp = copy.deepcopy(psi)
    for layer in H_mu_nu_by_layer:
        temp = apply_operator(layer, temp)
        temp.compress_no_normalization_max_bond(tol, max_bond = max_bond_layer)
    return temp


def apply_thc_mpo_and_compress(sub_H_list_as_layer, psi, trunc_tol, max_bond_global, r_THC):
    '''  
    It is long, but actually doesn't contain too much information.
    
    Purpose: apply all THC-MPOs on state \psi, seperately. And add the results together. 
    
    Input: sub_H_list_as_layer: THC-MPO; psi: quantum state as MPS.
    
    Output: compressed H\psi> as MPS, implemented with THC-MPO.
    
    Why is this so long? What does try/ except do? In the case singular values are small, the SVD doesn't converge, so that we must truncate more. 
    When the SVD convergence error occurs, we increase the trunc_tol. It only works when tunc_tol is not set as 0.
    
    For add, we allow a little larger bond-dims, since it is cheap. But the final Krylov vectors are still kept as the same.
    '''
    psi_original = copy.deepcopy(psi)
    for nu in range (r_THC):
        for s1 in range (2):
            #print("nu, s1:", nu, s1)
            H_nu_layers = sub_H_list_as_layer[find_indices_spin(0, nu, s1 ,0 ,r_THC)]
            temp_layers_nu = [H_nu_layers[0], H_nu_layers[1]]
            try:
                H_nu_psi =  H_on_mps_compress_by_layer(temp_layers_nu, psi_original, trunc_tol, max_bond_global)
            except Exception:
                #svd_small_count += 1
                print(nu, s1)
                try:
                    H_nu_psi =  H_on_mps_compress_by_layer(temp_layers_nu, psi_original, 10*trunc_tol, max_bond_global)
                except Exception:
                    try:
                        H_nu_psi =  H_on_mps_compress_by_layer(temp_layers_nu, psi_original, 100*trunc_tol, max_bond_global)
                    except Exception:
                        try:
                            H_nu_psi =  H_on_mps_compress_by_layer(temp_layers_nu, psi_original, 1000*trunc_tol, max_bond_global)
                        except Exception:
                            print("still fail for 4th attempt")
                    

            for mu in range(r_THC):
                for s2 in range (2):
                    #print("mu, s2:", mu, s2)
                    H_mu_layers = sub_H_list_as_layer[find_indices_spin (mu, nu, s1 ,s2 ,r_THC)]
                    temp_layers_mu = [H_mu_layers[2], H_mu_layers[3]]
                    if mu == 0 and s1 == 0 and nu == 0 and s2 == 0:
                        try:
                            H_on_psi = H_on_mps_compress_by_layer(temp_layers_mu, H_nu_psi, trunc_tol, max_bond_global)
                
                        except Exception:
                            #svd_small_count += 1
                            print(mu, nu, s1, s2)
                            try:
                                H_on_psi = H_on_mps_compress_by_layer(temp_layers_mu, H_nu_psi, 10*trunc_tol, max_bond_global)
                            except Exception:
                                try:
                                    H_on_psi = H_on_mps_compress_by_layer(temp_layers_mu, H_nu_psi, 100*trunc_tol, max_bond_global)
                                except Exception:
                                    try:
                                        H_on_psi = H_on_mps_compress_by_layer(temp_layers_mu, H_nu_psi, 1000*trunc_tol, max_bond_global)
                                    except Exception:
                                        print("still fail for 4th attempt")
                    else: 
                        try:
                            temp_mps = H_on_mps_compress_by_layer(temp_layers_mu, H_nu_psi, trunc_tol, max_bond_global)
                            #we allow a little larger bond-dims for addition
                            #H_on_psi = add_mps_and_compress_direct_SVD(H_on_psi, temp_mps, trunc_tol, 2*max_bond_global) 
                            H_on_psi = add_mps_and_compress(H_on_psi, temp_mps, trunc_tol, max_bond_global) 
                            #H_on_psi = add_mps_and_compress(H_on_psi, temp_mps, trunc_tol, max_bond_global) 
                        except Exception:
                            #svd_small_count += 1
                            print(mu, nu, s1, s2)
                            try:
                                temp_mps = H_on_mps_compress_by_layer(temp_layers_mu, H_nu_psi, 10*trunc_tol, max_bond_global)
                                H_on_psi = add_mps_and_compress(H_on_psi, temp_mps, 10*trunc_tol, max_bond_global) 
                                #H_on_psi = add_mps_and_compress_direct_SVD(H_on_psi, temp_mps, 10*trunc_tol, max_bond_global) 
                                
                            except Exception:
                                try:
                                    temp_mps = H_on_mps_compress_by_layer(temp_layers_mu, H_nu_psi, 100*trunc_tol, max_bond_global)
                                    H_on_psi = add_mps_and_compress(H_on_psi, temp_mps, 100*trunc_tol, max_bond_global)  
                                except Exception:
                                    try:
                                        temp_mps = H_on_mps_compress_by_layer(temp_layers_mu, H_nu_psi, 1000*trunc_tol, max_bond_global)
                                        H_on_psi = add_mps_and_compress(H_on_psi, temp_mps, 1000*trunc_tol, max_bond_global) 
                                    except Exception:
                                        print("still fail for 4th attempt")
    
    #The last step: apply kinetic term on \psi, and add it to the summation.
    #H_on_psi = add_mps_and_compress_direct_SVD(H_on_psi, Kinetic_on_psi, trunc_tol, max_bond_global) 
    
    Kinetic_on_psi = apply_operator_and_compress(sub_H_list_as_layer[-1][0], psi_original, trunc_tol, max_bond_global)                         
    H_on_psi = add_mps_and_compress(H_on_psi, Kinetic_on_psi, trunc_tol, max_bond_global) 
     
    return (H_on_psi)



def find_indices_spin (mu, nu, s1 ,s2 ,r_THC):
    ''' 
    Input: the four indices {mu, nu, spin1, spin2} for sub-Hamiltonian.
    
    Output: the index for MPO for sub-Hamiltonian, in the THC-MPO list.
    '''

    return (mu* 4* (r_THC) + nu* 4 + 2*s1 + s2)


# def generate_krylov_space_in_disk(N_Krylov, H_mu_nu_list_spin_layer, psi_original, max_bond_Krylov, trunc_tol, r_THC, foldername):  
#     '''  
#     generate orthogonalized Krylov space, and store in disk.
    
#     N_Krylov: the size of Krylov space.
    
#     H_mu_nu_list_spin_layer: THC-MPO, as a list of MPOs.
    
#     psi_original: initial state v_0 for Krylov methods.
    
#     max_bond_Krylov: maximum bond dimension for Krylov vectors.
    
#     foldername: one must input a foldername where the Krylov vectors are stored in.
#     '''
    
#     # store the v_0 in disk
#     filename = foldername + f"/Krylov_vec{0}.pkl"
#     with open(filename, 'wb') as file:
#         pickle.dump(copy.deepcopy(psi_original), file)

#     # H v_0 and orthogonalize it, then store in disk.
#     H_on_psi = apply_thc_mpo_and_compress(H_mu_nu_list_spin_layer, copy.deepcopy(psi_original), trunc_tol, max_bond_Krylov, r_THC)
#     H_on_psi.orthonormalize('right')
#     print(H_on_psi.bond_dims)

#     temp = copy.deepcopy(psi_original)
#     temp.A[0] = -vdot(H_on_psi, temp)* temp.A[0]
#     H_on_psi =  add_mps_and_compress(copy.deepcopy(H_on_psi), temp, trunc_tol, max_bond_Krylov)
#     H_on_psi.orthonormalize('right')

#     filename = foldername + f"/Krylov_vec{1}.pkl"
#     with open(filename, 'wb') as file:
#         pickle.dump(H_on_psi, file)

#     # from now on, all the H v_i should be orthogonalized to the previous two vectors.
#     for i in range (2, N_Krylov):
        
#         #show how many Krylov vectors are achieved     
#         print(i)    
#         # if i % 5 == 0:
#         #     print("implemented:", i)
        
#         #from disk load the previous two Krylov vectors.    
#         filename = foldername + f"/Krylov_vec{i-2}.pkl"
#         with open(filename, 'rb') as file:
#             orth_state1 = pickle.load(file)
#         filename = foldername + f"/Krylov_vec{i-1}.pkl"
#         with open(filename, 'rb') as file:
#             orth_state2 = pickle.load(file)
        
#         #first calculate H \v_i
#         this_state = apply_thc_mpo_and_compress(H_mu_nu_list_spin_layer, copy.deepcopy(orth_state2), trunc_tol, max_bond_Krylov, r_THC)
#         this_state.orthonormalize('right')
#         #print(this_state.bond_dims)
#         #orthogonalize "this state H \v_i" against the previous two”
#         this_state = ortho_to_previous_two(orth_state1, orth_state2, this_state, max_bond_Krylov, trunc_tol)
#         print(this_state.bond_dims)
#         # store orthogonalized H \v_i in disk.
#         filename = foldername + f"/Krylov_vec{i}.pkl"
#         with open(filename, 'wb') as file:
#             pickle.dump(this_state, file)
            
            
# def ortho_to_previous_two(orth_state1, orth_state2, this_state, max_bond, trunc_tol_ortho):
    
#     #orthoglnolize "this state" to previous two states, in MPS form.
    
#     temp_state = copy.deepcopy(this_state)
    
#     temp = copy.deepcopy(orth_state1)
#     temp.A[0] = -vdot(temp_state, temp)* temp.A[0]
#     this_state = add_mps_and_compress(copy.deepcopy(this_state), temp, trunc_tol_ortho, max_bond)
    
#     temp = copy.deepcopy(orth_state2)
#     temp.A[0] = -vdot(temp_state, temp)* temp.A[0]
#     this_state =  add_mps_and_compress(copy.deepcopy(this_state), temp, trunc_tol_ortho, max_bond)
    
#     this_state.orthonormalize('right')
    
#     return(this_state)


# def get_W(N_use, foldername):
#     #use stratege proposd in <Lanczos algorithm with Matrix Product States for dynamical correlation functions> to improve orthogonality
#     #get W in the paper
#     W = np.zeros([N_use, N_use])

#     for i in range (N_use):
#         for j in range (N_use):
#             filename = foldername + f"/Krylov_vec{i}.pkl"
#             with open(filename, 'rb') as file:
#                 temp1 = pickle.load(file)
                
#             filename = foldername + f"/Krylov_vec{j}.pkl"
#             with open(filename, 'rb') as file:
#                 temp2 = pickle.load(file)
#             W[i,j] = np.vdot(temp1.as_vector(), temp2.as_vector())
    
#     return(W)

# def get_S(W):
#     #use stratege proposd in <Lanczos algorithm with Matrix Product States for dynamical correlation functions> to improve orthogonality
#     #get S using W in the paper
#     x = W.shape[0]
#     S = []
#     S.append(np.zeros(x))
#     S[0][0] = 1

#     S_tilde = []
#     S_tilde.append(np.zeros(x))
#     S_tilde[0][0] = 1

#     k = np.zeros([x,x])
#     k[0, 0] = 1
#     k[0,:] = W[:,0]

#     Normalization = np.zeros(x)
#     Normalization[0] = 1

#     for n in range (1, x):
        
#         S_tilde.append(np.zeros(x))
#         S.append(np.zeros(x))
        
#         S[n][n] = 1
#         for i in range (n):
#             for k in range (i+1):
#                 for k_prime in range (i+1):
#                     S[n][k_prime] += -W[n,k] * S[i][k_prime]* S[i][k]     
                
#         Normalization[n] = np.sqrt(sum(S[n][p] * S[n][q] * W[p, q] for p in range(n + 1) for q in range(n + 1)))

#         S[n] = S[n]/Normalization[n]

#     return (S)

# def generate_re_ortho_space(N_use, W, foldername):
#     #use stratege proposd in <Lanczos algorithm with Matrix Product States for dynamical correlation functions> to improve orthogonality
#     #generate a list of post-orthogonalized Krylov vectors (in np.array)
#     S = get_S(W)
#     vector_list = []
    
#     filename = foldername + f"/Krylov_vec{0}.pkl"
#     with open(filename, 'rb') as file:
#         shape_test = pickle.load(file)
#     L = shape_test.nsites
    
#     for i in range (N_use):
#         temp2 = np.zeros([2**L], dtype = 'complex128')
#         for j in range (i+1):
#             filename = foldername + f"/Krylov_vec{j}.pkl"
#             with open(filename, 'rb') as file:
#                 temp1 = pickle.load(file)
#             temp2 += S[i][j]* temp1.as_vector()
        
#         temp2 /= np.linalg.norm(temp2)
#         vector_list.append(temp2)
#     return(vector_list)

# def generate_reduced_H(vector_list, H):
#     H_reduced = np.zeros([len(vector_list), len(vector_list)])
#     for i in range (len(vector_list)):
#         for j in range (len(vector_list)):
#             H_reduced[i, j] = np.vdot(vector_list[i], H@vector_list[j])
#     return(H_reduced)

# def popcount(x):
#     return bin(x).count('1')

# def generate_Hamiltonian_with_occupation_number(H, n):
#     """
#     Efficiently extract a Hamiltonian with a fixed occupation number from a given Hamiltonian H.
#     """
#     rows, cols = H.nonzero()
#     size = H.shape
#     H_occu = sparse.csr_matrix((size[0], size[0]), dtype=float)
    
#     valid_indices = [(i, j) for i, j in zip(rows, cols) if popcount(i) == n and popcount(j) == n]
    
#     if valid_indices:
#         valid_rows, valid_cols = zip(*valid_indices)
#         H_occu[valid_rows, valid_cols] = H[valid_rows, valid_cols]

#     return H_occu

def get_num_op_mpo(site_ind, L):
    #attention:
    #in our case, we start from the ends to count Z operators...
    #but for Hamiltonian, due to the Z operators only exist BETWEEN two sites, so that which site we 
    #start the Z operators doesn't matter.
    
    d = 2
    qd = np.zeros(d, dtype=int)
    
    DO =  [1] * (L + 1) 
    qDO = [np.zeros(Di, dtype=int) for Di in DO]
    
    I2 = np.identity(2)
    num_op = np.array([[0., 0.],[0, 1.]])
    
    mpo = MPO(qd, qDO, fill= 'random' )
    # for i in range(L):
    #     (mpo.A[i]).fill(0)
        
    for i in range (L):
        mpo.A[i][:, :, 0, 0] = I2
        #mpo.A[i][:, :, 0, 0] = pauli_Z
    
    mpo.A[site_ind][:, :, 0, 0] = num_op
        
    return mpo