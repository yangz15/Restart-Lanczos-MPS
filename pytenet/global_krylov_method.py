import numpy as np
import pickle
import copy
from .mps import add_mps
from .operation import vdot, operator_inner_product, operator_average, add_mps_and_compress, apply_operator_and_compress,apply_operator
#from .operation import add_mps_and_compress_direct_SVD
from .operation_thc import apply_thc_mpo_and_compress
from scipy import sparse

__all__ = ['generate_krylov_space_in_disk','generate_krylov_space' ,'ortho_to_previous_two', 'get_W', 'get_S', 'ortho_to_previous_two', 'generate_re_ortho_space', 'generate_reduced_H', 'generate_Hamiltonian_with_occupation_number']


def generate_krylov_space_in_disk(N_Krylov, H_mu_nu_list_spin_layer, psi_original, max_bond_Krylov, trunc_tol, r_THC, foldername):  
    '''  
    generate orthogonalized Krylov space, and store in disk.
    
    N_Krylov: the size of Krylov space.
    
    H_mu_nu_list_spin_layer: THC-MPO, as a list of MPOs.
    
    psi_original: initial state v_0 for Krylov methods.
    
    max_bond_Krylov: maximum bond dimension for Krylov vectors.
    
    foldername: one must input a foldername where the Krylov vectors are stored in.
    '''
    
    # store the v_0 in disk
    filename = foldername + f"/Krylov_vec{0}.pkl"
    with open(filename, 'wb') as file:
        pickle.dump(copy.deepcopy(psi_original), file)

    # H v_0 and orthogonalize it, then store in disk.
    H_on_psi = apply_thc_mpo_and_compress(H_mu_nu_list_spin_layer, copy.deepcopy(psi_original), trunc_tol, max_bond_Krylov, r_THC)
    H_on_psi.orthonormalize('right')
    print(H_on_psi.bond_dims)

    temp = copy.deepcopy(psi_original)
    temp.A[0] = -vdot(H_on_psi, temp)* temp.A[0]
    #compress the bond dims back 
    H_on_psi =  add_mps_and_compress(copy.deepcopy(H_on_psi), temp, trunc_tol, max_bond_Krylov)
    #H_on_psi =  add_mps_and_compress(copy.deepcopy(H_on_psi), temp, trunc_tol, max_bond_Krylov)
    H_on_psi.orthonormalize('right')

    filename = foldername + f"/Krylov_vec{1}.pkl"
    with open(filename, 'wb') as file:
        pickle.dump(H_on_psi, file)

    # from now on, all the H v_i should be orthogonalized to the previous two vectors.
    for i in range (2, N_Krylov):
        
        #show how many Krylov vectors are achieved     
        print(i)    
        # if i % 5 == 0:
        #     print("implemented:", i)
        
        #from disk load the previous two Krylov vectors.    
        filename = foldername + f"/Krylov_vec{i-2}.pkl"
        with open(filename, 'rb') as file:
            orth_state1 = pickle.load(file)
        filename = foldername + f"/Krylov_vec{i-1}.pkl"
        with open(filename, 'rb') as file:
            orth_state2 = pickle.load(file)
        
        #first calculate H \v_i
        this_state = apply_thc_mpo_and_compress(H_mu_nu_list_spin_layer, copy.deepcopy(orth_state2), trunc_tol, 2*max_bond_Krylov, r_THC)
        this_state.orthonormalize('right')
        #print(this_state.bond_dims)
        #orthogonalize "this state H \v_i" against the previous two” and compress the bond dims back 
        this_state = ortho_to_previous_two(orth_state1, orth_state2, this_state, max_bond_Krylov, trunc_tol)
        print(this_state.bond_dims)
        # store orthogonalized H \v_i in disk.
        filename = foldername + f"/Krylov_vec{i}.pkl"
        with open(filename, 'wb') as file:
            pickle.dump(this_state, file)




def generate_krylov_space_test_poly(N_Krylov, H,coeff, psi_original, max_bond_Krylov, trunc_tol, foldername):  
    '''  
    generate orthogonalized Krylov space, and store in disk.
    
    N_Krylov: the size of Krylov space.
    
    H: MPO.
    
    psi_original: initial state v_0 for Krylov methods.
    
    max_bond_Krylov: maximum bond dimension for Krylov vectors.
    
    foldername: one must input a foldername where the Krylov vectors are stored in.
    '''
    
    # store the v_0 in disk
    filename = foldername + f"/Krylov_vec{0}.pkl"
    with open(filename, 'wb') as file:
        pickle.dump(copy.deepcopy(psi_original), file)

    # H v_0 and orthogonalize it, then store in disk.
    H_on_psi_temp = apply_operator_and_compress(H, copy.deepcopy(psi_original), trunc_tol, max_bond_Krylov)
    #H_on_psi_temp.orthonormalize('right')  
    H_on_psi_list = [H_on_psi_temp]
    for x in range(1,len(coeff)):
        H_on_psi_temp = apply_operator_and_compress(H, H_on_psi_temp, trunc_tol,  max_bond_Krylov)
        #H_on_psi_temp.orthonormalize('right')
        H_on_psi_list.append(H_on_psi_temp)
    for j in range(len(coeff)):
        H_on_psi_list[j].A[0] = H_on_psi_list[j].A[0] * coeff[j]
    H_on_psi = H_on_psi_list[0]
    for k in range(1,len(coeff)):
        H_on_psi = add_mps_and_compress(H_on_psi, H_on_psi_list[k], trunc_tol, max_bond_Krylov)

    H_on_psi.orthonormalize('right')
    print(H_on_psi.bond_dims)

    temp = copy.deepcopy(psi_original)
    temp.A[0] = -vdot(H_on_psi, temp)* temp.A[0]
    #compress the bond dims back 
    H_on_psi =  add_mps_and_compress(copy.deepcopy(H_on_psi), temp, trunc_tol, max_bond_Krylov)
    #H_on_psi =  add_mps_and_compress(copy.deepcopy(H_on_psi), temp, trunc_tol, max_bond_Krylov)
    H_on_psi.orthonormalize('right')

    filename = foldername + f"/Krylov_vec{1}.pkl"
    with open(filename, 'wb') as file:
        pickle.dump(H_on_psi, file)

    # from now on, all the H v_i should be orthogonalized to the previous two vectors.
    for i in range (2, N_Krylov):
        
        #show how many Krylov vectors are achieved     
        print(i)    
        # if i % 5 == 0:
        #     print("implemented:", i)
        
        #from disk load the previous two Krylov vectors.    
        filename = foldername + f"/Krylov_vec{i-2}.pkl"
        with open(filename, 'rb') as file:
            orth_state1 = pickle.load(file)
        filename = foldername + f"/Krylov_vec{i-1}.pkl"
        with open(filename, 'rb') as file:
            orth_state2 = pickle.load(file)
        
        #first calculate H \v_i

        this_state_temp = apply_operator_and_compress(H, copy.deepcopy(orth_state2), trunc_tol, max_bond_Krylov)
        #this_state_temp.orthonormalize('right')  
        this_state_list = [this_state_temp]
        for x in range(1,len(coeff)):
            this_state_temp = apply_operator_and_compress(H, this_state_temp, trunc_tol, max_bond_Krylov)
            #this_state_temp.orthonormalize('right')
            this_state_list.append(this_state_temp)
        for j in range(len(coeff)):
            this_state_list[j].A[0] = this_state_list[j].A[0] * coeff[j]
        this_state = this_state_list[0]
        for k in range(1,len(coeff)):
            this_state = add_mps(this_state, this_state_list[k])
        this_state.compress_no_normalization_max_bond(trunc_tol,max_bond_Krylov)
        this_state.orthonormalize('right')
        #print(this_state.bond_dims)
        #orthogonalize "this state H \v_i" against the previous two” and compress the bond dims back 
        this_state = ortho_to_previous_two(orth_state1, orth_state2, this_state, max_bond_Krylov, trunc_tol)
        print(this_state.bond_dims)
        # store orthogonalized H \v_i in disk.
        filename = foldername + f"/Krylov_vec{i}.pkl"
        with open(filename, 'wb') as file:
            pickle.dump(this_state, file)


def generate_krylov_space_test_poly_with_constant(N_Krylov, H,coeff, psi_original, max_bond_Krylov, trunc_tol, foldername):  
    '''  
    generate orthogonalized Krylov space, and store in disk.
    
    N_Krylov: the size of Krylov space.
    
    H: MPO.
    
    psi_original: initial state v_0 for Krylov methods.
    
    max_bond_Krylov: maximum bond dimension for Krylov vectors.
    
    foldername: one must input a foldername where the Krylov vectors are stored in.
    '''
    
    # store the v_0 in disk
    filename = foldername + f"/Krylov_vec{0}.pkl"
    with open(filename, 'wb') as file:
        pickle.dump(copy.deepcopy(psi_original), file)

    # H v_0 and orthogonalize it, then store in disk.
    H_on_psi_list = [copy.deepcopy(psi_original)]
    H_on_psi_temp = copy.deepcopy(psi_original)
    for x in range(1,len(coeff)):
        H_on_psi_temp = apply_operator_and_compress(H, H_on_psi_temp, trunc_tol,  max_bond_Krylov)
        #H_on_psi_temp.orthonormalize('right')
        H_on_psi_list.append(H_on_psi_temp)
    for j in range(len(coeff)):
        H_on_psi_list[j].A[0] = H_on_psi_list[j].A[0] * coeff[j]
    H_on_psi = H_on_psi_list[0]
    for k in range(1,len(coeff)):
        H_on_psi = add_mps_and_compress(H_on_psi, H_on_psi_list[k], trunc_tol, max_bond_Krylov)

    H_on_psi.orthonormalize('right')
    print(H_on_psi.bond_dims)

    temp = copy.deepcopy(psi_original)
    temp.A[0] = -vdot(H_on_psi, temp)* temp.A[0]
    #compress the bond dims back 
    H_on_psi =  add_mps_and_compress(copy.deepcopy(H_on_psi), temp, trunc_tol, max_bond_Krylov)
    #H_on_psi =  add_mps_and_compress(copy.deepcopy(H_on_psi), temp, trunc_tol, max_bond_Krylov)
    H_on_psi.orthonormalize('right')

    filename = foldername + f"/Krylov_vec{1}.pkl"
    with open(filename, 'wb') as file:
        pickle.dump(H_on_psi, file)

    # from now on, all the H v_i should be orthogonalized to the previous two vectors.
    for i in range (2, N_Krylov):
        
        #show how many Krylov vectors are achieved     
        print(i)    
        # if i % 5 == 0:
        #     print("implemented:", i)
        
        #from disk load the previous two Krylov vectors.    
        filename = foldername + f"/Krylov_vec{i-2}.pkl"
        with open(filename, 'rb') as file:
            orth_state1 = pickle.load(file)
        filename = foldername + f"/Krylov_vec{i-1}.pkl"
        with open(filename, 'rb') as file:
            orth_state2 = pickle.load(file)
        
        #first calculate H \v_i

        this_state_list = [copy.deepcopy(orth_state2)]
        this_state_temp = copy.deepcopy(orth_state2)        #this_state_temp.orthonormalize('right')  
        for x in range(1,len(coeff)):
            this_state_temp = apply_operator_and_compress(H, this_state_temp, trunc_tol, max_bond_Krylov)
            #this_state_temp.orthonormalize('right')
            this_state_list.append(this_state_temp)
        for j in range(len(coeff)):
            this_state_list[j].A[0] = this_state_list[j].A[0] * coeff[j]
        this_state = this_state_list[0]
        for k in range(1,len(coeff)):
            this_state = add_mps(this_state, this_state_list[k])
        this_state.compress_no_normalization_max_bond(trunc_tol,max_bond_Krylov)
        this_state.orthonormalize('right')
        #print(this_state.bond_dims)
        #orthogonalize "this state H \v_i" against the previous two” and compress the bond dims back 
        this_state = ortho_to_previous_two(orth_state1, orth_state2, this_state, max_bond_Krylov, trunc_tol)
        print(this_state.bond_dims)
        # store orthogonalized H \v_i in disk.
        filename = foldername + f"/Krylov_vec{i}.pkl"
        with open(filename, 'wb') as file:
            pickle.dump(this_state, file)


def generate_krylov_space(N_Krylov, H, psi_original, max_bond_Krylov, trunc_tol, foldername):  
    '''  
    generate orthogonalized Krylov space, and store in disk.
    
    N_Krylov: the size of Krylov space.
    
    H: MPO.
    
    psi_original: initial state v_0 for Krylov methods.
    
    max_bond_Krylov: maximum bond dimension for Krylov vectors.
    
    foldername: one must input a foldername where the Krylov vectors are stored in.
    '''
    
    # store the v_0 in disk
    filename = foldername + f"/Krylov_vec{0}.pkl"
    with open(filename, 'wb') as file:
        pickle.dump(copy.deepcopy(psi_original), file)

    # H v_0 and orthogonalize it, then store in disk.
    H_on_psi = apply_operator_and_compress(H, copy.deepcopy(psi_original), trunc_tol, max_bond_Krylov)
    #H_on_psi.orthonormalize('right')
    print(H_on_psi.bond_dims)

    temp = copy.deepcopy(psi_original)
    temp.A[0] = -vdot(H_on_psi, temp)* temp.A[0]
    #compress the bond dims back 
    H_on_psi =  add_mps_and_compress(copy.deepcopy(H_on_psi), temp, trunc_tol, max_bond_Krylov)
    #H_on_psi =  add_mps_and_compress(copy.deepcopy(H_on_psi), temp, trunc_tol, max_bond_Krylov)
    H_on_psi.orthonormalize('right')

    filename = foldername + f"/Krylov_vec{1}.pkl"
    with open(filename, 'wb') as file:
        pickle.dump(H_on_psi, file)
    # from now on, all the H v_i should be orthogonalized to the previous two vectors.
    for i in range (2, N_Krylov):
        
        #show how many Krylov vectors are achieved     
        print(i)    
        # if i % 5 == 0:
        #     print("implemented:", i)
        
        #from disk load the previous two Krylov vectors.    
        filename = foldername + f"/Krylov_vec{i-2}.pkl"
        with open(filename, 'rb') as file:
            orth_state1 = pickle.load(file)
        filename = foldername + f"/Krylov_vec{i-1}.pkl"
        with open(filename, 'rb') as file:
            orth_state2 = pickle.load(file)
        
        #first calculate H \v_i
        this_state = apply_operator_and_compress(H, copy.deepcopy(orth_state2), trunc_tol, max_bond_Krylov)
        #this_state.orthonormalize('right')
        #print(this_state.bond_dims)
        #orthogonalize "this state H \v_i" against the previous two” and compress the bond dims back 
        this_state = ortho_to_previous_two(orth_state1, orth_state2, this_state, max_bond_Krylov, trunc_tol)
        print(this_state.bond_dims)
        # store orthogonalized H \v_i in disk.
        filename = foldername + f"/Krylov_vec{i}.pkl"
        with open(filename, 'wb') as file:
            pickle.dump(this_state, file)
            
def ortho_to_previous_two(orth_state1, orth_state2, this_state, max_bond, trunc_tol_ortho):
    
    #orthoglnolize "this state" to previous two states, in MPS form.
    
    temp_state = copy.deepcopy(this_state)
    
    temp = copy.deepcopy(orth_state1)
    temp.A[0] = -vdot(temp_state, temp)* temp.A[0]
    #this_state = add_mps_and_compress(copy.deepcopy(this_state), temp, trunc_tol_ortho, 2*max_bond)
    this_state = add_mps(copy.deepcopy(this_state), temp)
    
    temp = copy.deepcopy(orth_state2)
    temp.A[0] = -vdot(temp_state, temp)* temp.A[0]
    this_state =  add_mps_and_compress(copy.deepcopy(this_state), temp, trunc_tol_ortho, max_bond)

    this_state.orthonormalize('right')
    
    return(this_state) 


def get_W(N_use, foldername):
    #use stratege proposd in <Lanczos algorithm with Matrix Product States for dynamical correlation functions> to improve orthogonality
    #get W in the paper
    W = np.zeros([N_use, N_use])

    for i in range (N_use):
        for j in range (N_use):
            filename = foldername + f"/Krylov_vec{i}.pkl"
            with open(filename, 'rb') as file:
                temp1 = pickle.load(file)
                
            filename = foldername + f"/Krylov_vec{j}.pkl"
            with open(filename, 'rb') as file:
                temp2 = pickle.load(file)
            W[i,j] = vdot(temp1, temp2)
    
    return(W)

def get_S(W):
    #use stratege proposd in <Lanczos algorithm with Matrix Product States for dynamical correlation functions> to improve orthogonality
    #get S using W in the paper
    x = W.shape[0]
    S = []
    S.append(np.zeros(x))
    S[0][0] = 1

    S_tilde = []
    S_tilde.append(np.zeros(x))
    S_tilde[0][0] = 1

    k = np.zeros([x,x])
    k[0, 0] = 1
    k[0,:] = W[:,0]

    Normalization = np.zeros(x)
    Normalization[0] = 1

    for n in range (1, x):
        
        S_tilde.append(np.zeros(x))
        S.append(np.zeros(x))
        
        S[n][n] = 1
        for i in range (n):
            for k in range (i+1):
                for k_prime in range (i+1):
                    S[n][k_prime] += -W[n,k] * S[i][k_prime]* S[i][k]     
                
        Normalization[n] = np.sqrt(sum(S[n][p] * S[n][q] * W[p, q] for p in range(n + 1) for q in range(n + 1)))

        S[n] = S[n]/Normalization[n]

    return (S)

def coeff_gram_schmidt(N, foldername):
    '''
    actually, it is just a simple combination of above two functions.
    input: Krylov subspace size, Krylov vectors stored in foldername
    output: orthogonal coeff as numpy array
    Attention: here the coeff is as numpy array, instead of a list as above.
    '''
    W = get_W(N, foldername)
    coeff = get_S(W)
    coeff = np.array(coeff) 
    
    return coeff
    


def generate_re_ortho_space(N_use, foldername):
    #use stratege proposd in <Lanczos algorithm with Matrix Product States for dynamical correlation functions> to improve orthogonality
    #generate a list of post-orthogonalized Krylov vectors (in np.array)
    #Note: the vectors in this list are all np.array, we will have another MPS version.
    W = get_W(N_use, foldername)
    S = get_S(W)
    vector_list = []
    
    filename = foldername + f"/Krylov_vec{0}.pkl"
    with open(filename, 'rb') as file:
        shape_test = pickle.load(file)
    L = shape_test.nsites
    
    for i in range (N_use):
        temp2 = np.zeros([4**L], dtype = 'complex128')
        for j in range (i+1):
            filename = foldername + f"/Krylov_vec{j}.pkl"
            with open(filename, 'rb') as file:
                temp1 = pickle.load(file)
            temp2 += S[i][j]* temp1.as_vector()
        
        temp2 /= np.linalg.norm(temp2)
        vector_list.append(temp2)
    return(vector_list)

def generate_reduced_H_non_ortho(N, foldername, H_mpo):
    """ 
    Args: vectors (which are not orthogonal) as MPS, stored in folder 'foldername'
    
    H_mpo: hamiltonian as mpo
    
    Return: matrix as np.array: <v_i|H|v_j>
    
    The calculation is done using only MPS/MPO
    """
    H_reduced = np.zeros([N, N])
    for i in range (N):
        for j in range (N):
            filename = foldername + f"/Krylov_vec{i}.pkl"
            with open(filename, 'rb') as file:
                temp1 = pickle.load(file)
                
            filename = foldername + f"/Krylov_vec{j}.pkl"
            with open(filename, 'rb') as file:
                temp2 = pickle.load(file)
            
            H_reduced[i, j] = operator_inner_product(temp1, H_mpo, temp2)
    
    return(H_reduced)
    

def generate_reduced_H(vector_list, H):
    H_reduced = np.zeros([len(vector_list), len(vector_list)])
    for i in range (len(vector_list)):
        for j in range (len(vector_list)):
            H_reduced[i, j] = np.vdot(vector_list[i], H@vector_list[j])
    return(H_reduced)



def popcount(x):
    return bin(x).count('1')


def generate_Hamiltonian_with_occupation_number(H, n):
    """
    Efficiently extract a Hamiltonian with a fixed occupation number from a given Hamiltonian H.
    """
    rows, cols = H.nonzero()
    size = H.shape
    H_occu = sparse.csr_matrix((size[0], size[0]), dtype=float)
    
    valid_indices = [(i, j) for i, j in zip(rows, cols) if popcount(i) == n and popcount(j) == n]
    
    if valid_indices:
        valid_rows, valid_cols = zip(*valid_indices)
        H_occu[valid_rows, valid_cols] = H[valid_rows, valid_cols]

    return H_occu


def generate_re_ortho_space_with_coeff(N_use, C, foldername):
    #Input: a list of vectors stored in foldername
    #C: the coeff which tell us how to combine above vectors into orthogonal basis
    # |\psi_n> = \sum C_ni |\phi_i>, where \psi is new orthogonal basis, \phi are 'material' vectors
    #Note: the vectors in this list are all np.array, we will have another MPS version.
    vector_list = []
    
    filename = foldername + f"/Krylov_vec{0}.pkl"
    with open(filename, 'rb') as file:
        shape_test = pickle.load(file)
    L = shape_test.nsites
    
    for i in range (N_use):
        temp2 = np.zeros([2**L], dtype = 'complex128')
        for j in range (N_use):
            filename = foldername + f"/Krylov_vec{j}.pkl"
            with open(filename, 'rb') as file:
                temp1 = pickle.load(file)
            temp2 += C[i][j]* temp1.as_vector()
        
        temp2 /= np.linalg.norm(temp2)
        vector_list.append(temp2)
    return(vector_list)


def coeff_canonical_orthogonalization(N_Krylov, foldername):
    '''  
    Input: original vectors stored in foldername
    output: coeff matrix which can transform original vectors into orthogonal basis
    Advantage: such a ortho method can orthogonalize the vectors very well (better than Gram-Schmidt)
    Disadvantage: need to iteration to get ground state, since the first several ortho basis are more different 
    from original basis, which could be nice approx. to ground state (e.g., Hartree-Fock state).
    '''
    
    W = get_W(N_Krylov, foldername)
    D, U = np.linalg.eigh(W) 
    sqrt_D = np.sqrt(D)
    inverse_sqrt_D = 1 / sqrt_D
    D_invers_sq_root = np.diag(inverse_sqrt_D)
    S_inv_sqrt = U @ D_invers_sq_root @ U.T
    
    return (S_inv_sqrt)


def remain_only_tridiagonal_elements(H):
    # Create a mask for the tridiagonal elements
    tridiag_mask = np.eye(H.shape[0], k=0, dtype=bool) | \
                np.eye(H.shape[0], k=1, dtype=bool) | \
                np.eye(H.shape[0], k=-1, dtype=bool)

    # Apply the mask to retain only tridiagonal elements
    return (np.where(tridiag_mask, H, 0))


def solve_ritz(folder_containing_Krylov, H_reduced, N_subspace, coeff, max_bond, e_ref, mpo_ref):
    ''' 
    Given orthogonal Krylov vectors (represented by coeff), solve ritz vector.
    
    folder_containing_Krylov: foldername which contains Krylov vector
    H_reduced: reduced Hamiltonian calculated by orthogonal basis
    coeff: post-orthogonalization coeff, which linearly combine the Krylov vectors into orthogonal basis 
    max_bond: the max_bond used to represent the ritz vector
    e_ref: target energy
    mpo_ref: conventional mpo. Using conventional MPO to  calculate expectation value doenn't require large memory.

    return: ritz value and ritz vector (only smallest Ritz)
    '''
    for n in range (5, N_subspace+1, 5):
        e, v = np.linalg.eigh(H_reduced[:n, :n])
        C_ritz = ((v[:,0].reshape(n, 1)).transpose(1, 0))@ coeff[:n, :n]
        C_ritz  = C_ritz.reshape(C_ritz.shape[1],)
        
        filename = folder_containing_Krylov + f"/Krylov_vec{0}.pkl"
        with open(filename, 'rb') as file:
            ritz_vec = pickle.load(file)
        ritz_vec.A[0] = C_ritz[0]* ritz_vec.A[0]
        
        for i in range (1, n, 1):
            filename = folder_containing_Krylov + f"/Krylov_vec{i}.pkl"
            with open(filename, 'rb') as file:
                temp = pickle.load(file)
            temp.A[0] = C_ritz[i]* temp.A[0]
            #ritz_vec = add_mps_and_compress_direct_SVD(ritz_vec, temp, 0, 2* max_bond)
            ritz_vec = add_mps_and_compress(ritz_vec, temp, 0, 2* max_bond)
        #using svd to go back to max_bond, and left canonical-form
        ritz_vec.compress_direct_svd_right_max_bond(0, max_bond)
        #orthonormalize
        ritz_vec.orthonormalize('right')
        e_ritz = operator_average(ritz_vec, mpo_ref)
        print(e_ritz-e_ref)
    
    return e_ritz, ritz_vec
            


def solve_ritz_two_vec(folder_containing_Krylov, H_reduced, N_subspace, coeff, max_bond, e1, e2, mpo_ref):
    ''' 
    Given orthogonal Krylov vectors (represented by coeff), solve TWO ritz vectors with smallest values.
    
    folder_containing_Krylov: foldername which contains Krylov vector
    H_reduced: reduced Hamiltonian calculated by orthogonal basis
    coeff: post-orthogonalization coeff, which linearly combine the Krylov vectors into orthogonal basis 
    max_bond: the max_bond used to represent the ritz vector
    e1: ground energy
    e2: 1st_ex energy
    mpo_ref: conventional mpo. Using conventional MPO to  calculate expectation value doenn't require large memory.

    return: ritz value and ritz vector (TWO smallest Ritz values)
    '''
    for n in range (5, N_subspace+1, 5):
        e, v = np.linalg.eigh(H_reduced[:n, :n])
        
        #1st Ritz:
        C_ritz = ((v[:,0].reshape(n, 1)).transpose(1, 0))@ coeff[:n, :n]
        C_ritz  = C_ritz.reshape(C_ritz.shape[1],)
        
        filename = folder_containing_Krylov + f"/Krylov_vec{0}.pkl"
        with open(filename, 'rb') as file:
            ritz_vec = pickle.load(file)
        ritz_vec.A[0] = C_ritz[0]* ritz_vec.A[0]
        
        for i in range (1, n, 1):
            filename = folder_containing_Krylov + f"/Krylov_vec{i}.pkl"
            with open(filename, 'rb') as file:
                temp = pickle.load(file)
            temp.A[0] = C_ritz[i]* temp.A[0]
            #ritz_vec = add_mps_and_compress_direct_SVD(ritz_vec, temp, 0, 2* max_bond)
            ritz_vec = add_mps_and_compress(ritz_vec, temp, 0, 2* max_bond)
        #using svd to go back to max_bond, and left canonical-form
        ritz_vec.compress_direct_svd_right_max_bond(0, max_bond)
        #orthonormalize
        ritz_vec.orthonormalize('right')
        e_ritz = operator_average(ritz_vec, mpo_ref)
        e_ritz = operator_average(ritz_vec, mpo_ref)
        print('ground error', e_ritz - e1)
        
        #2nd Ritz:
        C_ritz2 = ((v[:,1].reshape(n, 1)).transpose(1, 0))@ coeff[:n, :n]
        C_ritz2  = C_ritz2.reshape(C_ritz2.shape[1],)
        
        filename = folder_containing_Krylov + f"/Krylov_vec{0}.pkl"
        with open(filename, 'rb') as file:
            ritz_vec2 = pickle.load(file)
        ritz_vec2.A[0] = C_ritz2[0]* ritz_vec2.A[0]
        
        for i in range (1, n, 1):
            filename2 = folder_containing_Krylov + f"/Krylov_vec{i}.pkl"
            with open(filename2, 'rb') as file:
                temp2 = pickle.load(file)
            temp2.A[0] = C_ritz2[i]* temp2.A[0]
            #ritz_vec = add_mps_and_compress_direct_SVD(ritz_vec, temp, 0, 2* max_bond)
            ritz_vec = add_mps_and_compress(ritz_vec, temp, 0, 2* max_bond)
        #using svd to go back to max_bond, and left canonical-form
        #ritz_vec.compress_direct_svd_right_max_bond(0, max_bond)
        #orthonormalize
        ritz_vec.orthonormalize('right')
        e_ritz2 = operator_average(ritz_vec2, mpo_ref)
        e_ritz2 = operator_average(ritz_vec2, mpo_ref)
        print('first excited error', e_ritz2 - e2)
    
    return [e_ritz, e_ritz2], [ritz_vec, ritz_vec2]
    