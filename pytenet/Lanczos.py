import numpy as np
from pytenet.global_krylov_method import get_W, get_S
import copy
import pickle
from pytenet.operation import vdot, apply_operator_and_compress,apply_operator, operator_average, add_mps_and_compress,operator_inner_product,add_mps
from pytenet.global_krylov_method import ortho_to_previous_two
from .mps import MPS, local_orthonormalize_left_qr, local_orthonormalize_right_qr, merge_mps_tensor_pair, split_mps_tensor
from .mpo import MPO, merge_mpo_tensor_pair
from .operation import (
        contraction_operator_step_right,
        contraction_operator_step_left,
        compute_right_operator_blocks,
        apply_local_hamiltonian)
from .krylov import eigh_krylov
from .qnumber import is_qsparse

__all__ = ['dmrg_singlesite', 'dmrg_twosite','solve_ritz_vecs',"ortho_to_list", "generate_restart_with_ritz", "generate_re_ortho_space", "generate_Krylov_space_TRL", "generate_band_krylov"
           "generate_single_state_spin", "generate_single_state_bose","generate_band_thick_krylov"]

def solve_ritz_vecs(folder_containing_Krylov, H_reduced, n, coeff, max_bond, num_ritz, mpo_ref):
    ''' 
    Given orthogonal Krylov vectors (represented by coeff), solve num_ritz ritz vectors with smallest values.
    folder_containing_Krylov: foldername which contains Krylov vector
    H_reduced: reduced Hamiltonian calculated by orthogonal basis
    coeff: post-orthogonalization coeff, which linearly combine the Krylov vectors into orthogonal basis 
    num_ritz: number of Ritz vectors to be calculated
    max_bond: the max_bond used to represent the ritz vector
    mpo_ref: conventional mpo. Using conventional MPO to  calculate expectation value doenn't require large memory.

    return: ritz value and ritz vector (chosen num of Ritz values)
    '''
    krylov_vec = []
    test_energy = []
    for count in range(num_ritz):
        e,v = np.linalg.eigh(H_reduced[:n, :n])
        C_ritz = ((v[:,count].reshape(n, 1)).transpose(1, 0))@ coeff
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
            ritz_vec = add_mps_and_compress(ritz_vec, temp, 0, 10* max_bond)
            #ritz_vec = add_mps_and_compress(ritz_vec, temp, 0, 2* max_bond)
        #using svd to go back to max_bond, and left canonical-form
        ritz_vec.compress_direct_svd_right_max_bond(0, max_bond)
        #orthonormalize
        ritz_vec.orthonormalize('right')
        ### use e to check the error
        e_ritz = operator_average(ritz_vec, mpo_ref)
        krylov_vec.append(ritz_vec)
        test_energy.append(e_ritz)

  
    
    return  test_energy, krylov_vec


def ortho_to_list(vec_list, this_state, max_bond, trunc_tol_ortho):
    
    #orthoglnolize "this state" to all mpss in the list, in MPS form, using the Gram-Schmidt process.
    
    temp_state = copy.deepcopy(this_state)
    for i in range (len(vec_list) - 1):
        temp = copy.deepcopy(vec_list[i])
        temp.A[0] = -vdot(temp_state, temp)* temp.A[0]
        this_state = add_mps(copy.deepcopy(this_state), temp)
    
    temp = copy.deepcopy(vec_list[len(vec_list) - 1])
    temp.A[0] = -vdot(temp_state, temp)* temp.A[0]
    this_state = add_mps_and_compress(copy.deepcopy(this_state), temp, trunc_tol_ortho, max_bond)
    #temp_state = add_mps(copy.deepcopy(temp_state), temp)
    
    this_state.orthonormalize('right')
    
    return(this_state)


def ortho_to_list_mgs(ritz_list, this_state, max_bond, trunc_tol_ortho):
    # Orthogonalize "this_state" to all previous states in MPS form using Modified Gram–Schmidt.
    # The updated this_state is used to compute each subsequent projection.

    for i, ritz in enumerate(ritz_list):
        # Work with a copy of the current ritz state.
        current_ritz = copy.deepcopy(ritz)
        # Compute the projection coefficient using the updated state.
        coeff = vdot(this_state, current_ritz)
        # Modify the coefficient tensor; here we subtract the projection.
        current_ritz.A[0] = -coeff * current_ritz.A[0]
        
        # For all but the last state, add without compression.
        if i < len(ritz_list) - 1:
            this_state = add_mps(this_state, current_ritz)
        else:
            # For the last one, we add and compress.
            this_state = add_mps_and_compress(this_state, current_ritz, trunc_tol_ortho, max_bond)

    # Finally, reorthonormalize the MPS (here along the 'right' direction).
    this_state.orthonormalize('right')
    return this_state



def generate_restart_with_ritz(ritz_list,N_Krylov, H, max_bond_Krylov, trunc_tol, foldername):  
    '''  
    generate orthogonalized Krylov space, and store in disk.

    ritz_list: the list of selected ritz vectors, calculated from previous Lanczos loop.
    
    N_Krylov: the size of Krylov space.
    
    H: MPO.
    
    max_bond_Krylov: maximum bond dimension for Krylov vectors.

    trunc_tol: truncation tolerance for Krylov vectors.
    
    foldername: one must input a foldername where the Krylov vectors are stored in.
    '''
    
    # store the all selected ritz vectors and the last krylov vector in disk
    k = len(ritz_list)
    for i in range (k):
        filename = foldername + f"/Krylov_vec{i}.pkl"
        with open(filename, 'wb') as file:
            pickle.dump(copy.deepcopy(ritz_list[i]), file)
    
    # apply H to the last Ritz vector, and orthogonalize it to the previous Ritz vectors

    r_re = apply_operator(H, copy.deepcopy(ritz_list[-1]))
    r_re = ortho_to_list(ritz_list, r_re, max_bond_Krylov, trunc_tol)
    filename = foldername + f"/Krylov_vec{k}.pkl"
    with open(filename, 'wb') as file:
        pickle.dump(r_re, file) 
    
    # from now on, all the H v_i should be orthogonalized to the previous two vectors.
    for i in range (k+1, N_Krylov):
        
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
        #this_state = apply_operator(H, copy.deepcopy(orth_state2))
        this_state.orthonormalize('right')
        #print(this_state.bond_dims)
        #orthogonalize "this state H \v_i" against the previous two” and compress the bond dims back 
        this_state = ortho_to_previous_two(orth_state1, orth_state2, this_state, max_bond_Krylov, trunc_tol)
        #this_state = ortho_to_ritz(ritz_list, this_state, max_bond_Krylov, trunc_tol)
        #ritz_list.append(this_state)
        print(this_state.bond_dims)
        # store orthogonalized H \v_i in disk.
        filename = foldername + f"/Krylov_vec{i}.pkl"
        with open(filename, 'wb') as file:
            pickle.dump(this_state, file)



def generate_re_ortho_space(N_use, foldername, max_bond_apply):
    """
    Use the strategy proposed in
      "Lanczos algorithm with Matrix Product States for dynamical correlation functions"
    to improve orthogonality.
    
    This function generates a list of post-orthogonalized Krylov vectors stored as MPS.
    """
    W = get_W(N_use, foldername)
    S = get_S(W)
    vector_list = []
    
    for i in range(N_use):
        # Load the first Krylov MPS vector
        filename = foldername + f"/Krylov_vec{0}.pkl"
        with open(filename, 'rb') as file:
            temp2 = pickle.load(file)
        # Multiply the first MPS by the corresponding coefficient S[i][0]
        temp2.A[0] = temp2.A[0] * S[i][0]
        
        # For j = 1,...,i, load the j-th vector, multiply by S[i][j] and add to temp2
        for j in range(1, i + 1):
            filename = foldername + f"/Krylov_vec{j}.pkl"
            with open(filename, 'rb') as file:
                temp1 = pickle.load(file)
            # Multiply the first tensor of temp1 by the scalar S[i][j]
            temp1.A[0] = temp1.A[0] * S[i][j]
            # Add temp1 into temp2 with compression (summing two MPS objects)
            temp2 = add_mps_and_compress(temp2, temp1, 0, max_bond_apply)
        
        # Re-orthonormalize the resulting MPS (here along the right)
        temp2.orthonormalize('right')
        vector_list.append(temp2)
    
    return vector_list




def generate_Krylov_space_TRL(ritz_list,r0, N_Krylov, H, max_bond_Krylov, trunc_tol, foldername):  
    '''  
    generate orthogonalized Krylov space, and store in disk, using modified TRL/IRLM method.
    
    ritz_list: the list of selected ritz vectors, calculated from previous Lanczos loop.

    r0: the restarted state for TRL.
    
    N_Krylov: the size of Krylov space.
    
    H: MPO.
   
    trunc_tol: truncation tolerance for Krylov vectors.
    
    max_bond_Krylov: maximum bond dimension for Krylov vectors.
    
    foldername: one must input a foldername where the Krylov vectors are stored in.
    '''
    
    # store the all selected ritz vectors and the last krylov vector in disk
    ritz_list.append(r0)
    k = len(ritz_list)
    for i in range (k):
        filename = foldername + f"/Krylov_vec{i}.pkl"
        with open(filename, 'wb') as file:
            pickle.dump(copy.deepcopy(ritz_list[i]), file)
    

    r_re = apply_operator(H, copy.deepcopy(r0))
    #r_re = apply_operator(H, copy.deepcopy(ritz_list[-1]))
    r_re = ortho_to_list_mgs(ritz_list, r_re, max_bond_Krylov, trunc_tol)
    filename = foldername + f"/Krylov_vec{k}.pkl"
    with open(filename, 'wb') as file:
        pickle.dump(r_re, file) 
    ritz_list.append(r_re)

    # from now on, all the H v_i should be orthogonalized to the previous two vectors.
    for i in range (k+1, N_Krylov):
        
        #show how many Krylov vectors are achieved     
        print(i)    
        # if i % 5 == 0:
        #     print("implemented:", i)
        
        #from disk load the previous Krylov vectors.    
        filename = foldername + f"/Krylov_vec{i-1}.pkl"
        with open(filename, 'rb') as file:
            orth_state2 = pickle.load(file)
        
        #first calculate H \v_i
        this_state = apply_operator_and_compress(H, copy.deepcopy(orth_state2), trunc_tol, max_bond_Krylov)
        #this_state = apply_operator(H, copy.deepcopy(orth_state2))
        this_state.orthonormalize('right')
        #print(this_state.bond_dims)
        #orthogonalize "this state H \v_i" against the previous two” and compress the bond dims back 
        this_state = ortho_to_list_mgs(ritz_list, orth_state2, max_bond_Krylov, trunc_tol)
        #this_state = ortho_to_ritz(ritz_list, this_state, max_bond_Krylov, trunc_tol)
        #ritz_list.append(this_state)
        print(this_state.bond_dims)
        # store orthogonalized H \v_i in disk.
        filename = foldername + f"/Krylov_vec{i}.pkl"
        with open(filename, 'wb') as file:
            pickle.dump(this_state, file)
        ritz_list.append(this_state)





def generate_band_thick_krylov(mps_block,error_block,N_Krylov, H, max_bond_Krylov, trunc_tol, foldername):  
    '''  
    generate orthogonalized Krylov space, and store in disk.

    mps_block: the block of selected ritz vectors, calculated from previous Lanczos loop.
    
    N_Krylov: the size of Krylov space.
    
    H: MPO.
    
    max_bond_Krylov: maximum bond dimension for Krylov vectors.

    trunc_tol: truncation tolerance for Krylov vectors.

    foldername: one must input a foldername where the Krylov vectors are stored in.


    '''
    
    # store the all blocked vectors in disk
    k = copy.deepcopy(len(mps_block))
    mps_block = mps_block+ error_block
    l = copy.deepcopy(len(mps_block))
    for j in range (l):
        filename = foldername + f"/Krylov_vec{j}.pkl"
        with open(filename, 'wb') as file:
            pickle.dump(copy.deepcopy(mps_block[j]), file)
    
    for i in range (k,N_Krylov+k):
        
        #show how many Krylov vectors are achieved     
        print(i+l)    
        # if i % 5 == 0:
        #     print("implemented:", i)
        
        #first calculate H \v_i
        #this_state = apply_operator_and_compress(H, copy.deepcopy(ritz_list[i]), trunc_tol, max_bond_Krylov)

        this_state = apply_operator(H, copy.deepcopy(mps_block[i]))
        this_state.orthonormalize('right')

        #if len(mps_block) < 2*l+1:
        #    this_state = ortho_to_list_mgs(mps_block, this_state, max_bond_Krylov, trunc_tol)
        #if len(mps_block) >= 2*l+1:
        #    this_state = ortho_to_list_mgs(mps_block[-2*l-1:], this_state, max_bond_Krylov, trunc_tol)
        if len(mps_block) < 2*k-1:
            this_state = ortho_to_list_mgs(mps_block, this_state, max_bond_Krylov, trunc_tol)
        if len(mps_block) >= 2*k-1:
            this_state = ortho_to_list_mgs(mps_block[-2*k+1:], this_state, max_bond_Krylov, trunc_tol)     

        mps_block.append(this_state)
        print(this_state.bond_dims)
        # store orthogonalized H \v_i in disk.
        filename = foldername + f"/Krylov_vec{i+l}.pkl"
        with open(filename, 'wb') as file:
            pickle.dump(this_state, file)





def generate_band_krylov(mps_block,N_Krylov, H, max_bond_Krylov, trunc_tol, foldername, lock_idx):  
    '''  
    generate orthogonalized Krylov space, and store in disk.

    mps_block: the block of selected ritz vectors, calculated from previous Lanczos loop.
    
    N_Krylov: the size of Krylov space.
    
    H: MPO.
    
    max_bond_Krylov: maximum bond dimension for Krylov vectors.

    trunc_tol: truncation tolerance for Krylov vectors.

    foldername: one must input a foldername where the Krylov vectors are stored in.


    '''
    
    # store the all blocked vectors in disk
    l = copy.deepcopy(len(mps_block))
    for j in range (l):
        filename = foldername + f"/Krylov_vec{j}.pkl"
        with open(filename, 'wb') as file:
            pickle.dump(copy.deepcopy(mps_block[j]), file)
    l0 = l - lock_idx
    for i in range (lock_idx,N_Krylov):
        
        #show how many Krylov vectors are achieved     
        print(i+l)    
        # if i % 5 == 0:
        #     print("implemented:", i)
        
        #first calculate H \v_i
        #this_state = apply_operator_and_compress(H, copy.deepcopy(ritz_list[i]), trunc_tol, max_bond_Krylov)

        this_state = apply_operator(H, copy.deepcopy(mps_block[i]))
        this_state.orthonormalize('right')

        #if len(mps_block) < 2*l+1:
        #    this_state = ortho_to_list_mgs(mps_block, this_state, max_bond_Krylov, trunc_tol)
        #if len(mps_block) >= 2*l+1:
        #    this_state = ortho_to_list_mgs(mps_block[-2*l-1:], this_state, max_bond_Krylov, trunc_tol)
        if len(mps_block) < 2*l0-1:
            this_state = ortho_to_list_mgs(mps_block, this_state, max_bond_Krylov, trunc_tol)
        if len(mps_block) >= 2*l0-1:
            this_state = ortho_to_list_mgs(mps_block[-2*l0+1:], this_state, max_bond_Krylov, trunc_tol)     

        mps_block.append(this_state)
        print(this_state.bond_dims)
        # store orthogonalized H \v_i in disk.
        filename = foldername + f"/Krylov_vec{i+l}.pkl"
        with open(filename, 'wb') as file:
            pickle.dump(this_state, file)





def create_0_mps_spin(bond_set):
    """
    bond_set: list of bond dimensions
    Create an MPS with all entries 0 for a spin-1/2 chain (physical dimension = 2).

    Physical basis mapping:
      0: spin-up   (quantum number: +1)
      1: spin-down (quantum number: -1)

    The resulting physical quantum numbers list is exactly: qd = [1, -1].
    """
    D = bond_set
    # Directly assign the quantum numbers for the two states
    qd = [1, -1]  
    qD = [np.zeros(Di, dtype=int) for Di in D]
    A = MPS(qd, qD, fill=0.)
    return A

def generate_single_state_spin(L, state):
    """
    Generate an MPS for a spin-1/2 chain with L sites.

    state: list of length L containing the computational basis state,
           where each entry is either 0 (spin-up) or 1 (spin-down).

    The MPS carries the following quantum numbers:
      0 -> spin-up:   +1
      1 -> spin-down: -1

    At each site the accumulated quantum number on the bond is updated
    by adding the quantum number corresponding to the site state.
    """
    assert L == len(state), "State vector length must equal L"
    bond_set = [1] * (L + 1)
    mps_single = create_0_mps_spin(bond_set)

    # Initialize the first bond with zero accumulated quantum number.
    q_accumulation = 0
    mps_single.qD[0] = np.array([q_accumulation])
    
    # Process each site
    for i in range(L):
        # Set the tensor entry corresponding to the state at site i
        mps_single.A[i][state[i]] = 1  
        # Update the accumulated quantum number:
        # If state[i] is 0 (spin-up), add +1;
        # if state[i] is 1 (spin-down), add -1.
        if state[i] == 0:
            q_accumulation += 1
        elif state[i] == 1:
            q_accumulation += -1
        mps_single.qD[i + 1] = np.array([q_accumulation])
    
    return mps_single


def create_0_mps_bose(bond_set, d):
    """
    bond_set: list of bond dimensions
    Create an MPS with all entries 0 for a Bose-Hubbard chain (physical dimension = d).

    Physical basis mapping:
      0: 0 bosons
      1: 1 boson
      ...
      d-1: d-1 bosons

    The resulting physical quantum numbers list is exactly: qd = [0, 1, 2, ..., d-1].
    """
    D = bond_set
    # Define the physical quantum numbers for the bosonic states using tolist()
    qd = np.arange(d).tolist()
    # Initialize bond quantum numbers for each bond as zeros
    qD = [np.zeros(Di, dtype=int) for Di in D]
    A = MPS(qd, qD, fill=0.)
    return A

def generate_single_state_bose(L, state, d):
    """
    Generate an MPS for a Bose-Hubbard chain with L sites.

    Args:
        L: number of sites.
        state: list of length L containing the occupation number at each site,
               where each entry is an integer from 0 to d-1.
        d: physical dimension (i.e. maximum local occupancy is d-1).

    The MPS carries the following physical quantum numbers:
      For a site in state n, the quantum number is n.

    The accumulated quantum number on each bond represents the total number of bosons
    up to that site.
    """
    assert L == len(state), "State vector length must equal L"
    bond_set = [1] * (L + 1)
    mps_single = create_0_mps_bose(bond_set, d)

    # Initialize the first bond with zero accumulated boson number.
    boson_accumulation = 0
    mps_single.qD[0] = np.array([boson_accumulation])
    
    # Process each site, setting the appropriate tensor entry and updating the total boson count.
    for i in range(L):
        occ = state[i]  # local occupancy: an integer in 0,1,...,d-1
        mps_single.A[i][occ] = 1  # set the entry corresponding to the occupation number
        boson_accumulation += occ  # update total boson number
        mps_single.qD[i + 1] = np.array([boson_accumulation])
    
    return mps_single


def create_0_mps_spin1(bond_set):
    """
    Create an MPS with all entries 0 for a spin-1 chain (physical dimension = 3).

    Args:
        bond_set (list): List of bond dimensions for the MPS.

    Returns:
        MPS: An MPS object initialized with zeros.

    Physical basis mapping (indices to Sz quantum numbers):
        0: Sz = +1 (quantum number: +1)
        1: Sz =  0 (quantum number:  0)
        2: Sz = -1 (quantum number: -1)

    The resulting physical quantum numbers list is exactly: qd = [1, 0, -1].
    """
    D = bond_set
    # Directly assign the quantum numbers for the three states (Sz=+1, 0, -1)
    qd = [1, 0, -1]
    # Initialize bond quantum numbers as arrays of zeros for each bond
    qD = [np.zeros(Di, dtype=int) for Di in D]
    # Create the MPS object, filling tensors with 0.0
    A = MPS(qd, qD, fill=0.)
    return A

def generate_single_state_spin1(L, state):
    """
    Generate an MPS for a spin-1 chain with L sites, representing a single
    computational basis state (product state).

    Args:
        L (int): The number of sites in the spin chain.
        state (list): A list of length L containing the computational basis state.
                      Each entry must be 0, 1, or 2, corresponding to the
                      physical states (Sz=+1, 0, -1 respectively).

    Returns:
        MPS: An MPS object representing the specified product state.

    The MPS carries the following quantum numbers based on Sz values:
        Index 0 -> Sz=+1: +1
        Index 1 -> Sz= 0:  0
        Index 2 -> Sz=-1: -1

    At each site, the accumulated quantum number on the bond is updated
    by adding the quantum number corresponding to the site's state.
    """
    assert L == len(state), "State vector length must equal L"
    assert all(s in [0, 1, 2] for s in state), "State entries must be 0, 1, or 2"

    # Bond dimension is 1 for a product state
    bond_set = [1] * (L + 1)
    # Create the basic zero-filled MPS structure for spin-1
    mps_single = create_0_mps_spin1(bond_set)

    # Map state index to its quantum number (Sz value)
    qn_map = {0: 1, 1: 0, 2: -1}

    # Initialize the first bond with zero accumulated quantum number.
    q_accumulation = 0
    mps_single.qD[0] = np.array([q_accumulation]) # Bond 0 has dimension 1

    # Process each site to set the state and update quantum numbers
    for i in range(L):
        # Set the tensor entry corresponding to the state at site i to 1.
        # Since bond dimensions are 1, A[i] is effectively 1x3x1.
        # state[i] selects the physical index (0, 1, or 2).
        mps_single.A[i][state[i]] = 1.0

        # Get the quantum number (Sz value) for the state at site i
        site_qn = qn_map[state[i]]

        # Update the accumulated quantum number
        q_accumulation += site_qn

        # Store the new accumulated quantum number on the next bond
        mps_single.qD[i + 1] = np.array([q_accumulation]) # Bond i+1 has dimension 1

    return mps_single

def dmrg_singlesite(H: MPO, psi: MPS, numsweeps: int, numiter_lanczos: int = 25):
    """
    Approximate the ground state MPS by left and right sweeps and local single-site optimizations
    (single-site DMRG algorithm). Virtual bond dimensions of starting state `psi` can only decrease.

    Args:
        H: Hamiltonian as MPO
        psi: initial MPS used for optimization; will be overwritten
        numsweeps: maximum number of left and right sweeps
        numiter_lanczos: number of local Lanczos iterations

    Returns:
        numpy.ndarray: array of approximate ground state energies after each iteration

    Reference:
        Ulrich Schollw"ock
        The density-matrix renormalization group in the age of matrix product states
        Annals of Physics 326, 96-192 (2011)
    """

    # number of lattice sites
    L = H.nsites
    assert L == psi.nsites

    # right-normalize input matrix product state
    psi.orthonormalize(mode='right')

    # left and right operator blocks
    # initialize leftmost block by 1x1x1 identity
    BR = compute_right_operator_blocks(psi, H)
    BL = [None for _ in range(L)]
    BL[0] = np.array([[[1]]], dtype=BR[0].dtype)

    # consistency check
    for i in range(len(BR)):
        assert is_qsparse(BR[i], [psi.qD[i+1], H.qD[i+1], -psi.qD[i+1]]), \
            'sparsity pattern of operator blocks must match quantum numbers'

    en_min = np.zeros(numsweeps)

    # TODO: number of iterations should be determined by tolerance and some convergence measure
    for n in range(numsweeps):
        en = 0

        # sweep from left to right
        for i in range(L - 1):
            en, psi.A[i] = _minimize_local_energy(BL[i], BR[i], H.A[i], psi.A[i], numiter_lanczos)
            # left-orthonormalize current psi.A[i]
            psi.A[i], psi.A[i+1], psi.qD[i+1] = local_orthonormalize_left_qr(
                                        psi.A[i], psi.A[i+1], psi.qd, psi.qD[i:i+2])
            # update the left blocks
            BL[i+1] = contraction_operator_step_left(psi.A[i], psi.A[i], H.A[i], BL[i])

        # sweep from right to left
        for i in reversed(range(1, L)):
            en, psi.A[i] = _minimize_local_energy(BL[i], BR[i], H.A[i], psi.A[i], numiter_lanczos)
            # right-orthonormalize current psi.A[i]
            psi.A[i], psi.A[i-1], psi.qD[i] = local_orthonormalize_right_qr(
                                        psi.A[i], psi.A[i-1], psi.qd, psi.qD[i:i+2])
            # update the right blocks
            BR[i-1] = contraction_operator_step_right(psi.A[i], psi.A[i], H.A[i], BR[i])

        # right-normalize leftmost tensor to ensure that 'psi' is normalized
        psi.A[0], _, psi.qD[0] = local_orthonormalize_right_qr(
                                psi.A[0], np.array([[[1]]]), psi.qd, psi.qD[:2])

        # record energy after each sweep
        en_min[n] = en

    return en_min


def dmrg_twosite(H: MPO, psi: MPS, numsweeps: int, numiter_lanczos: int = 25, tol_split: float = 0):
    """
    Approximate the ground state MPS by left and right sweeps and local two-site optimizations
    (two-site DMRG algorithm). Virtual bond dimensions of starting state `psi` can only decrease.

    Args:
        H: Hamiltonian as MPO
        psi: initial MPS used for optimization; will be overwritten
        numsweeps: maximum number of left and right sweeps
        numiter_lanczos: number of local Lanczos iterations

    Returns:
        numpy.ndarray: array of approximate ground state energies after each iteration

    Reference:
        Ulrich Schollw"ock
        The density-matrix renormalization group in the age of matrix product states
        Annals of Physics 326, 96-192 (2011)
    """

    # number of lattice sites
    L = H.nsites
    assert L == psi.nsites

    # right-normalize input matrix product state
    psi.orthonormalize(mode='right')

    # left and right operator blocks
    # initialize leftmost block by 1x1x1 identity
    BR = compute_right_operator_blocks(psi, H)
    BL = [None for _ in range(L)]
    BL[0] = np.array([[[1]]], dtype=BR[0].dtype)

    # consistency check
    for i in range(len(BR)):
        assert is_qsparse(BR[i], [psi.qD[i+1], H.qD[i+1], -psi.qD[i+1]]), \
            'sparsity pattern of operator blocks must match quantum numbers'

    en_min = np.zeros(numsweeps)

    # TODO: number of iterations should be determined by tolerance and some convergence measure
    for n in range(numsweeps):
        en = 0

        # sweep from left to right
        for i in range(L - 2):
            # merge neighboring tensors
            Am = merge_mps_tensor_pair(psi.A[i], psi.A[i+1])
            Hm = merge_mpo_tensor_pair(H.A[i], H.A[i+1])
            # minimize local energy
            en, Am = _minimize_local_energy(BL[i], BR[i+1], Hm, Am, numiter_lanczos)
            # split Am
            psi.A[i], psi.A[i+1], psi.qD[i+1] = split_mps_tensor(Am, psi.qd, psi.qd, [psi.qD[i], psi.qD[i+2]], 'right', tol=tol_split)
            # update the left blocks
            BL[i+1] = contraction_operator_step_left(psi.A[i], psi.A[i], H.A[i], BL[i])

        # sweep from right to left
        for i in reversed(range(L - 1)):
            # merge neighboring tensors
            Am = merge_mps_tensor_pair(psi.A[i], psi.A[i+1])
            Hm = merge_mpo_tensor_pair(H.A[i], H.A[i+1])
            # minimize local energy
            en, Am = _minimize_local_energy(BL[i], BR[i+1], Hm, Am, numiter_lanczos)
            # split Am
            psi.A[i], psi.A[i+1], psi.qD[i+1] = split_mps_tensor(Am, psi.qd, psi.qd, [psi.qD[i], psi.qD[i+2]], 'left', tol=tol_split)
            # update the right blocks
            BR[i] = contraction_operator_step_right(psi.A[i+1], psi.A[i+1], H.A[i+1], BR[i+1])

        # right-normalize leftmost tensor to ensure that 'psi' is normalized
        psi.A[0], _, psi.qD[0] = local_orthonormalize_right_qr(
                                psi.A[0], np.array([[[1]]]), psi.qd, psi.qD[:2])

        # record energy after each sweep
        en_min[n] = en

    return en_min


def _minimize_local_energy(L, R, W, Astart, numiter: int):
    """
    Minimize single-site local energy by Lanczos iteration.
    """
    w, u_ritz = eigh_krylov(
        lambda x: apply_local_hamiltonian(L, R, W, x.reshape(Astart.shape)).reshape(-1),
            Astart.reshape(-1), numiter, 1)

    Aopt = u_ritz[:, 0].reshape(Astart.shape)

    return w[0], Aopt