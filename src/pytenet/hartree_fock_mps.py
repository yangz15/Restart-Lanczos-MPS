import numpy as np
from pytenet.mps import MPS
from pytenet.hamiltonian import encode_quantum_number_pair
import copy


# def create_0_mps(bond_set, dir):
#     ''' 
#     bond_set: bond dims
#     create a MPS with all entries 0
#     physical dimension:
#     0: no electron
#     1: only spin-down
#     2: only spin-up
#     3: spin-up and spin-down
    
#     '''
#     #d = 4
#     D = bond_set
#     #down: 1
#     #up: 2**16
#     qd = np.array([0, 1, 2**16, 2**16 +1])
#     qD = [np.zeros(Di, dtype=int) for Di in D]
#     A = MPS(qd, qD, fill=0.)
#     #A.orthonormalize(mode = dir)
#     return A

# def _encode_quantum_number_pair(qa: int, qb: int):
#     """
#     Encode a pair of quantum numbers into a single quantum number.
#     """
#     return (qa << 16) + qb

def create_0_mps(bond_set):
    ''' 
    bond_set: bond dims
    create a MPS with all entries 0
    physical dimension:
    0: no electron
    1: only spin-down
    2: only spin-up
    3: spin-up and spin-down
    
    '''
    #d = 4
    D = bond_set
    #down: 1
    #up: 2**16
    qN = [0,  1,  1,  2]
    qS = [0, -1,  1,  0]
    qd = [encode_quantum_number_pair(q[0], q[1]) for q in zip(qN, qS)]
    qD = [np.zeros(Di, dtype=int) for Di in D]
    A = MPS(qd, qD, fill=0.)
    #A.orthonormalize(mode = dir)
    return A

# def generate_single_state(L, state):
#     '''  
#     site: (up, down) , (up, down), (up, down), (up, down) ......
#     physical dimension:
#     0: no electron (0, 0)
#     1: only spin-down (0, down)
#     2: only spin-up (up, 0)
#     3: spin-up and spin-down (up, down)
    
#     state: a single computational basis e.g., |3,3,1,1,0,0,0> 
#     L: MPS nsites (number of spatial orbitals when using d = 4 MPS.)
#     state: state vector written as computational basis, written as list.
#     Given a state vector e.g., |3,3,1,1,0,0,0> , generate its mps. state should be given as list, e.g., [3,3,1,1,0,0,0]
#     MPS qnumber direction:
#           |
#           |
#           V
#     --->  O  --->
    
#     qnumber: down: 1 up: 2**16
#     '''
#     assert L == len(state)
#     bond_set_1 = [1] * (L+1)
#     mps_single = create_0_mps(bond_set_1)
    
#     #count electron number:
#     n_electron_up = 0
#     n_electron_down = 0
#     for i in range (L):
#         if state[i] == 1: 
#             n_electron_down += 1
#         if state[i] == 2: 
#             n_electron_up += 1
#         if state[i] == 3: 
#             n_electron_down += 1
#             n_electron_up += 1
    
#     #initial qnumber for first bond
#     n_accumulation = 0
#     mps_single.qD[0] = np.array([n_accumulation])
#     for i in range (L):
#         mps_single.A[i][state[i]] = 1
#         #update qnumber for next site
#         if state[i] == 1: 
#             n_accumulation += 1
#         if state[i] == 2:
#             n_accumulation += 2**16
#         if state[i] == 3:
#             n_accumulation += 1 + 2**16
#         mps_single.qD[i+1] = np.array([n_accumulation])
    
#     least_sixteen_bits, higher_bits = extract_least_sixteen_bits(n_accumulation)
      
#     assert n_accumulation == n_electron_down + n_electron_down* (2**16)
#     assert n_electron_down == least_sixteen_bits
#     assert n_electron_up == higher_bits
    
#     return mps_single

def generate_single_state(L, state):
    '''  
    site: (up, down) , (up, down), (up, down), (up, down) ......
    physical dimension:
    0: no electron (0, 0)
    1: only spin-down (0, down)
    2: only spin-up (up, 0)
    3: spin-up and spin-down (up, down)
    
    state: a single computational basis e.g., |3,3,1,1,0,0,0> 
    L: MPS nsites (number of spatial orbitals when using d = 4 MPS.)
    state: state vector written as computational basis, written as list.
    Given a state vector e.g., |3,3,1,1,0,0,0> , generate its mps. state should be given as list, e.g., [3,3,1,1,0,0,0]
    MPS qnumber direction:
          |
          |
          V
    --->  O  --->
    
    qnumber: down: 1 up: 2**16
    '''
    qN = [0,  1,  1,  2]
    qS = [0, -1,  1,  0]
    qd = [encode_quantum_number_pair(q[0], q[1]) for q in zip(qN, qS)]
    
    assert L == len(state)
    bond_set_1 = [1] * (L+1)
    mps_single = create_0_mps(bond_set_1)
    
    #count electron number:
    n_electron_up = 0
    n_electron_down = 0
    for i in range (L):
        if state[i] == 1: 
            n_electron_down += 1
        if state[i] == 2: 
            n_electron_up += 1
        if state[i] == 3: 
            n_electron_down += 1
            n_electron_up += 1
    
    # n_electron = n_electron_down + n_electron_up
    # n_spin = (n_electron_up - n_electron_down)/2       
    
    #initial qnumber for first bond
    n_accumulation = 0
    mps_single.qD[0] = np.array([n_accumulation])
    for i in range (L):
        mps_single.A[i][state[i]] = 1
        #update qnumber for next site
        if state[i] == 1: #down
            n_accumulation += qd[1]
        if state[i] == 2: #up
            n_accumulation += qd[2]
        if state[i] == 3: #up and down
            n_accumulation += qd[3]
        mps_single.qD[i+1] = np.array([n_accumulation])
    
    # least_sixteen_bits, higher_bits = extract_least_sixteen_bits(n_accumulation)
    
    # if round(least_sixteen_bits/2**16) == 1: #total spin is down spin
    #     assert n_spin == least_sixteen_bits
        
    
    
    # assert n_electron == higher_bits
   
    return mps_single


def extract_least_sixteen_bits(n):
    # Extract the lowest sixteen bits
    least_sixteen_bits = n & 0xFFFF
    # Extract the higher bits
    higher_bits = n >> 16
    return least_sixteen_bits, higher_bits




