# API reference

All reusable functions live in `.py` modules. Solver names from the original notebooks remain available. Import public solver functions with `from restart_lanczos_mps import ...`; import example helpers from `restart_lanczos_mps.examples`. The bundled backend remains accessible as `pytenet`.

See [algorithm notes](algorithms.md) for numerical limitations and solver-specific convergence conventions. Inputs are legacy research interfaces and are not comprehensively validated.

## lanczos

Source: [src/restart_lanczos_mps/lanczos.py](../src/restart_lanczos_mps/lanczos.py).

### lanczos_lr_py

Generate a rank-truncated, nonorthogonal Lanczos basis.

```python
lanczos_lr_py(H, v0_mps, maxiter, tol, rmax)
```

| Parameter | Meaning |
| --- | --- |
| `H` | Hamiltonian as an MPO. |
| `v0_mps` | Normalized initial MPS. |
| `maxiter` | Number of plain-Lanczos iterations; use a positive integer. |
| `tol` | Routine-specific compression or convergence threshold; see the convergence table in algorithms.md. |
| `rmax` | Maximum bond parameter passed to compression operations. |

**Returns:** `(V, P, W, eigenvalue_history)`. `V` normally has one continuation vector beyond the prefix represented by `P` and `W`; that last vector need not be normalized.

### trl_lr_compact

Run the conventional thick-restart comparison method with an in-memory basis.

```python
trl_lr_compact(H, v0, m=30, k=10, nrestart=20, tol=1e-10, rmax=32, epsW=1e-12, orthog_last_to_ritz=False, orthog_new_to_ritz=False)
```

| Parameter | Meaning |
| --- | --- |
| `H` | Hamiltonian as an MPO. |
| `v0` | Normalized initial MPS. |
| `m` | Projected basis size for the conventional thick-restart routine; choose m > k > 0. |
| `k` | Number of requested eigenpairs; the initial basis must support at least this many independent directions. |
| `nrestart` | Maximum conventional thick-restart cycles; positive integer. |
| `tol` | Routine-specific compression or convergence threshold; see the convergence table in algorithms.md. |
| `rmax` | Maximum bond parameter passed to compression operations. |
| `epsW` | Relative floor applied to Gram eigenvalues in conventional TRL. |
| `orthog_last_to_ritz` | Whether to project the continuation vector against retained Ritz vectors. |
| `orthog_new_to_ritz` | Whether to project new vectors against retained Ritz vectors. |

**Returns:** A list of projected eigenvalue arrays (up to five values per restart). No explicit eigenvectors are returned.

### band_lanczos

Generate a band-style basis using a recent-vector orthogonalization window of up to `2*l - 1`.

```python
band_lanczos(H, initial_block, num_new_vecs, tol, rmax, lock_idx=0)
```

| Parameter | Meaning |
| --- | --- |
| `H` | Hamiltonian as an MPO. |
| `initial_block` | Nonempty list of normalized starting MPS objects. |
| `num_new_vecs` | Number of attempted band expansions. |
| `tol` | Routine-specific compression or convergence threshold; see the convergence table in algorithms.md. |
| `rmax` | Maximum bond parameter passed to compression operations. |
| `lock_idx` | Number of initial block slots excluded from seeding; 0 <= lock_idx < len(initial_block). |

**Returns:** `(V, P, W, eigenvalue_history)`. A dependent initial block returns `(V, None, None, [])`. The initial projection uses real arrays; see the complex-arithmetic limitation in the algorithm notes.

## ritz

Source: [src/restart_lanczos_mps/ritz.py](../src/restart_lanczos_mps/ritz.py).

### solve_ritz_vecs_generalized

Reconstruct Ritz vectors from legacy `Krylov_vec{i}.pkl` files.

```python
solve_ritz_vecs_generalized(folder_containing_Krylov, P, W, n, max_bond, num_ritz, mpo_ref)
```

| Parameter | Meaning |
| --- | --- |
| `folder_containing_Krylov` | String path containing trusted Krylov_vec0.pkl, Krylov_vec1.pkl, etc. |
| `P` | Projected Hamiltonian matrix. |
| `W` | Hermitian Gram/overlap matrix. |
| `n` | Number of basis vectors used for the projected solve. |
| `max_bond` | Maximum bond parameter passed to compression operations; backend and Ritz reconstruction limits are documented in algorithms.md. |
| `num_ritz` | Number of Ritz vectors to reconstruct; must not exceed n. |
| `mpo_ref` | Hamiltonian used for expectation-value diagnostics. |

**Returns:** `(ritz_values, ritz_vectors)`, or two empty lists if the generalized solve fails. Final vectors are compressed to `max_bond`. Only load trusted pickle files.

### solve_ritz_vecs_from_list

Reconstruct Ritz vectors from an in-memory basis.

```python
solve_ritz_vecs_from_list(V_basis, P, W, n, max_bond, num_ritz, mpo_ref)
```

| Parameter | Meaning |
| --- | --- |
| `V_basis` | List of basis MPS objects. |
| `P` | Projected Hamiltonian matrix. |
| `W` | Hermitian Gram/overlap matrix. |
| `n` | Number of basis vectors used for the projected solve. |
| `max_bond` | Maximum bond parameter passed to compression operations; backend and Ritz reconstruction limits are documented in algorithms.md. |
| `num_ritz` | Number of Ritz vectors to reconstruct; must not exceed n. |
| `mpo_ref` | Hamiltonian used for expectation-value diagnostics. |

**Returns:** `(ritz_values, ritz_vectors)`, or two empty lists if the generalized solve fails. Intermediate compression uses `10 * max_bond`; there is no final compression to `max_bond` in the supplied code.

### get_S0

Construct a recursive reorthogonalization transform for a Hermitian Gram matrix.

```python
get_S0(W)
```

| Parameter | Meaning |
| --- | --- |
| `W` | Hermitian Gram/overlap matrix. |

**Returns:** A complex square matrix `S`. For a well-conditioned positive-definite `W`, `S.conj().T @ W @ S` is approximately the identity. Dependent directions can produce zero columns.

### get_W1

Build the Hermitian Gram matrix for a basis prefix.

```python
get_W1(N_use, V0)
```

| Parameter | Meaning |
| --- | --- |
| `N_use` | Number of leading basis vectors. |
| `V0` | List of basis MPS objects. |

**Returns:** Complex array of shape `(N_use, N_use)`.

### generate_reduced_H

Build the Hermitian projected Hamiltonian for a basis prefix.

```python
generate_reduced_H(N, V0, H_mpo)
```

| Parameter | Meaning |
| --- | --- |
| `N` | Number of leading basis vectors. |
| `V0` | List of basis MPS objects. |
| `H_mpo` | Hamiltonian as an MPO. |

**Returns:** Complex array of shape `(N, N)`.

### form_linear_combination

Form a compressed linear combination of basis MPS objects.

```python
form_linear_combination(basis, coeffs_vec, max_bond, trunc_tol)
```

| Parameter | Meaning |
| --- | --- |
| `basis` | Nonempty list of basis MPS objects. |
| `coeffs_vec` | One coefficient per basis vector. |
| `max_bond` | Maximum bond parameter passed to compression operations; backend and Ritz reconstruction limits are documented in algorithms.md. |
| `trunc_tol` | Tolerance forwarded to compressed MPS addition. |

**Returns:** A normalized MPS, except that an all-negligible coefficient vector returns a zero MPS. Coefficients smaller than or equal to `1e-14` in magnitude are ignored.

## solvers

Source: [src/restart_lanczos_mps/solvers.py](../src/restart_lanczos_mps/solvers.py).

### trl_plus_r_solver

Run residual-enriched restarts from an initial block, tracking errors against reference energies.

```python
trl_plus_r_solver(H_mpo, k, max_bond, initial_guess_block, realeig, num_restarts=20, lanczos_steps=10, tol=1e-12, trunc_tol=1e-12)
```

| Parameter | Meaning |
| --- | --- |
| `H_mpo` | Hamiltonian as an MPO. |
| `k` | Number of requested eigenpairs; the initial basis must support at least this many independent directions. |
| `max_bond` | Maximum bond parameter passed to compression operations; backend and Ritz reconstruction limits are documented in algorithms.md. |
| `initial_guess_block` | Starting MPS block with enough independent directions for k target states. |
| `realeig` | Reference energy array in ascending order, with at least k entries. |
| `num_restarts` | Maximum restart cycles; positive integer. |
| `lanczos_steps` | Expansion budget per restart. The main variance routine allocates this per active slot; other variants use their documented legacy loop budgets. |
| `tol` | Routine-specific compression or convergence threshold; see the convergence table in algorithms.md. |
| `trunc_tol` | Tolerance forwarded to compressed MPS addition. |

**Returns:** `(eigenvalues, eigenvectors, error_history)`; history has one row per completed cycle and `k` columns.

### trl_plus_r_solver_variance

Run residual-enriched restarts from a block, recording squared residuals.

```python
trl_plus_r_solver_variance(H_mpo, k, max_bond, initial_guess_block, num_restarts, lanczos_steps, tol, trunc_tol)
```

| Parameter | Meaning |
| --- | --- |
| `H_mpo` | Hamiltonian as an MPO. |
| `k` | Number of requested eigenpairs; the initial basis must support at least this many independent directions. |
| `max_bond` | Maximum bond parameter passed to compression operations; backend and Ritz reconstruction limits are documented in algorithms.md. |
| `initial_guess_block` | Starting MPS block with enough independent directions for k target states. |
| `num_restarts` | Maximum restart cycles; positive integer. |
| `lanczos_steps` | Expansion budget per restart. The main variance routine allocates this per active slot; other variants use their documented legacy loop budgets. |
| `tol` | Routine-specific compression or convergence threshold; see the convergence table in algorithms.md. |
| `trunc_tol` | Tolerance forwarded to compressed MPS addition. |

**Returns:** `(eigenvalues, eigenvectors, variance_history)`. Convergence uses residual norms, not the squared diagnostic.

### trl_solver

Initialize from one MPS with plain Lanczos, then run residual-enriched restarts.

```python
trl_solver(H_mpo, k, max_bond, v0, realeig, initial_lanczos_steps=20, num_restarts=20, lanczos_steps=10, tol=1e-12, trunc_tol=1e-12)
```

| Parameter | Meaning |
| --- | --- |
| `H_mpo` | Hamiltonian as an MPO. |
| `k` | Number of requested eigenpairs; the initial basis must support at least this many independent directions. |
| `max_bond` | Maximum bond parameter passed to compression operations; backend and Ritz reconstruction limits are documented in algorithms.md. |
| `v0` | Normalized initial MPS. |
| `realeig` | Reference energy array in ascending order, with at least k entries. |
| `initial_lanczos_steps` | Initial plain-Lanczos iteration count. |
| `num_restarts` | Maximum restart cycles; positive integer. |
| `lanczos_steps` | Expansion budget per restart. The main variance routine allocates this per active slot; other variants use their documented legacy loop budgets. |
| `tol` | Routine-specific compression or convergence threshold; see the convergence table in algorithms.md. |
| `trunc_tol` | Tolerance forwarded to compressed MPS addition. |

**Returns:** `(eigenvalues, eigenvectors, error_history)` with absolute energy errors against `realeig`.

### trl_variance_solver

MTBL (Modified Thick-Block Lanczos) example routine: initialize from one MPS, enrich restarts with residuals, and lock slots using squared residuals.

```python
trl_variance_solver(H_mpo, k, max_bond, v0, initial_lanczos_steps=20, num_restarts=200, lanczos_steps=10, tol=1e-12, trunc_tol=1e-16)
```

| Parameter | Meaning |
| --- | --- |
| `H_mpo` | Hamiltonian as an MPO. |
| `k` | Number of requested eigenpairs; the initial basis must support at least this many independent directions. |
| `max_bond` | Maximum bond parameter passed to compression operations; backend and Ritz reconstruction limits are documented in algorithms.md. |
| `v0` | Normalized initial MPS. |
| `initial_lanczos_steps` | Initial plain-Lanczos iteration count. |
| `num_restarts` | Maximum restart cycles; positive integer. |
| `lanczos_steps` | Expansion budget per restart. The main variance routine allocates this per active slot; other variants use their documented legacy loop budgets. |
| `tol` | Lock when the squared compressed residual is below this threshold. |
| `trunc_tol` | Tolerance forwarded to compressed MPS addition. |

**Returns:** `(eigenvalues, eigenvectors, variance_history)`. History shape is `(completed_cycles, states_in_block)`. Zero entries for previously locked slots are placeholders. Legacy partial-lock assembly can repeat returned slots; see algorithm notes.

## inverse_iteration

Source: [src/restart_lanczos_mps/inverse_iteration.py](../src/restart_lanczos_mps/inverse_iteration.py).

### inverse_iteration

Perform single-site sweeps for a shifted projected linear system.

```python
inverse_iteration(H: MPO, X: MPS, Xk_ref: MPS, sigma: float, numsweeps: int, local_solver: str='gmres', local_maxit: int=3, local_tol: float=1e-10)
```

| Parameter | Meaning |
| --- | --- |
| `H` | Hamiltonian as an MPO. |
| `X` | MPS to update in place (or inspect for diagnostics). |
| `Xk_ref` | Reference/right-hand-side MPS; inverse sweeps normalize it in place. |
| `sigma` | Spectral shift. |
| `numsweeps` | Number of local sweep pairs. |
| `local_solver` | Local solver name, normally gmres; legacy fallback selects minres. |
| `local_maxit` | Maximum iterations forwarded to the local iterative solver. |
| `local_tol` | Local iterative solver relative tolerance. |

**Returns:** The modified `X` MPS. Both `X` and `Xk_ref` are normalized in place; `X` is updated. No convergence history is returned.

### inverse_iteration_twosite

Perform two-site sweeps for a shifted projected linear system.

```python
inverse_iteration_twosite(H: MPO, X: MPS, Xk_ref: MPS, sigma: float, numsweeps: int, tol_split: float=0, local_solver: str='gmres', local_maxit: int=2, local_tol: float=1e-08)
```

| Parameter | Meaning |
| --- | --- |
| `H` | Hamiltonian as an MPO. |
| `X` | MPS to update in place (or inspect for diagnostics). |
| `Xk_ref` | Reference/right-hand-side MPS; inverse sweeps normalize it in place. |
| `sigma` | Spectral shift. |
| `numsweeps` | Number of local sweep pairs. |
| `tol_split` | Two-site SVD splitting tolerance. |
| `local_solver` | Local solver name, normally gmres; legacy fallback selects minres. |
| `local_maxit` | Maximum iterations forwarded to the local iterative solver. |
| `local_tol` | Local iterative solver relative tolerance. |

**Returns:** The modified `X` MPS. Both inputs are normalized in place. Splitting uses `tol_split` without an explicit maximum bond cap.

### norm_test

Measure a relative shifted-system residual using dense vectors.

```python
norm_test(H: MPO, X: MPS, Xk_ref: MPS, sigma: float)
```

| Parameter | Meaning |
| --- | --- |
| `H` | Hamiltonian as an MPO. |
| `X` | MPS to update in place (or inspect for diagnostics). |
| `Xk_ref` | Reference/right-hand-side MPS; inverse sweeps normalize it in place. |
| `sigma` | Spectral shift. |

**Returns:** Float `||(H - sigma I) X - Xk_ref|| / ||Xk_ref||`. Requires a nonzero reference state and memory for dense vectors.

### compute_right_mixed_blocks

Contract right environments between two MPS objects.

```python
compute_right_mixed_blocks(X: MPS, Xk_ref: MPS)
```

| Parameter | Meaning |
| --- | --- |
| `X` | MPS to update in place (or inspect for diagnostics). |
| `Xk_ref` | Reference/right-hand-side MPS; inverse sweeps normalize it in place. |

**Returns:** A list of right contraction matrices, one per site.

### apply_local_mps

Apply left and right overlap environments to a local MPS tensor.

```python
apply_local_mps(L: np.ndarray, R: np.ndarray, A: np.ndarray)
```

| Parameter | Meaning |
| --- | --- |
| `L` | Left overlap environment matrix. |
| `R` | Right overlap environment matrix. |
| `A` | Local MPS tensor. |

**Returns:** A rank-three tensor. `L` and `R` must be matrices; `A` must have physical/left/right axes.

### robust_inverse_iteration

Call the two-site inverse sweep with a copied reference and handle a possible `None` return.

```python
robust_inverse_iteration(H, mps_input, sigma, numsweeps)
```

| Parameter | Meaning |
| --- | --- |
| `H` | Hamiltonian as an MPO. |
| `mps_input` | MPS supplied to the inverse-iteration wrapper; may be mutated. |
| `sigma` | Spectral shift. |
| `numsweeps` | Number of local sweep pairs. |

**Returns:** An MPS. The supplied routine passes `mps_input` as the updated state, so callers must assume it is mutated.

## shift_invert

Source: [src/restart_lanczos_mps/shift_invert.py](../src/restart_lanczos_mps/shift_invert.py).

### shift_invert_trl_solver

Target Ritz values near `sigma` and match states across cycles by overlap.

```python
shift_invert_trl_solver(H_mpo, k, max_bond, initial_guess_block, sigma, num_restarts=50, lanczos_steps=20, tol=1e-12, trunc_tol=1e-12)
```

| Parameter | Meaning |
| --- | --- |
| `H_mpo` | Hamiltonian as an MPO. |
| `k` | Number of requested eigenpairs; the initial basis must support at least this many independent directions. |
| `max_bond` | Maximum bond parameter passed to compression operations; backend and Ritz reconstruction limits are documented in algorithms.md. |
| `initial_guess_block` | Starting MPS block with enough independent directions for k target states. |
| `sigma` | Spectral shift. |
| `num_restarts` | Maximum restart cycles; positive integer. |
| `lanczos_steps` | Expansion budget per restart. The main variance routine allocates this per active slot; other variants use their documented legacy loop budgets. |
| `tol` | Routine-specific compression or convergence threshold; see the convergence table in algorithms.md. |
| `trunc_tol` | Tolerance forwarded to compressed MPS addition. |

**Returns:** `(eigenvalues, eigenvectors, variance_history, energy_history)`. State order follows overlap matching after the first cycle, not necessarily sorted energy order. Convergence uses the shift-invert residual.

## examples

Source: [src/restart_lanczos_mps/examples.py](../src/restart_lanczos_mps/examples.py).

### make_hubbard_problem

Construct the model and deterministic random initial state used by the examples.

```python
make_hubbard_problem(nsites=3, interaction=8.0, hopping=1.0, chemical_potential=0.0, seed=42)
```

| Parameter | Meaning |
| --- | --- |
| `nsites` | Number of Hubbard sites, integer >= 2. |
| `interaction` | On-site interaction U. |
| `hopping` | Hopping amplitude t. |
| `chemical_potential` | Chemical potential mu. |
| `seed` | Seed for NumPy default_rng. |

**Returns:** `(hamiltonian, initial_state)`. Quantum numbers are disabled and half filling is not enforced.

### plot_variance_history

Plot squared compressed residuals against recorded restart cycles.

```python
plot_variance_history(variance_history, *, ax=None)
```

| Parameter | Meaning |
| --- | --- |
| `variance_history` | Two-dimensional array of squared residual diagnostics, cycles by states. |
| `ax` | Optional existing Matplotlib axes. |

**Returns:** A Matplotlib `Axes`. Nonpositive or nonfinite entries are masked; existing axes can be supplied.

## Private local solver

`inverse_iteration._solve_local_projected_system` constructs the effective shifted operator and mixed right-hand side from local/environment tensors, then calls GMRES or MINRES. It is an internal implementation detail, not re-exported from the package root. The solver status is currently not propagated.

## Tensor-network backend

The original `pytenet` source is included because the solvers rely on its modified MPS compression behavior. Its classes and low-level routines retain their in-source documentation. The public entry points most relevant here are `pytenet.mps.MPS`, `pytenet.mpo.MPO`, `pytenet.hamiltonian.fermi_hubbard_mpo`, `pytenet.mps.create_random_mps_4`, and contraction/compression routines in `pytenet.operation`. This refactor does not redesign that backend API.
