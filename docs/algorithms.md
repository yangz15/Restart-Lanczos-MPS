# Algorithms and implementation notes

## MTBL naming

**MTBL** stands for **Modified Thick-Block Lanczos**, as introduced in [An accurate Lanczos method for the matrix product state representation](https://doi.org/10.1038/s42005-026-02896-x), *Communications Physics* **9**, 326 (2026).

The method retains Ritz vectors and uses their individual residual directions in a restarted search. This addresses the loss of a shared residual direction under MPS truncation.

The supplied experiments invoke `trl_variance_solver`. This package labels that routine and its examples as MTBL-related while keeping the original callable name. The other residual-enriched routines (`trl_solver`, `trl_plus_r_solver`, and `trl_plus_r_solver_variance`) are related variants. `shift_invert_trl_solver` adds spectral targeting around a shift. `lanczos_lr_py` and `trl_lr_compact` are the plain-Lanczos and conventional thick-restart comparison routines; they are not relabeled MTBL.

This refactor does not assert that every implementation detail matches Algorithm 1 of the paper. The supplied main routine expands from the most recent vector, uses a window of roughly `k` recent vectors for orthogonalization, and tracks convergence by rank slot. Those choices are preserved rather than replaced by a new algorithm.

## Projection and residuals

The basis is generally nonorthogonal after compression. The solvers construct the Gram matrix `W[i, j] = <V_i | V_j>` and projected Hamiltonian `P[i, j] = <V_i | H | V_j>`, and solve `P c = theta W c` using SciPy's generalized Hermitian eigensolver. This requires a sufficiently independent basis and a positive-definite Gram matrix.

Ritz vectors are reconstructed as compressed MPS linear combinations. Residuals are formed approximately using variational MPO application and compressed addition. Consequently, the recorded squared residual can differ from both an exact dense residual and the energy variance of the reconstructed state. Small residuals alone do not establish that all requested states have been found.

## Convergence conventions

| Routine | Meaning of `tol` | History |
| --- | --- | --- |
| `lanczos_lr_py`, `band_lanczos`, `trl_lr_compact` | Compression tolerance | Projected eigenvalue history |
| `trl_plus_r_solver`, `trl_solver` | Compressed residual norm threshold | Absolute errors against `realeig` |
| `trl_plus_r_solver_variance` | Compressed residual norm threshold | Squared compressed residuals |
| `trl_variance_solver` | **Squared** compressed residual threshold for locking | Squared compressed residuals |
| `shift_invert_trl_solver` | Shift-invert residual norm threshold | Physical residual squares and energies |

The original comment for `trl_variance_solver` described a norm threshold, but the executed condition is `variances[j] < tol`. The documentation follows the code; the threshold has not been silently squared or square-rooted.

## Main solver bookkeeping

`initial_lanczos_steps` controls the initial plain-Lanczos run. Its returned list includes a final continuation vector beyond the prefix used by the returned projection matrices. The restart solver uses the full returned list, as in the original code.

Each restart forms a projected problem and records one history row. Expansion has a budget of `active_slots * lanczos_steps` additional Hamiltonian applications, with a target basis dimension of `k * (lanczos_steps + 2)`. Residual construction requires additional applications. Breakdown can shorten expansion. The history does not contain a reliable cumulative MPO-operation count.

Locking is attached to sorted Ritz **slots**, not overlap-tracked eigenstates. Locked slots remain in the projected basis and their diagnostic entries are set to zero without remeasurement. The supplied implementation recomputes the projected Ritz vectors in subsequent cycles, so locking does not guarantee a frozen vector identity inside that basis. Near crossings or degeneracies, this can matter.

The legacy final-output assembly collects locked slots first and then fills from the last Ritz block, potentially repeating a slot under partial locking. Returned results should be checked for completeness, independence, and residual quality before scientific use. This behavior is preserved in this structural refactor and is not exercised by the no-lock regression case.

## Other inherited limitations

- A singular Gram matrix may stop a restart routine. Some legacy variants can fail before their return variables are initialized. These are research routines, not failure-tolerant production eigensolver interfaces.
- `trl_lr_compact` retains the original normalization and basis-padding behavior. It is a comparison implementation, not a replacement for a standard validated TRL library.
- `band_lanczos` initializes real projection matrices even for a complex starting block. Complex-valued band calculations require separate validation before use.
- `solve_ritz_vecs_from_list` uses an intermediate compression cap of `10 * max_bond` and has no final compression to `max_bond`; its returned vectors can exceed the requested nominal cap.
- The variational MPO application backend uses a supplied initial guess's bond dimensions. `max_bond` is not a guarantee that every intermediate or returned tensor has that exact bond size.
- The local inverse solvers mutate their state arguments, normalize the right-hand-side MPS, and do not propagate the iterative solver's convergence status. MINRES support for complex effective operators is not established.
- `norm_test` explicitly constructs dense vectors and scales exponentially with site count; it is a small-system diagnostic only.
- `solve_ritz_vecs_generalized` reads legacy Python pickle files. Use only trusted files generated by a compatible environment. No saved Krylov vectors are included in this package.

These limitations come from the supplied implementation and constrain scientific claims. They are not resolved by reorganizing the code into modules.

## Example conventions

The Hubbard parameters are `U = 8`, `t = 1`, `mu = 0`, with random seed 42. Quantum numbers are disabled, as in the originals. A random product MPS in the full local four-state basis does not enforce half filling. The example notebooks are therefore not claimed to reproduce a fixed-sector benchmark from the paper.

Plots use the recorded restart cycle as the horizontal coordinate and the unscaled squared compressed residual as the vertical coordinate. Locked-slot zeros are masked on the logarithmic scale. The original hard-coded iteration spacing and division by the final energy have been removed from plotting only.
