# Architecture (seed — R1-F1 expands)

See `docs/CONTRACT_API.md` for frozen public signatures.

Layering:

1. `core` — in-memory relational FE/test database
2. `io` — translators in/out of `FEModel` / `ModalResult` / `FRFResult`
3. `fea` — element library + sparse K/M/C + solvers
4. `dynamics` — FRF, harmonic, substructuring, TDS
5. `correlation` / `pretest` — V&V and test planning
6. `updating` / `optimization` — inverse and design
7. `mpe` / `rbpe` — test identification
8. `script` / `cli` / `gui` — user surfaces

All numeric kernels are vectorized numpy/scipy. Sparse matrices via `scipy.sparse.csr_matrix`. Eigen via ARPACK (`eigsh`) with shift-invert for constrained structures.
