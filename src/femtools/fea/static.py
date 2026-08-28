"""Linear static solution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from .assemble import AssemblyResult, assemble_km
from .loads import build_load_vector

__all__ = ["StaticResult", "solve_static", "reactions"]


@dataclass
class StaticResult:
    """Full field results of a linear static analysis."""

    u: np.ndarray
    reactions: np.ndarray
    assembly: AssemblyResult
    load: np.ndarray

    @property
    def displacements(self) -> np.ndarray:
        return self.u

    def node_displacement(self, node_id: Any) -> np.ndarray:
        return self.u[self.assembly.dof_map.node_dofs(node_id)]

    def __array__(self, dtype: Any = None) -> np.ndarray:  # pragma: no cover - numpy protocol
        return np.asarray(self.u, dtype=dtype)


def _reject_framed_rotation(assembly: AssemblyResult, node_id: Any, index: int) -> None:
    """Refuse an enforced rotation on a node solved in a local triad.

    Prescribing one basic-frame rotation of such a node is a constraint across
    all three of its equations, not a value for one of them, so writing it
    straight into the analysis-frame vector would silently mean something else.
    """
    if assembly.frames is None or assembly.frames.is_identity:
        return
    if index % assembly.dofs_per_node < 3 or not assembly.frames.is_framed(node_id):
        return
    raise ValueError(
        f"node {node_id!r} solves its rotations in a local shell triad (its averaged "
        "normal is not a global axis), so a single enforced rotation about a basic axis "
        "is not one degree of freedom there; enforce the translations, or assemble with "
        "nodal_frames=False to work in the basic frame"
    )


def _reject_dependent(assembly: AssemblyResult, node_id: Any, index: int) -> None:
    """Refuse an enforced value on a DOF a rigid body already eliminated.

    Such a DOF has no equation of its own -- it follows its independent node --
    so writing a value into it would be quietly ignored rather than enforced.
    """
    if assembly.mpc is None or index not in set(assembly.mpc_dof.tolist()):
        return
    comp = index % assembly.dofs_per_node + 1
    raise ValueError(
        f"node {node_id!r} component {comp} is dependent on rigid body "
        f"{assembly.mpc.sources.get(index)!r}, so it has no equation to enforce a "
        "displacement in; drive the independent node instead"
    )


def _factorized_solve(A: sp.csr_matrix, b: np.ndarray) -> np.ndarray:
    if A.shape[0] == 0:
        return np.zeros_like(b)
    lu = splu(sp.csc_matrix(A))
    return lu.solve(b)


def solve_static(
    model: Any,
    loads: Any = None,
    *,
    assembly: AssemblyResult | None = None,
    enforced: dict[Any, float] | None = None,
    return_reactions: bool = False,
    full_result: bool = False,
    **assemble_kwargs: Any,
) -> np.ndarray | tuple[np.ndarray, np.ndarray] | StaticResult:
    """Solve ``K u = f`` with single point constraints eliminated.

    Returns the displacement vector over **all** ``6 * n_nodes`` DOFs, with
    enforced displacements written back into the constrained positions.  Loads
    are read in the basic (global) frame; the answer comes back in the
    assembly's analysis frame, which differs from the basic one only at the
    rotations of an obliquely oriented shell node (see
    :class:`~femtools.fea.assemble.AssemblyResult`).  ``assembly.to_basic(u)``
    rotates it back when those rotations are wanted globally.

    Rigid bodies (``model.rbe2``) are honoured through the assembly: loads land
    on the independent nodes as force *and* moment, and the returned field
    carries the motion of the dependent nodes (see :mod:`femtools.fea.mpc`).

    Parameters
    ----------
    model
        Model database (duck typed, see :mod:`femtools.fea.protocols`).
    loads
        Anything accepted by :func:`femtools.fea.loads.build_load_vector`.
        ``None`` uses ``model.loads``.
    assembly
        Reuse a previously assembled system instead of rebuilding it.
    enforced
        Extra enforced displacements ``{(node_id, dof): value}`` merged on top
        of the SPC values taken from the model.  A DOF named here is held at
        its value whether or not the model constrains it, so this is also the
        way to drive a DOF the assembler left free.
    return_reactions
        Also return the reaction vector (non-zero only on constrained DOFs).
    full_result
        Return a :class:`StaticResult` instead of a bare array.
    """
    asm = assembly if assembly is not None else assemble_km(model, **assemble_kwargs)
    dof_map = asm.dof_map
    # Applied loads are given in the basic frame; the equations are written in
    # the analysis frame.  The map is orthogonal, so forces use the same one as
    # displacements.
    # A load on a DOF a rigid body eliminated is carried to the independent
    # node -- with the moment of its offset -- by the same transform that built
    # the constrained stiffness.
    f = asm.constrain_load(asm.from_basic(build_load_vector(loads, dof_map, model=model)))

    u_prescribed = asm.spc_values.copy()
    held = np.zeros(asm.n_dof, dtype=bool)
    if enforced:
        for key, value in enforced.items():
            node_id, comp = key if isinstance(key, tuple) else (key, 0)
            index = dof_map.index(node_id, comp)
            _reject_framed_rotation(asm, node_id, index)
            _reject_dependent(asm, node_id, index)
            u_prescribed[index] = float(value)
            held[index] = True

    free = asm.free_dof
    K = asm.K.tocsr()
    multi = f.ndim == 2

    # An enforced value on a DOF the assembler left in the free set has to be
    # removed from the solved set, not merely carried to the right hand side.
    # Moving its column across is only half of the elimination: leaving its own
    # equilibrium row in place solves the DOF as if it were free and the
    # enforced value is then written over the result, which corrupts the whole
    # field rather than just that one entry.
    solve_dof, Kss = free, asm.Kff
    if held[free].any():
        solve_dof = free[~held[free]]
        Kss = sp.csr_matrix(K[solve_dof][:, solve_dof])

    rhs = f[solve_dof].copy()
    fixed = np.flatnonzero(u_prescribed != 0.0)
    if fixed.size:
        contribution = K[solve_dof][:, fixed] @ u_prescribed[fixed]
        rhs = rhs - (contribution[:, None] if multi else contribution)

    u_solved = _factorized_solve(Kss, rhs)

    u = np.zeros((asm.n_dof, f.shape[1])) if multi else np.zeros(asm.n_dof)
    if multi:
        u[solve_dof, :] = u_solved
        u[fixed, :] = u_prescribed[fixed][:, None]
    else:
        u[solve_dof] = u_solved
        u[fixed] = u_prescribed[fixed]

    # Rigid bodies: fill the motion of the DOFs that were eliminated, so the
    # answer describes the whole structure and not only its independent nodes.
    u = asm.recover_dependent(u)

    if not (return_reactions or full_result):
        return u

    r = reactions(asm, u, f, extra_dof=np.flatnonzero(held))
    if full_result:
        return StaticResult(u=u, reactions=r, assembly=asm, load=f)
    return u, r


def reactions(
    assembly: AssemblyResult,
    u: np.ndarray,
    load: np.ndarray | None = None,
    *,
    extra_dof: np.ndarray | None = None,
) -> np.ndarray:
    """Recover ``K u - f`` on the constrained DOFs (zero elsewhere).

    ``extra_dof`` adds DOFs held by an enforced displacement rather than by the
    model's own constraints; they carry a reaction just the same.  ``u`` and the
    returned reactions are both in the assembly's analysis frame.
    """
    u = np.asarray(u)
    residual = assembly.K @ u
    if load is not None:
        residual = residual - np.asarray(load)
    out = np.zeros_like(residual)
    idx = np.union1d(assembly.spc_dof, assembly.drilling_dof)
    if extra_dof is not None and np.size(extra_dof):
        idx = np.union1d(idx, np.asarray(extra_dof, dtype=int))
    if idx.size:
        out[idx] = residual[idx] if residual.ndim == 1 else residual[idx, :]
    return out
