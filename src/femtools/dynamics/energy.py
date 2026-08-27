"""Modal energy diagnostics: where a mode keeps its strain and its kinetic energy.

For a mode shape ``phi_r`` the two quadratic forms

    MSE_r = phi_r^T K phi_r / 2      MKE_r = phi_r^T M phi_r / 2

are the strain energy stored and the kinetic energy carried by that mode at unit modal
amplitude. On a **mass-normalised** basis (``Phi^T M Phi = I``, which is what
:func:`femtools.fea.eigen.solve_modes` returns) they collapse to ``omega_r^2 / 2`` and
``1/2``, so the per-mode numbers are only a normalisation check. What the diagnostic is
actually for is the **distribution**: split the same quadratic form element by element and
the fraction of a mode's strain energy carried by each element is the classical ranking
behind MSE-based model updating and damping-treatment placement (Kim & Bartkowicz;
Johnson & Kienholz for the modal-strain-energy damping estimate), while the kinetic
fraction ranks where the mode has mass participation.

The split is exact rather than approximate: the assembled ``K`` is the sum of the element
matrices scattered into the global DOF space, so

    sum_e  phi_e^T k_e phi_e  =  phi^T K phi

with ``phi_e`` the element's rows of the mode. :func:`element_modal_energy` evaluates the
left-hand side and reports the closure against the right-hand side in its ``meta``.

Both entry points are duck-typed the same way as the rest of :mod:`femtools.dynamics`:
they take ``(modes, K)``, ``(modes, M)`` — in either order, see :func:`modal_strain_energy`
— and the matrix argument may equally be an ``AssemblyResult``, a ``ModalResult`` carrying
one, or a model database, in which case the free partition is used and a global-DOF mode
matrix is restricted to it. :mod:`femtools.fea` is imported only inside those branches.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp

from .modal import ModalModel, as_modal
from .system import SystemMatrices, as_system, is_matrix_like

__all__ = [
    "ElementEnergy",
    "element_modal_energy",
    "modal_kinetic_energy",
    "modal_strain_energy",
]

_KINDS = ("strain", "kinetic")


# ---------------------------------------------------------------------------
# argument plumbing
# ---------------------------------------------------------------------------
def _as_shape_matrix(obj: Any) -> np.ndarray:
    """Coerce a plain array of mode shapes to ``(ndof, n_modes)``, keeping complex data."""
    arr = obj.toarray() if sp.issparse(obj) else np.asarray(obj)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"mode shapes must be 1-D or 2-D, got shape {arr.shape}")
    if not np.issubdtype(arr.dtype, np.number):
        raise TypeError(f"mode shapes must be numeric, got dtype {arr.dtype}")
    return arr


def _numeric_array(obj: Any) -> Any | None:
    """``obj`` as a numeric array (sparse ones pass through), or ``None``."""
    if sp.issparse(obj):
        return obj
    if isinstance(obj, np.ndarray):
        return obj if np.issubdtype(obj.dtype, np.number) else None
    if isinstance(obj, (list, tuple)):
        try:
            arr = np.asarray(obj)
        except ValueError:  # ragged, e.g. a (freq_hz, modes) tuple
            return None
        return arr if arr.ndim in (1, 2) and np.issubdtype(arr.dtype, np.number) else None
    return None


def _shape_of(obj: Any) -> tuple[int, int]:
    """Shape of an array-like argument, a 1-D array counting as one column."""
    shape = tuple(np.shape(obj))
    if len(shape) == 1:
        return (int(shape[0]), 1)
    if len(shape) != 2:
        raise ValueError(f"expected a 1-D or 2-D array, got shape {shape}")
    return (int(shape[0]), int(shape[1]))


def _is_symmetric(obj: Any) -> bool:
    """True when the argument is a symmetric matrix (cheap for sparse input)."""
    if sp.issparse(obj):
        gap = abs(obj - obj.T)
        scale = abs(obj).max() if obj.nnz else 0.0
        return bool(gap.nnz == 0 or gap.max() <= 1e-12 * max(float(scale), 1.0))
    arr = np.asarray(obj)
    return bool(arr.ndim == 2 and arr.shape[0] == arr.shape[1] and np.allclose(arr, arr.T))


def _classify(obj: Any) -> str:
    """``"modes"``, ``"array"`` or ``"source"`` (an assembly / model / modal result)."""
    if isinstance(obj, ModalModel):
        return "modes"
    if isinstance(obj, Mapping):
        return "modes" if "modes" in obj else "source"
    if _numeric_array(obj) is not None:
        return "array"
    if hasattr(obj, "modes") and hasattr(obj, "freq_hz"):
        return "modes"
    if isinstance(obj, (list, tuple)) and 2 <= len(obj) <= 3:
        return "modes"
    return "source"


def _split(first: Any, second: Any, matrix_name: str) -> tuple[Any, Any]:
    """Return ``(mode_source, matrix_source)`` from two positional arguments.

    Both orders are accepted because the two arguments are almost always
    distinguishable: only one of them can be a modal object, only one of them can be an
    assembly or a model, and of two plain arrays only one can have a shape that makes the
    triple product ``Phi^T A Phi`` conformable. The remaining case — two square arrays of
    the same size — is decided on symmetry (``K`` and ``M`` are symmetric, a mode matrix
    is not), and if that is inconclusive too the first argument is taken as ``Phi``.
    """
    kinds = (_classify(first), _classify(second))
    if kinds == ("modes", "modes"):
        raise TypeError(
            f"both arguments look like mode shapes; one of them must be {matrix_name} "
            "(a matrix, an assembly or a model)"
        )
    if kinds == ("source", "source"):
        raise TypeError(
            f"neither argument looks like a mode matrix; pass the modes and {matrix_name}"
        )
    if kinds[0] == "modes" or kinds[1] == "source":
        return first, second
    if kinds[1] == "modes" or kinds[0] == "source":
        return second, first
    # A sparse argument is a system matrix: this package never stores modes sparsely.
    if sp.issparse(second) and not sp.issparse(first):
        return first, second
    if sp.issparse(first) and not sp.issparse(second):
        return second, first

    a, b = _shape_of(first), _shape_of(second)
    first_is_phi = b[0] == b[1] and b[0] == a[0]
    first_is_matrix = a[0] == a[1] and a[0] == b[0]
    if first_is_phi and not first_is_matrix:
        return first, second
    if first_is_matrix and not first_is_phi:
        return second, first
    if not (first_is_phi or first_is_matrix):
        raise ValueError(
            f"shapes {a} and {b} do not form a Phi^T {matrix_name} Phi product; "
            f"{matrix_name} must be square with one row per row of Phi"
        )
    a_sym, b_sym = _is_symmetric(first), _is_symmetric(second)
    if b_sym and not a_sym:
        return first, second
    if a_sym and not b_sym:
        return second, first
    return first, second


def _mode_shapes(source: Any, mass_normalize: bool) -> tuple[np.ndarray, ModalModel | None]:
    """Mode shape matrix and, when the source carried one, the modal model behind it."""
    if _classify(source) != "modes":
        return _as_shape_matrix(source), None
    mm = as_modal(source)
    if mass_normalize:
        mm = mm.mass_normalized()
    return _as_shape_matrix(mm.modes), mm


def _resolve(
    first: Any, second: Any, which: str, mass_normalize: bool
) -> tuple[np.ndarray, Any, SystemMatrices | None, ModalModel | None]:
    """Resolve the ``(modes, matrix)`` argument pair of the two public functions."""
    matrix_name = "K" if which == "strain" else "M"
    system: SystemMatrices | None
    if second is None:
        if _classify(first) != "modes":
            raise TypeError(
                f"{matrix_name} is required unless the single argument is a modal result "
                "that carries the assembly it was solved from"
            )
        system = as_system(first)
        mode_source: Any = first
        matrix_source: Any = system
    else:
        mode_source, matrix_source = _split(first, second, matrix_name)
        system = None if is_matrix_like(matrix_source) else as_system(matrix_source)

    matrix = matrix_source if system is None else (system.K if which == "strain" else system.M)
    phi, mm = _mode_shapes(mode_source, mass_normalize)

    n = int(matrix.shape[0])
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{matrix_name} must be square, got shape {matrix.shape}")
    if phi.shape[0] != n:
        n_global = None if system is None else system.n_dof_global
        if system is not None and system.free_dof is not None and phi.shape[0] == n_global:
            phi = phi[system.free_dof, :]
        else:
            raise ValueError(
                f"mode shapes have {phi.shape[0]} rows but {matrix_name} has {n}"
                + (f" ({n_global} before constraints)" if n_global is not None else "")
            )
    return phi, matrix, system, mm


def _half_quadratic_diagonal(A: Any, phi: np.ndarray) -> np.ndarray:  # noqa: N803
    """``diag(Phi^H A Phi) / 2`` without forming the full product."""
    if phi.size == 0:
        return np.zeros(phi.shape[1])
    Aphi = A @ phi
    if sp.issparse(Aphi):
        Aphi = Aphi.toarray()
    Aphi = np.asarray(Aphi)
    if np.iscomplexobj(phi) or np.iscomplexobj(Aphi):
        return 0.5 * np.real(np.einsum("ir,ir->r", np.conj(phi), Aphi))
    return 0.5 * np.einsum("ir,ir->r", phi, Aphi)


# ---------------------------------------------------------------------------
# per-mode energies
# ---------------------------------------------------------------------------
def modal_strain_energy(
    modes: Any = None, K: Any = None, *, mass_normalize: bool = True
) -> np.ndarray:
    """Per-mode modal strain energy ``diag(Phi^T K Phi) / 2``.

    Parameters
    ----------
    modes:
        Mode shapes: a ``(ndof, n_modes)`` array (a single mode may be a 1-D vector), a
        :class:`~femtools.dynamics.modal.ModalModel`, a ``ModalResult`` or anything else
        :func:`~femtools.dynamics.modal.as_modal` accepts. Complex modes are contracted
        as ``Phi^H K Phi`` and the real part is returned.
    K:
        Stiffness matrix, dense or sparse, or an ``AssemblyResult`` / ``ModalResult`` /
        model database to take it from — in which case the free partition ``Kff`` is used
        and modes given over the global DOF space are restricted to it. May be omitted
        when ``modes`` is itself a modal result carrying its assembly.
    mass_normalize:
        Scale a modal source to ``Phi^T M Phi = I`` first, using the ``generalized_mass``
        it declares. On an already normalised basis (the usual case) this is a no-op; it
        never touches a plain array, whose scaling is by definition the caller's.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_modes,)``, in energy units. For a mass-normalised basis the result is
        ``omega_r^2 / 2`` with ``omega_r`` in rad/s, so
        ``sqrt(2 * mse) / (2 pi)`` recovers the natural frequencies in Hz.

    Notes
    -----
    The two arguments may be given in either order: ``modal_strain_energy(Phi, K)`` and
    ``modal_strain_energy(K, Phi)`` both work, because a stiffness matrix and a mode
    matrix are distinguishable by type, shape or symmetry in every case but one (two
    square symmetric arrays of the same size), where the first argument is read as
    ``Phi``. Pass the arguments by keyword if that case is yours.

    Examples
    --------
    ::

        modes = solve_modes(model, n_modes=10)
        mse = modal_strain_energy(modes)            # (10,), = omega^2 / 2
        mke = modal_kinetic_energy(modes)           # (10,), = 1/2
    """
    phi, matrix, _system, _mm = _resolve(modes, K, "strain", mass_normalize)
    return _half_quadratic_diagonal(matrix, phi)


def modal_kinetic_energy(
    modes: Any = None, M: Any = None, *, mass_normalize: bool = True
) -> np.ndarray:
    """Per-mode modal kinetic energy ``diag(Phi^T M Phi) / 2``.

    Arguments and argument-order handling are those of :func:`modal_strain_energy`, with
    the mass matrix in place of the stiffness matrix.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_modes,)``. Exactly ``1/2`` per mode on a mass-normalised basis, which
        makes this the cheapest check that a basis really is normalised the way the rest
        of the package assumes. It is the *generalised* kinetic energy at unit modal
        amplitude; the peak kinetic energy of mode ``r`` oscillating at its own frequency
        is ``omega_r^2`` times this, i.e. equal to :func:`modal_strain_energy` — the
        equipartition that a normal mode satisfies by definition.
    """
    phi, matrix, _system, _mm = _resolve(modes, M, "kinetic", mass_normalize)
    return _half_quadratic_diagonal(matrix, phi)


# ---------------------------------------------------------------------------
# per-element breakdown
# ---------------------------------------------------------------------------
@dataclass
class ElementEnergy:
    """Per-element, per-mode strain and kinetic energy.

    Attributes
    ----------
    strain, kinetic:
        Energy of every element in every mode, shape ``(n_elements, n_modes)``. Column
        ``r`` sums to :func:`modal_strain_energy` / :func:`modal_kinetic_energy` of mode
        ``r`` (see ``meta["closure"]`` for how well, measured).
    element_ids:
        Element identifiers, one per row, in the order the model lists them.
    freq_hz:
        Natural frequencies of the modes when the source carried them, else ``None``.
    """

    strain: np.ndarray
    kinetic: np.ndarray
    element_ids: list[Any]
    freq_hz: np.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_elements(self) -> int:
        """Number of elements in the breakdown."""
        return int(self.strain.shape[0])

    @property
    def n_modes(self) -> int:
        """Number of modes in the breakdown."""
        return int(self.strain.shape[1])

    def _matrix(self, kind: str) -> np.ndarray:
        if kind not in _KINDS:
            raise ValueError(f"unknown energy kind {kind!r}; expected one of {_KINDS}")
        return self.strain if kind == "strain" else self.kinetic

    def total(self, kind: str = "strain") -> np.ndarray:
        """Energy summed over the elements, shape ``(n_modes,)``."""
        return np.asarray(self._matrix(kind).sum(axis=0))

    def fraction(self, kind: str = "strain") -> np.ndarray:
        """Per-mode energy fractions, shape ``(n_elements, n_modes)``, columns summing to 1.

        This is the quantity MSE-based updating and damping-treatment placement rank on.
        A mode with no energy at all (a rigid-body mode, for the strain fraction) gives a
        zero column rather than ``nan``.
        """
        energy = self._matrix(kind)
        total = self.total(kind)
        safe = np.where(np.abs(total) > 0.0, total, 1.0)
        return np.where(np.abs(total)[None, :] > 0.0, energy / safe[None, :], 0.0)

    def ranking(
        self, mode: int = 0, kind: str = "strain", n: int | None = None
    ) -> list[tuple[Any, float]]:
        """``(element_id, fraction)`` for one mode, largest first, truncated to ``n``."""
        frac = self.fraction(kind)[:, int(mode)]
        order = np.argsort(-frac)
        if n is not None:
            order = order[: int(n)]
        return [(self.element_ids[int(i)], float(frac[int(i)])) for i in order]


def element_modal_energy(
    model: Any,
    modes: Any = None,
    *,
    elements: Sequence[Any] | None = None,
    mass_normalize: bool = True,
    assemble: Mapping[str, Any] | None = None,
) -> ElementEnergy:
    """Split the modal strain and kinetic energy element by element.

    Parameters
    ----------
    model:
        Model database. Splitting the quadratic form means rebuilding the element
        matrices, and only the model carries the geometry, materials and properties they
        are built from, so an ``AssemblyResult`` on its own is not enough here.
    modes:
        Mode shapes over the global or the free DOF space, or any modal object.
    elements:
        Restrict the breakdown to these element ids; defaults to every element that the
        assembly actually contributed.
    mass_normalize:
        As in :func:`modal_strain_energy`.
    assemble:
        Keyword arguments for :func:`femtools.fea.assemble.assemble_km`. Pass the *same*
        ones the modes were solved with (``lumped_mass``, ``drill_factor``, ``options``),
        otherwise both the split and the global form it is checked against are those of a
        different model than the modes.

    Returns
    -------
    ElementEnergy

    Notes
    -----
    Two numbers in ``meta`` say whether the answer means anything.

    ``meta["closure"]`` is the largest relative difference between the element sum and
    the global quadratic form, over the modes and over both energies. It should be at
    round-off (measured: ``4e-16`` on a HEX8 cube, ``3e-13`` on a QUAD4 plate whose
    drilling penalty is part of the element stiffness); anything else means the elements
    being split are not the elements that were assembled.

    ``meta["modal_kinetic_energy"]`` is the per-mode ``diag(Phi^T M Phi) / 2`` of this
    assembly, which is ``1/2`` exactly when the modes really are the mass-normalised
    modes of it. That is the number that catches an ``assemble`` mismatch, which the
    closure cannot: solving with ``lumped_mass=True`` and splitting with the default
    consistent mass leaves the closure at ``8e-13`` and drops these to ``0.46, 0.42,
    0.30``.
    """
    from femtools.fea.elements import ModelIndex, element_matrices, element_spec
    from femtools.fea.protocols import get_any, iter_records

    system = as_system(model, assemble=assemble)
    if system.dof_map is None or system.free_dof is None:
        raise TypeError(
            "an element breakdown needs a model database (nodes, elements, properties), "
            f"got {type(model).__name__}"
        )
    source = system.model if system.model is not None else model
    if not (hasattr(source, "nodes") or isinstance(source, Mapping)):
        raise TypeError(
            "an element breakdown needs the model itself, not only its assembled "
            f"matrices; got {type(model).__name__}"
        )

    if modes is None:
        raise TypeError("modes are required for an element energy breakdown")
    phi_free, _matrix, _system, mm = _resolve(modes, system, "strain", mass_normalize)

    n_modes = phi_free.shape[1]
    phi = np.zeros((system.dof_map.n_dof, n_modes), dtype=phi_free.dtype)
    phi[system.free_dof, :] = phi_free

    wanted: set[Any] | None = None
    if elements is not None:
        wanted = set(elements)
    elif system.assembly is not None and system.assembly.element_ids:
        wanted = set(system.assembly.element_ids)

    # Only the options that shape an element matrix carry over; the rest of the
    # assemble_km keywords describe the DOF partition, which is already fixed.
    element_options = {
        key: value
        for key, value in dict(assemble or {}).items()
        if key in ("lumped_mass", "drill_factor", "options")
    }
    index = ModelIndex.build(source)

    strain: list[np.ndarray] = []
    kinetic: list[np.ndarray] = []
    element_ids: list[Any] = []
    records = get_any(source, ("elements", "elems", "element"), None)
    for eid, element in iter_records(records):
        if element is None or (wanted is not None and eid not in wanted):
            continue
        etype = str(get_any(element, ("type", "etype", "element_type", "kind"), "")).upper()
        try:
            element_spec(etype)
        except KeyError:
            continue
        em = element_matrices(source, eid, element, index=index, **element_options)
        gdof = np.fromiter(
            (system.dof_map.index(nid, comp) for nid, comp in em.dofs),
            dtype=int,
            count=len(em.dofs),
        )
        phi_e = phi[gdof, :]
        strain.append(
            _half_quadratic_diagonal(em.k, phi_e)
            if em.k is not None
            else np.zeros(n_modes)
        )
        kinetic.append(
            _half_quadratic_diagonal(em.m, phi_e)
            if em.m is not None
            else np.zeros(n_modes)
        )
        element_ids.append(eid)

    shape = (len(element_ids), n_modes)
    mse = np.array(strain).reshape(shape) if element_ids else np.zeros(shape)
    mke = np.array(kinetic).reshape(shape) if element_ids else np.zeros(shape)

    global_mse = _half_quadratic_diagonal(system.K, phi_free)
    global_mke = _half_quadratic_diagonal(system.M, phi_free)
    closure = max(
        _relative_gap(mse.sum(axis=0), global_mse),
        _relative_gap(mke.sum(axis=0), global_mke),
    )
    return ElementEnergy(
        strain=mse,
        kinetic=mke,
        element_ids=element_ids,
        freq_hz=None if mm is None else np.asarray(mm.freq_hz, dtype=float).copy(),
        meta={
            "n_elements": len(element_ids),
            "n_modes": n_modes,
            "closure": closure,
            "modal_strain_energy": global_mse,
            "modal_kinetic_energy": global_mke,
        },
    )


def _relative_gap(a: np.ndarray, b: np.ndarray) -> float:
    """Largest ``|a - b|`` relative to the scale of ``b``."""
    scale = float(np.max(np.abs(b))) if b.size else 0.0
    if scale <= 0.0:
        return float(np.max(np.abs(a - b))) if a.size else 0.0
    return float(np.max(np.abs(a - b)) / scale)
