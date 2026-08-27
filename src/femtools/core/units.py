"""Unit systems and conversions.

femtools stores every :class:`~femtools.core.model.FEModel` in a single,
explicit :class:`UnitSystem`.  The default is SI (m, N, kg, s, Hz).  All
conversions are explicit -- unknown unit names raise :class:`UnitError`
instead of silently passing values through.

Two kinds of API are provided:

* free functions ``convert_length`` / ``convert_force`` / ``convert_mass`` /
  ``convert_time`` / ``convert_frequency`` for one-off conversions between
  named units, and
* :class:`UnitSystem`, a frozen dataclass describing the base units of a
  model, with ``si_factor`` / ``to_si`` / ``from_si`` / ``convert`` helpers
  for derived quantities (stress, density, ...).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TypeVar

import numpy as np

__all__ = [
    "UnitError",
    "UnitSystem",
    "convert",
    "convert_length",
    "convert_force",
    "convert_mass",
    "convert_time",
    "convert_frequency",
]

_T = TypeVar("_T", float, np.ndarray)


class UnitError(ValueError):
    """Raised for unknown unit names or inconsistent unit systems."""


# Factors are "SI units per one named unit":  value_si = value * factor.
_LENGTH_TO_M: dict[str, float] = {
    "m": 1.0,
    "mm": 1e-3,
    "cm": 1e-2,
    "km": 1e3,
    "um": 1e-6,
    "in": 0.0254,
    "ft": 0.3048,
}

_FORCE_TO_N: dict[str, float] = {
    "N": 1.0,
    "kN": 1e3,
    "MN": 1e6,
    "mN": 1e-3,
    "dyn": 1e-5,
    "lbf": 4.4482216152605,
    "kgf": 9.80665,
}

_MASS_TO_KG: dict[str, float] = {
    "kg": 1.0,
    "g": 1e-3,
    "mg": 1e-6,
    "t": 1e3,
    "tonne": 1e3,
    "lbm": 0.45359237,
    "slug": 14.59390293720636,
    "slinch": 175.12683524647636,
}

_TIME_TO_S: dict[str, float] = {
    "s": 1.0,
    "ms": 1e-3,
    "us": 1e-6,
    "min": 60.0,
    "h": 3600.0,
}

_FREQ_TO_HZ: dict[str, float] = {
    "Hz": 1.0,
    "kHz": 1e3,
    "MHz": 1e6,
    "rpm": 1.0 / 60.0,
    "cpm": 1.0 / 60.0,
    "rad/s": 1.0 / (2.0 * math.pi),
}

_TABLES: dict[str, dict[str, float]] = {
    "length": _LENGTH_TO_M,
    "force": _FORCE_TO_N,
    "mass": _MASS_TO_KG,
    "time": _TIME_TO_S,
    "frequency": _FREQ_TO_HZ,
}


def _factor(quantity: str, unit: str) -> float:
    try:
        table = _TABLES[quantity]
    except KeyError as exc:
        raise UnitError(
            f"unknown base quantity {quantity!r}; expected one of {sorted(_TABLES)}"
        ) from exc
    try:
        return table[unit]
    except KeyError as exc:
        raise UnitError(
            f"unknown {quantity} unit {unit!r}; expected one of {sorted(table)}"
        ) from exc


def convert(value: _T, quantity: str, from_unit: str, to_unit: str) -> _T:
    """Convert ``value`` of ``quantity`` between two named units.

    Works on scalars and numpy arrays.  Raises :class:`UnitError` for
    unknown quantities or units.
    """
    f_src = _factor(quantity, from_unit)
    f_dst = _factor(quantity, to_unit)
    return value * (f_src / f_dst)


def convert_length(value: _T, from_unit: str, to_unit: str) -> _T:
    """Convert a length (e.g. ``convert_length(25.4, "mm", "in") == 1.0``)."""
    return convert(value, "length", from_unit, to_unit)


def convert_force(value: _T, from_unit: str, to_unit: str) -> _T:
    """Convert a force."""
    return convert(value, "force", from_unit, to_unit)


def convert_mass(value: _T, from_unit: str, to_unit: str) -> _T:
    """Convert a mass."""
    return convert(value, "mass", from_unit, to_unit)


def convert_time(value: _T, from_unit: str, to_unit: str) -> _T:
    """Convert a time."""
    return convert(value, "time", from_unit, to_unit)


def convert_frequency(value: _T, from_unit: str, to_unit: str) -> _T:
    """Convert a frequency (Hz, kHz, rpm, cpm, rad/s)."""
    return convert(value, "frequency", from_unit, to_unit)


# Derived quantities expressed as (length_exp, force_exp, mass_exp, time_exp).
_DERIVED: dict[str, tuple[int, int, int, int]] = {
    "length": (1, 0, 0, 0),
    "area": (2, 0, 0, 0),
    "volume": (3, 0, 0, 0),
    "inertia": (4, 0, 0, 0),  # second moment of area
    "force": (0, 1, 0, 0),
    "moment": (1, 1, 0, 0),
    "stress": (-2, 1, 0, 0),
    "pressure": (-2, 1, 0, 0),
    "mass": (0, 0, 1, 0),
    "density": (-3, 0, 1, 0),
    "time": (0, 0, 0, 1),
    "frequency": (0, 0, 0, -1),
    "velocity": (1, 0, 0, -1),
    "acceleration": (1, 0, 0, -2),
    "stiffness": (-1, 1, 0, 0),  # translational spring, force/length
    "damping": (-1, 1, 0, 1),  # force/(length/time)
}


@dataclass(frozen=True)
class UnitSystem:
    """Base units of a model: length, force, mass, time.

    Frequency is derived from time.  The default is SI.  A system is
    *consistent* when ``1 force-unit == 1 mass-unit * 1 length-unit /
    (1 time-unit)^2`` so that no hidden factors appear in ``F = M a``
    (e.g. SI, or mm / N / tonne / s).
    """

    length: str = "m"
    force: str = "N"
    mass: str = "kg"
    time: str = "s"

    def __post_init__(self) -> None:
        _factor("length", self.length)
        _factor("force", self.force)
        _factor("mass", self.mass)
        _factor("time", self.time)

    # -- constructors ---------------------------------------------------
    @classmethod
    def si(cls) -> UnitSystem:
        """SI: m, N, kg, s."""
        return cls()

    @classmethod
    def mm_n_tonne_s(cls) -> UnitSystem:
        """Consistent millimetre system: mm, N, tonne, s (stress in MPa)."""
        return cls(length="mm", force="N", mass="tonne", time="s")

    @classmethod
    def in_lbf_slinch_s(cls) -> UnitSystem:
        """Consistent US system: in, lbf, slinch (= lbf s^2/in), s."""
        return cls(length="in", force="lbf", mass="slinch", time="s")

    # -- consistency ----------------------------------------------------
    @property
    def is_si(self) -> bool:
        return (self.length, self.force, self.mass, self.time) == ("m", "N", "kg", "s")

    @property
    def is_consistent(self) -> bool:
        """True when F = M L / T^2 holds without hidden factors."""
        f = _factor("force", self.force)
        m = _factor("mass", self.mass)
        length = _factor("length", self.length)
        t = _factor("time", self.time)
        return math.isclose(f, m * length / t**2, rel_tol=1e-9)

    def check_consistent(self) -> None:
        """Raise :class:`UnitError` when the system is dynamically inconsistent."""
        if not self.is_consistent:
            raise UnitError(
                f"unit system {self} is not dynamically consistent "
                "(force != mass * length / time^2); FE matrices built in these "
                "units would mix scales silently"
            )

    # -- factors ----------------------------------------------------------
    def si_factor(self, quantity: str) -> float:
        """SI units per one unit of ``quantity`` in this system.

        ``value_si = value * si_factor(quantity)``.  Supports base and
        derived quantities (stress, density, velocity, stiffness, ...).
        """
        try:
            le, fe, me, te = _DERIVED[quantity]
        except KeyError as exc:
            raise UnitError(
                f"unknown quantity {quantity!r}; expected one of {sorted(_DERIVED)}"
            ) from exc
        return (
            _factor("length", self.length) ** le
            * _factor("force", self.force) ** fe
            * _factor("mass", self.mass) ** me
            * _factor("time", self.time) ** te
        )

    def to_si(self, value: _T, quantity: str) -> _T:
        """Convert ``value`` expressed in this system to SI."""
        return value * self.si_factor(quantity)

    def from_si(self, value: _T, quantity: str) -> _T:
        """Convert an SI ``value`` into this system."""
        return value / self.si_factor(quantity)

    def convert(self, value: _T, quantity: str, to: UnitSystem) -> _T:
        """Convert ``value`` of ``quantity`` from this system into ``to``."""
        return value * (self.si_factor(quantity) / to.si_factor(quantity))

    # -- (de)serialization helpers ---------------------------------------
    def to_dict(self) -> dict[str, str]:
        return {
            "length": self.length,
            "force": self.force,
            "mass": self.mass,
            "time": self.time,
        }

    @classmethod
    def from_dict(cls, d: dict[str, str]) -> UnitSystem:
        return cls(
            length=d.get("length", "m"),
            force=d.get("force", "N"),
            mass=d.get("mass", "kg"),
            time=d.get("time", "s"),
        )
