"""Configuration, physical constants, and ordered enumerations for ACPF flux computation.

Mirrors the symbol glossary, physical constants, default QA thresholds, and ordered
ErrorBar severity scale defined in ``doc/acpf_math.md`` (Sections 2 and 6.3).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import timedelta
from enum import StrEnum

import polars as pl

# ---------------------------------------------------------------------------
# Temporal Defaults (doc/acpf_math.md Section 2.1)
# ---------------------------------------------------------------------------
# Δt          : duration of one discrete aggregation period (default 1 hr = 3600 s)
# Δt_gap,max  : maximum raw telemetry gap bridged by interpolation (default 3 Δt)
DEFAULT_TIME_INTERVAL: timedelta = timedelta(hours=1)
DEFAULT_MAX_INTERP_GAP: timedelta = 3 * DEFAULT_TIME_INTERVAL

# ---------------------------------------------------------------------------
# WGS 84 Ellipsoid Constants (doc/acpf_math.md Section 2.2)
# ---------------------------------------------------------------------------
# a_wgs       : exact WGS 84 equatorial radius [m]
# f_wgs       : exact WGS 84 flattening (1 / 298.257223563) [1]
# e_wgs^2     : first eccentricity squared (2 f_wgs - f_wgs^2) [1]
WGS84_SEMI_MAJOR_M: float = 6_378_137.0
WGS84_FLATTENING: float = 1.0 / 298.257223563
WGS84_E2: float = 2.0 * WGS84_FLATTENING - WGS84_FLATTENING**2

# ---------------------------------------------------------------------------
# Atmospheric Thermodynamics & Unit Conversion Constants (doc/acpf_math.md Section 2.4)
# ---------------------------------------------------------------------------
# R_gas       : CODATA 2018/2022 exact molar gas constant [J mol^-1 K^-1]
# M_CO2       : IUPAC/NIST molar mass of CO2 [kg mol^-1]
# M_air       : U.S. Standard Atmosphere 1976 dry-air molar mass [kg mol^-1]
# R_d         : specific gas constant for dry air (R_gas / M_air) [J kg^-1 K^-1]
# g_0         : exact standard gravity [m s^-2]
# γ_unit      : exact conversion from kg s^-1 to tCO2 hr^-1 (3600 / 1000)
# T_air, P_atm: default surface air temperature (20 °C) [K] and pressure [Pa]
UNIVERSAL_GAS_CONSTANT: float = 8.314462618
MOLAR_MASS_CO2_KG: float = 0.0440095
MOLAR_MASS_DRY_AIR_KG: float = 0.0289644
DRY_AIR_GAS_CONSTANT: float = UNIVERSAL_GAS_CONSTANT / MOLAR_MASS_DRY_AIR_KG
STANDARD_GRAVITY: float = 9.80665
KG_PER_S_TO_T_PER_HR: float = 3.6
DEFAULT_T_AIR_K: float = 293.15
DEFAULT_P_ATM_PA: float = 101_325.0

# ---------------------------------------------------------------------------
# Numerical Guards & Default QA Thresholds (doc/acpf_math.md Section 2.5)
# ---------------------------------------------------------------------------
# ε_w         : zero-wind speed guard for constancy denominator [m s^-1]
# A_min       : minimum enclosing convex-hull area [m^2] (1 km^2)
# Q_iso,min   : minimum Polsby-Popper compactness 4π A / L^2 [1]
# δ_CO2,max   : dual-sensor CO2 discrepancy warning threshold [ppm]
# r_const,min : minimum intra-period wind vector constancy [1]
# Δθ_w,max    : maximum inter-period wind direction change [deg]
# η_w,max     : maximum relative inter-period wind speed change [1]
# η_H,max     : maximum relative inter-period MBLH change [1]
MIN_WIND_SPEED_EPS: float = 1e-9
DEFAULT_MIN_HULL_AREA_M2: float = 1e6
DEFAULT_MIN_HULL_COMPACTNESS: float = 0.05
DEFAULT_MAX_CO2_SENSOR_DIFF_PPM: float = 2.0
DEFAULT_MIN_WIND_CONSTANCY: float = 0.80
DEFAULT_MAX_WIND_DIR_CHANGE_DEG: float = 30.0
DEFAULT_MAX_WIND_SPEED_REL_CHANGE: float = 0.35
DEFAULT_MAX_MBLH_REL_CHANGE: float = 0.25


class InterpolationMethod(StrEnum):
    """Pluggable Stage 1 per-ASV temporal interpolation method."""

    LINEAR = "linear"
    PCHIP = "pchip"
    AKIMA = "akima"


class WindMode(StrEnum):
    """Horizontal wind field model applied along convex-hull perimeter edges."""

    UNIFORM_MEAN = "uniform_mean"
    EXTERNAL = "external"
    PER_ASV = "per_asv"


class ErrorBar(StrEnum):
    """Ordered uncertainty / QA classification for each ACPF aggregation period.

    Members are ordered from lowest (0) to highest (4) severity:
    ``STABLE < UNSTABLE_WIND_OR_MBLH < CO2_SENSOR_DISCREPANCY < TWO_ASV_EXTRAPOLATED < INSUFFICIENT_ASVS``.

    Relational comparisons (``<``, ``<=``, ``>``, ``>=``) compare by declared
    severity rank rather than lexicographical string order, matching the Polars
    ``ERROR_BAR_DTYPE = pl.Enum(ErrorBar)`` category ordering.
    """

    STABLE = "stable"
    UNSTABLE_WIND_OR_MBLH = "unstable_wind_or_mblh"
    CO2_SENSOR_DISCREPANCY = "co2_sensor_discrepancy"
    TWO_ASV_EXTRAPOLATED = "two_asv_extrapolated"
    INSUFFICIENT_ASVS = "insufficient_asvs"

    def _compare_rank(self, other: object) -> tuple[int, int] | None:
        if isinstance(other, ErrorBar):
            return _ERROR_BAR_RANK[self], _ERROR_BAR_RANK[other]
        if isinstance(other, str):
            raise TypeError(
                f"Ordering comparison not supported between 'ErrorBar' and "
                f"'{type(other).__name__}'; convert strings with ErrorBar(value)."
            )
        return None

    def __lt__(self, other: object) -> bool:
        ranks = self._compare_rank(other)
        return NotImplemented if ranks is None else ranks[0] < ranks[1]

    def __le__(self, other: object) -> bool:
        ranks = self._compare_rank(other)
        return NotImplemented if ranks is None else ranks[0] <= ranks[1]

    def __gt__(self, other: object) -> bool:
        ranks = self._compare_rank(other)
        return NotImplemented if ranks is None else ranks[0] > ranks[1]

    def __ge__(self, other: object) -> bool:
        ranks = self._compare_rank(other)
        return NotImplemented if ranks is None else ranks[0] >= ranks[1]


_ERROR_BAR_RANK: dict[ErrorBar, int] = {member: idx for idx, member in enumerate(ErrorBar)}
ERROR_BAR_DTYPE: pl.Enum = pl.Enum(ErrorBar)


def _validate_positive_finite(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and > 0, got {value!r}.")


def _validate_bounded_finite(
    name: str,
    value: float,
    *,
    low: float,
    high: float,
    low_inclusive: bool,
    high_inclusive: bool = True,
) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}.")
    above_low = value >= low if low_inclusive else value > low
    below_high = value <= high if high_inclusive else value < high
    if not (above_low and below_high):
        left = "[" if low_inclusive else "("
        right = "]" if high_inclusive else ")"
        raise ValueError(f"{name} must be in {left}{low}, {high}{right}, got {value!r}.")


@dataclass(frozen=True, slots=True)
class ACPFConfig:
    """Immutable configuration and QA thresholds for the ACPF computation pipeline."""

    time_interval: timedelta = DEFAULT_TIME_INTERVAL
    max_interp_gap: timedelta = DEFAULT_MAX_INTERP_GAP
    interpolation_method: InterpolationMethod = InterpolationMethod.LINEAR
    wind_mode: WindMode = WindMode.UNIFORM_MEAN
    min_hull_area_m2: float = DEFAULT_MIN_HULL_AREA_M2
    min_hull_compactness: float = DEFAULT_MIN_HULL_COMPACTNESS
    max_co2_sensor_diff_ppm: float = DEFAULT_MAX_CO2_SENSOR_DIFF_PPM
    min_wind_constancy: float = DEFAULT_MIN_WIND_CONSTANCY
    max_wind_dir_change_deg: float = DEFAULT_MAX_WIND_DIR_CHANGE_DEG
    max_wind_speed_rel_change: float = DEFAULT_MAX_WIND_SPEED_REL_CHANGE
    max_mblh_rel_change: float = DEFAULT_MAX_MBLH_REL_CHANGE
    default_t_air_k: float = DEFAULT_T_AIR_K
    default_p_atm_pa: float = DEFAULT_P_ATM_PA

    def __post_init__(self) -> None:
        if not isinstance(self.time_interval, timedelta):
            raise TypeError(
                f"time_interval must be a timedelta, got {type(self.time_interval).__name__}."
            )
        if self.time_interval <= timedelta(0):
            raise ValueError(f"time_interval must be > 0, got {self.time_interval!r}.")

        if not isinstance(self.max_interp_gap, timedelta):
            raise TypeError(
                f"max_interp_gap must be a timedelta, got {type(self.max_interp_gap).__name__}."
            )
        if self.max_interp_gap <= timedelta(0):
            raise ValueError(f"max_interp_gap must be > 0, got {self.max_interp_gap!r}.")

        if not isinstance(self.interpolation_method, InterpolationMethod):
            raise TypeError(
                "interpolation_method must be an InterpolationMethod enum member, "
                f"got {type(self.interpolation_method).__name__}."
            )
        if not isinstance(self.wind_mode, WindMode):
            raise TypeError(
                f"wind_mode must be a WindMode enum member, got {type(self.wind_mode).__name__}."
            )

        _validate_positive_finite("min_hull_area_m2", self.min_hull_area_m2)
        _validate_bounded_finite(
            "min_hull_compactness",
            self.min_hull_compactness,
            low=0.0,
            high=1.0,
            low_inclusive=False,
        )
        _validate_positive_finite("max_co2_sensor_diff_ppm", self.max_co2_sensor_diff_ppm)
        _validate_bounded_finite(
            "min_wind_constancy",
            self.min_wind_constancy,
            low=0.0,
            high=1.0,
            low_inclusive=True,
        )
        _validate_bounded_finite(
            "max_wind_dir_change_deg",
            self.max_wind_dir_change_deg,
            low=0.0,
            high=180.0,
            low_inclusive=False,
        )
        _validate_positive_finite("max_wind_speed_rel_change", self.max_wind_speed_rel_change)
        _validate_positive_finite("max_mblh_rel_change", self.max_mblh_rel_change)
        _validate_positive_finite("default_t_air_k", self.default_t_air_k)
        _validate_positive_finite("default_p_atm_pa", self.default_p_atm_pa)
