"""Pytest test suite for ACPF configuration, constants, and ordered enumerations."""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError
from datetime import timedelta
from enum import StrEnum

import polars as pl
import pytest

from perimeter_flux.config import (
    ACPFConfig,
    DEFAULT_MAX_CO2_SENSOR_DIFF_PPM,
    DEFAULT_MAX_INTERP_GAP,
    DEFAULT_MAX_MBLH_REL_CHANGE,
    DEFAULT_MAX_WIND_DIR_CHANGE_DEG,
    DEFAULT_MAX_WIND_SPEED_REL_CHANGE,
    DEFAULT_MIN_HULL_AREA_M2,
    DEFAULT_MIN_HULL_COMPACTNESS,
    DEFAULT_MIN_WIND_CONSTANCY,
    DEFAULT_P_ATM_PA,
    DEFAULT_T_AIR_K,
    DEFAULT_TIME_INTERVAL,
    DRY_AIR_GAS_CONSTANT,
    ERROR_BAR_DTYPE,
    ErrorBar,
    InterpolationMethod,
    KG_PER_S_TO_T_PER_HR,
    MIN_WIND_SPEED_EPS,
    MOLAR_MASS_CO2_KG,
    MOLAR_MASS_DRY_AIR_KG,
    STANDARD_GRAVITY,
    UNIVERSAL_GAS_CONSTANT,
    WGS84_E2,
    WGS84_FLATTENING,
    WGS84_SEMI_MAJOR_M,
    WindMode,
)


class TestConstants:
    """Verify exact physical, geodetic, temporal, and QA default constants."""

    def test_default_time_interval_and_gap(self):
        assert DEFAULT_TIME_INTERVAL == timedelta(hours=1)
        assert DEFAULT_MAX_INTERP_GAP == timedelta(hours=3)
        assert DEFAULT_MAX_INTERP_GAP == 3 * DEFAULT_TIME_INTERVAL

    def test_wgs84_constants(self):
        assert WGS84_SEMI_MAJOR_M == 6_378_137.0
        assert math.isclose(WGS84_FLATTENING, 1.0 / 298.257223563, rel_tol=1e-15)
        expected_e2 = 2.0 * WGS84_FLATTENING - WGS84_FLATTENING**2
        assert math.isclose(WGS84_E2, expected_e2, rel_tol=1e-15)
        assert math.isclose(WGS84_E2, 0.00669437999014132, rel_tol=1e-12)

    def test_atmospheric_and_unit_constants(self):
        assert UNIVERSAL_GAS_CONSTANT == 8.314462618
        assert MOLAR_MASS_CO2_KG == 0.0440095
        assert MOLAR_MASS_DRY_AIR_KG == 0.0289644
        assert math.isclose(
            DRY_AIR_GAS_CONSTANT,
            UNIVERSAL_GAS_CONSTANT / MOLAR_MASS_DRY_AIR_KG,
            rel_tol=1e-15,
        )
        assert math.isclose(DRY_AIR_GAS_CONSTANT, 287.05796, rel_tol=1e-6)
        assert STANDARD_GRAVITY == 9.80665
        assert KG_PER_S_TO_T_PER_HR == 3.6
        assert DEFAULT_T_AIR_K == 293.15
        assert DEFAULT_P_ATM_PA == 101_325.0

    def test_derived_scale_height_and_co2_factor_sanity(self):
        # Cross-check against doc/acpf_math.md Section 5.5 reference numbers
        scale_height_m = DRY_AIR_GAS_CONSTANT * DEFAULT_T_AIR_K / STANDARD_GRAVITY
        assert math.isclose(scale_height_m, 8581.0, rel_tol=1e-3)

        eff_1000m = scale_height_m * (1.0 - math.exp(-1000.0 / scale_height_m))
        assert math.isclose(eff_1000m, 943.9, rel_tol=1e-3)

        kappa_co2 = (
            1e-6 * DEFAULT_P_ATM_PA * MOLAR_MASS_CO2_KG / (UNIVERSAL_GAS_CONSTANT * DEFAULT_T_AIR_K)
        )
        assert math.isclose(kappa_co2, 1.8297e-6, rel_tol=1e-3)

    def test_default_qa_threshold_constants(self):
        assert MIN_WIND_SPEED_EPS == 1e-9
        assert DEFAULT_MIN_HULL_AREA_M2 == 1e6
        assert DEFAULT_MIN_HULL_COMPACTNESS == 0.05
        assert DEFAULT_MAX_CO2_SENSOR_DIFF_PPM == 2.0
        assert DEFAULT_MIN_WIND_CONSTANCY == 0.80
        assert DEFAULT_MAX_WIND_DIR_CHANGE_DEG == 30.0
        assert DEFAULT_MAX_WIND_SPEED_REL_CHANGE == 0.35
        assert DEFAULT_MAX_MBLH_REL_CHANGE == 0.25


class TestModeEnums:
    """Verify InterpolationMethod and WindMode StrEnums."""

    def test_interpolation_method_members(self):
        assert issubclass(InterpolationMethod, StrEnum)
        assert list(InterpolationMethod) == [
            InterpolationMethod.LINEAR,
            InterpolationMethod.PCHIP,
            InterpolationMethod.AKIMA,
        ]
        assert InterpolationMethod.LINEAR == "linear"
        assert InterpolationMethod.PCHIP == "pchip"
        assert InterpolationMethod.AKIMA == "akima"
        assert InterpolationMethod("pchip") is InterpolationMethod.PCHIP

    def test_wind_mode_members(self):
        assert issubclass(WindMode, StrEnum)
        assert list(WindMode) == [
            WindMode.UNIFORM_MEAN,
            WindMode.EXTERNAL,
            WindMode.PER_ASV,
        ]
        assert WindMode.UNIFORM_MEAN == "uniform_mean"
        assert WindMode.EXTERNAL == "external"
        assert WindMode.PER_ASV == "per_asv"
        assert WindMode("uniform_mean") is WindMode.UNIFORM_MEAN


class TestErrorBar:
    """Verify ErrorBar severity ordering in Python and Polars pl.Enum."""

    EXPECTED_ORDER = [
        ErrorBar.STABLE,
        ErrorBar.UNSTABLE_WIND_OR_MBLH,
        ErrorBar.CO2_SENSOR_DISCREPANCY,
        ErrorBar.TWO_ASV_EXTRAPOLATED,
        ErrorBar.INSUFFICIENT_ASVS,
    ]

    def test_members_and_string_values(self):
        assert issubclass(ErrorBar, StrEnum)
        assert list(ErrorBar) == self.EXPECTED_ORDER
        assert [m.value for m in ErrorBar] == [
            "stable",
            "unstable_wind_or_mblh",
            "co2_sensor_discrepancy",
            "two_asv_extrapolated",
            "insufficient_asvs",
        ]
        assert ErrorBar.STABLE == "stable"
        assert hash(ErrorBar.STABLE) == hash("stable")

    def test_python_severity_ordering_overrides_lexicographical(self):
        # Alphabetically "co2_sensor_discrepancy" < "stable", but by severity STABLE < CO2_SENSOR_DISCREPANCY.
        assert ErrorBar.STABLE < ErrorBar.CO2_SENSOR_DISCREPANCY
        assert ErrorBar.CO2_SENSOR_DISCREPANCY > ErrorBar.STABLE

        for i, low in enumerate(self.EXPECTED_ORDER):
            for j, high in enumerate(self.EXPECTED_ORDER):
                assert (low < high) is (i < j)
                assert (low <= high) is (i <= j)
                assert (low > high) is (i > j)
                assert (low >= high) is (i >= j)

    def test_python_sorted_min_max(self):
        shuffled = [
            ErrorBar.TWO_ASV_EXTRAPOLATED,
            ErrorBar.STABLE,
            ErrorBar.INSUFFICIENT_ASVS,
            ErrorBar.CO2_SENSOR_DISCREPANCY,
            ErrorBar.UNSTABLE_WIND_OR_MBLH,
        ]
        assert sorted(shuffled) == self.EXPECTED_ORDER
        assert min(shuffled) is ErrorBar.STABLE
        assert max(shuffled) is ErrorBar.INSUFFICIENT_ASVS
        assert (
            max(
                ErrorBar.STABLE,
                ErrorBar.UNSTABLE_WIND_OR_MBLH,
                ErrorBar.CO2_SENSOR_DISCREPANCY,
            )
            is ErrorBar.CO2_SENSOR_DISCREPANCY
        )

    def test_ordering_against_raw_str_raises_type_error(self):
        # Prevent silent fallback to lexicographical str comparisons.
        with pytest.raises(TypeError):
            _ = ErrorBar.STABLE < "co2_sensor_discrepancy"
        with pytest.raises(TypeError):
            _ = "co2_sensor_discrepancy" < ErrorBar.STABLE
        with pytest.raises(TypeError):
            _ = ErrorBar.STABLE <= "stable"
        with pytest.raises(TypeError):
            _ = ErrorBar.STABLE > "insufficient_asvs"
        with pytest.raises(TypeError):
            _ = ErrorBar.STABLE >= "stable"

    def test_polars_enum_dtype_sorting_and_filtering(self):
        assert ERROR_BAR_DTYPE == pl.Enum(ErrorBar)
        assert ERROR_BAR_DTYPE.categories.to_list() == [m.value for m in self.EXPECTED_ORDER]

        df = pl.DataFrame(
            {
                "hour": [0, 1, 2, 3, 4],
                "error_bar": pl.Series(
                    [
                        ErrorBar.INSUFFICIENT_ASVS,
                        ErrorBar.CO2_SENSOR_DISCREPANCY,
                        ErrorBar.STABLE,
                        ErrorBar.TWO_ASV_EXTRAPOLATED,
                        ErrorBar.UNSTABLE_WIND_OR_MBLH,
                    ],
                    dtype=ERROR_BAR_DTYPE,
                ),
            }
        )

        sorted_values = df.sort("error_bar")["error_bar"].to_list()
        assert sorted_values == [m.value for m in self.EXPECTED_ORDER]

        acceptable = df.filter(pl.col("error_bar") <= ErrorBar.UNSTABLE_WIND_OR_MBLH)
        assert acceptable["error_bar"].to_list() == [
            ErrorBar.STABLE.value,
            ErrorBar.UNSTABLE_WIND_OR_MBLH.value,
        ]

        # Reflected operator with ErrorBar on the left of a Polars expression
        acceptable_reflected = df.filter(ErrorBar.UNSTABLE_WIND_OR_MBLH >= pl.col("error_bar"))
        assert acceptable_reflected["hour"].to_list() == acceptable["hour"].to_list()

        severe = df.filter(pl.col("error_bar") > ErrorBar.CO2_SENSOR_DISCREPANCY)
        assert severe["error_bar"].to_list() == [
            ErrorBar.INSUFFICIENT_ASVS.value,
            ErrorBar.TWO_ASV_EXTRAPOLATED.value,
        ]

    def test_polars_enum_rejects_unknown_category(self):
        with pytest.raises((pl.exceptions.InvalidOperationError, pl.exceptions.ComputeError)):
            pl.Series(["not_a_valid_error_bar"], dtype=ERROR_BAR_DTYPE)


class TestACPFConfig:
    """Verify ACPFConfig defaults, immutability, slots, and validation."""

    def test_default_config_values(self):
        cfg = ACPFConfig()
        assert cfg.time_interval == DEFAULT_TIME_INTERVAL
        assert cfg.max_interp_gap == DEFAULT_MAX_INTERP_GAP
        assert cfg.interpolation_method is InterpolationMethod.LINEAR
        assert cfg.wind_mode is WindMode.UNIFORM_MEAN
        assert cfg.min_hull_area_m2 == DEFAULT_MIN_HULL_AREA_M2
        assert cfg.min_hull_compactness == DEFAULT_MIN_HULL_COMPACTNESS
        assert cfg.max_co2_sensor_diff_ppm == DEFAULT_MAX_CO2_SENSOR_DIFF_PPM
        assert cfg.min_wind_constancy == DEFAULT_MIN_WIND_CONSTANCY
        assert cfg.max_wind_dir_change_deg == DEFAULT_MAX_WIND_DIR_CHANGE_DEG
        assert cfg.max_wind_speed_rel_change == DEFAULT_MAX_WIND_SPEED_REL_CHANGE
        assert cfg.max_mblh_rel_change == DEFAULT_MAX_MBLH_REL_CHANGE
        assert cfg.default_t_air_k == DEFAULT_T_AIR_K
        assert cfg.default_p_atm_pa == DEFAULT_P_ATM_PA

    def test_frozen_and_slots(self):
        cfg = ACPFConfig()
        assert not hasattr(cfg, "__dict__")
        with pytest.raises(FrozenInstanceError):
            cfg.min_hull_area_m2 = 5e5  # type: ignore[misc]

    def test_valid_custom_boundary_values(self):
        cfg = ACPFConfig(
            time_interval=timedelta(minutes=30),
            max_interp_gap=timedelta(minutes=15),  # Valid: raw gap guard tighter than period step
            interpolation_method=InterpolationMethod.PCHIP,
            wind_mode=WindMode.PER_ASV,
            min_hull_area_m2=100.0,
            min_hull_compactness=1.0,
            max_co2_sensor_diff_ppm=0.5,
            min_wind_constancy=0.0,
            max_wind_dir_change_deg=180.0,
            max_wind_speed_rel_change=0.5,
            max_mblh_rel_change=0.5,
            default_t_air_k=298.15,
            default_p_atm_pa=100_000.0,
        )
        assert cfg.time_interval == timedelta(minutes=30)
        assert cfg.max_interp_gap == timedelta(minutes=15)
        assert cfg.interpolation_method is InterpolationMethod.PCHIP
        assert cfg.wind_mode is WindMode.PER_ASV

        # Upper boundary for min_wind_constancy == 1.0
        cfg_strict_constancy = ACPFConfig(min_wind_constancy=1.0)
        assert cfg_strict_constancy.min_wind_constancy == 1.0

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"time_interval": timedelta(0)},
            {"time_interval": timedelta(minutes=-10)},
            {"max_interp_gap": timedelta(0)},
            {"max_interp_gap": timedelta(seconds=-1)},
            {"min_hull_area_m2": 0.0},
            {"min_hull_area_m2": -100.0},
            {"min_hull_area_m2": float("nan")},
            {"min_hull_area_m2": float("inf")},
            {"min_hull_compactness": 0.0},
            {"min_hull_compactness": -0.1},
            {"min_hull_compactness": 1.01},
            {"min_hull_compactness": float("nan")},
            {"max_co2_sensor_diff_ppm": 0.0},
            {"max_co2_sensor_diff_ppm": -1.0},
            {"max_co2_sensor_diff_ppm": float("nan")},
            {"max_co2_sensor_diff_ppm": float("inf")},
            {"min_wind_constancy": -0.01},
            {"min_wind_constancy": 1.01},
            {"min_wind_constancy": float("nan")},
            {"max_wind_dir_change_deg": 0.0},
            {"max_wind_dir_change_deg": -5.0},
            {"max_wind_dir_change_deg": 180.1},
            {"max_wind_dir_change_deg": float("nan")},
            {"max_wind_speed_rel_change": 0.0},
            {"max_wind_speed_rel_change": -0.1},
            {"max_wind_speed_rel_change": float("inf")},
            {"max_mblh_rel_change": 0.0},
            {"max_mblh_rel_change": -0.2},
            {"max_mblh_rel_change": float("nan")},
            {"default_t_air_k": 0.0},
            {"default_t_air_k": -273.15},
            {"default_t_air_k": float("nan")},
            {"default_p_atm_pa": 0.0},
            {"default_p_atm_pa": -1000.0},
            {"default_p_atm_pa": float("inf")},
        ],
    )
    def test_invalid_numeric_or_interval_bounds_raise_value_error(self, kwargs):
        with pytest.raises(ValueError):
            ACPFConfig(**kwargs)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"time_interval": 3600},
            {"max_interp_gap": 10800},
            {"interpolation_method": "linear"},
            {"wind_mode": "uniform_mean"},
        ],
    )
    def test_invalid_types_raise_type_error(self, kwargs):
        with pytest.raises(TypeError):
            ACPFConfig(**kwargs)
