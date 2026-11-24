"""Pytest test suite for Stage 1 asynchronous telemetry alignment and resampling."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from perimeter_flux.config import (
    WGS84_E2,
    WGS84_SEMI_MAJOR_M,
    ACPFConfig,
    InterpolationMethod,
)
from perimeter_flux.resample import (
    build_interpolator,
    interval_average_channel,
    resample_asv_telemetry,
    resample_met_series,
)


def _utc_hours(count: int, start: datetime | None = None) -> pl.Series:
    base = start or datetime(2026, 6, 1, 0, 0, tzinfo=UTC)
    return pl.Series(
        "period_start",
        [base + timedelta(hours=h) for h in range(count)],
        dtype=pl.Datetime("us", "UTC"),
    )


class TestBuildInterpolator:
    """Verify pluggable SciPy interpolator construction, fallback, and validation."""

    @pytest.mark.parametrize("method", list(InterpolationMethod))
    def test_linear_ramp_exact_across_all_methods(self, method: InterpolationMethod):
        t = np.array([0.0, 500.0, 1700.0, 3100.0, 5000.0, 7200.0])
        y = 412.5 + 0.002 * t
        interp = build_interpolator(t, y, method)
        integral = float(interp.integrate(1000.0, 4600.0))
        expected = (4600.0 - 1000.0) * (412.5 + 0.002 * 2800.0)
        assert math.isclose(integral, expected, rel_tol=1e-12, abs_tol=1e-12)

    @pytest.mark.parametrize("n_pts", [2, 3, 4])
    def test_akima_graceful_fallback_for_few_points(self, n_pts: int):
        t = np.linspace(0.0, 3600.0, n_pts)
        y = 10.0 + 0.01 * t
        interp = build_interpolator(t, y, InterpolationMethod.AKIMA)
        avg = float(interp.integrate(0.0, 3600.0)) / 3600.0
        assert math.isclose(avg, 10.0 + 0.01 * 1800.0, rel_tol=1e-12)

    def test_invalid_inputs_raise_value_error(self):
        with pytest.raises(ValueError, match="at least 2"):
            build_interpolator([0.0], [1.0], InterpolationMethod.LINEAR)
        with pytest.raises(ValueError, match="same 1D shape"):
            build_interpolator([0.0, 1.0], [1.0, 2.0, 3.0], InterpolationMethod.LINEAR)
        with pytest.raises(ValueError, match="strictly increasing"):
            build_interpolator([0.0, 1.0, 1.0], [1.0, 2.0, 3.0], InterpolationMethod.PCHIP)
        with pytest.raises(ValueError, match="finite"):
            build_interpolator([0.0, 1.0, 2.0], [1.0, float("nan"), 3.0], InterpolationMethod.LINEAR)


class TestIntervalAverageChannel:
    """Verify time-weighted interval averages, PCHIP properties, duplicates, and guards."""

    @pytest.mark.parametrize("method", list(InterpolationMethod))
    def test_reproduces_constant_and_linear_ramp_on_irregular_timestamps(
        self, method: InterpolationMethod
    ):
        # Irregular asynchronous timestamps spanning [0, 4 hr]
        t_obs = np.array([0.0, 310.0, 1450.0, 2900.0, 4120.0, 6800.0, 9500.0, 12100.0, 14400.0])
        period_starts = np.array([0.0, 3600.0, 7200.0, 10800.0])
        dt = 3600.0

        # Constant signal
        const_vals = np.full_like(t_obs, 418.75)
        avg_const = interval_average_channel(
            t_obs, const_vals, period_starts, dt, method=method, max_gap_sec=3 * dt
        )
        np.testing.assert_allclose(avg_const, 418.75, rtol=0.0, atol=1e-12)

        # Linear ramp q(t) = a + b*t -> exact period mean a + b*(t_k + dt/2)
        a, b = 410.0, -3.5e-4
        ramp_vals = a + b * t_obs
        avg_ramp = interval_average_channel(
            t_obs, ramp_vals, period_starts, dt, method=method, max_gap_sec=3 * dt
        )
        expected_ramp = a + b * (period_starts + 0.5 * dt)
        np.testing.assert_allclose(avg_ramp, expected_ramp, rtol=1e-11, atol=1e-11)

    def test_pchip_zero_overshoot_and_total_integral_conservation(self):
        # Step jump from 410 to 420 ppm across irregular points
        t_obs = np.array([0.0, 1800.0, 3600.0, 5400.0, 7200.0, 9000.0, 10800.0, 14400.0])
        vals = np.array([410.0, 410.0, 410.0, 420.0, 420.0, 420.0, 420.0, 420.0])
        period_starts = np.array([0.0, 3600.0, 7200.0, 10800.0])
        dt = 3600.0

        avgs = interval_average_channel(
            t_obs, vals, period_starts, dt, method=InterpolationMethod.PCHIP, max_gap_sec=3 * dt
        )
        # Zero overshoot: every hourly average stays strictly inside [410.0, 420.0]
        assert np.all(avgs >= 410.0 - 1e-12)
        assert np.all(avgs <= 420.0 + 1e-12)

        # Total integral conservation across contiguous periods
        interp = build_interpolator(t_obs, vals, InterpolationMethod.PCHIP)
        total_from_periods = float(np.sum(avgs) * dt)
        total_direct = float(interp.integrate(0.0, 14400.0))
        assert math.isclose(total_from_periods, total_direct, rel_tol=1e-12)

    def test_duplicate_and_unsorted_timestamps_are_averaged(self):
        # Duplicate timestamps at t=0 and t=3600 (out of order)
        t_obs = np.array([3600.0, 0.0, 7200.0, 0.0, 3600.0])
        vals = np.array([422.0, 408.0, 430.0, 412.0, 418.0])
        # Averaged unique points: (0 -> 410), (3600 -> 420), (7200 -> 430) => exact linear ramp
        period_starts = np.array([0.0, 3600.0])
        avgs = interval_average_channel(
            t_obs, vals, period_starts, 3600.0, method=InterpolationMethod.LINEAR, max_gap_sec=10800.0
        )
        np.testing.assert_allclose(avgs, [415.0, 425.0], rtol=0.0, atol=1e-12)

    def test_strict_no_extrapolation_and_max_gap_guard(self):
        # Gap of 5 hours between t=2 hr and t=7 hr (exceeds max_gap = 3 hr)
        t_obs = np.array([3600.0, 7200.0, 25200.0, 28800.0])
        vals = np.array([10.0, 20.0, 30.0, 40.0])
        # Periods: [0,1) pre-span, [1,2) valid, [2,3)..[6,7) bad gap, [7,8) valid, [8,9) post-span
        period_starts = np.arange(9, dtype=np.float64) * 3600.0
        avgs = interval_average_channel(
            t_obs, vals, period_starts, 3600.0, method=InterpolationMethod.LINEAR, max_gap_sec=3 * 3600.0
        )

        assert np.isnan(avgs[0])  # Before observed span (no extrapolation)
        assert math.isclose(avgs[1], 15.0, abs_tol=1e-12)  # Adjacent to gap, does not overlap gap
        assert np.all(np.isnan(avgs[2:7]))  # Overlaps 5-hour gap > 3 hr
        assert math.isclose(avgs[7], 35.0, abs_tol=1e-12)  # Resumes cleanly after gap
        assert np.isnan(avgs[8])  # Past observed span (no extrapolation)

    def test_nan_samples_dropped_and_create_gap_if_too_wide(self):
        # Hourly samples from 0..5 hr, with NaNs at hours 1, 2, 3 -> 4-hr gap [0, 4] > 3 hr
        t_obs = np.arange(6, dtype=np.float64) * 3600.0
        vals = np.array([400.0, np.nan, np.nan, np.nan, 404.0, 405.0])
        period_starts = np.arange(5, dtype=np.float64) * 3600.0
        avgs = interval_average_channel(
            t_obs, vals, period_starts, 3600.0, method=InterpolationMethod.LINEAR, max_gap_sec=3 * 3600.0
        )
        assert np.all(np.isnan(avgs[0:4]))
        assert math.isclose(avgs[4], 404.5, abs_tol=1e-12)


class TestResampleASVTelemetry:
    """Verify Stage 1 ASV resampling: wind oscillation, constancy, IDL ECEF, drift, and outages."""

    def test_symmetric_sailboat_wind_oscillation_cancels_and_lowers_constancy(self):
        base = datetime(2026, 6, 1, 0, 0, tzinfo=UTC)
        # 5-minute samples across 2 hours (25 points: 0..120 min)
        # Exact 12-sample (1-hour) symmetric triangle/square-like oscillation on v_wind around mean (6.0, 0.0)
        n_steps = 24
        times = [base + timedelta(minutes=5 * i) for i in range(n_steps + 1)]
        # Periodic cross-wind bobbing: 0, +8, 0, -8 (20 min period = 3 full cycles/hr)
        pattern = [0.0, 8.0, 0.0, -8.0]
        v_osc = [pattern[i % 4] for i in range(n_steps + 1)]
        u_vals = [6.0] * (n_steps + 1)

        df = pl.DataFrame(
            {
                "asv_id": ["Oshen1"] * (n_steps + 1),
                "timestamp": times,
                "latitude": [13.4] * (n_steps + 1),
                "longitude": [-101.7] * (n_steps + 1),
                "co2_1": [415.0] * (n_steps + 1),
                "co2_2": [415.2] * (n_steps + 1),
                "u_wind": u_vals,
                "v_wind": v_osc,
            }
        ).with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))

        periods = _utc_hours(2, base)
        aligned = resample_asv_telemetry(df, periods, ACPFConfig())

        assert aligned.height == 2
        # Mean wind vector recovers exact (6.0, 0.0)
        np.testing.assert_allclose(aligned["u_wind"].to_numpy(), [6.0, 6.0], atol=1e-12)
        np.testing.assert_allclose(aligned["v_wind"].to_numpy(), [0.0, 0.0], atol=1e-12)

        # Scalar speed oscillates between 6.0 and hypot(6, 8) = 10.0; linear average is (6 + 10)/2 = 8.0
        # Constancy r_const = ||w_bar|| / s_bar = 6.0 / 8.0 = 0.75 < 1.0
        np.testing.assert_allclose(aligned["wind_constancy"].to_numpy(), [0.75, 0.75], atol=1e-12)

    def test_wind_constancy_edge_cases_steady_zero_and_reversal(self):
        base = datetime(2026, 6, 1, 0, 0, tzinfo=UTC)
        # Three ASVs over 1 hour [0, 60 min]:
        # Oshen1: perfectly steady (5, 0) -> constancy = 1.0
        # Oshen2: calm zero wind (0, 0) -> constancy = 1.0 (epsilon guard)
        # Oshen3: 180-deg symmetric reversal (+5 at t=0, 0 at t=30, -5 at t=60) -> vector mean = 0, constancy = 0.0
        times = [base, base + timedelta(minutes=30), base + timedelta(minutes=60)]
        rows = []
        for asv_id, u_seq in [
            ("Oshen1", [5.0, 5.0, 5.0]),
            ("Oshen2", [0.0, 0.0, 0.0]),
            ("Oshen3", [5.0, 0.0, -5.0]),
        ]:
            for ts, u_val in zip(times, u_seq, strict=True):
                rows.append(
                    {
                        "asv_id": asv_id,
                        "timestamp": ts,
                        "latitude": 13.4,
                        "longitude": -101.7,
                        "co2_1": 415.0,
                        "co2_2": 415.0,
                        "u_wind": u_val,
                        "v_wind": 0.0,
                    }
                )

        df = pl.DataFrame(rows).with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))
        aligned = resample_asv_telemetry(df, _utc_hours(1, base), ACPFConfig())
        by_asv = dict(zip(aligned["asv_id"].to_list(), aligned["wind_constancy"].to_list(), strict=True))

        assert math.isclose(by_asv["Oshen1"], 1.0, abs_tol=1e-12)
        assert math.isclose(by_asv["Oshen2"], 1.0, abs_tol=1e-12)
        assert math.isclose(by_asv["Oshen3"], 0.0, abs_tol=1e-12)

    def test_ecef_position_interpolation_across_international_date_line(self):
        base = datetime(2026, 6, 1, 0, 0, tzinfo=UTC)
        # ASV sails steadily East across +/-180 deg from +179.6 -> +180.0 -> -179.6 over 2 hours
        times = [base, base + timedelta(hours=1), base + timedelta(hours=2)]
        lons = [179.6, 180.0, -179.6]
        df = pl.DataFrame(
            {
                "asv_id": ["Oshen1"] * 3,
                "timestamp": times,
                "latitude": [13.4, 13.4, 13.4],
                "longitude": lons,
                "co2_1": [415.0] * 3,
                "co2_2": [415.0] * 3,
                "u_wind": [5.0] * 3,
                "v_wind": [0.0] * 3,
            }
        ).with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))

        aligned = resample_asv_telemetry(df, _utc_hours(2, base), ACPFConfig())
        assert aligned.height == 2
        lat_out = aligned["latitude"].to_numpy()
        lon_out = aligned["longitude"].to_numpy()

        np.testing.assert_allclose(lat_out, [13.4, 13.4], atol=1e-4)
        assert math.isclose(lon_out[0], 179.8, abs_tol=1e-4)
        assert math.isclose(lon_out[1], -179.8, abs_tol=1e-4)

    def test_asv_drift_speed_accuracy_along_meridian(self):
        base = datetime(2026, 6, 1, 0, 0, tzinfo=UTC)
        lat0_deg = 13.4
        lat0_rad = math.radians(lat0_deg)
        # Exact WGS 84 meridional radius M(phi)
        m_radius = WGS84_SEMI_MAJOR_M * (1.0 - WGS84_E2) / ((1.0 - WGS84_E2 * math.sin(lat0_rad) ** 2) ** 1.5)
        true_speed_m_s = 1.25
        deg_per_hour = math.degrees((true_speed_m_s * 3600.0) / m_radius)

        # 4 hourly observations covering 3 periods [0,1), [1,2), [2,3)
        times = [base + timedelta(hours=h) for h in range(4)]
        lats = [lat0_deg + h * deg_per_hour for h in range(4)]
        df = pl.DataFrame(
            {
                "asv_id": ["Oshen1"] * 4,
                "timestamp": times,
                "latitude": lats,
                "longitude": [-101.7] * 4,
                "co2_1": [415.0] * 4,
                "co2_2": [415.0] * 4,
                "u_wind": [5.0] * 4,
                "v_wind": [0.0] * 4,
            }
        ).with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))

        aligned = resample_asv_telemetry(df, _utc_hours(3, base), ACPFConfig())
        drift = aligned["asv_drift_speed_m_s"].to_numpy()

        assert np.isnan(drift[0])  # First period has no previous period
        np.testing.assert_allclose(drift[1:], [true_speed_m_s, true_speed_m_s], rtol=1e-4)

    def test_single_co2_sensor_kept_and_dual_outage_omitted(self):
        base = datetime(2026, 6, 1, 0, 0, tzinfo=UTC)
        times = [base, base + timedelta(hours=1), base + timedelta(hours=2)]

        # Oshen1: co2_2 is null throughout -> kept with co2_1 finite, co2_2 NaN
        # Oshen2: both co2_1 and co2_2 are null -> omitted completely
        # Oshen3: both sensors valid -> kept
        df = pl.DataFrame(
            {
                "asv_id": ["Oshen1"] * 3 + ["Oshen2"] * 3 + ["Oshen3"] * 3,
                "timestamp": times * 3,
                "latitude": [13.4] * 9,
                "longitude": [-101.7] * 9,
                "co2_1": [415.0, 416.0, 417.0, None, None, None, 414.0, 414.0, 414.0],
                "co2_2": [None, None, None, None, None, None, 414.2, 414.2, 414.2],
                "u_wind": [5.0] * 9,
                "v_wind": [1.0] * 9,
            },
            schema={
                "asv_id": pl.String,
                "timestamp": pl.Datetime("us", "UTC"),
                "latitude": pl.Float64,
                "longitude": pl.Float64,
                "co2_1": pl.Float64,
                "co2_2": pl.Float64,
                "u_wind": pl.Float64,
                "v_wind": pl.Float64,
            },
        )

        aligned = resample_asv_telemetry(df, _utc_hours(2, base), ACPFConfig())
        assert aligned["asv_id"].unique(maintain_order=True).to_list() == ["Oshen1", "Oshen3"]
        assert aligned.height == 4

        oshen1 = aligned.filter(pl.col("asv_id") == "Oshen1")
        np.testing.assert_allclose(oshen1["co2_1"].to_numpy(), [415.5, 416.5], atol=1e-12)
        assert np.all(np.isnan(oshen1["co2_2"].to_numpy()))


class TestResampleMetSeries:
    """Verify meteorological time series direct match and interval-integral alignment."""

    def test_direct_match_when_already_on_period_grid(self):
        base = datetime(2026, 6, 1, 0, 0, tzinfo=UTC)
        periods = _utc_hours(3, base)
        met_df = pl.DataFrame(
            {
                "timestamp": periods,
                "mblh_m": [800.0, 950.0, 1100.0],
                "t_air_k": [292.0, 293.0, 294.0],
                "p_atm_pa": [101_200.0, 101_300.0, 101_400.0],
                "u_wind_ext": [4.0, 5.0, 6.0],
                "v_wind_ext": [-1.0, -2.0, -3.0],
            }
        )

        aligned = resample_met_series(met_df, periods, ACPFConfig())
        assert aligned["period_start"].to_list() == periods.to_list()
        np.testing.assert_allclose(aligned["mblh_m"].to_numpy(), [800.0, 950.0, 1100.0])
        np.testing.assert_allclose(aligned["t_air_k"].to_numpy(), [292.0, 293.0, 294.0])
        np.testing.assert_allclose(aligned["p_atm_pa"].to_numpy(), [101_200.0, 101_300.0, 101_400.0])
        np.testing.assert_allclose(aligned["u_wind_ext"].to_numpy(), [4.0, 5.0, 6.0])
        np.testing.assert_allclose(aligned["v_wind_ext"].to_numpy(), [-1.0, -2.0, -3.0])

    def test_subhourly_and_coarse_met_resampled_via_interval_integral(self):
        base = datetime(2026, 6, 1, 0, 0, tzinfo=UTC)
        # 3-hourly linear ramp: mblh(t_hr) = 800 + 100 * t_hr over [0, 6 hr]
        met_df = pl.DataFrame(
            {
                "timestamp": [base + timedelta(hours=h) for h in [0, 3, 6]],
                "mblh_m": [800.0, 1100.0, 1400.0],
            }
        ).with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))

        periods = _utc_hours(6, base)
        # Default max_interp_gap = 3 hr bridges the 3-hourly spacing exactly
        aligned = resample_met_series(met_df, periods, ACPFConfig())
        expected = [800.0 + 100.0 * (h + 0.5) for h in range(6)]
        np.testing.assert_allclose(aligned["mblh_m"].to_numpy(), expected, atol=1e-12)

        # Tighter max_interp_gap = 2 hr rejects the 3-hourly gap
        tight_cfg = ACPFConfig(max_interp_gap=timedelta(hours=2))
        aligned_tight = resample_met_series(met_df, periods, tight_cfg)
        assert np.all(np.isnan(aligned_tight["mblh_m"].to_numpy()))

    def test_subhourly_met_resampled_via_interval_integral(self):
        base = datetime(2026, 6, 1, 0, 0, tzinfo=UTC)
        # 15-minute linear ramp over [0, 2 hr]: mblh(t_hr) = 900 + 60 * t_hr
        times = [base + timedelta(minutes=15 * i) for i in range(9)]
        mblh = [900.0 + 15.0 * i for i in range(9)]
        met_df = pl.DataFrame(
            {
                "timestamp": times,
                "mblh_m": mblh,
                "t_air_k": [293.0 + 0.25 * i for i in range(9)],
            }
        ).with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))

        aligned = resample_met_series(met_df, _utc_hours(2, base), ACPFConfig())
        np.testing.assert_allclose(aligned["mblh_m"].to_numpy(), [930.0, 990.0], atol=1e-12)
        np.testing.assert_allclose(aligned["t_air_k"].to_numpy(), [293.5, 294.5], atol=1e-12)

    def test_missing_required_column_raises_value_error(self):
        base = datetime(2026, 6, 1, 0, 0, tzinfo=UTC)
        bad_met = pl.DataFrame({"timestamp": _utc_hours(2, base), "t_air_k": [293.0, 294.0]})
        with pytest.raises(ValueError, match="mblh_m"):
            resample_met_series(bad_met, _utc_hours(2, base), ACPFConfig())
