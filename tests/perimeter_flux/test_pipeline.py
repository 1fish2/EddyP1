"""Pytest test suite for Stage 3 multi-period ACPF orchestration, storage term, and window summaries."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from perimeter_flux.config import (
    DEFAULT_P_ATM_PA,
    DEFAULT_T_AIR_K,
    ERROR_BAR_DTYPE,
    KG_PER_S_TO_T_PER_HR,
    ErrorBar,
)
from perimeter_flux.flux import co2_mass_per_ppm_kg_m3, effective_mblh_m
from perimeter_flux.geodesy import project_latlon_to_local_en
from perimeter_flux.pipeline import (
    ACPFResult,
    compute_acpf,
    compute_acpf_from_aligned,
    summarize_window,
)
from perimeter_flux.sample_data import ACAPULCO_EDDY_VERTICES


def _build_aligned_acapulco_inputs(
    n_hours: int,
    *,
    start: datetime | None = None,
    u_by_hour: list[float] | None = None,
    v_by_hour: list[float] | None = None,
    mblh_by_hour: list[float] | None = None,
    co2_fn=None,
    active_asv_counts: list[int] | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Helper producing pre-aligned Stage 1 DataFrames on ACAPULCO_EDDY_VERTICES."""
    base = start or datetime(2026, 6, 1, 0, 0, tzinfo=UTC)
    lats = [p[0] for p in ACAPULCO_EDDY_VERTICES]
    lons = [p[1] for p in ACAPULCO_EDDY_VERTICES]
    xy_m, _, _ = project_latlon_to_local_en(lats, lons)

    u_seq = u_by_hour if u_by_hour is not None else [6.0] * n_hours
    v_seq = v_by_hour if v_by_hour is not None else [0.0] * n_hours
    mblh_seq = mblh_by_hour if mblh_by_hour is not None else [1000.0] * n_hours
    counts = active_asv_counts if active_asv_counts is not None else [6] * n_hours

    asv_rows: list[dict[str, object]] = []
    met_rows: list[dict[str, object]] = []
    for h in range(n_hours):
        ts = base + timedelta(hours=h)
        met_rows.append({"period_start": ts, "mblh_m": float(mblh_seq[h])})
        for idx in range(counts[h]):
            x_m = float(xy_m[idx, 0])
            y_m = float(xy_m[idx, 1])
            if co2_fn is None:
                c1, c2 = 420.0, 420.0
            else:
                c1, c2 = co2_fn(h, idx, x_m, y_m)
            asv_rows.append(
                {
                    "period_start": ts,
                    "asv_id": f"Oshen{idx + 1}",
                    "latitude": lats[idx],
                    "longitude": lons[idx],
                    "u_wind": float(u_seq[h]),
                    "v_wind": float(v_seq[h]),
                    "wind_constancy": 1.0,
                    "co2_1": c1,
                    "co2_2": c2,
                    "asv_drift_speed_m_s": 0.0 if h > 0 else float("nan"),
                }
            )

    aligned_asvs = pl.DataFrame(
        asv_rows,
        schema={
            "period_start": pl.Datetime("us", "UTC"),
            "asv_id": pl.String,
            "latitude": pl.Float64,
            "longitude": pl.Float64,
            "u_wind": pl.Float64,
            "v_wind": pl.Float64,
            "wind_constancy": pl.Float64,
            "co2_1": pl.Float64,
            "co2_2": pl.Float64,
            "asv_drift_speed_m_s": pl.Float64,
        },
    )
    aligned_met = pl.DataFrame(
        met_rows,
        schema={"period_start": pl.Datetime("us", "UTC"), "mblh_m": pl.Float64},
    )
    return aligned_asvs, aligned_met


class TestEndToEndAsyncPipeline:
    """Verify full Stage 1 -> Stage 2 -> Stage 3 async pipeline and output schemas."""

    def test_async_steady_drawdown_exact_flux_and_schemas(self):
        base = datetime(2026, 6, 1, 0, 0, tzinfo=UTC)
        lats = [p[0] for p in ACAPULCO_EDDY_VERTICES]
        lons = [p[1] for p in ACAPULCO_EDDY_VERTICES]
        xy_m, _, _ = project_latlon_to_local_en(lats, lons)

        u_wind, v_wind = 7.0, -2.5
        gx, gy = -1.2e-5, 0.4e-5
        mblh = 950.0

        # Irregular asynchronous timestamps per ASV spanning [0, 5 hr]
        asv_rows: list[dict[str, object]] = []
        for idx in range(6):
            c = 418.0 + gx * float(xy_m[idx, 0]) + gy * float(xy_m[idx, 1])
            # Different irregular spacing per ASV covering [0, 300 min]
            minute_marks = sorted({0, 300, *(range(7 * (idx + 1), 300, 17 + idx * 3))})
            for m in minute_marks:
                asv_rows.append(
                    {
                        "asv_id": f"Oshen{idx + 1}",
                        "timestamp": base + timedelta(minutes=m),
                        "latitude": lats[idx],
                        "longitude": lons[idx],
                        "co2_1": c,
                        "co2_2": c + 0.1,
                        "u_wind": u_wind,
                        "v_wind": v_wind,
                    }
                )

        asv_df = pl.DataFrame(asv_rows).with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))
        met_df = pl.DataFrame(
            {
                "timestamp": [base + timedelta(hours=h) for h in range(5)],
                "mblh_m": [mblh] * 5,
            }
        ).with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))

        result = compute_acpf(asv_df, met_df)
        assert isinstance(result, ACPFResult)
        assert result.hourly.height == 5
        assert result.edges.height == 5 * 6
        assert result.aligned_asvs.height == 5 * 6

        # Verify ErrorBar Polars Enum dtype & ordering
        assert result.hourly.schema["error_bar"] == ERROR_BAR_DTYPE
        assert result.hourly.filter(pl.col("error_bar") == ErrorBar.STABLE).height == 5

        # Verify exact analytical CDRR and zero storage flux in steady state
        area_m2 = float(result.hourly["hull_area_km2"][0]) * 1e6
        kappa = co2_mass_per_ppm_kg_m3(DEFAULT_T_AIR_K, DEFAULT_P_ATM_PA)
        h_eff = effective_mblh_m(mblh, DEFAULT_T_AIR_K)
        expected_cdrr = KG_PER_S_TO_T_PER_HR * kappa * h_eff * (-(gx * u_wind + gy * v_wind)) * area_m2

        np.testing.assert_allclose(
            result.hourly["lateral_cdrr_tco2_hr"].to_numpy(), expected_cdrr, rtol=1e-10
        )
        # 1 ULP on ~420 ppm over a ~38,000 km^2 MBL column is ~4e-9 tCO2/hr
        np.testing.assert_allclose(
            result.hourly["storage_flux_tco2_hr"].to_numpy(), 0.0, atol=1e-7
        )
        np.testing.assert_allclose(
            result.hourly["storage_adjusted_cdrr_tco2_hr"].to_numpy(), expected_cdrr, rtol=1e-10
        )

        # Verify canonical hull start in hourly and edges tables
        assert result.hourly["hull_start_asv_id"].to_list() == ["Oshen1"] * 5
        first_edges = result.edges.filter(pl.col("edge_idx") == 0)
        assert first_edges["start_asv_id"].to_list() == ["Oshen1"] * 5
        assert result.aligned_asvs["on_hull"].all()

    def test_empty_inputs_return_empty_typed_frames(self):
        empty_asv = pl.DataFrame(
            schema={
                "asv_id": pl.String,
                "timestamp": pl.Datetime("us", "UTC"),
                "latitude": pl.Float64,
                "longitude": pl.Float64,
                "co2_1": pl.Float64,
                "co2_2": pl.Float64,
                "u_wind": pl.Float64,
                "v_wind": pl.Float64,
            }
        )
        empty_met = pl.DataFrame(
            schema={"timestamp": pl.Datetime("us", "UTC"), "mblh_m": pl.Float64}
        )
        res = compute_acpf(empty_asv, empty_met)
        assert res.hourly.height == 0
        assert res.edges.height == 0
        assert res.aligned_asvs.height == 0
        assert res.hourly.schema["error_bar"] == ERROR_BAR_DTYPE


class TestInterPeriodStabilityAndPrecedence:
    """Verify circular wind wrap-around, MBLH/wind shifts, and ErrorBar precedence."""

    def test_circular_wrap_safe_wind_shift_350_to_10_degrees(self):
        speed = 6.0
        headings_deg = [350.0, 10.0, 350.0]
        u_seq = [speed * math.cos(math.radians(a)) for a in headings_deg]
        v_seq = [speed * math.sin(math.radians(a)) for a in headings_deg]
        aligned_asvs, aligned_met = _build_aligned_acapulco_inputs(
            3, u_by_hour=u_seq, v_by_hour=v_seq
        )

        res = compute_acpf_from_aligned(aligned_asvs, aligned_met)
        deltas = res.hourly["wind_dir_delta_deg"].to_numpy()
        assert np.isnan(deltas[0])
        assert math.isclose(deltas[1], 20.0, abs_tol=1e-9)
        assert math.isclose(deltas[2], -20.0, abs_tol=1e-9)
        assert res.hourly["error_bar"].to_list() == [ErrorBar.STABLE.value] * 3

    def test_wind_shift_speed_jump_mblh_ramp_and_worst_case_precedence(self):
        # Hour 0: baseline (heading 0 deg, speed 5 m/s, mblh 1000 m) -> STABLE
        # Hour 1: +50 deg wind shift (> 30 deg) -> UNSTABLE_WIND_OR_MBLH
        # Hour 2: speed jump 5 -> 10 m/s (rel delta = +0.5 > 0.35) -> UNSTABLE_WIND_OR_MBLH
        # Hour 3: MBLH ramp 1000 -> 1500 m (rel delta = +0.333 > 0.25) -> UNSTABLE_WIND_OR_MBLH
        # Hour 4: +60 deg wind shift AND Oshen3 4.0 ppm CO2 discrepancy -> CO2_SENSOR_DISCREPANCY wins!
        # Hour 5: +60 deg wind shift AND only 2 ASVs -> TWO_ASV_EXTRAPOLATED wins!
        headings = [0.0, 50.0, 50.0, 50.0, 110.0, 170.0]
        speeds = [5.0, 5.0, 10.0, 10.0, 10.0, 10.0]
        mblhs = [1000.0, 1000.0, 1000.0, 1500.0, 1500.0, 1500.0]
        counts = [6, 6, 6, 6, 6, 2]
        u_seq = [s * math.cos(math.radians(a)) for s, a in zip(speeds, headings, strict=True)]
        v_seq = [s * math.sin(math.radians(a)) for s, a in zip(speeds, headings, strict=True)]

        def co2_fn(h: int, idx: int, _x: float, _y: float) -> tuple[float, float]:
            if h == 4 and idx == 2:  # Oshen3 sensor discrepancy during wind shift
                return (418.0, 422.0)
            return (420.0, 420.0)

        aligned_asvs, aligned_met = _build_aligned_acapulco_inputs(
            6,
            u_by_hour=u_seq,
            v_by_hour=v_seq,
            mblh_by_hour=mblhs,
            co2_fn=co2_fn,
            active_asv_counts=counts,
        )
        res = compute_acpf_from_aligned(aligned_asvs, aligned_met)

        assert res.hourly["error_bar"].to_list() == [
            ErrorBar.STABLE.value,
            ErrorBar.UNSTABLE_WIND_OR_MBLH.value,
            ErrorBar.UNSTABLE_WIND_OR_MBLH.value,
            ErrorBar.UNSTABLE_WIND_OR_MBLH.value,
            ErrorBar.CO2_SENSOR_DISCREPANCY.value,
            ErrorBar.TWO_ASV_EXTRAPOLATED.value,
        ]
        # Diagnostics preserved even when higher-severity ErrorBar wins in Hour 4
        assert math.isclose(float(res.hourly["wind_dir_delta_deg"][4]), 60.0, abs_tol=1e-9)
        assert math.isclose(float(res.hourly["max_co2_sensor_diff_ppm"][4]), 4.0, abs_tol=1e-9)
        assert int(res.hourly["n_co2_flagged_asvs"][4]) == 1
        assert math.isnan(float(res.hourly["lateral_cdrr_tco2_hr"][5]))


class TestStorageTermAndSynopticFrontCancellation:
    """Verify synoptic front cancellation, zero-wind pure storage, and gapped stencils."""

    def test_propagating_linear_synoptic_front_cancels_to_zero_net_cdrr(self):
        # Passive tracer c(x, t) = 420 + gx * (x - u * t) with zero ocean uptake.
        # Advection causes a large transient lateral flux, canceled exactly by storage flux.
        u = 8.0
        gx = -1.5e-5  # ppm / m -> dc/dt = -gx * u = +1.2e-4 ppm / s
        dt_sec = 3600.0

        def front_co2(h: int, _idx: int, x_m: float, _y_m: float) -> tuple[float, float]:
            t_mid = (h + 0.5) * dt_sec
            c = 420.0 + gx * (x_m - u * t_mid)
            return (c, c)

        aligned_asvs, aligned_met = _build_aligned_acapulco_inputs(
            6, u_by_hour=[u] * 6, v_by_hour=[0.0] * 6, co2_fn=front_co2
        )
        res = compute_acpf_from_aligned(aligned_asvs, aligned_met)

        lat_flux = res.hourly["lateral_cdrr_tco2_hr"].to_numpy()
        stor_flux = res.hourly["storage_flux_tco2_hr"].to_numpy()
        net_flux = res.hourly["storage_adjusted_cdrr_tco2_hr"].to_numpy()

        # Lateral spike is substantially non-zero (positive fake uptake)
        assert np.all(lat_flux > 100.0)
        # Storage flux is equal and opposite; net CDRR cancels to zero within float64 ULP (~1e-8 tCO2/hr)
        np.testing.assert_allclose(stor_flux, -lat_flux, rtol=1e-10)
        np.testing.assert_allclose(net_flux, 0.0, atol=1e-7)

    def test_zero_wind_pure_storage_drawdown(self):
        # Zero wind => Phi_lat == 0; uniform box CO2 drops at -0.5 ppm/hr => positive storage CDRR
        rate_ppm_per_hr = -0.5

        def drawdown_co2(h: int, _idx: int, _x_m: float, _y_m: float) -> tuple[float, float]:
            c = 420.0 + rate_ppm_per_hr * h
            return (c, c)

        aligned_asvs, aligned_met = _build_aligned_acapulco_inputs(
            5, u_by_hour=[0.0] * 5, v_by_hour=[0.0] * 5, co2_fn=drawdown_co2
        )
        res = compute_acpf_from_aligned(aligned_asvs, aligned_met)

        area_m2 = float(res.hourly["hull_area_km2"][0]) * 1e6
        kappa = co2_mass_per_ppm_kg_m3(DEFAULT_T_AIR_K, DEFAULT_P_ATM_PA)
        h_eff = effective_mblh_m(1000.0, DEFAULT_T_AIR_K)
        expected_stor = (
            -KG_PER_S_TO_T_PER_HR * kappa * area_m2 * h_eff * (rate_ppm_per_hr / 3600.0)
        )
        assert expected_stor > 0.0

        np.testing.assert_allclose(res.hourly["lateral_cdrr_tco2_hr"].to_numpy(), 0.0, atol=1e-12)
        np.testing.assert_allclose(
            res.hourly["storage_flux_tco2_hr"].to_numpy(), expected_stor, rtol=1e-10
        )
        np.testing.assert_allclose(
            res.hourly["storage_adjusted_cdrr_tco2_hr"].to_numpy(), expected_stor, rtol=1e-10
        )

    def test_contiguous_boundary_centered_and_isolated_storage_stencils(self):
        # 8 hours with quadratic box CO2 c(h) = 420 + 0.2 * h^2 so forward/centered/backward differ!
        # Valid pattern: [V, V, V, X, V, X, V, V]
        #   h=0: start of run -> forward  (c[1] - c[0]) / dt
        #   h=1: interior     -> centered (c[2] - c[0]) / (2 dt)
        #   h=2: end of run   -> backward (c[2] - c[1]) / dt
        #   h=3: invalid (1 ASV) -> NaN
        #   h=4: isolated valid  -> NaN
        #   h=5: invalid (1 ASV) -> NaN
        #   h=6: start of run -> forward  (c[7] - c[6]) / dt
        #   h=7: end of run   -> backward (c[7] - c[6]) / dt
        counts = [6, 6, 6, 1, 6, 1, 6, 6]
        c_vals = [420.0 + 0.2 * (h**2) for h in range(8)]

        def quad_co2(h: int, _idx: int, _x_m: float, _y_m: float) -> tuple[float, float]:
            return (c_vals[h], c_vals[h])

        aligned_asvs, aligned_met = _build_aligned_acapulco_inputs(
            8, u_by_hour=[0.0] * 8, v_by_hour=[0.0] * 8, co2_fn=quad_co2, active_asv_counts=counts
        )
        res = compute_acpf_from_aligned(aligned_asvs, aligned_met)

        area_m2 = float(res.hourly["hull_area_km2"][0]) * 1e6
        kappa = co2_mass_per_ppm_kg_m3(DEFAULT_T_AIR_K, DEFAULT_P_ATM_PA)
        h_eff = effective_mblh_m(1000.0, DEFAULT_T_AIR_K)
        scale = -KG_PER_S_TO_T_PER_HR * kappa * area_m2 * h_eff / 3600.0

        stor = res.hourly["storage_flux_tco2_hr"].to_numpy()
        assert math.isclose(stor[0], scale * (c_vals[1] - c_vals[0]), rel_tol=1e-10)
        assert math.isclose(stor[1], scale * (c_vals[2] - c_vals[0]) / 2.0, rel_tol=1e-10)
        assert math.isclose(stor[2], scale * (c_vals[2] - c_vals[1]), rel_tol=1e-10)
        assert np.isnan(stor[3])
        assert np.isnan(stor[4])  # Isolated valid period
        assert np.isnan(stor[5])
        assert math.isclose(stor[6], scale * (c_vals[7] - c_vals[6]), rel_tol=1e-10)
        assert math.isclose(stor[7], scale * (c_vals[7] - c_vals[6]), rel_tol=1e-10)


class TestSummarizeWindow:
    """Verify multi-period rolling tau_win summaries with ErrorBar severity filtering."""

    def test_rolling_window_filtering_and_worst_error_bar(self):
        base = datetime(2026, 6, 1, 0, 0, tzinfo=UTC)
        hourly = pl.DataFrame(
            {
                "period_start": [base + timedelta(hours=h) for h in range(6)],
                "lateral_cdrr_tco2_hr": [10.0, 20.0, 999.0, float("nan"), 40.0, 50.0],
                "storage_flux_tco2_hr": [1.0, 2.0, -50.0, float("nan"), 4.0, 5.0],
                "storage_adjusted_cdrr_tco2_hr": [11.0, 22.0, 949.0, float("nan"), 44.0, 55.0],
                "storage_adjusted_cdrr_tco2_km2_hr": [0.11, 0.22, 9.49, float("nan"), 0.44, 0.55],
                "error_bar": pl.Series(
                    [
                        ErrorBar.STABLE,
                        ErrorBar.UNSTABLE_WIND_OR_MBLH,
                        ErrorBar.CO2_SENSOR_DISCREPANCY,
                        ErrorBar.TWO_ASV_EXTRAPOLATED,
                        ErrorBar.STABLE,
                        ErrorBar.STABLE,
                    ],
                    dtype=ERROR_BAR_DTYPE,
                ),
            }
        ).with_columns(pl.col("period_start").cast(pl.Datetime("us", "UTC")))

        # 3-hour rolling window accepting <= UNSTABLE_WIND_OR_MBLH
        summary = summarize_window(
            hourly, timedelta(hours=3), max_error_bar=ErrorBar.UNSTABLE_WIND_OR_MBLH
        )
        assert summary.height == 6
        assert summary.schema["worst_error_bar"] == ERROR_BAR_DTYPE

        # Windows:
        # h=0: {0}       -> valid {0}: mean_net = 11.0, n_valid = 1, worst = STABLE
        # h=1: {0,1}     -> valid {0,1}: mean_net = 16.5, n_valid = 2, worst = UNSTABLE
        # h=2: {0,1,2}   -> valid {0,1} (2 excluded): mean_net = 16.5, n_valid = 2, worst = CO2_DISCREPANCY
        # h=3: {1,2,3}   -> valid {1}: mean_net = 22.0, n_valid = 1, worst = TWO_ASV
        # h=4: {2,3,4}   -> valid {4}: mean_net = 44.0, n_valid = 1, worst = TWO_ASV
        # h=5: {3,4,5}   -> valid {4,5}: mean_net = 49.5, n_valid = 2, worst = TWO_ASV
        assert summary["n_periods_valid"].to_list() == [1, 2, 2, 1, 1, 2]
        np.testing.assert_allclose(
            summary["mean_storage_adjusted_cdrr_tco2_hr"].to_numpy(),
            [11.0, 16.5, 16.5, 22.0, 44.0, 49.5],
            atol=1e-12,
        )
        assert summary["worst_error_bar"].to_list() == [
            ErrorBar.STABLE.value,
            ErrorBar.UNSTABLE_WIND_OR_MBLH.value,
            ErrorBar.CO2_SENSOR_DISCREPANCY.value,
            ErrorBar.TWO_ASV_EXTRAPOLATED.value,
            ErrorBar.TWO_ASV_EXTRAPOLATED.value,
            ErrorBar.TWO_ASV_EXTRAPOLATED.value,
        ]

        # Strict STABLE-only filter leaves hour 3 ({1,2,3}) with 0 valid periods -> NaN
        strict = summarize_window(hourly, timedelta(hours=3), max_error_bar=ErrorBar.STABLE)
        assert strict["n_periods_valid"].to_list() == [1, 1, 1, 0, 1, 2]
        assert math.isnan(float(strict["mean_storage_adjusted_cdrr_tco2_hr"][3]))

    def test_invalid_window_duration_raises_value_error(self):
        # base = datetime(2026, 6, 1, 0, 0, tzinfo=timezone.utc)
        empty = pl.DataFrame(
            schema={
                "period_start": pl.Datetime("us", "UTC"),
                "lateral_cdrr_tco2_hr": pl.Float64,
                "storage_flux_tco2_hr": pl.Float64,
                "storage_adjusted_cdrr_tco2_hr": pl.Float64,
                "storage_adjusted_cdrr_tco2_km2_hr": pl.Float64,
                "error_bar": ERROR_BAR_DTYPE,
            }
        )
        with pytest.raises(ValueError, match="window_duration"):
            summarize_window(empty, timedelta(0))
        with pytest.raises(ValueError, match="window_duration"):
            summarize_window(empty, timedelta(hours=-1))
