"""Stage 3: Multi-period ACPF orchestration, column storage term, QA classification, and window summaries.

Mirrors Section 6 of ``doc/acpf_math.md``:

Eulerian Column Storage & Storage-Adjusted CDRR (Section 6.1)
-------------------------------------------------------------
- ``Phi_stor,k = -gamma_unit * kappa_CO2,k * A_hull,k * H_eff,k * (d c_bar_box / dt)_k``
- ``Phi_net,k  = Phi_lat,k + Phi_stor,k``
- ``psi_net,k  = Phi_net,k / (A_hull,k * 1e-6)``
- ``(d c_bar_box / dt)_k`` uses a 2nd-order centered difference inside contiguous
  valid runs, one-sided forward/backward differences at run endpoints, and ``NaN``
  for isolated valid periods.

Inter-Period Stability & Ordered ErrorBar Precedence (Section 6.3)
------------------------------------------------------------------
- Circular wrap-safe wind shift:
  ``Delta_theta_w,k = atan2(sin(theta_k - theta_{k-1}), cos(theta_k - theta_{k-1}))``
- Relative wind speed change: ``(s_poly,k - s_poly,k-1) / s_poly,k``
- Relative MBLH change: ``(H_MBL,k - H_MBL,k-1) / H_MBL,k``
- Worst-severity precedence via ordered ``ErrorBar``:
  ``STABLE < UNSTABLE_WIND_OR_MBLH < CO2_SENSOR_DISCREPANCY < TWO_ASV_EXTRAPOLATED < INSUFFICIENT_ASVS``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import polars as pl

from perimeter_flux.config import (
    ERROR_BAR_DTYPE,
    KG_PER_S_TO_T_PER_HR,
    MIN_WIND_SPEED_EPS,
    ACPFConfig,
    ErrorBar,
    WindMode,
)
from perimeter_flux.flux import (
    ASVObservation,
    PeriodFluxResult,
    co2_mass_per_ppm_kg_m3,
    compute_period_flux,
)
from perimeter_flux.resample import resample_asv_telemetry, resample_met_series

_DEFAULT_CONFIG = ACPFConfig()

_HOURLY_SCHEMA = pl.Schema(
    {
        "period_start": pl.Datetime("us", "UTC"),
        "lateral_cdrr_tco2_hr": pl.Float64,
        "storage_flux_tco2_hr": pl.Float64,
        "storage_adjusted_cdrr_tco2_hr": pl.Float64,
        "lateral_cdrr_tco2_km2_hr": pl.Float64,
        "storage_adjusted_cdrr_tco2_km2_hr": pl.Float64,
        "error_bar": ERROR_BAR_DTYPE,
        "n_asvs_total": pl.Int64,
        "n_asvs_hull": pl.Int64,
        "hull_start_asv_id": pl.String,
        "hull_area_km2": pl.Float64,
        "hull_perimeter_km": pl.Float64,
        "hull_compactness": pl.Float64,
        "mblh_m": pl.Float64,
        "effective_mblh_m": pl.Float64,
        "t_air_k": pl.Float64,
        "p_atm_pa": pl.Float64,
        "box_mean_co2_ppm": pl.Float64,
        "co2_ref_ppm": pl.Float64,
        "mean_wind_u": pl.Float64,
        "mean_wind_v": pl.Float64,
        "mean_wind_speed_m_s": pl.Float64,
        "wind_divergence_m2_s": pl.Float64,
        "wind_constancy_min": pl.Float64,
        "wind_dir_delta_deg": pl.Float64,
        "wind_speed_rel_delta": pl.Float64,
        "mblh_rel_delta": pl.Float64,
        "max_co2_sensor_diff_ppm": pl.Float64,
        "n_co2_flagged_asvs": pl.Int64,
        "max_asv_drift_speed_m_s": pl.Float64,
    }
)

_EDGES_SCHEMA = pl.Schema(
    {
        "period_start": pl.Datetime("us", "UTC"),
        "edge_idx": pl.Int64,
        "start_asv_id": pl.String,
        "end_asv_id": pl.String,
        "x1_km": pl.Float64,
        "y1_km": pl.Float64,
        "x2_km": pl.Float64,
        "y2_km": pl.Float64,
        "mid_x_km": pl.Float64,
        "mid_y_km": pl.Float64,
        "normal_x": pl.Float64,
        "normal_y": pl.Float64,
        "edge_length_km": pl.Float64,
        "edge_co2_ppm": pl.Float64,
        "edge_co2_anomaly_ppm": pl.Float64,
        "edge_wind_u": pl.Float64,
        "edge_wind_v": pl.Float64,
        "wind_dot_normal_m_s": pl.Float64,
        "edge_outward_flux_tco2_hr": pl.Float64,
    }
)

_ALIGNED_ASVS_SCHEMA = pl.Schema(
    {
        "period_start": pl.Datetime("us", "UTC"),
        "asv_id": pl.String,
        "latitude": pl.Float64,
        "longitude": pl.Float64,
        "x_km": pl.Float64,
        "y_km": pl.Float64,
        "on_hull": pl.Boolean,
        "co2_1": pl.Float64,
        "co2_2": pl.Float64,
        "co2_ppm": pl.Float64,
        "co2_sensor_diff_ppm": pl.Float64,
        "co2_sensor_flagged": pl.Boolean,
        "u_wind": pl.Float64,
        "v_wind": pl.Float64,
        "wind_constancy": pl.Float64,
        "asv_drift_speed_m_s": pl.Float64,
    }
)

_WINDOW_SUMMARY_SCHEMA = pl.Schema(
    {
        "period_start": pl.Datetime("us", "UTC"),
        "n_periods_total": pl.Int64,
        "n_periods_valid": pl.Int64,
        "mean_lateral_cdrr_tco2_hr": pl.Float64,
        "mean_storage_flux_tco2_hr": pl.Float64,
        "mean_storage_adjusted_cdrr_tco2_hr": pl.Float64,
        "mean_storage_adjusted_cdrr_tco2_km2_hr": pl.Float64,
        "worst_error_bar": ERROR_BAR_DTYPE,
    }
)


@dataclass(frozen=True, slots=True)
class ACPFResult:
    """Multi-period ACPF output bundle of flat, Altair-ready Polars DataFrames."""

    hourly: pl.DataFrame
    edges: pl.DataFrame
    aligned_asvs: pl.DataFrame


def _normalize_ts_col(df: pl.DataFrame, col: str) -> pl.DataFrame:
    dtype = df.schema[col]
    if not isinstance(dtype, pl.Datetime):
        raise TypeError(f"Column '{col}' must be Datetime, got {dtype}.")
    if dtype.time_zone is None:
        expr = pl.col(col).cast(pl.Datetime("us")).dt.replace_time_zone("UTC")
    else:
        expr = pl.col(col).dt.convert_time_zone("UTC").cast(pl.Datetime("us", "UTC"))
    return df.with_columns(expr.alias(col))


def _empty_result() -> ACPFResult:
    return ACPFResult(
        hourly=pl.DataFrame(schema=_HOURLY_SCHEMA),
        edges=pl.DataFrame(schema=_EDGES_SCHEMA),
        aligned_asvs=pl.DataFrame(schema=_ALIGNED_ASVS_SCHEMA),
    )


def _infer_period_starts(asv_telemetry: pl.DataFrame, config: ACPFConfig) -> pl.Series:
    """Build a uniform UTC period_start grid covering the observed telemetry span."""
    if asv_telemetry.height == 0:
        return pl.Series("period_start", [], dtype=pl.Datetime("us", "UTC"))

    ts_utc = _normalize_ts_col(asv_telemetry.select("timestamp"), "timestamp")["timestamp"]
    epochs_us = ts_utc.dt.epoch("us").drop_nulls()
    if epochs_us.len() == 0:
        return pl.Series("period_start", [], dtype=pl.Datetime("us", "UTC"))

    epochs_arr = epochs_us.to_numpy()
    min_us = int(epochs_arr.min())
    max_us = int(epochs_arr.max())
    dt_us = round(config.time_interval.total_seconds() * 1_000_000)

    # Align grid lines to exact UTC multiples of dt_us fully inside [min_us, max_us]
    first_us = -(-min_us // dt_us) * dt_us
    last_end_us = (max_us // dt_us) * dt_us
    if last_end_us - first_us < dt_us:
        return pl.Series("period_start", [], dtype=pl.Datetime("us", "UTC"))

    return pl.Series("period_start", range(first_us, last_end_us, dt_us), dtype=pl.Int64).cast(
        pl.Datetime("us", "UTC")
    )


def _opt_float(val: object) -> float | None:
    if not isinstance(val, (int, float)):
        return None
    f = float(val)
    return None if math.isnan(f) else f


def _float_or_nan(val: object) -> float:
    return float(val) if isinstance(val, (int, float)) else float("nan")


def compute_acpf_from_aligned(
    aligned_asvs: pl.DataFrame,
    aligned_met: pl.DataFrame,
    *,
    config: ACPFConfig = _DEFAULT_CONFIG,
) -> ACPFResult:
    """Run Stages 2 and 3 on already period-aligned ASV and meteorological tables."""
    if aligned_asvs.height == 0 and aligned_met.height == 0:
        return _empty_result()

    asv_df = _normalize_ts_col(aligned_asvs, "period_start") if aligned_asvs.height > 0 else aligned_asvs
    met_df = _normalize_ts_col(aligned_met, "period_start") if aligned_met.height > 0 else aligned_met

    period_series_parts: list[pl.Series] = []
    if met_df.height > 0:
        period_series_parts.append(met_df["period_start"])
    if asv_df.height > 0:
        period_series_parts.append(asv_df["period_start"])
    periods_series = pl.concat(period_series_parts).unique().sort()
    periods_list: list[datetime] = periods_series.to_list()
    n_periods = len(periods_list)
    if n_periods == 0:
        return _empty_result()

    periods_us = periods_series.dt.epoch("us").to_numpy()
    dt_sec = config.time_interval.total_seconds()
    dt_us = round(dt_sec * 1_000_000)

    # Index met rows by period_start
    met_by_period: dict[datetime, dict[str, object]] = {}
    if met_df.height > 0:
        for row in met_df.iter_rows(named=True):
            met_by_period[row["period_start"]] = row

    # Group aligned ASV observations by period_start
    has_constancy = "wind_constancy" in asv_df.columns
    has_drift = "asv_drift_speed_m_s" in asv_df.columns
    obs_by_period: dict[datetime, list[ASVObservation]] = {ts: [] for ts in periods_list}
    if asv_df.height > 0:
        for row in asv_df.iter_rows(named=True):
            constancy_val = row["wind_constancy"] if has_constancy else 1.0
            drift_val = row["asv_drift_speed_m_s"] if has_drift else float("nan")
            obs_by_period[row["period_start"]].append(
                ASVObservation(
                    asv_id=str(row["asv_id"]),
                    latitude=float(row["latitude"]) if row["latitude"] is not None else float("nan"),
                    longitude=float(row["longitude"]) if row["longitude"] is not None else float("nan"),
                    u_wind=float(row["u_wind"]) if row["u_wind"] is not None else float("nan"),
                    v_wind=float(row["v_wind"]) if row["v_wind"] is not None else float("nan"),
                    co2_1=_opt_float(row["co2_1"]),
                    co2_2=_opt_float(row["co2_2"]),
                    wind_constancy=float(constancy_val) if constancy_val is not None else float("nan"),
                    asv_drift_speed_m_s=float(drift_val) if drift_val is not None else float("nan"),
                )
            )

    # Evaluate Stage 2 per period
    stage2_results: list[PeriodFluxResult] = []
    for ts in periods_list:
        met = met_by_period.get(ts)
        if met is None:
            mblh_val = float("nan")
            t_air_val = None
            p_atm_val = None
            ext_wind = (float("nan"), float("nan")) if config.wind_mode is WindMode.EXTERNAL else None
        else:
            mblh_val = _float_or_nan(met.get("mblh_m"))
            t_air_val = _opt_float(met.get("t_air_k"))
            p_atm_val = _opt_float(met.get("p_atm_pa"))
            if config.wind_mode is WindMode.EXTERNAL:
                ext_wind = (
                    _float_or_nan(met.get("u_wind_ext")),
                    _float_or_nan(met.get("v_wind_ext")),
                )
            else:
                ext_wind = None

        stage2_results.append(
            compute_period_flux(
                obs_by_period[ts],
                mblh_m=mblh_val,
                config=config,
                t_air_k=t_air_val,
                p_atm_pa=p_atm_val,
                external_wind=ext_wind,
            )
        )

    # Stage 3A: Inter-period wind & MBLH stability deltas + ordered ErrorBar precedence
    nan = float("nan")
    wind_dir_delta_deg = np.full(n_periods, nan, dtype=np.float64)
    wind_speed_rel_delta = np.full(n_periods, nan, dtype=np.float64)
    mblh_rel_delta = np.full(n_periods, nan, dtype=np.float64)
    final_error_bars: list[ErrorBar] = [r.error_bar for r in stage2_results]

    for k in range(1, n_periods):
        if int(periods_us[k] - periods_us[k - 1]) != dt_us:
            continue
        prev_r = stage2_results[k - 1]
        curr_r = stage2_results[k]
        unstable = False

        # Circular wrap-safe wind heading delta (-180, 180] deg
        if (
            math.isfinite(curr_r.mean_wind_speed_m_s)
            and math.isfinite(prev_r.mean_wind_speed_m_s)
            and curr_r.mean_wind_speed_m_s > MIN_WIND_SPEED_EPS
            and prev_r.mean_wind_speed_m_s > MIN_WIND_SPEED_EPS
        ):
            theta_prev = math.atan2(prev_r.mean_wind_v, prev_r.mean_wind_u)
            theta_curr = math.atan2(curr_r.mean_wind_v, curr_r.mean_wind_u)
            dtheta = math.atan2(
                math.sin(theta_curr - theta_prev),
                math.cos(theta_curr - theta_prev),
            )
            dtheta_deg = math.degrees(dtheta)
            wind_dir_delta_deg[k] = dtheta_deg
            if abs(dtheta_deg) > config.max_wind_dir_change_deg:
                unstable = True

        # Relative wind speed change (s_k - s_{k-1}) / s_k
        if (
            math.isfinite(curr_r.mean_wind_speed_m_s)
            and math.isfinite(prev_r.mean_wind_speed_m_s)
            and curr_r.mean_wind_speed_m_s > MIN_WIND_SPEED_EPS
        ):
            rel_ws = (curr_r.mean_wind_speed_m_s - prev_r.mean_wind_speed_m_s) / curr_r.mean_wind_speed_m_s
            wind_speed_rel_delta[k] = rel_ws
            if abs(rel_ws) > config.max_wind_speed_rel_change:
                unstable = True

        # Relative MBLH change (H_k - H_{k-1}) / H_k
        if (
            math.isfinite(curr_r.mblh_m)
            and math.isfinite(prev_r.mblh_m)
            and curr_r.mblh_m > 0.0
            and prev_r.mblh_m > 0.0
        ):
            rel_h = (curr_r.mblh_m - prev_r.mblh_m) / curr_r.mblh_m
            mblh_rel_delta[k] = rel_h
            if abs(rel_h) > config.max_mblh_rel_change:
                unstable = True

        if unstable:
            final_error_bars[k] = max(final_error_bars[k], ErrorBar.UNSTABLE_WIND_OR_MBLH)

    # Stage 3B: Eulerian column storage term Phi_stor & storage-adjusted CDRR Phi_net
    valid_flux = np.array(
        [math.isfinite(r.lateral_cdrr_tco2_hr) and math.isfinite(r.box_mean_co2_ppm) for r in stage2_results],
        dtype=bool,
    )
    box_co2 = np.array([r.box_mean_co2_ppm for r in stage2_results], dtype=np.float64)

    storage_flux = np.full(n_periods, nan, dtype=np.float64)
    storage_adj_cdrr = np.full(n_periods, nan, dtype=np.float64)
    storage_adj_km2 = np.full(n_periods, nan, dtype=np.float64)

    for k in range(n_periods):
        if not valid_flux[k]:
            continue
        prev_ok = k > 0 and valid_flux[k - 1] and int(periods_us[k] - periods_us[k - 1]) == dt_us
        next_ok = (
            k + 1 < n_periods
            and valid_flux[k + 1]
            and int(periods_us[k + 1] - periods_us[k]) == dt_us
        )
        if prev_ok and next_ok:
            dc_dt = (box_co2[k + 1] - box_co2[k - 1]) / (2.0 * dt_sec)
        elif next_ok:
            dc_dt = (box_co2[k + 1] - box_co2[k]) / dt_sec
        elif prev_ok:
            dc_dt = (box_co2[k] - box_co2[k - 1]) / dt_sec
        else:
            continue

        r = stage2_results[k]
        kappa_k = co2_mass_per_ppm_kg_m3(r.t_air_k, r.p_atm_pa)
        phi_stor = -KG_PER_S_TO_T_PER_HR * kappa_k * r.hull_area_m2 * r.effective_mblh_m * dc_dt
        phi_net = r.lateral_cdrr_tco2_hr + phi_stor
        storage_flux[k] = phi_stor
        storage_adj_cdrr[k] = phi_net
        storage_adj_km2[k] = phi_net / (r.hull_area_m2 * 1e-6)

    # Assemble flat Polars DataFrames (hourly, edges, aligned_asvs)
    hourly_df = pl.DataFrame(
        {
            "period_start": periods_series,
            "lateral_cdrr_tco2_hr": [r.lateral_cdrr_tco2_hr for r in stage2_results],
            "storage_flux_tco2_hr": storage_flux,
            "storage_adjusted_cdrr_tco2_hr": storage_adj_cdrr,
            "lateral_cdrr_tco2_km2_hr": [r.lateral_cdrr_tco2_km2_hr for r in stage2_results],
            "storage_adjusted_cdrr_tco2_km2_hr": storage_adj_km2,
            "error_bar": pl.Series(final_error_bars, dtype=ERROR_BAR_DTYPE),
            "n_asvs_total": [r.n_asvs_total for r in stage2_results],
            "n_asvs_hull": [r.n_asvs_hull for r in stage2_results],
            "hull_start_asv_id": [r.hull_start_asv_id for r in stage2_results],
            "hull_area_km2": [r.hull_area_m2 * 1e-6 for r in stage2_results],
            "hull_perimeter_km": [r.hull_perimeter_m * 1e-3 for r in stage2_results],
            "hull_compactness": [r.hull_compactness for r in stage2_results],
            "mblh_m": [r.mblh_m for r in stage2_results],
            "effective_mblh_m": [r.effective_mblh_m for r in stage2_results],
            "t_air_k": [r.t_air_k for r in stage2_results],
            "p_atm_pa": [r.p_atm_pa for r in stage2_results],
            "box_mean_co2_ppm": [r.box_mean_co2_ppm for r in stage2_results],
            "co2_ref_ppm": [r.co2_ref_ppm for r in stage2_results],
            "mean_wind_u": [r.mean_wind_u for r in stage2_results],
            "mean_wind_v": [r.mean_wind_v for r in stage2_results],
            "mean_wind_speed_m_s": [r.mean_wind_speed_m_s for r in stage2_results],
            "wind_divergence_m2_s": [r.wind_divergence_m2_s for r in stage2_results],
            "wind_constancy_min": [r.wind_constancy_min for r in stage2_results],
            "wind_dir_delta_deg": wind_dir_delta_deg,
            "wind_speed_rel_delta": wind_speed_rel_delta,
            "mblh_rel_delta": mblh_rel_delta,
            "max_co2_sensor_diff_ppm": [r.max_co2_sensor_diff_ppm for r in stage2_results],
            "n_co2_flagged_asvs": [len(r.flagged_co2_asv_ids) for r in stage2_results],
            "max_asv_drift_speed_m_s": [r.max_asv_drift_speed_m_s for r in stage2_results],
        },
        schema=_HOURLY_SCHEMA,
    )

    edge_rows: list[dict[str, object]] = []
    asv_detail_rows: list[dict[str, object]] = []
    for ts, r in zip(periods_list, stage2_results, strict=True):
        for e in r.edges:
            x1_km = e.x1_m * 1e-3
            y1_km = e.y1_m * 1e-3
            x2_km = e.x2_m * 1e-3
            y2_km = e.y2_m * 1e-3
            edge_rows.append(
                {
                    "period_start": ts,
                    "edge_idx": e.edge_idx,
                    "start_asv_id": e.start_asv_id,
                    "end_asv_id": e.end_asv_id,
                    "x1_km": x1_km,
                    "y1_km": y1_km,
                    "x2_km": x2_km,
                    "y2_km": y2_km,
                    "mid_x_km": 0.5 * (x1_km + x2_km),
                    "mid_y_km": 0.5 * (y1_km + y2_km),
                    "normal_x": e.normal_x,
                    "normal_y": e.normal_y,
                    "edge_length_km": e.edge_length_m * 1e-3,
                    "edge_co2_ppm": e.edge_co2_ppm,
                    "edge_co2_anomaly_ppm": e.edge_co2_anomaly_ppm,
                    "edge_wind_u": e.edge_wind_u,
                    "edge_wind_v": e.edge_wind_v,
                    "wind_dot_normal_m_s": e.wind_dot_normal_m_s,
                    "edge_outward_flux_tco2_hr": e.edge_outward_flux_tco2_hr,
                }
            )
        for d in r.asv_details:
            asv_detail_rows.append(
                {
                    "period_start": ts,
                    "asv_id": d.asv_id,
                    "latitude": d.latitude,
                    "longitude": d.longitude,
                    "x_km": d.x_m * 1e-3,
                    "y_km": d.y_m * 1e-3,
                    "on_hull": d.on_hull,
                    "co2_1": d.co2_1,
                    "co2_2": d.co2_2,
                    "co2_ppm": d.co2_ppm,
                    "co2_sensor_diff_ppm": d.co2_sensor_diff_ppm,
                    "co2_sensor_flagged": d.co2_sensor_flagged,
                    "u_wind": d.u_wind,
                    "v_wind": d.v_wind,
                    "wind_constancy": d.wind_constancy,
                    "asv_drift_speed_m_s": d.asv_drift_speed_m_s,
                }
            )

    edges_df = pl.DataFrame(edge_rows, schema=_EDGES_SCHEMA)
    aligned_asvs_df = pl.DataFrame(asv_detail_rows, schema=_ALIGNED_ASVS_SCHEMA)
    return ACPFResult(hourly=hourly_df, edges=edges_df, aligned_asvs=aligned_asvs_df)


def compute_acpf(
    asv_telemetry: pl.DataFrame,
    met_series: pl.DataFrame,
    *,
    config: ACPFConfig = _DEFAULT_CONFIG,
    period_starts: pl.Series | None = None,
) -> ACPFResult:
    """Run the full 3-stage ACPF pipeline on raw asynchronous ASV telemetry and met inputs."""
    grid = _infer_period_starts(asv_telemetry, config) if period_starts is None else period_starts
    if grid.len() == 0:
        return _empty_result()

    aligned_asvs = resample_asv_telemetry(asv_telemetry, grid, config=config)
    aligned_met = resample_met_series(met_series, grid, config=config)
    return compute_acpf_from_aligned(aligned_asvs, aligned_met, config=config)


def summarize_window(
    hourly_df: pl.DataFrame,
    window_duration: timedelta,
    *,
    max_error_bar: ErrorBar = ErrorBar.UNSTABLE_WIND_OR_MBLH,
) -> pl.DataFrame:
    """Compute trailing rolling-window summaries over duration ``window_duration`` (tau_win).

    Aggregates only periods satisfying ``error_bar <= max_error_bar`` with finite
    flux values (doc/acpf_math.md Section 6.2), while reporting ``worst_error_bar``
    across all periods in each trailing window ``(t - window_duration, t]``.
    """
    if not isinstance(window_duration, timedelta) or window_duration <= timedelta(0):
        raise ValueError(f"window_duration must be a positive timedelta, got {window_duration!r}.")
    if not isinstance(max_error_bar, ErrorBar):
        raise TypeError(f"max_error_bar must be an ErrorBar member, got {type(max_error_bar).__name__}.")

    if hourly_df.height == 0:
        return pl.DataFrame(schema=_WINDOW_SUMMARY_SCHEMA)

    df = _normalize_ts_col(hourly_df, "period_start").sort("period_start")
    if df.schema["error_bar"] != ERROR_BAR_DTYPE:
        df = df.with_columns(pl.col("error_bar").cast(ERROR_BAR_DTYPE))

    valid_lat = (pl.col("error_bar") <= max_error_bar) & pl.col("lateral_cdrr_tco2_hr").is_finite()
    valid_stor = valid_lat & pl.col("storage_flux_tco2_hr").is_finite()
    valid_net = valid_lat & pl.col("storage_adjusted_cdrr_tco2_hr").is_finite()
    valid_net_km2 = valid_lat & pl.col("storage_adjusted_cdrr_tco2_km2_hr").is_finite()

    return (
        df.rolling(index_column="period_start", period=window_duration)
        .agg(
            pl.len().cast(pl.Int64).alias("n_periods_total"),
            valid_lat.sum().cast(pl.Int64).alias("n_periods_valid"),
            pl.col("lateral_cdrr_tco2_hr")
            .filter(valid_lat)
            .mean()
            .fill_null(float("nan"))
            .alias("mean_lateral_cdrr_tco2_hr"),
            pl.col("storage_flux_tco2_hr")
            .filter(valid_stor)
            .mean()
            .fill_null(float("nan"))
            .alias("mean_storage_flux_tco2_hr"),
            pl.col("storage_adjusted_cdrr_tco2_hr")
            .filter(valid_net)
            .mean()
            .fill_null(float("nan"))
            .alias("mean_storage_adjusted_cdrr_tco2_hr"),
            pl.col("storage_adjusted_cdrr_tco2_km2_hr")
            .filter(valid_net_km2)
            .mean()
            .fill_null(float("nan"))
            .alias("mean_storage_adjusted_cdrr_tco2_km2_hr"),
            pl.col("error_bar").max().alias("worst_error_bar"),
        )
        .cast(_WINDOW_SUMMARY_SCHEMA)
    )
