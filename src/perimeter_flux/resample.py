"""Stage 1: Asynchronous ASV telemetry and meteorological temporal alignment.

Mirrors Section 4 of ``doc/acpf_math.md``:

Time-Weighted Period Average (Section 4.1)
------------------------------------------
For each ASV ``i`` and scalar channel ``q(t)`` (3D ECEF ``X, Y, Z``, wind
``u, v``, scalar wind speed ``s = sqrt(u^2 + v^2)``, and dual CO2 channels
``c_1, c_2``), exact duplicate timestamps are averaged, a pluggable interpolant
``q_hat_i(t)`` is fit, and each period ``[t_k, t_k + dt)`` is integrated::

    q_bar_{i,k} = (1 / dt) * integral_{t_k}^{t_k + dt} q_hat_i(t) dt

Guards & Diagnostics (Sections 4.2 - 4.3)
-----------------------------------------
- No extrapolation: valid iff ``t_obs[0] <= t_k`` and ``t_k + dt <= t_obs[-1]``.
- Max gap guard: every consecutive sample interval overlapping ``(t_k, t_k + dt)``
  must satisfy ``t_{m+1} - t_m <= max_interp_gap``.
- Wind vector constancy:
  ``r_const = ||w_bar||_2 / s_bar`` if ``s_bar > MIN_WIND_SPEED_EPS`` else ``1.0``.
- Inter-period ASV drift speed: ``||X_{i,k} - X_{i,k-1}||_2 / dt`` across adjacent
  valid periods separated by ``dt``.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import (
    Akima1DInterpolator,
    BSpline,
    PPoly,
    PchipInterpolator,
    make_interp_spline,
)

from perimeter_flux.config import ACPFConfig, InterpolationMethod, MIN_WIND_SPEED_EPS
from perimeter_flux.geodesy import ecef_to_latlon, latlon_to_ecef

_REQUIRED_ASV_COLUMNS: tuple[str, ...] = (
    "asv_id",
    "timestamp",
    "latitude",
    "longitude",
    "co2_1",
    "co2_2",
    "u_wind",
    "v_wind",
)
_OPTIONAL_MET_COLUMNS: tuple[str, ...] = (
    "t_air_k",
    "p_atm_pa",
    "u_wind_ext",
    "v_wind_ext",
)
_ALIGNED_ASV_SCHEMA: dict[str, pl.DataType] = {
    "period_start": pl.Datetime("us", "UTC"),
    "asv_id": pl.String,
    "ecef_x_m": pl.Float64,
    "ecef_y_m": pl.Float64,
    "ecef_z_m": pl.Float64,
    "latitude": pl.Float64,
    "longitude": pl.Float64,
    "u_wind": pl.Float64,
    "v_wind": pl.Float64,
    "wind_constancy": pl.Float64,
    "co2_1": pl.Float64,
    "co2_2": pl.Float64,
    "asv_drift_speed_m_s": pl.Float64,
}


def _to_utc_us_series(series: pl.Series, name: str | None = None) -> pl.Series:
    """Normalize a Polars Datetime series to microsecond UTC precision."""
    if not isinstance(series.dtype, pl.Datetime):
        raise TypeError(f"Expected a Datetime series, got {series.dtype}.")
    if series.dtype.time_zone is None:
        out = series.cast(pl.Datetime("us")).dt.replace_time_zone("UTC")
    else:
        out = series.dt.convert_time_zone("UTC").cast(pl.Datetime("us", "UTC"))
    return out.alias(name) if name is not None else out


def _series_to_float64(series: pl.Series) -> NDArray[np.float64]:
    """Convert a numeric Polars Series to a float64 NumPy array with nulls as NaN."""
    return series.cast(pl.Float64).fill_null(float("nan")).to_numpy()


def build_interpolator(
    t_sec: ArrayLike,
    values: ArrayLike,
    method: InterpolationMethod,
) -> BSpline | PPoly:
    """Build a SciPy spline/piecewise-polynomial interpolant supporting ``.integrate(a, b)``.

    - ``LINEAR``: ``make_interp_spline(t_sec, values, k=1)``
    - ``PCHIP``: ``PchipInterpolator(t_sec, values, extrapolate=False)``
    - ``AKIMA``: ``Akima1DInterpolator(t_sec, values, extrapolate=False)`` for >= 5
      points, falling back to ``PchipInterpolator`` (3-4 points) or linear (2 points).
    """
    t = np.asarray(t_sec, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    if t.ndim != 1 or y.ndim != 1 or t.shape != y.shape:
        raise ValueError(
            f"t_sec and values must have the same 1D shape, got {t.shape} and {y.shape}."
        )
    if t.size < 2:
        raise ValueError(f"Interpolation requires at least 2 points, got {t.size}.")
    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(y))):
        raise ValueError("t_sec and values must contain only finite numbers.")
    if np.any(np.diff(t) <= 0.0):
        raise ValueError("t_sec must be strictly increasing.")

    if method is InterpolationMethod.LINEAR:
        return make_interp_spline(t, y, k=1)
    if method is InterpolationMethod.PCHIP:
        return PchipInterpolator(t, y, extrapolate=False)
    if method is InterpolationMethod.AKIMA:
        if t.size >= 5:
            return Akima1DInterpolator(t, y, extrapolate=False)
        if t.size >= 3:
            return PchipInterpolator(t, y, extrapolate=False)
        return make_interp_spline(t, y, k=1)

    raise TypeError(f"Unsupported interpolation method: {method!r}.")


def interval_average_channel(
    t_obs_sec: ArrayLike,
    values: ArrayLike,
    period_starts_sec: ArrayLike,
    dt_sec: float,
    method: InterpolationMethod = InterpolationMethod.LINEAR,
    max_gap_sec: float = 10_800.0,
) -> NDArray[np.float64]:
    """Compute time-weighted interval averages over ``[t_k, t_k + dt_sec)``.

    Drops non-finite samples, sorts timestamps ascending, averages exact
    duplicate timestamps, enforces strict no-extrapolation and ``max_gap_sec``
    guards, and integrates the chosen interpolant over each valid period.
    """
    if not math.isfinite(dt_sec) or dt_sec <= 0.0:
        raise ValueError(f"dt_sec must be finite and > 0, got {dt_sec!r}.")
    if not math.isfinite(max_gap_sec) or max_gap_sec <= 0.0:
        raise ValueError(f"max_gap_sec must be finite and > 0, got {max_gap_sec!r}.")

    t_raw = np.asarray(t_obs_sec, dtype=np.float64)
    v_raw = np.asarray(values, dtype=np.float64)
    p_starts = np.asarray(period_starts_sec, dtype=np.float64)
    if t_raw.ndim != 1 or v_raw.ndim != 1 or t_raw.shape != v_raw.shape:
        raise ValueError(
            f"t_obs_sec and values must have the same 1D shape, got {t_raw.shape} and {v_raw.shape}."
        )
    if p_starts.ndim != 1:
        raise ValueError(f"period_starts_sec must be 1D, got shape {p_starts.shape}.")

    n_periods = p_starts.size
    if n_periods == 0:
        return np.empty((0,), dtype=np.float64)

    result = np.full(n_periods, np.nan, dtype=np.float64)

    # Filter non-finite samples before sorting and deduplicating
    finite_mask = np.isfinite(t_raw) & np.isfinite(v_raw)
    t_fin = t_raw[finite_mask]
    v_fin = v_raw[finite_mask]
    if t_fin.size < 2:
        return result

    order = np.argsort(t_fin, kind="mergesort")
    t_sorted = t_fin[order]
    v_sorted = v_fin[order]

    # Average exact duplicate timestamps
    t_unique, inv, counts = np.unique(t_sorted, return_inverse=True, return_counts=True)
    if t_unique.size < 2:
        return result
    if t_unique.size == t_sorted.size:
        v_unique = v_sorted
    else:
        v_unique = np.bincount(inv, weights=v_sorted) / counts

    # Shift time origin to t_unique[0] for spline numerical conditioning
    t0 = float(t_unique[0])
    t_rel = t_unique - t0
    a_rel = p_starts - t0
    b_rel = a_rel + dt_sec

    in_span = np.isfinite(a_rel) & (a_rel >= 0.0) & (b_rel <= t_rel[-1])
    if not np.any(in_span):
        return result

    # Vectorized max-gap guard: check every raw interval [t_m, t_{m+1}] strictly
    # overlapping (a_k, b_k), i.e., t_m < b_k and t_{m+1} > a_k.
    bad_gaps = np.diff(t_rel) > max_gap_sec
    bad_prefix = np.concatenate(([0], np.cumsum(bad_gaps, dtype=np.int64)))

    candidate_idx = np.flatnonzero(in_span)
    a_cand = a_rel[candidate_idx]
    b_cand = b_rel[candidate_idx]
    idx_start = np.searchsorted(t_rel, a_cand, side="right") - 1
    idx_end = np.searchsorted(t_rel, b_cand, side="left") - 1
    gap_ok = (bad_prefix[idx_end + 1] - bad_prefix[idx_start]) == 0
    valid_idx = candidate_idx[gap_ok]
    if valid_idx.size == 0:
        return result

    interp = build_interpolator(t_rel, v_unique, method)
    for k in valid_idx:
        result[k] = float(interp.integrate(float(a_rel[k]), float(b_rel[k]))) / dt_sec

    return result


def resample_asv_telemetry(
    asv_telemetry: pl.DataFrame,
    period_starts: pl.Series,
    config: ACPFConfig = ACPFConfig(),
) -> pl.DataFrame:
    """Align asynchronous per-ASV telemetry onto the discrete ``period_starts`` grid.

    Interpolates sea-level 3D WGS 84 ECEF positions (IDL-safe), wind ``(u, v)``
    and scalar speed ``s`` (yielding ``wind_constancy``), and dual CO2 channels
    ``(co2_1, co2_2)`` via time-weighted interval integration. Omits ASV-period
    rows lacking valid position, valid wind, or at least one valid CO2 sensor.
    """
    missing = [c for c in _REQUIRED_ASV_COLUMNS if c not in asv_telemetry.columns]
    if missing:
        raise ValueError(f"asv_telemetry is missing required columns: {missing}.")

    periods_utc = _to_utc_us_series(period_starts, name="period_start")
    n_periods = periods_utc.len()
    if n_periods == 0 or asv_telemetry.height == 0:
        return pl.DataFrame(schema=_ALIGNED_ASV_SCHEMA)

    periods_us = periods_utc.dt.epoch("us").to_numpy()
    if n_periods > 1 and np.any(np.diff(periods_us) <= 0):
        raise ValueError("period_starts must be strictly increasing.")

    dt_sec = config.time_interval.total_seconds()
    dt_us = int(round(dt_sec * 1_000_000))
    max_gap_sec = config.max_interp_gap.total_seconds()
    method = config.interpolation_method

    ref_us = int(periods_us[0])
    period_starts_sec = (periods_us - ref_us) / 1_000_000.0
    adjacent_prev = np.zeros(n_periods, dtype=bool)
    if n_periods > 1:
        adjacent_prev[1:] = np.diff(periods_us) == dt_us

    telemetry = asv_telemetry.with_columns(_to_utc_us_series(asv_telemetry["timestamp"], "timestamp"))

    frames: list[pl.DataFrame] = []
    for asv_id in sorted(telemetry["asv_id"].unique().to_list()):
        sub = telemetry.filter(pl.col("asv_id") == asv_id)
        t_obs_sec = (sub["timestamp"].dt.epoch("us").to_numpy() - ref_us) / 1_000_000.0

        lat_raw = _series_to_float64(sub["latitude"])
        lon_raw = _series_to_float64(sub["longitude"])
        pos_finite = np.isfinite(t_obs_sec) & np.isfinite(lat_raw) & np.isfinite(lon_raw)

        ecef_x = np.full(n_periods, np.nan, dtype=np.float64)
        ecef_y = np.full(n_periods, np.nan, dtype=np.float64)
        ecef_z = np.full(n_periods, np.nan, dtype=np.float64)
        if np.count_nonzero(pos_finite) >= 2:
            raw_xyz = latlon_to_ecef(lat_raw[pos_finite], lon_raw[pos_finite])
            t_pos = t_obs_sec[pos_finite]
            ecef_x = interval_average_channel(
                t_pos, raw_xyz[:, 0], period_starts_sec, dt_sec, method, max_gap_sec
            )
            ecef_y = interval_average_channel(
                t_pos, raw_xyz[:, 1], period_starts_sec, dt_sec, method, max_gap_sec
            )
            ecef_z = interval_average_channel(
                t_pos, raw_xyz[:, 2], period_starts_sec, dt_sec, method, max_gap_sec
            )

        pos_ok = np.isfinite(ecef_x) & np.isfinite(ecef_y) & np.isfinite(ecef_z)
        lat_bar = np.full(n_periods, np.nan, dtype=np.float64)
        lon_bar = np.full(n_periods, np.nan, dtype=np.float64)
        ecef_sea = np.full((n_periods, 3), np.nan, dtype=np.float64)
        if np.any(pos_ok):
            valid_xyz = np.column_stack((ecef_x[pos_ok], ecef_y[pos_ok], ecef_z[pos_ok]))
            lat_v, lon_v = ecef_to_latlon(valid_xyz)
            lat_bar[pos_ok] = lat_v
            lon_bar[pos_ok] = lon_v
            ecef_sea[pos_ok] = latlon_to_ecef(lat_v, lon_v)

        # Inter-period ASV drift speed across adjacent dt steps
        drift_speed = np.full(n_periods, np.nan, dtype=np.float64)
        if n_periods > 1:
            drift_valid = adjacent_prev[1:] & pos_ok[1:] & pos_ok[:-1]
            if np.any(drift_valid):
                disp = ecef_sea[1:][drift_valid] - ecef_sea[:-1][drift_valid]
                drift_speed[1:][drift_valid] = np.linalg.norm(disp, axis=1) / dt_sec

        # Wind vector (u, v) and scalar speed s = hypot(u, v) on rows where both u & v are finite
        u_raw = _series_to_float64(sub["u_wind"])
        v_raw = _series_to_float64(sub["v_wind"])
        wind_finite = np.isfinite(t_obs_sec) & np.isfinite(u_raw) & np.isfinite(v_raw)

        u_bar = np.full(n_periods, np.nan, dtype=np.float64)
        v_bar = np.full(n_periods, np.nan, dtype=np.float64)
        s_bar = np.full(n_periods, np.nan, dtype=np.float64)
        if np.count_nonzero(wind_finite) >= 2:
            t_w = t_obs_sec[wind_finite]
            u_w = u_raw[wind_finite]
            v_w = v_raw[wind_finite]
            s_w = np.hypot(u_w, v_w)
            u_bar = interval_average_channel(t_w, u_w, period_starts_sec, dt_sec, method, max_gap_sec)
            v_bar = interval_average_channel(t_w, v_w, period_starts_sec, dt_sec, method, max_gap_sec)
            s_bar = interval_average_channel(t_w, s_w, period_starts_sec, dt_sec, method, max_gap_sec)

        wind_ok = np.isfinite(u_bar) & np.isfinite(v_bar) & np.isfinite(s_bar)
        constancy = np.full(n_periods, np.nan, dtype=np.float64)
        if np.any(wind_ok):
            vec_mag = np.hypot(u_bar[wind_ok], v_bar[wind_ok])
            s_valid = s_bar[wind_ok]
            ratio = np.where(
                s_valid > MIN_WIND_SPEED_EPS,
                np.divide(vec_mag, s_valid, out=np.ones_like(vec_mag), where=s_valid > MIN_WIND_SPEED_EPS),
                1.0,
            )
            constancy[wind_ok] = np.clip(ratio, 0.0, 1.0)

        co2_1_bar = interval_average_channel(
            t_obs_sec, _series_to_float64(sub["co2_1"]), period_starts_sec, dt_sec, method, max_gap_sec
        )
        co2_2_bar = interval_average_channel(
            t_obs_sec, _series_to_float64(sub["co2_2"]), period_starts_sec, dt_sec, method, max_gap_sec
        )
        co2_ok = np.isfinite(co2_1_bar) | np.isfinite(co2_2_bar)

        keep = pos_ok & wind_ok & co2_ok
        if not np.any(keep):
            continue

        keep_idx = np.flatnonzero(keep)
        frames.append(
            pl.DataFrame(
                {
                    "period_start": periods_utc.gather(keep_idx.tolist()),
                    "asv_id": [asv_id] * keep_idx.size,
                    "ecef_x_m": ecef_sea[keep_idx, 0],
                    "ecef_y_m": ecef_sea[keep_idx, 1],
                    "ecef_z_m": ecef_sea[keep_idx, 2],
                    "latitude": lat_bar[keep_idx],
                    "longitude": lon_bar[keep_idx],
                    "u_wind": u_bar[keep_idx],
                    "v_wind": v_bar[keep_idx],
                    "wind_constancy": constancy[keep_idx],
                    "co2_1": co2_1_bar[keep_idx],
                    "co2_2": co2_2_bar[keep_idx],
                    "asv_drift_speed_m_s": drift_speed[keep_idx],
                },
                schema=_ALIGNED_ASV_SCHEMA,
            )
        )

    if not frames:
        return pl.DataFrame(schema=_ALIGNED_ASV_SCHEMA)

    return pl.concat(frames, how="vertical").sort(["period_start", "asv_id"])


def resample_met_series(
    met_df: pl.DataFrame,
    period_starts: pl.Series,
    config: ACPFConfig = ACPFConfig(),
) -> pl.DataFrame:
    """Align meteorological series (``mblh_m`` and optional met columns) onto ``period_starts``.

    Matches directly when ``met_df`` is already provided strictly on the ``dt``
    period grid covering all requested ``period_starts``; otherwise aligns every
    channel via ``interval_average_channel`` with the configured interpolant and
    ``max_interp_gap`` guard (doc/acpf_math.md Section 4.4).
    """
    for req in ("timestamp", "mblh_m"):
        if req not in met_df.columns:
            raise ValueError(f"met_df is missing required column '{req}'.")

    periods_utc = _to_utc_us_series(period_starts, name="period_start")
    value_cols = ["mblh_m", *[c for c in _OPTIONAL_MET_COLUMNS if c in met_df.columns]]
    out_schema: dict[str, pl.DataType] = {
        "period_start": pl.Datetime("us", "UTC"),
        **{c: pl.Float64 for c in value_cols},
    }

    n_periods = periods_utc.len()
    if n_periods == 0:
        return pl.DataFrame(schema=out_schema)

    periods_us = periods_utc.dt.epoch("us").to_numpy()
    if n_periods > 1 and np.any(np.diff(periods_us) <= 0):
        raise ValueError("period_starts must be strictly increasing.")

    if met_df.height == 0:
        nan_col = np.full(n_periods, np.nan, dtype=np.float64)
        return pl.DataFrame(
            {"period_start": periods_utc, **{c: nan_col.copy() for c in value_cols}},
            schema=out_schema,
        )

    dt_sec = config.time_interval.total_seconds()
    dt_us = int(round(dt_sec * 1_000_000))
    max_gap_sec = config.max_interp_gap.total_seconds()
    method = config.interpolation_method

    met_utc = met_df.with_columns(_to_utc_us_series(met_df["timestamp"], "timestamp"))
    met_ts_us = met_utc["timestamp"].dt.epoch("us").to_numpy()
    ref_us = int(periods_us[0])

    # Direct match when all met timestamps fall on the dt grid and cover every period_start
    on_grid = bool(np.all((met_ts_us - ref_us) % dt_us == 0))
    covers_all = set(periods_us.tolist()).issubset(set(met_ts_us.tolist()))
    if on_grid and covers_all:
        deduped = met_utc.group_by("timestamp").agg(
            [pl.col(c).cast(pl.Float64).fill_nan(None).mean().fill_null(float("nan")) for c in value_cols]
        )
        grid = pl.DataFrame({"period_start": periods_utc})
        joined = grid.join(
            deduped.rename({"timestamp": "period_start"}),
            on="period_start",
            how="left",
            maintain_order="left",
        )
        return joined.select(["period_start", *value_cols]).cast(out_schema)

    t_obs_sec = (met_ts_us - ref_us) / 1_000_000.0
    period_starts_sec = (periods_us - ref_us) / 1_000_000.0
    cols: dict[str, object] = {"period_start": periods_utc}
    for col in value_cols:
        cols[col] = interval_average_channel(
            t_obs_sec,
            _series_to_float64(met_utc[col]),
            period_starts_sec,
            dt_sec,
            method=method,
            max_gap_sec=max_gap_sec,
        )
    return pl.DataFrame(cols, schema=out_schema)
