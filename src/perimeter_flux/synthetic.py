"""Synthetic multi-ASV telemetry scenario generators for ACPF testing and exploration.

Generates realistic synchronous and asynchronous telemetry for `"Oshen1"`..`"Oshen6"`
around an Acapulco-scale (~200 km) ocean eddy (or any custom vertex set such as
``IDL_EQUATOR_EDDY_VERTICES``), returning ``(asv_telemetry_df, met_df, metadata)``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl
from numpy.typing import NDArray
from scipy.spatial import ConvexHull

from perimeter_flux.config import (
    DEFAULT_P_ATM_PA,
    DEFAULT_T_AIR_K,
    ErrorBar,
    KG_PER_S_TO_T_PER_HR,
)
from perimeter_flux.flux import co2_mass_per_ppm_kg_m3, effective_mblh_m
from perimeter_flux.geodesy import (
    ecef_origin_from_points,
    ecef_to_enu,
    ecef_to_latlon,
    enu_rotation_matrix,
    latlon_to_ecef,
)
from perimeter_flux.sample_data import ACAPULCO_EDDY_VERTICES

SCENARIO_NAMES: tuple[str, ...] = (
    "steady_drawdown",
    "async_sailboat_motion",
    "synoptic_front_storage",
    "sharp_front_7asv",
    "unstable_wind_and_mblh",
    "dual_co2_sensor_fault",
    "degraded_geometry_and_outages",
    "combined_showcase",
)

_DEFAULT_START = datetime(2026, 6, 1, 0, 0, tzinfo=timezone.utc)

_ASV_SCHEMA: dict[str, pl.DataType] = {
    "asv_id": pl.String,
    "timestamp": pl.Datetime("us", "UTC"),
    "latitude": pl.Float64,
    "longitude": pl.Float64,
    "co2_1": pl.Float64,
    "co2_2": pl.Float64,
    "u_wind": pl.Float64,
    "v_wind": pl.Float64,
}

_MET_SCHEMA: dict[str, pl.DataType] = {
    "timestamp": pl.Datetime("us", "UTC"),
    "mblh_m": pl.Float64,
    "t_air_k": pl.Float64,
    "p_atm_pa": pl.Float64,
}


@dataclass(frozen=True, slots=True)
class ScenarioMetadata:
    """Ground-truth and expected QA characteristics for a synthetic ACPF scenario."""

    name: str
    description: str
    expected_cdrr_tco2_hr: float | None
    expected_error_bars: tuple[ErrorBar, ...]
    start_time: datetime
    n_hours: int
    seed: int


def _prepare_frame(
    vertices: Sequence[tuple[float, float]],
) -> tuple[NDArray[np.float64], float, float, NDArray[np.float64], NDArray[np.float64]]:
    """Compute base local EN meters and ENU frame parameters for an eddy vertex ring."""
    lats = np.array([v[0] for v in vertices], dtype=np.float64)
    lons = np.array([v[1] for v in vertices], dtype=np.float64)
    xyz = latlon_to_ecef(lats, lons)
    lat0_rad, lon0_rad, origin_xyz = ecef_origin_from_points(xyz)
    enu = ecef_to_enu(xyz, lat0_rad, lon0_rad, origin_xyz)
    rot = enu_rotation_matrix(lat0_rad, lon0_rad)
    return np.ascontiguousarray(enu[:, :2]), lat0_rad, lon0_rad, origin_xyz, rot


def _en_to_latlon(
    xy_m: NDArray[np.float64],
    origin_xyz: NDArray[np.float64],
    rot: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Map local 2D East-North meters back to WGS 84 (lat_deg, lon_deg) via ECEF."""
    enu = np.column_stack((xy_m, np.zeros(xy_m.shape[0], dtype=np.float64)))
    # Since enu = (xyz - origin) @ rot.T and rot is orthonormal, xyz = origin + enu @ rot
    xyz = origin_xyz + enu @ rot
    return ecef_to_latlon(xyz)


def _analytical_linear_cdrr(
    xy_m: NDArray[np.float64],
    u_wind: float,
    v_wind: float,
    gx: float,
    gy: float,
    mblh_m: float,
    t_air_k: float = DEFAULT_T_AIR_K,
    p_atm_pa: float = DEFAULT_P_ATM_PA,
) -> float:
    area_m2 = float(ConvexHull(xy_m).volume)
    kappa = co2_mass_per_ppm_kg_m3(t_air_k, p_atm_pa)
    h_eff = effective_mblh_m(mblh_m, t_air_k)
    return KG_PER_S_TO_T_PER_HR * kappa * h_eff * (-(gx * u_wind + gy * v_wind)) * area_m2


def _regular_times_sec(n_hours: int, step_minutes: float = 15.0) -> NDArray[np.float64]:
    n_steps = int(round(n_hours * 60.0 / step_minutes))
    return np.linspace(0.0, n_hours * 3600.0, n_steps + 1, dtype=np.float64)


def _async_times_sec(
    n_hours: int,
    rng: np.random.Generator,
    min_step_min: float = 5.0,
    max_step_min: float = 12.0,
) -> NDArray[np.float64]:
    """Generate strictly increasing irregular timestamps spanning [0, n_hours * 3600]."""
    end_sec = n_hours * 3600.0
    times = [0.0]
    curr = 0.0
    while True:
        step = float(rng.uniform(min_step_min * 60.0, max_step_min * 60.0))
        nxt = curr + step
        if nxt >= end_sec - 60.0:
            break
        times.append(nxt)
        curr = nxt
    times.append(end_sec)
    return np.array(times, dtype=np.float64)


def _hourly_met(
    start_time: datetime,
    mblh_values: Sequence[float],
    t_air_k: float = DEFAULT_T_AIR_K,
    p_atm_pa: float = DEFAULT_P_ATM_PA,
) -> pl.DataFrame:
    n = len(mblh_values)
    return pl.DataFrame(
        {
            "timestamp": [start_time + timedelta(hours=h) for h in range(n)],
            "mblh_m": [float(v) for v in mblh_values],
            "t_air_k": [t_air_k] * n,
            "p_atm_pa": [p_atm_pa] * n,
        },
        schema=_MET_SCHEMA,
    )


def _build_asv_df(rows: list[dict[str, object]]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=_ASV_SCHEMA).sort(["timestamp", "asv_id"])


def _scenario_steady_drawdown(
    start_time: datetime,
    seed: int,
    vertices: Sequence[tuple[float, float]],
) -> tuple[pl.DataFrame, pl.DataFrame, ScenarioMetadata]:
    n_hours = 12
    xy_m, _, _, _, _ = _prepare_frame(vertices)
    u_wind, v_wind = 6.5, -2.0
    gx, gy = -1.2e-5, 0.3e-5
    mblh = 950.0
    expected_cdrr = _analytical_linear_cdrr(xy_m, u_wind, v_wind, gx, gy, mblh)

    times_sec = _regular_times_sec(n_hours, step_minutes=15.0)
    rows: list[dict[str, object]] = []
    for idx, (lat, lon) in enumerate(vertices):
        c = 418.0 + gx * float(xy_m[idx, 0]) + gy * float(xy_m[idx, 1])
        asv_id = f"Oshen{idx + 1}"
        for t_s in times_sec:
            rows.append(
                {
                    "asv_id": asv_id,
                    "timestamp": start_time + timedelta(seconds=float(t_s)),
                    "latitude": float(lat),
                    "longitude": float(lon),
                    "co2_1": c,
                    "co2_2": c,
                    "u_wind": u_wind,
                    "v_wind": v_wind,
                }
            )

    met_df = _hourly_met(start_time, [mblh] * n_hours)
    meta = ScenarioMetadata(
        name="steady_drawdown",
        description="Steady uniform wind, constant MBLH, and linear spatial CO2 drawdown gradient.",
        expected_cdrr_tco2_hr=expected_cdrr,
        expected_error_bars=(ErrorBar.STABLE,),
        start_time=start_time,
        n_hours=n_hours,
        seed=seed,
    )
    return _build_asv_df(rows), met_df, meta


def _scenario_async_sailboat_motion(
    start_time: datetime,
    seed: int,
    vertices: Sequence[tuple[float, float]],
) -> tuple[pl.DataFrame, pl.DataFrame, ScenarioMetadata]:
    n_hours = 12
    rng = np.random.default_rng(seed)
    xy_base, _, _, origin_xyz, rot = _prepare_frame(vertices)
    u_mean, v_mean = 6.5, -2.0
    speed_mean = math.hypot(u_mean, v_mean)
    # Unit vector perpendicular to mean wind for cross-wind yaw/bobbing noise
    perp_u, perp_v = -v_mean / speed_mean, u_mean / speed_mean

    gx, gy = -1.2e-5, 0.3e-5
    mblh = 950.0
    expected_cdrr = _analytical_linear_cdrr(xy_base, u_mean, v_mean, gx, gy, mblh)

    # Anticyclonic (clockwise) orbital swirl at ~0.5 m/s around eddy radius (~110 km)
    mean_radius_m = float(np.mean(np.linalg.norm(xy_base, axis=1)))
    omega_rad_s = -0.5 / mean_radius_m

    rows: list[dict[str, object]] = []
    for idx in range(len(vertices)):
        asv_id = f"Oshen{idx + 1}"
        t_sec = _async_times_sec(n_hours, rng, min_step_min=5.0, max_step_min=12.0)
        phase = float(rng.uniform(0.0, 2.0 * math.pi))
        bob_period_sec = float(rng.uniform(14.0 * 60.0, 22.0 * 60.0))

        # Rotate base vertex slowly around eddy center
        angles = omega_rad_s * t_sec
        cos_a = np.cos(angles)
        sin_a = np.sin(angles)
        x_t = xy_base[idx, 0] * cos_a - xy_base[idx, 1] * sin_a
        y_t = xy_base[idx, 0] * sin_a + xy_base[idx, 1] * cos_a
        lats_t, lons_t = _en_to_latlon(np.column_stack((x_t, y_t)), origin_xyz, rot)

        bob = 2.8 * np.sin(2.0 * math.pi * t_sec / bob_period_sec + phase)
        u_t = u_mean + bob * perp_u
        v_t = v_mean + bob * perp_v
        c_true = 418.0 + gx * x_t + gy * y_t
        c1_t = c_true + rng.normal(0.0, 0.05, size=t_sec.size)
        c2_t = c_true + rng.normal(0.0, 0.05, size=t_sec.size)

        for k in range(t_sec.size):
            rows.append(
                {
                    "asv_id": asv_id,
                    "timestamp": start_time + timedelta(seconds=float(t_sec[k])),
                    "latitude": float(lats_t[k]),
                    "longitude": float(lons_t[k]),
                    "co2_1": float(c1_t[k]),
                    "co2_2": float(c2_t[k]),
                    "u_wind": float(u_t[k]),
                    "v_wind": float(v_t[k]),
                }
            )

    met_df = _hourly_met(start_time, [mblh] * n_hours)
    meta = ScenarioMetadata(
        name="async_sailboat_motion",
        description="Irregular 5-12 min async timestamps, sailboat turning/bobbing wind noise, and ~0.5 m/s orbital drift.",
        expected_cdrr_tco2_hr=expected_cdrr,
        expected_error_bars=(ErrorBar.STABLE,),
        start_time=start_time,
        n_hours=n_hours,
        seed=seed,
    )
    return _build_asv_df(rows), met_df, meta


def _scenario_synoptic_front_storage(
    start_time: datetime,
    seed: int,
    vertices: Sequence[tuple[float, float]],
) -> tuple[pl.DataFrame, pl.DataFrame, ScenarioMetadata]:
    n_hours = 24
    xy_m, _, _, _, _ = _prepare_frame(vertices)
    u_wind, v_wind = 8.0, 0.0
    mblh = 1000.0
    t_center_sec = 12.0 * 3600.0
    front_width_m = 200_000.0  # Synoptic-scale +2.0 ppm front sweeping across the eddy

    times_sec = _regular_times_sec(n_hours, step_minutes=15.0)
    rows: list[dict[str, object]] = []
    for idx, (lat, lon) in enumerate(vertices):
        x_i = float(xy_m[idx, 0])
        asv_id = f"Oshen{idx + 1}"
        for t_s in times_sec:
            # Passive advection solution c(x, t) = f(u * (t - t_center) - x) -> dc/dt + u dc/dx = 0
            arg = (u_wind * (float(t_s) - t_center_sec) - x_i) / front_width_m
            c = 418.0 + 1.0 * (1.0 + math.tanh(arg))
            rows.append(
                {
                    "asv_id": asv_id,
                    "timestamp": start_time + timedelta(seconds=float(t_s)),
                    "latitude": float(lat),
                    "longitude": float(lon),
                    "co2_1": c,
                    "co2_2": c,
                    "u_wind": u_wind,
                    "v_wind": v_wind,
                }
            )

    met_df = _hourly_met(start_time, [mblh] * n_hours)
    meta = ScenarioMetadata(
        name="synoptic_front_storage",
        description="Zero ocean uptake with a +2.0 ppm synoptic CO2 front advecting eastward across the eddy.",
        expected_cdrr_tco2_hr=0.0,
        expected_error_bars=(ErrorBar.STABLE,),
        start_time=start_time,
        n_hours=n_hours,
        seed=seed,
    )
    return _build_asv_df(rows), met_df, meta


def _scenario_sharp_front_7asv(
    start_time: datetime,
    seed: int,
    vertices: Sequence[tuple[float, float]],
) -> tuple[pl.DataFrame, pl.DataFrame, ScenarioMetadata]:
    """Sharp +2.0 ppm synoptic front (100 km width) with a 7th interior ASV at the eddy center."""
    n_hours = 20
    xy_hull, _, _, origin_xyz, rot = _prepare_frame(vertices)
    u_wind, v_wind = 8.0, 0.0
    mblh = 1000.0
    t_center_sec = 10.0 * 3600.0
    front_width_m = 100_000.0  # Sharp 100 km front where interior sampling helps box-mean CO2

    # 7th interior ASV placed at local EN origin (0, 0) -> sea-level WGS 84 centroid
    center_lat_arr, center_lon_arr = _en_to_latlon(np.zeros((1, 2), dtype=np.float64), origin_xyz, rot)
    all_vertices = [*vertices, (float(center_lat_arr[0]), float(center_lon_arr[0]))]
    all_x_m = [*xy_hull[:, 0].tolist(), 0.0]

    times_sec = _regular_times_sec(n_hours, step_minutes=15.0)
    rows: list[dict[str, object]] = []
    for idx, ((lat, lon), x_i) in enumerate(zip(all_vertices, all_x_m, strict=True)):
        asv_id = f"Oshen{idx + 1}"
        for t_s in times_sec:
            arg = (u_wind * (float(t_s) - t_center_sec) - x_i) / front_width_m
            c = 418.0 + 1.0 * (1.0 + math.tanh(arg))
            rows.append(
                {
                    "asv_id": asv_id,
                    "timestamp": start_time + timedelta(seconds=float(t_s)),
                    "latitude": float(lat),
                    "longitude": float(lon),
                    "co2_1": c,
                    "co2_2": c,
                    "u_wind": u_wind,
                    "v_wind": v_wind,
                }
            )

    met_df = _hourly_met(start_time, [mblh] * n_hours)
    meta = ScenarioMetadata(
        name="sharp_front_7asv",
        description=(
            "Sharp +2.0 ppm synoptic front (100 km width, zero ocean uptake) with a 7th interior "
            "ASV (Oshen7) at the eddy center to improve box-mean CO2 storage cancellation."
        ),
        expected_cdrr_tco2_hr=0.0,
        expected_error_bars=(ErrorBar.STABLE,),
        start_time=start_time,
        n_hours=n_hours,
        seed=seed,
    )
    return _build_asv_df(rows), met_df, meta


def _scenario_unstable_wind_and_mblh(
    start_time: datetime,
    seed: int,
    vertices: Sequence[tuple[float, float]],
) -> tuple[pl.DataFrame, pl.DataFrame, ScenarioMetadata]:
    n_hours = 16
    xy_m, _, _, _, _ = _prepare_frame(vertices)
    gx, gy = -1.0e-5, 0.2e-5

    # Sudden +60 deg wind shift at hour 5; rapid MBLH ramp 900 -> 1350 m at hour 10
    times_sec = _regular_times_sec(n_hours, step_minutes=15.0)
    rows: list[dict[str, object]] = []
    for idx, (lat, lon) in enumerate(vertices):
        c = 419.0 + gx * float(xy_m[idx, 0]) + gy * float(xy_m[idx, 1])
        asv_id = f"Oshen{idx + 1}"
        for t_s in times_sec:
            hr = float(t_s) / 3600.0
            heading_deg = 0.0 if hr < 5.0 else 60.0
            speed = 6.0
            u_w = speed * math.cos(math.radians(heading_deg))
            v_w = speed * math.sin(math.radians(heading_deg))
            rows.append(
                {
                    "asv_id": asv_id,
                    "timestamp": start_time + timedelta(seconds=float(t_s)),
                    "latitude": float(lat),
                    "longitude": float(lon),
                    "co2_1": c,
                    "co2_2": c,
                    "u_wind": u_w,
                    "v_wind": v_w,
                }
            )

    mblh_seq = [900.0 if h < 10 else 1350.0 for h in range(n_hours)]
    met_df = _hourly_met(start_time, mblh_seq)
    meta = ScenarioMetadata(
        name="unstable_wind_and_mblh",
        description="Sudden +60 deg wind direction shift at hour 5 and +50% morning MBLH ramp at hour 10.",
        expected_cdrr_tco2_hr=None,
        expected_error_bars=(ErrorBar.STABLE, ErrorBar.UNSTABLE_WIND_OR_MBLH),
        start_time=start_time,
        n_hours=n_hours,
        seed=seed,
    )
    return _build_asv_df(rows), met_df, meta


def _scenario_dual_co2_sensor_fault(
    start_time: datetime,
    seed: int,
    vertices: Sequence[tuple[float, float]],
) -> tuple[pl.DataFrame, pl.DataFrame, ScenarioMetadata]:
    n_hours = 16
    xy_m, _, _, _, _ = _prepare_frame(vertices)
    u_wind, v_wind = 6.5, -1.5
    gx, gy = -1.1e-5, 0.2e-5
    mblh = 950.0

    times_sec = _regular_times_sec(n_hours, step_minutes=15.0)
    rows: list[dict[str, object]] = []
    for idx, (lat, lon) in enumerate(vertices):
        c_base = 418.5 + gx * float(xy_m[idx, 0]) + gy * float(xy_m[idx, 1])
        asv_id = f"Oshen{idx + 1}"
        for t_s in times_sec:
            hr = float(t_s) / 3600.0
            c1 = c_base
            c2: float | None = c_base + 0.08
            if asv_id == "Oshen3":
                if 4.0 <= hr < 7.0:
                    # Gradual drift approaching warning threshold (+0.5 to +1.7 ppm)
                    c2 = c_base + 0.5 + 0.4 * (hr - 4.0)
                elif 7.0 <= hr < 10.5:
                    # +6.0 ppm sensor spike (> 2.0 ppm threshold)
                    c2 = c_base + 6.0
                elif 10.5 <= hr <= 14.0:
                    # 3.5-hour null dropout (> 3 hr max_interp_gap) -> single-sensor fallback
                    c2 = None
            rows.append(
                {
                    "asv_id": asv_id,
                    "timestamp": start_time + timedelta(seconds=float(t_s)),
                    "latitude": float(lat),
                    "longitude": float(lon),
                    "co2_1": c1,
                    "co2_2": c2,
                    "u_wind": u_wind,
                    "v_wind": v_wind,
                }
            )

    met_df = _hourly_met(start_time, [mblh] * n_hours)
    meta = ScenarioMetadata(
        name="dual_co2_sensor_fault",
        description="Oshen3 co2_2 drifts, spikes by +6 ppm, and drops to null while preserving 6-ASV hull.",
        expected_cdrr_tco2_hr=None,
        expected_error_bars=(ErrorBar.STABLE, ErrorBar.CO2_SENSOR_DISCREPANCY),
        start_time=start_time,
        n_hours=n_hours,
        seed=seed,
    )
    return _build_asv_df(rows), met_df, meta


def _scenario_degraded_geometry_and_outages(
    start_time: datetime,
    seed: int,
    vertices: Sequence[tuple[float, float]],
) -> tuple[pl.DataFrame, pl.DataFrame, ScenarioMetadata]:
    n_hours = 20
    xy_base, _, _, origin_xyz, rot = _prepare_frame(vertices)
    u_wind, v_wind = 6.0, -1.0
    gx, gy = -1.2e-5, 0.2e-5
    mblh = 950.0

    # Compute interior coordinates for Oshen5 (idx 4) near eddy centroid (15% of radial offset)
    xy_interior = (0.15 * xy_base[4]).reshape(1, 2)
    lat_int_arr, lon_int_arr = _en_to_latlon(xy_interior, origin_xyz, rot)
    lat_int, lon_int = float(lat_int_arr[0]), float(lon_int_arr[0])

    times_sec = _regular_times_sec(n_hours, step_minutes=15.0)
    rows: list[dict[str, object]] = []
    for idx, (lat, lon) in enumerate(vertices):
        asv_id = f"Oshen{idx + 1}"
        for t_s in times_sec:
            hr = float(t_s) / 3600.0
            # Outages longer than 3 hr max_interp_gap:
            # - Oshen3..6 absent for hr in (7.75, 16.0) -> hours 8..15 (8.25 hr gap)
            # - Oshen2 absent for hr in (11.75, 16.0) -> hours 12..15 (4.25 hr gap)
            if idx >= 2 and 7.75 < hr < 16.0:
                continue
            if idx == 1 and 11.75 < hr < 16.0:
                continue

            if idx == 4 and 4.0 <= hr <= 7.75:
                cur_lat, cur_lon = lat_int, lon_int
                x_cur, y_cur = float(xy_interior[0, 0]), float(xy_interior[0, 1])
            else:
                cur_lat, cur_lon = float(lat), float(lon)
                x_cur, y_cur = float(xy_base[idx, 0]), float(xy_base[idx, 1])

            c = 418.0 + gx * x_cur + gy * y_cur
            rows.append(
                {
                    "asv_id": asv_id,
                    "timestamp": start_time + timedelta(seconds=float(t_s)),
                    "latitude": cur_lat,
                    "longitude": cur_lon,
                    "co2_1": c,
                    "co2_2": c,
                    "u_wind": u_wind,
                    "v_wind": v_wind,
                }
            )

    met_df = _hourly_met(start_time, [mblh] * n_hours)
    meta = ScenarioMetadata(
        name="degraded_geometry_and_outages",
        description="6-ASV hull -> interior Oshen5 (5 hull ASVs) -> 2-ASV outage -> 1-ASV outage -> recovery.",
        expected_cdrr_tco2_hr=None,
        expected_error_bars=(
            ErrorBar.STABLE,
            ErrorBar.TWO_ASV_EXTRAPOLATED,
            ErrorBar.INSUFFICIENT_ASVS,
        ),
        start_time=start_time,
        n_hours=n_hours,
        seed=seed,
    )
    return _build_asv_df(rows), met_df, meta


def _scenario_combined_showcase(
    start_time: datetime,
    seed: int,
    vertices: Sequence[tuple[float, float]],
) -> tuple[pl.DataFrame, pl.DataFrame, ScenarioMetadata]:
    n_hours = 48
    rng = np.random.default_rng(seed)
    xy_base, _, _, _, _ = _prepare_frame(vertices)

    gx_bg, gy_bg = -1.0e-5, 0.2e-5  # Steady ocean uptake (~200 tCO2/hr)
    front_center_sec = 15.0 * 3600.0
    front_width_m = 200_000.0

    rows: list[dict[str, object]] = []
    for idx, (lat, lon) in enumerate(vertices):
        asv_id = f"Oshen{idx + 1}"
        t_sec = _async_times_sec(n_hours, rng, min_step_min=6.0, max_step_min=11.0)
        phase = float(rng.uniform(0.0, 2.0 * math.pi))
        x_i, y_i = float(xy_base[idx, 0]), float(xy_base[idx, 1])

        for t_s in t_sec:
            hr = float(t_s) / 3600.0
            # Brief 4-hour 2-ASV outage during hours 38..41 (Oshen3..6 silent on (37.75, 42.0))
            if idx >= 2 and 37.75 < hr < 42.0:
                continue

            # Wind heading shifts +55 deg at hour 22, relaxes smoothly back by hour 28
            heading_deg = 0.0 if hr < 22.0 else (55.0 if hr < 28.0 else 15.0)
            u_base = 7.5 * math.cos(math.radians(heading_deg))
            v_base = 7.5 * math.sin(math.radians(heading_deg))
            # Mild sailboat bobbing
            bob = 1.5 * math.sin(2.0 * math.pi * float(t_s) / 1080.0 + phase)
            u_w = u_base - bob * math.sin(math.radians(heading_deg))
            v_w = v_base + bob * math.cos(math.radians(heading_deg))

            # Background linear drawdown + propagating +2.0 ppm synoptic front around hr 10..20
            c_drawdown = 418.0 + gx_bg * x_i + gy_bg * y_i
            front_arg = (7.5 * (float(t_s) - front_center_sec) - x_i) / front_width_m
            c_front = 1.0 * (1.0 + math.tanh(front_arg))
            c_true = c_drawdown + c_front

            c1 = c_true + float(rng.normal(0.0, 0.04))
            c2: float | None = c_true + float(rng.normal(0.0, 0.04))
            if asv_id == "Oshen3":
                if 28.0 <= hr < 30.0:
                    c2 = c_true + 1.0 + 0.8 * (hr - 28.0)
                elif 30.0 <= hr < 33.0:
                    c2 = c_true + 5.5
                elif 33.0 <= hr <= 36.5:
                    c2 = None

            rows.append(
                {
                    "asv_id": asv_id,
                    "timestamp": start_time + timedelta(seconds=float(t_s)),
                    "latitude": float(lat),
                    "longitude": float(lon),
                    "co2_1": c1,
                    "co2_2": c2,
                    "u_wind": u_w,
                    "v_wind": v_w,
                }
            )

    # Morning MBLH ramp at hour 24 (900 -> 1300 m)
    mblh_seq = [900.0 if h < 24 else (1300.0 if h < 34 else 1050.0) for h in range(n_hours)]
    met_df = _hourly_met(start_time, mblh_seq)
    meta = ScenarioMetadata(
        name="combined_showcase",
        description=(
            "48-hour multi-phase mission combining steady drawdown, a passing synoptic front, "
            "wind/MBLH shifts, Oshen3 dual-sensor drift/spike/dropout, and a 2-ASV outage."
        ),
        expected_cdrr_tco2_hr=None,
        expected_error_bars=(
            ErrorBar.STABLE,
            ErrorBar.UNSTABLE_WIND_OR_MBLH,
            ErrorBar.CO2_SENSOR_DISCREPANCY,
            ErrorBar.TWO_ASV_EXTRAPOLATED,
        ),
        start_time=start_time,
        n_hours=n_hours,
        seed=seed,
    )
    return _build_asv_df(rows), met_df, meta


_SCENARIO_BUILDERS = {
    "steady_drawdown": _scenario_steady_drawdown,
    "async_sailboat_motion": _scenario_async_sailboat_motion,
    "synoptic_front_storage": _scenario_synoptic_front_storage,
    "sharp_front_7asv": _scenario_sharp_front_7asv,
    "unstable_wind_and_mblh": _scenario_unstable_wind_and_mblh,
    "dual_co2_sensor_fault": _scenario_dual_co2_sensor_fault,
    "degraded_geometry_and_outages": _scenario_degraded_geometry_and_outages,
    "combined_showcase": _scenario_combined_showcase,
}


def generate_scenario(
    name: str,
    *,
    seed: int = 42,
    vertices: Sequence[tuple[float, float]] = ACAPULCO_EDDY_VERTICES,
    start_time: datetime | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame, ScenarioMetadata]:
    """Generate ``(asv_telemetry_df, met_df, metadata)`` for a named synthetic scenario."""
    builder = _SCENARIO_BUILDERS.get(name)
    if builder is None:
        valid = ", ".join(SCENARIO_NAMES)
        raise ValueError(f"Unknown scenario {name!r}; expected one of: {valid}.")
    t0 = _DEFAULT_START if start_time is None else start_time.astimezone(timezone.utc)
    return builder(t0, seed, vertices)
