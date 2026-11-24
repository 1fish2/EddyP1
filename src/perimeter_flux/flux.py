"""Stage 2: Stateless per-period convex-hull ACPF perimeter flux kernel.

Mirrors Section 5 of ``doc/acpf_math.md``:

Dual-Sensor CO2 Reconciliation (Section 5.1)
--------------------------------------------
- Both finite: ``c_i = (c_{i,1} + c_{i,2}) / 2``, ``delta_c_i = |c_{i,1} - c_{i,2}|``.
- Single finite: fall back to the finite sensor, ``delta_c_i = +inf``, flagged.
- Flagged iff one sensor missing or ``delta_c_i > max_co2_sensor_diff_ppm``.

Convex Hull & Outward Normals (Sections 5.2 - 5.3)
--------------------------------------------------
- Project valid ASVs to local WGS 84 East-North meters ``p_i = (x_i, y_i)``.
- Construct 2D ``scipy.spatial.ConvexHull(P)`` (CCW vertices, never ``"QJ"``).
- Rotate CCW ``hull.vertices`` so the alphabetically smallest hull ``asv_id`` is
  index 0.
- For CCW edge ``e: e_0 -> e_1`` with ``(dx, dy) = p_{e1} - p_{e0}`` and length
  ``L_e = hypot(dx, dy)``, outward unit normal is ``n_e = (dy / L_e, -dx / L_e)``.

Hydrostatic Scaling & Perimeter Flux (Sections 5.4 - 5.5)
---------------------------------------------------------
- ``kappa_CO2 = 1e-6 * P_atm * M_CO2 / (R_gas * T_air)`` [kg m^-3 ppm^-1]
- ``H_scale   = R_d * T_air / g_0`` [m]
- ``H_eff     = H_scale * (1 - exp(-H_MBL / H_scale))`` [m]
- ``D_w       = sum_e (w_{e,local} . n_e) * L_e`` [m^2 s^-1]
- ``Phi_edge  = gamma_unit * kappa_CO2 * H_eff * (c_bar_e - c_ref) * (w_e . n_e) * L_e``
- ``Phi_lat   = -sum_e Phi_edge`` [tCO2 hr^-1]  (positive = net ocean uptake)
- ``psi_lat   = Phi_lat / (A_hull * 1e-6)`` [tCO2 km^-2 hr^-1]
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from perimeter_flux.config import (
    DEFAULT_P_ATM_PA,
    DEFAULT_T_AIR_K,
    DRY_AIR_GAS_CONSTANT,
    KG_PER_S_TO_T_PER_HR,
    MOLAR_MASS_CO2_KG,
    STANDARD_GRAVITY,
    UNIVERSAL_GAS_CONSTANT,
    ACPFConfig,
    ErrorBar,
    WindMode,
)
from perimeter_flux.geodesy import project_latlon_to_local_en


@dataclass(frozen=True, slots=True)
class ASVObservation:
    """Single-period per-ASV state for direct Stage 2 flux evaluation."""

    asv_id: str
    latitude: float
    longitude: float
    u_wind: float
    v_wind: float
    co2_1: float | None = None
    co2_2: float | None = None
    wind_constancy: float = 1.0
    asv_drift_speed_m_s: float = float("nan")


@dataclass(frozen=True, slots=True)
class ASVPeriodDetail:
    """Reconciled per-ASV state and local EN coordinates for a single period."""

    asv_id: str
    latitude: float
    longitude: float
    x_m: float
    y_m: float
    on_hull: bool
    co2_1: float
    co2_2: float
    co2_ppm: float
    co2_sensor_diff_ppm: float
    co2_sensor_flagged: bool
    u_wind: float
    v_wind: float
    wind_constancy: float
    asv_drift_speed_m_s: float


@dataclass(frozen=True, slots=True)
class EdgeFluxDetail:
    """Per-edge geometry, outward unit normal, wind projection, and outward CO2 flux."""

    edge_idx: int
    start_asv_id: str
    end_asv_id: str
    x1_m: float
    y1_m: float
    x2_m: float
    y2_m: float
    normal_x: float
    normal_y: float
    edge_length_m: float
    edge_co2_ppm: float
    edge_co2_anomaly_ppm: float
    edge_wind_u: float
    edge_wind_v: float
    wind_dot_normal_m_s: float
    edge_outward_flux_tco2_hr: float


@dataclass(frozen=True, slots=True)
class PeriodFluxResult:
    """Complete Stage 2 per-period ACPF output, geometry metrics, and diagnostics."""

    lateral_cdrr_tco2_hr: float
    lateral_cdrr_tco2_km2_hr: float
    error_bar: ErrorBar
    n_asvs_total: int
    n_asvs_hull: int
    hull_start_asv_id: str | None
    hull_area_m2: float
    hull_perimeter_m: float
    hull_compactness: float
    mblh_m: float
    effective_mblh_m: float
    t_air_k: float
    p_atm_pa: float
    box_mean_co2_ppm: float
    co2_ref_ppm: float
    mean_wind_u: float
    mean_wind_v: float
    mean_wind_speed_m_s: float
    wind_divergence_m2_s: float
    wind_constancy_min: float
    max_co2_sensor_diff_ppm: float
    flagged_co2_asv_ids: tuple[str, ...]
    max_asv_drift_speed_m_s: float
    edges: tuple[EdgeFluxDetail, ...]
    asv_details: tuple[ASVPeriodDetail, ...]


def effective_mblh_m(mblh_m: float, t_air_k: float = DEFAULT_T_AIR_K) -> float:
    """Compute hydrostatic barometric effective MBL height H_eff [m].

    Equation (doc/acpf_math.md Section 5.5)::

        H_scale = R_d * T_air / g_0
        H_eff   = H_scale * (1 - exp(-H_MBL / H_scale))
    """
    if not math.isfinite(mblh_m) or mblh_m <= 0.0:
        raise ValueError(f"mblh_m must be finite and > 0, got {mblh_m!r}.")
    if not math.isfinite(t_air_k) or t_air_k <= 0.0:
        raise ValueError(f"t_air_k must be finite and > 0, got {t_air_k!r}.")

    scale_height_m = DRY_AIR_GAS_CONSTANT * t_air_k / STANDARD_GRAVITY
    return -scale_height_m * math.expm1(-mblh_m / scale_height_m)


def co2_mass_per_ppm_kg_m3(
    t_air_k: float = DEFAULT_T_AIR_K,
    p_atm_pa: float = DEFAULT_P_ATM_PA,
) -> float:
    """Compute surface CO2 mass concentration per 1 ppm dry-air mole fraction [kg m^-3 ppm^-1].

    Equation (doc/acpf_math.md Section 5.5)::

        kappa_CO2 = 1e-6 * P_atm * M_CO2 / (R_gas * T_air)
    """
    if not math.isfinite(t_air_k) or t_air_k <= 0.0:
        raise ValueError(f"t_air_k must be finite and > 0, got {t_air_k!r}.")
    if not math.isfinite(p_atm_pa) or p_atm_pa <= 0.0:
        raise ValueError(f"p_atm_pa must be finite and > 0, got {p_atm_pa!r}.")

    return 1e-6 * p_atm_pa * MOLAR_MASS_CO2_KG / (UNIVERSAL_GAS_CONSTANT * t_air_k)


def _as_optional_float(val: float | None) -> float:
    return float("nan") if val is None else float(val)


def _resolve_met_param(name: str, value: float | None, default: float) -> float:
    if value is None:
        return default
    fval = float(value)
    if math.isnan(fval):
        return default
    if not math.isfinite(fval) or fval <= 0.0:
        raise ValueError(f"{name} must be finite and > 0, got {value!r}.")
    return fval


_DEFAULT_CONFIG = ACPFConfig()


def compute_period_flux(
    observations: Sequence[ASVObservation],
    mblh_m: float,
    *,
    config: ACPFConfig = _DEFAULT_CONFIG,
    t_air_k: float | None = None,
    p_atm_pa: float | None = None,
    external_wind: tuple[float, float] | None = None,
) -> PeriodFluxResult:
    """Compute Stage 2 lateral ACPF CO2 perimeter flux for a single aggregation period."""
    t_air = _resolve_met_param("t_air_k", t_air_k, config.default_t_air_k)
    p_atm = _resolve_met_param("p_atm_pa", p_atm_pa, config.default_p_atm_pa)
    mblh = float(mblh_m)

    if config.wind_mode is WindMode.EXTERNAL and external_wind is None:
        raise ValueError("external_wind is required when config.wind_mode is WindMode.EXTERNAL.")

    # Check for duplicate ASV IDs across input observations
    seen_ids: set[str] = set()
    for obs in observations:
        if obs.asv_id in seen_ids:
            raise ValueError(f"Duplicate asv_id in observations: {obs.asv_id!r}.")
        seen_ids.add(obs.asv_id)

    # Filter to valid ASVs and reconcile dual CO2 sensors (Section 5.1)
    reconciled: list[tuple[ASVObservation, float, float, float, float, bool]] = []
    for obs in sorted(observations, key=lambda o: o.asv_id):
        lat = float(obs.latitude)
        lon = float(obs.longitude)
        u_w = float(obs.u_wind)
        v_w = float(obs.v_wind)
        if not (math.isfinite(lat) and math.isfinite(lon) and math.isfinite(u_w) and math.isfinite(v_w)):
            continue

        c1 = _as_optional_float(obs.co2_1)
        c2 = _as_optional_float(obs.co2_2)
        c1_ok = math.isfinite(c1)
        c2_ok = math.isfinite(c2)
        if not (c1_ok or c2_ok):
            continue

        if c1_ok and c2_ok:
            c_i = 0.5 * (c1 + c2)
            diff_i = abs(c1 - c2)
            flagged_i = diff_i > config.max_co2_sensor_diff_ppm
        elif c1_ok:
            c_i = c1
            diff_i = float("inf")
            flagged_i = True
        else:
            c_i = c2
            diff_i = float("inf")
            flagged_i = True

        reconciled.append((obs, c1, c2, c_i, diff_i, flagged_i))

    n_total = len(reconciled)
    nan = float("nan")

    if n_total > 0:
        lats = np.array([r[0].latitude for r in reconciled], dtype=np.float64)
        lons = np.array([r[0].longitude for r in reconciled], dtype=np.float64)
        xy_m, _, _ = project_latlon_to_local_en(lats, lons)

        u_arr = np.array([r[0].u_wind for r in reconciled], dtype=np.float64)
        v_arr = np.array([r[0].v_wind for r in reconciled], dtype=np.float64)
        c_arr = np.array([r[3] for r in reconciled], dtype=np.float64)

        mean_u = float(np.mean(u_arr))
        mean_v = float(np.mean(v_arr))
        mean_speed = math.hypot(mean_u, mean_v)
        box_mean_co2 = float(np.mean(c_arr))

        constancies = [float(r[0].wind_constancy) for r in reconciled if math.isfinite(r[0].wind_constancy)]
        wind_constancy_min = min(constancies) if constancies else nan

        drifts = [
            float(r[0].asv_drift_speed_m_s)
            for r in reconciled
            if math.isfinite(r[0].asv_drift_speed_m_s)
        ]
        max_drift = max(drifts) if drifts else nan

        max_co2_diff = max(r[4] for r in reconciled)
        flagged_ids = tuple(r[0].asv_id for r in reconciled if r[5])
    else:
        xy_m = np.empty((0, 2), dtype=np.float64)
        u_arr = np.empty((0,), dtype=np.float64)
        v_arr = np.empty((0,), dtype=np.float64)
        c_arr = np.empty((0,), dtype=np.float64)
        mean_u = nan
        mean_v = nan
        mean_speed = nan
        box_mean_co2 = nan
        wind_constancy_min = nan
        max_drift = nan
        max_co2_diff = nan
        flagged_ids = ()

    def _build_asv_details(hull_set: set[int]) -> tuple[ASVPeriodDetail, ...]:
        return tuple(
            ASVPeriodDetail(
                asv_id=obs.asv_id,
                latitude=float(obs.latitude),
                longitude=float(obs.longitude),
                x_m=float(xy_m[idx, 0]),
                y_m=float(xy_m[idx, 1]),
                on_hull=idx in hull_set,
                co2_1=c1,
                co2_2=c2,
                co2_ppm=c_i,
                co2_sensor_diff_ppm=diff_i,
                co2_sensor_flagged=flagged_i,
                u_wind=float(obs.u_wind),
                v_wind=float(obs.v_wind),
                wind_constancy=float(obs.wind_constancy),
                asv_drift_speed_m_s=float(obs.asv_drift_speed_m_s),
            )
            for idx, (obs, c1, c2, c_i, diff_i, flagged_i) in enumerate(reconciled)
        )

    def _degraded_result(
        error_bar: ErrorBar,
        *,
        h_eff: float = nan,
        hull_area_m2: float = nan,
        hull_perimeter_m: float = nan,
        hull_compactness: float = nan,
    ) -> PeriodFluxResult:
        return PeriodFluxResult(
            lateral_cdrr_tco2_hr=nan,
            lateral_cdrr_tco2_km2_hr=nan,
            error_bar=error_bar,
            n_asvs_total=n_total,
            n_asvs_hull=0,
            hull_start_asv_id=None,
            hull_area_m2=hull_area_m2,
            hull_perimeter_m=hull_perimeter_m,
            hull_compactness=hull_compactness,
            mblh_m=mblh,
            effective_mblh_m=h_eff,
            t_air_k=t_air,
            p_atm_pa=p_atm,
            box_mean_co2_ppm=box_mean_co2,
            co2_ref_ppm=nan,
            mean_wind_u=mean_u,
            mean_wind_v=mean_v,
            mean_wind_speed_m_s=mean_speed,
            wind_divergence_m2_s=nan,
            wind_constancy_min=wind_constancy_min,
            max_co2_sensor_diff_ppm=max_co2_diff,
            flagged_co2_asv_ids=flagged_ids,
            max_asv_drift_speed_m_s=max_drift,
            edges=(),
            asv_details=_build_asv_details(set()),
        )

    # Insufficient ASV count or invalid MBLH
    mblh_valid = math.isfinite(mblh) and mblh > 0.0
    if not mblh_valid:
        return _degraded_result(ErrorBar.INSUFFICIENT_ASVS)

    h_eff = effective_mblh_m(mblh, t_air)

    if config.wind_mode is WindMode.EXTERNAL:
        assert external_wind is not None
        ext_u, ext_v = float(external_wind[0]), float(external_wind[1])
        if not (math.isfinite(ext_u) and math.isfinite(ext_v)):
            return _degraded_result(ErrorBar.INSUFFICIENT_ASVS, h_eff=h_eff)
    else:
        ext_u = nan
        ext_v = nan

    if n_total < 2:
        return _degraded_result(ErrorBar.INSUFFICIENT_ASVS, h_eff=h_eff)
    if n_total == 2:
        return _degraded_result(ErrorBar.TWO_ASV_EXTRAPOLATED, h_eff=h_eff)

    # Collinearity and convex hull construction (Section 5.2, never use Qhull "QJ")
    centered = xy_m - np.mean(xy_m, axis=0, keepdims=True)
    if int(np.linalg.matrix_rank(centered)) < 2:
        return _degraded_result(ErrorBar.TWO_ASV_EXTRAPOLATED, h_eff=h_eff)

    try:
        hull = ConvexHull(xy_m)
    except QhullError:
        return _degraded_result(ErrorBar.TWO_ASV_EXTRAPOLATED, h_eff=h_eff)

    hull_area_m2 = float(hull.volume)  # 2D area in SciPy ConvexHull
    hull_perimeter_m = float(hull.area)  # 1D perimeter in SciPy ConvexHull
    hull_compactness = (
        (4.0 * math.pi * hull_area_m2) / (hull_perimeter_m * hull_perimeter_m)
        if hull_perimeter_m > 0.0
        else 0.0
    )

    if hull_area_m2 < config.min_hull_area_m2 or hull_compactness < config.min_hull_compactness:
        return _degraded_result(
            ErrorBar.TWO_ASV_EXTRAPOLATED,
            h_eff=h_eff,
            hull_area_m2=hull_area_m2,
            hull_perimeter_m=hull_perimeter_m,
            hull_compactness=hull_compactness,
        )

    # Rotate guaranteed-CCW hull.vertices so alphabetically smallest hull asv_id is vertex 0
    raw_hull_idx = hull.vertices
    min_pos = min(range(len(raw_hull_idx)), key=lambda pos: reconciled[int(raw_hull_idx[pos])][0].asv_id)
    hull_idx = np.roll(raw_hull_idx, -min_pos)
    n_hull = int(hull_idx.size)
    hull_start_asv_id = reconciled[int(hull_idx[0])][0].asv_id

    co2_ref = float(np.mean(c_arr[hull_idx]))
    kappa = co2_mass_per_ppm_kg_m3(t_air, p_atm)
    flux_scale = KG_PER_S_TO_T_PER_HR * kappa * h_eff

    edge_details: list[EdgeFluxDetail] = []
    wind_divergence = 0.0
    total_outward_flux = 0.0

    for edge_idx in range(n_hull):
        i0 = int(hull_idx[edge_idx])
        i1 = int(hull_idx[(edge_idx + 1) % n_hull])
        x1, y1 = float(xy_m[i0, 0]), float(xy_m[i0, 1])
        x2, y2 = float(xy_m[i1, 0]), float(xy_m[i1, 1])
        dx = x2 - x1
        dy = y2 - y1
        length_m = math.hypot(dx, dy)

        # Outward unit normal for CCW edge: (dy, -dx) / L_e
        nx = dy / length_m
        ny = -dx / length_m

        # Raw local edge wind divergence diagnostic D_w (always recorded)
        w_loc_u = 0.5 * (float(u_arr[i0]) + float(u_arr[i1]))
        w_loc_v = 0.5 * (float(v_arr[i0]) + float(v_arr[i1]))
        wind_divergence += (w_loc_u * nx + w_loc_v * ny) * length_m

        if config.wind_mode is WindMode.UNIFORM_MEAN:
            w_u, w_v = mean_u, mean_v
        elif config.wind_mode is WindMode.EXTERNAL:
            w_u, w_v = ext_u, ext_v
        else:
            w_u, w_v = w_loc_u, w_loc_v

        w_dot_n = w_u * nx + w_v * ny
        edge_co2 = 0.5 * (float(c_arr[i0]) + float(c_arr[i1]))
        edge_co2_anom = edge_co2 - co2_ref
        edge_flux = flux_scale * edge_co2_anom * w_dot_n * length_m
        total_outward_flux += edge_flux

        edge_details.append(
            EdgeFluxDetail(
                edge_idx=edge_idx,
                start_asv_id=reconciled[i0][0].asv_id,
                end_asv_id=reconciled[i1][0].asv_id,
                x1_m=x1,
                y1_m=y1,
                x2_m=x2,
                y2_m=y2,
                normal_x=nx,
                normal_y=ny,
                edge_length_m=length_m,
                edge_co2_ppm=edge_co2,
                edge_co2_anomaly_ppm=edge_co2_anom,
                edge_wind_u=w_u,
                edge_wind_v=w_v,
                wind_dot_normal_m_s=w_dot_n,
                edge_outward_flux_tco2_hr=edge_flux,
            )
        )

    # Positive CDRR = net ocean CO2 uptake/removal = -outward flux
    lateral_cdrr = -total_outward_flux
    lateral_cdrr_per_km2 = lateral_cdrr / (hull_area_m2 * 1e-6)

    # Worst-severity Stage 2 ErrorBar classification
    error_bar = ErrorBar.STABLE
    if math.isfinite(wind_constancy_min) and wind_constancy_min < config.min_wind_constancy:
        error_bar = max(error_bar, ErrorBar.UNSTABLE_WIND_OR_MBLH)
    if flagged_ids:
        error_bar = max(error_bar, ErrorBar.CO2_SENSOR_DISCREPANCY)

    return PeriodFluxResult(
        lateral_cdrr_tco2_hr=lateral_cdrr,
        lateral_cdrr_tco2_km2_hr=lateral_cdrr_per_km2,
        error_bar=error_bar,
        n_asvs_total=n_total,
        n_asvs_hull=n_hull,
        hull_start_asv_id=hull_start_asv_id,
        hull_area_m2=hull_area_m2,
        hull_perimeter_m=hull_perimeter_m,
        hull_compactness=hull_compactness,
        mblh_m=mblh,
        effective_mblh_m=h_eff,
        t_air_k=t_air,
        p_atm_pa=p_atm,
        box_mean_co2_ppm=box_mean_co2,
        co2_ref_ppm=co2_ref,
        mean_wind_u=mean_u,
        mean_wind_v=mean_v,
        mean_wind_speed_m_s=mean_speed,
        wind_divergence_m2_s=wind_divergence,
        wind_constancy_min=wind_constancy_min,
        max_co2_sensor_diff_ppm=max_co2_diff,
        flagged_co2_asv_ids=flagged_ids,
        max_asv_drift_speed_m_s=max_drift,
        edges=tuple(edge_details),
        asv_details=_build_asv_details(set(hull_idx.tolist())),
    )
