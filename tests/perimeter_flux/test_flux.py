"""Pytest test suite for Stage 2 per-period convex-hull ACPF flux computation."""

from __future__ import annotations

import math

import pytest

from perimeter_flux.config import (
    ACPFConfig,
    DEFAULT_P_ATM_PA,
    DEFAULT_T_AIR_K,
    ErrorBar,
    KG_PER_S_TO_T_PER_HR,
    WindMode,
)
from perimeter_flux.flux import (
    ASVObservation,
    co2_mass_per_ppm_kg_m3,
    compute_period_flux,
    effective_mblh_m,
)
from perimeter_flux.geodesy import project_latlon_to_local_en
from perimeter_flux.sample_data import ACAPULCO_EDDY_VERTICES

def _make_acapulco_observations(
    *,
    u_wind: float = 6.5,
    v_wind: float = -3.2,
    base_co2: float = 420.0,
    grad_x_ppm_per_m: float = -1.5e-5,
    grad_y_ppm_per_m: float = 0.8e-5,
) -> list[ASVObservation]:
    """Build 6 Oshen ASV observations on the Acapulco eddy with an exact linear EN CO2 field."""
    lats = [p[0] for p in ACAPULCO_EDDY_VERTICES]
    lons = [p[1] for p in ACAPULCO_EDDY_VERTICES]
    xy_m, _, _ = project_latlon_to_local_en(lats, lons)

    obs: list[ASVObservation] = []
    for idx, (lat, lon) in enumerate(ACAPULCO_EDDY_VERTICES):
        c = base_co2 + grad_x_ppm_per_m * float(xy_m[idx, 0]) + grad_y_ppm_per_m * float(xy_m[idx, 1])
        obs.append(
            ASVObservation(
                asv_id=f"Oshen{idx + 1}",
                latitude=lat,
                longitude=lon,
                u_wind=u_wind,
                v_wind=v_wind,
                co2_1=c,
                co2_2=c,
            )
        )
    return obs


class TestThermodynamics:
    """Verify hydrostatic effective MBLH and ideal-gas CO2 mass scaling."""

    def test_effective_mblh_hydrostatic_and_small_limit(self):
        h_eff_1000 = effective_mblh_m(1000.0, DEFAULT_T_AIR_K)
        assert h_eff_1000 < 1000.0
        assert math.isclose(h_eff_1000, 943.9, rel_tol=1e-3)

        # Small-H_MBL linear limit: H_eff / H_MBL -> 1 accurately (expm1 stability)
        h_small = 1e-4
        assert math.isclose(effective_mblh_m(h_small, DEFAULT_T_AIR_K) / h_small, 1.0, rel_tol=1e-8)

        with pytest.raises(ValueError):
            effective_mblh_m(0.0)
        with pytest.raises(ValueError):
            effective_mblh_m(-100.0)
        with pytest.raises(ValueError):
            effective_mblh_m(1000.0, t_air_k=0.0)
        with pytest.raises(ValueError):
            effective_mblh_m(float("nan"))

    def test_co2_mass_per_ppm_ideal_gas_scaling(self):
        kappa_std = co2_mass_per_ppm_kg_m3(DEFAULT_T_AIR_K, DEFAULT_P_ATM_PA)
        assert math.isclose(kappa_std, 1.8297e-6, rel_tol=1e-3)

        # Linear in pressure, inverse in temperature
        assert math.isclose(
            co2_mass_per_ppm_kg_m3(DEFAULT_T_AIR_K, 2.0 * DEFAULT_P_ATM_PA),
            2.0 * kappa_std,
            rel_tol=1e-12,
        )
        assert math.isclose(
            co2_mass_per_ppm_kg_m3(2.0 * DEFAULT_T_AIR_K, DEFAULT_P_ATM_PA),
            0.5 * kappa_std,
            rel_tol=1e-12,
        )

        with pytest.raises(ValueError):
            co2_mass_per_ppm_kg_m3(t_air_k=-10.0)
        with pytest.raises(ValueError):
            co2_mass_per_ppm_kg_m3(p_atm_pa=0.0)


class TestAnalyticalBoxFluxAndInvariants:
    """Verify exact Divergence Theorem flux, sign convention, background invariance, and closure."""

    def test_exact_analytical_linear_gradient_on_acapulco_hexagon(self):
        u, v = 6.5, -3.2
        gx, gy = -1.5e-5, 0.8e-5  # grad(c) dot w = -9.75e-5 - 2.56e-5 = -1.231e-4 < 0 -> net uptake (+)
        mblh = 900.0
        obs = _make_acapulco_observations(
            u_wind=u, v_wind=v, grad_x_ppm_per_m=gx, grad_y_ppm_per_m=gy
        )

        res = compute_period_flux(obs, mblh_m=mblh)
        assert res.error_bar is ErrorBar.STABLE
        assert res.n_asvs_total == 6
        assert res.n_asvs_hull == 6
        assert len(res.edges) == 6

        kappa = co2_mass_per_ppm_kg_m3(DEFAULT_T_AIR_K, DEFAULT_P_ATM_PA)
        h_eff = effective_mblh_m(mblh, DEFAULT_T_AIR_K)
        expected_cdrr = KG_PER_S_TO_T_PER_HR * kappa * h_eff * (-(gx * u + gy * v)) * res.hull_area_m2

        assert expected_cdrr > 0.0  # Positive = net ocean CO2 removal/uptake
        assert math.isclose(res.lateral_cdrr_tco2_hr, expected_cdrr, rel_tol=1e-10)
        assert math.isclose(
            res.lateral_cdrr_tco2_km2_hr,
            expected_cdrr / (res.hull_area_m2 * 1e-6),
            rel_tol=1e-12,
        )

        # Reversing wind reverses sign (outgassing < 0)
        obs_rev = _make_acapulco_observations(
            u_wind=-u, v_wind=-v, grad_x_ppm_per_m=gx, grad_y_ppm_per_m=gy
        )
        res_rev = compute_period_flux(obs_rev, mblh_m=mblh)
        assert math.isclose(res_rev.lateral_cdrr_tco2_hr, -expected_cdrr, rel_tol=1e-10)

        # Wind orthogonal to CO2 gradient yields zero net flux
        obs_orth = _make_acapulco_observations(
            u_wind=-gy * 1e5, v_wind=gx * 1e5, grad_x_ppm_per_m=gx, grad_y_ppm_per_m=gy
        )
        res_orth = compute_period_flux(obs_orth, mblh_m=mblh)
        assert abs(res_orth.lateral_cdrr_tco2_hr) < 1e-10

    def test_exact_analytical_flux_on_square_and_closure_identity(self):
        # 4-ASV square around (13.4 N, -101.7 W)
        coords = [(13.0, -102.1), (13.0, -101.3), (13.8, -101.3), (13.8, -102.1)]
        lats = [c[0] for c in coords]
        lons = [c[1] for c in coords]
        xy_m, _, _ = project_latlon_to_local_en(lats, lons)
        u, v = 8.0, 2.0
        gx, gy = -2.0e-5, -0.5e-5
        obs = [
            ASVObservation(
                asv_id=f"Oshen{i + 1}",
                latitude=lat,
                longitude=lon,
                u_wind=u,
                v_wind=v,
                co2_1=415.0 + gx * float(xy_m[i, 0]) + gy * float(xy_m[i, 1]),
                co2_2=415.0 + gx * float(xy_m[i, 0]) + gy * float(xy_m[i, 1]),
            )
            for i, (lat, lon) in enumerate(coords)
        ]

        res = compute_period_flux(obs, mblh_m=1000.0)
        kappa = co2_mass_per_ppm_kg_m3()
        h_eff = effective_mblh_m(1000.0)
        expected = KG_PER_S_TO_T_PER_HR * kappa * h_eff * (-(gx * u + gy * v)) * res.hull_area_m2
        assert math.isclose(res.lateral_cdrr_tco2_hr, expected, rel_tol=1e-10)

        # Closure identity: sum(L_e * n_e) == (0, 0)
        closure_x = sum(e.edge_length_m * e.normal_x for e in res.edges)
        closure_y = sum(e.edge_length_m * e.normal_y for e in res.edges)
        assert abs(closure_x) < 1e-9
        assert abs(closure_y) < 1e-9

        # Every normal is unit length, orthogonal to edge, and points outward from centroid
        cx = sum(e.x1_m for e in res.edges) / len(res.edges)
        cy = sum(e.y1_m for e in res.edges) / len(res.edges)
        for edge in res.edges:
            assert math.isclose(math.hypot(edge.normal_x, edge.normal_y), 1.0, rel_tol=1e-12)
            dx = edge.x2_m - edge.x1_m
            dy = edge.y2_m - edge.y1_m
            assert abs(dx * edge.normal_x + dy * edge.normal_y) < 1e-9
            mid_x = 0.5 * (edge.x1_m + edge.x2_m)
            mid_y = 0.5 * (edge.y1_m + edge.y2_m)
            assert (mid_x - cx) * edge.normal_x + (mid_y - cy) * edge.normal_y > 0.0

    def test_background_invariance_adding_1000_ppm(self):
        obs_base = _make_acapulco_observations(base_co2=420.0)
        obs_shifted = _make_acapulco_observations(base_co2=1420.0)

        res_base = compute_period_flux(obs_base, mblh_m=1000.0)
        res_shifted = compute_period_flux(obs_shifted, mblh_m=1000.0)

        assert math.isclose(
            res_shifted.lateral_cdrr_tco2_hr, res_base.lateral_cdrr_tco2_hr, rel_tol=1e-10
        )
        assert math.isclose(res_shifted.co2_ref_ppm - res_base.co2_ref_ppm, 1000.0, abs_tol=1e-10)
        for e0, e1 in zip(res_base.edges, res_shifted.edges, strict=True):
            assert math.isclose(
                e1.edge_outward_flux_tco2_hr, e0.edge_outward_flux_tco2_hr, rel_tol=1e-10
            )


class TestCanonicalHullOrderingAndInteriorASVs:
    """Verify alphabetical CCW hull anchoring and interior ASV handling."""

    def test_shuffled_inputs_start_at_alphabetically_smallest_hull_asv(self):
        obs = _make_acapulco_observations()
        shuffled = [obs[3], obs[0], obs[5], obs[2], obs[4], obs[1]]

        res = compute_period_flux(shuffled, mblh_m=1000.0)
        assert res.hull_start_asv_id == "Oshen1"
        assert res.edges[0].start_asv_id == "Oshen1"
        # Closed CCW chain: end of edge e == start of edge (e + 1) % N
        for idx, edge in enumerate(res.edges):
            nxt = res.edges[(idx + 1) % len(res.edges)]
            assert edge.end_asv_id == nxt.start_asv_id
        # Signed shoelace area of edge vertices is positive (CCW)
        signed_area2 = sum(e.x1_m * e.y2_m - e.x2_m * e.y1_m for e in res.edges)
        assert signed_area2 > 0.0

    def test_interior_asv_excluded_from_hull_but_included_in_mean_wind_and_box_co2(self):
        # Place Oshen1 at the center of the Acapulco eddy; Oshen2..Oshen6 form the 5-vertex hull
        obs = _make_acapulco_observations(u_wind=5.0, v_wind=0.0, grad_x_ppm_per_m=0.0, grad_y_ppm_per_m=0.0)
        center_lat = sum(p[0] for p in ACAPULCO_EDDY_VERTICES) / 6.0
        center_lon = sum(p[1] for p in ACAPULCO_EDDY_VERTICES) / 6.0
        obs[0] = ASVObservation(
            asv_id="Oshen1",
            latitude=center_lat,
            longitude=center_lon,
            u_wind=11.0,  # Pulls mean wind from 5.0 to (11 + 5*5)/6 = 6.0
            v_wind=0.0,
            co2_1=426.0,  # Pulls box_mean_co2 from 420.0 to 421.0, while co2_ref on hull stays 420.0
            co2_2=426.0,
        )

        res = compute_period_flux(obs, mblh_m=1000.0)
        assert res.error_bar is ErrorBar.STABLE
        assert res.n_asvs_total == 6
        assert res.n_asvs_hull == 5
        assert res.hull_start_asv_id == "Oshen2"
        assert res.edges[0].start_asv_id == "Oshen2"
        assert all(e.start_asv_id != "Oshen1" and e.end_asv_id != "Oshen1" for e in res.edges)

        # Interior ASV contributes to polygon-mean wind & box mean CO2, but not hull co2_ref
        assert math.isclose(res.mean_wind_u, 6.0, abs_tol=1e-12)
        assert math.isclose(res.box_mean_co2_ppm, 421.0, abs_tol=1e-12)
        assert math.isclose(res.co2_ref_ppm, 420.0, abs_tol=1e-12)

        by_id = {d.asv_id: d for d in res.asv_details}
        assert by_id["Oshen1"].on_hull is False
        for k in range(2, 7):
            assert by_id[f"Oshen{k}"].on_hull is True


class TestWindModesAndDivergenceDiagnostic:
    """Verify UNIFORM_MEAN, EXTERNAL, PER_ASV modes and raw wind divergence D_w."""

    def test_biased_anemometer_divergence_diagnostic_and_background_suppression(self):
        # Uniform 420 ppm CO2 everywhere; Oshen1 (furthest North) has a +3 m/s outward North wind bias
        obs = _make_acapulco_observations(
            u_wind=5.0, v_wind=0.0, base_co2=420.0, grad_x_ppm_per_m=0.0, grad_y_ppm_per_m=0.0
        )
        obs[0] = ASVObservation(
            asv_id=obs[0].asv_id,
            latitude=obs[0].latitude,
            longitude=obs[0].longitude,
            u_wind=5.0,
            v_wind=3.0,  # Outward bias on Northernmost vertex -> D_w > 0
            co2_1=420.0,
            co2_2=420.0,
        )

        res_unif = compute_period_flux(obs, 1000.0, config=ACPFConfig(wind_mode=WindMode.UNIFORM_MEAN))
        res_per = compute_period_flux(obs, 1000.0, config=ACPFConfig(wind_mode=WindMode.PER_ASV))
        res_ext = compute_period_flux(
            obs,
            1000.0,
            config=ACPFConfig(wind_mode=WindMode.EXTERNAL),
            external_wind=(4.0, -1.0),
        )

        # Raw local divergence D_w > 0 is identical across all wind modes
        assert res_unif.wind_divergence_m2_s > 1e4
        assert math.isclose(res_per.wind_divergence_m2_s, res_unif.wind_divergence_m2_s, rel_tol=1e-12)
        assert math.isclose(res_ext.wind_divergence_m2_s, res_unif.wind_divergence_m2_s, rel_tol=1e-12)

        # Because c_ref is subtracted, all three modes give exact zero flux on a uniform 420 ppm field
        assert abs(res_unif.lateral_cdrr_tco2_hr) < 1e-12
        assert abs(res_per.lateral_cdrr_tco2_hr) < 1e-12
        assert abs(res_ext.lateral_cdrr_tco2_hr) < 1e-12

    def test_external_wind_mode_uses_external_vector_and_validates_presence(self):
        gx, gy = -1.5e-5, 0.5e-5
        obs = _make_acapulco_observations(
            u_wind=1.0, v_wind=1.0, grad_x_ppm_per_m=gx, grad_y_ppm_per_m=gy
        )
        ext_u, ext_v = 10.0, -4.0
        cfg = ACPFConfig(wind_mode=WindMode.EXTERNAL)

        res = compute_period_flux(obs, 1000.0, config=cfg, external_wind=(ext_u, ext_v))
        kappa = co2_mass_per_ppm_kg_m3()
        h_eff = effective_mblh_m(1000.0)
        expected = KG_PER_S_TO_T_PER_HR * kappa * h_eff * (-(gx * ext_u + gy * ext_v)) * res.hull_area_m2
        assert math.isclose(res.lateral_cdrr_tco2_hr, expected, rel_tol=1e-10)

        with pytest.raises(ValueError, match="external_wind"):
            compute_period_flux(obs, 1000.0, config=cfg)

    def test_per_asv_wind_differs_from_uniform_under_spatial_shear(self):
        obs = _make_acapulco_observations(u_wind=5.0, v_wind=0.0, grad_x_ppm_per_m=-2e-5, grad_y_ppm_per_m=0.0)
        # Shear wind strongly on Oshen2 (West) vs Oshen5 (East)
        obs[1] = ASVObservation(
            asv_id=obs[1].asv_id,
            latitude=obs[1].latitude,
            longitude=obs[1].longitude,
            u_wind=9.0,
            v_wind=0.0,
            co2_1=obs[1].co2_1,
            co2_2=obs[1].co2_2,
        )
        res_unif = compute_period_flux(obs, 1000.0, config=ACPFConfig(wind_mode=WindMode.UNIFORM_MEAN))
        res_per = compute_period_flux(obs, 1000.0, config=ACPFConfig(wind_mode=WindMode.PER_ASV))
        assert not math.isclose(res_per.lateral_cdrr_tco2_hr, res_unif.lateral_cdrr_tco2_hr, rel_tol=1e-3)


class TestDualSensorCO2Reconciliation:
    """Verify dual-sensor averaging, discrepancy flagging, fallback, and omission."""

    def test_discrepant_co2_sensors_flag_and_preserve_hull(self):
        obs = _make_acapulco_observations()
        # Inject a 4.5 ppm discrepancy on Oshen3 (> default 2.0 ppm threshold)
        c_mid = obs[2].co2_1
        assert c_mid is not None
        obs[2] = ASVObservation(
            asv_id="Oshen3",
            latitude=obs[2].latitude,
            longitude=obs[2].longitude,
            u_wind=obs[2].u_wind,
            v_wind=obs[2].v_wind,
            co2_1=c_mid - 2.25,
            co2_2=c_mid + 2.25,
        )

        res = compute_period_flux(obs, 1000.0)
        assert res.error_bar is ErrorBar.CO2_SENSOR_DISCREPANCY
        assert res.n_asvs_hull == 6
        assert math.isclose(res.max_co2_sensor_diff_ppm, 4.5, abs_tol=1e-12)
        assert res.flagged_co2_asv_ids == ("Oshen3",)
        assert math.isfinite(res.lateral_cdrr_tco2_hr)

    def test_single_sensor_fallback_flags_discrepancy_and_both_nan_omits_asv(self):
        obs = _make_acapulco_observations()
        # Oshen3 loses co2_2 -> stays in hull using co2_1, flagged with inf diff
        obs[2] = ASVObservation(
            asv_id="Oshen3",
            latitude=obs[2].latitude,
            longitude=obs[2].longitude,
            u_wind=obs[2].u_wind,
            v_wind=obs[2].v_wind,
            co2_1=obs[2].co2_1,
            co2_2=None,
        )
        res_single = compute_period_flux(obs, 1000.0)
        assert res_single.error_bar is ErrorBar.CO2_SENSOR_DISCREPANCY
        assert res_single.n_asvs_total == 6
        assert res_single.n_asvs_hull == 6
        assert math.isinf(res_single.max_co2_sensor_diff_ppm)
        assert res_single.flagged_co2_asv_ids == ("Oshen3",)

        # Oshen3 loses BOTH sensors -> omitted from period, remaining 5 ASVs are STABLE
        obs[2] = ASVObservation(
            asv_id="Oshen3",
            latitude=obs[2].latitude,
            longitude=obs[2].longitude,
            u_wind=obs[2].u_wind,
            v_wind=obs[2].v_wind,
            co2_1=float("nan"),
            co2_2=float("nan"),
        )
        res_omitted = compute_period_flux(obs, 1000.0)
        assert res_omitted.error_bar is ErrorBar.STABLE
        assert res_omitted.n_asvs_total == 5
        assert res_omitted.n_asvs_hull == 5
        assert res_omitted.flagged_co2_asv_ids == ()


class TestDegenerateGeometryAndInsufficientInputs:
    """Verify 2-ASV, collinear, sliver, tiny-area, low-constancy, and invalid-MBLH handling."""

    def test_two_asvs_returns_nan_and_two_asv_extrapolated(self):
        obs = _make_acapulco_observations()[:2]
        res = compute_period_flux(obs, 1000.0)
        assert res.error_bar is ErrorBar.TWO_ASV_EXTRAPOLATED
        assert math.isnan(res.lateral_cdrr_tco2_hr)
        assert math.isnan(res.lateral_cdrr_tco2_km2_hr)
        assert res.n_asvs_total == 2
        assert res.n_asvs_hull == 0
        assert res.edges == ()

    def test_collinear_sliver_and_tiny_area_hulls_return_two_asv_extrapolated(self):
        # 1. Strictly collinear 3 ASVs along same longitude
        collinear = [
            ASVObservation("Oshen1", 13.0, -101.7, 5.0, 0.0, 415.0, 415.0),
            ASVObservation("Oshen2", 13.5, -101.7, 5.0, 0.0, 414.5, 414.5),
            ASVObservation("Oshen3", 14.0, -101.7, 5.0, 0.0, 414.0, 414.0),
        ]
        res_col = compute_period_flux(collinear, 1000.0)
        assert res_col.error_bar is ErrorBar.TWO_ASV_EXTRAPOLATED
        assert math.isnan(res_col.lateral_cdrr_tco2_hr)

        # 2. Razor-thin sliver (~111 km long, ~100 m wide -> Q_iso << 0.05)
        sliver = [
            ASVObservation("Oshen1", 13.0, -101.7, 5.0, 0.0, 415.0, 415.0),
            ASVObservation("Oshen2", 13.5, -101.699, 5.0, 0.0, 414.5, 414.5),
            ASVObservation("Oshen3", 14.0, -101.7, 5.0, 0.0, 414.0, 414.0),
        ]
        res_sliver = compute_period_flux(sliver, 1000.0)
        assert res_sliver.error_bar is ErrorBar.TWO_ASV_EXTRAPOLATED
        assert res_sliver.hull_compactness < ACPFConfig().min_hull_compactness
        assert math.isnan(res_sliver.lateral_cdrr_tco2_hr)

        # 3. Tiny equilateral-ish triangle (< 1 km^2 area)
        tiny = [
            ASVObservation("Oshen1", 13.4000, -101.7000, 5.0, 0.0, 415.0, 415.0),
            ASVObservation("Oshen2", 13.4000, -101.6950, 5.0, 0.0, 414.9, 414.9),
            ASVObservation("Oshen3", 13.4040, -101.6975, 5.0, 0.0, 414.8, 414.8),
        ]
        res_tiny = compute_period_flux(tiny, 1000.0)
        assert res_tiny.error_bar is ErrorBar.TWO_ASV_EXTRAPOLATED
        assert res_tiny.hull_area_m2 < ACPFConfig().min_hull_area_m2
        assert math.isnan(res_tiny.lateral_cdrr_tco2_hr)

    def test_insufficient_asvs_and_invalid_mblh(self):
        obs = _make_acapulco_observations()

        res_empty = compute_period_flux([], 1000.0)
        assert res_empty.error_bar is ErrorBar.INSUFFICIENT_ASVS
        assert math.isnan(res_empty.lateral_cdrr_tco2_hr)

        res_one = compute_period_flux(obs[:1], 1000.0)
        assert res_one.error_bar is ErrorBar.INSUFFICIENT_ASVS
        assert math.isnan(res_one.lateral_cdrr_tco2_hr)

        for bad_mblh in (0.0, -50.0, float("nan")):
            res_bad_mblh = compute_period_flux(obs, bad_mblh)
            assert res_bad_mblh.error_bar is ErrorBar.INSUFFICIENT_ASVS
            assert math.isnan(res_bad_mblh.lateral_cdrr_tco2_hr)

    def test_low_wind_constancy_and_duplicate_asv_id(self):
        obs = _make_acapulco_observations()
        obs[0] = ASVObservation(
            asv_id=obs[0].asv_id,
            latitude=obs[0].latitude,
            longitude=obs[0].longitude,
            u_wind=obs[0].u_wind,
            v_wind=obs[0].v_wind,
            co2_1=obs[0].co2_1,
            co2_2=obs[0].co2_2,
            wind_constancy=0.5,
        )
        res = compute_period_flux(obs, 1000.0)
        assert res.error_bar is ErrorBar.UNSTABLE_WIND_OR_MBLH
        assert math.isclose(res.wind_constancy_min, 0.5, abs_tol=1e-12)
        assert math.isfinite(res.lateral_cdrr_tco2_hr)

        dup = _make_acapulco_observations()
        dup.append(dup[0])
        with pytest.raises(ValueError, match="Duplicate asv_id"):
            compute_period_flux(dup, 1000.0)
