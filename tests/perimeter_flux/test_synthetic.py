"""Pytest test suite for synthetic Oshen ASV scenario data generators."""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from perimeter_flux.config import ErrorBar
from perimeter_flux.pipeline import compute_acpf
from perimeter_flux.sample_data import ACAPULCO_EDDY_VERTICES, IDL_EQUATOR_EDDY_VERTICES
from perimeter_flux.synthetic import (
    SCENARIO_NAMES,
    ScenarioMetadata,
    generate_scenario,
)

EXPECTED_ASV_COLUMNS = {
    "asv_id": pl.String,
    "timestamp": pl.Datetime("us", "UTC"),
    "latitude": pl.Float64,
    "longitude": pl.Float64,
    "co2_1": pl.Float64,
    "co2_2": pl.Float64,
    "u_wind": pl.Float64,
    "v_wind": pl.Float64,
}


class TestSyntheticRegistryAndSchemas:
    """Verify all 7 scenarios exist, obey Polars schemas, use Oshen1..6, and are deterministic."""

    def test_expected_scenario_names_present(self):
        assert set(SCENARIO_NAMES) == {
            "steady_drawdown",
            "async_sailboat_motion",
            "synoptic_front_storage",
            "sharp_front_7asv",
            "unstable_wind_and_mblh",
            "dual_co2_sensor_fault",
            "degraded_geometry_and_outages",
            "combined_showcase",
        }

    @pytest.mark.parametrize("scenario_name", SCENARIO_NAMES)
    def test_schema_naming_and_deterministic_reproducibility(self, scenario_name: str):
        asv1, met1, meta1 = generate_scenario(scenario_name, seed=42)
        asv2, met2, meta2 = generate_scenario(scenario_name, seed=42)

        assert isinstance(meta1, ScenarioMetadata)
        assert meta1 == meta2
        assert asv1.equals(asv2)
        assert met1.equals(met2)

        for col, dtype in EXPECTED_ASV_COLUMNS.items():
            assert col in asv1.columns
            assert asv1.schema[col] == dtype

        assert "timestamp" in met1.columns
        assert met1.schema["timestamp"] == pl.Datetime("us", "UTC")
        assert "mblh_m" in met1.columns
        assert met1.schema["mblh_m"] == pl.Float64

        asv_ids = sorted(asv1["asv_id"].unique().to_list())
        expected_count = 7 if scenario_name == "sharp_front_7asv" else 6
        assert asv_ids == [f"Oshen{i}" for i in range(1, expected_count + 1)]

    def test_unknown_scenario_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown scenario"):
            generate_scenario("nonexistent_scenario")


class TestSyntheticPhysicsAndTargets:
    """Verify each scenario reproduces its analytical physics and triggers target ErrorBars."""

    @pytest.mark.parametrize(
        "vertices",
        [ACAPULCO_EDDY_VERTICES, IDL_EQUATOR_EDDY_VERTICES],
        ids=["acapulco", "idl_equator"],
    )
    def test_steady_drawdown_recovers_exact_ground_truth_including_idl_equator(self, vertices):
        asv_df, met_df, meta = generate_scenario("steady_drawdown", vertices=vertices)
        assert meta.expected_cdrr_tco2_hr is not None
        assert meta.expected_cdrr_tco2_hr > 0.0

        res = compute_acpf(asv_df, met_df)
        assert res.hourly.height == meta.n_hours
        assert set(res.hourly["error_bar"].to_list()) == {ErrorBar.STABLE.value}

        np.testing.assert_allclose(
            res.hourly["lateral_cdrr_tco2_hr"].to_numpy(),
            meta.expected_cdrr_tco2_hr,
            rtol=1e-9,
        )
        np.testing.assert_allclose(
            res.hourly["storage_adjusted_cdrr_tco2_hr"].to_numpy(),
            meta.expected_cdrr_tco2_hr,
            rtol=1e-9,
        )

    def test_async_sailboat_motion_shows_constancy_drop_drift_and_mean_recovery(self):
        asv_df, met_df, meta = generate_scenario("async_sailboat_motion", seed=7)
        asv_df_alt, _, _ = generate_scenario("async_sailboat_motion", seed=99)
        assert not asv_df.equals(asv_df_alt)

        res = compute_acpf(asv_df, met_df)
        assert res.hourly.height == meta.n_hours
        # Bobbing/turning noise lowers intra-period constancy below 1.0
        assert float(res.hourly["wind_constancy_min"].to_numpy().min()) < 0.98
        # Orbital eddy swirl produces measurable positive drift speed (~0.5 m/s)
        drifts = res.hourly["max_asv_drift_speed_m_s"].to_numpy()
        valid_drifts = drifts[np.isfinite(drifts)]
        assert valid_drifts.size > 0
        assert float(np.mean(valid_drifts)) > 0.2

        # Mean CDRR recovers analytical ground truth closely despite async jitter & bobbing
        assert meta.expected_cdrr_tco2_hr is not None
        mean_cdrr = float(np.nanmean(res.hourly["lateral_cdrr_tco2_hr"].to_numpy()))
        assert math.isclose(mean_cdrr, meta.expected_cdrr_tco2_hr, rel_tol=0.08)

    def test_synoptic_front_storage_cancels_transient_lateral_spike(self):
        asv_df, met_df, meta = generate_scenario("synoptic_front_storage")
        assert meta.expected_cdrr_tco2_hr == 0.0

        res = compute_acpf(asv_df, met_df)
        lat = res.hourly["lateral_cdrr_tco2_hr"].to_numpy()
        net = res.hourly["storage_adjusted_cdrr_tco2_hr"].to_numpy()

        max_lat = float(np.nanmax(np.abs(lat)))
        max_net = float(np.nanmax(np.abs(net)))
        assert max_lat > 100.0  # Strong transient fake uptake spike during front passage
        assert max_net < 0.15 * max_lat  # Storage term cancels > 85% of transient spike

    @pytest.mark.parametrize(
        "scenario_name",
        [
            "unstable_wind_and_mblh",
            "dual_co2_sensor_fault",
            "degraded_geometry_and_outages",
            "combined_showcase",
        ],
    )
    def test_scenarios_trigger_all_expected_error_bar_categories(self, scenario_name: str):
        asv_df, met_df, meta = generate_scenario(scenario_name)
        res = compute_acpf(asv_df, met_df)
        observed = set(res.hourly["error_bar"].to_list())
        for expected_eb in meta.expected_error_bars:
            assert expected_eb.value in observed, (
                f"Scenario {scenario_name!r} did not trigger {expected_eb.value}; got {observed}"
            )

    def test_dual_co2_sensor_fault_preserves_hull_and_includes_spike_and_null(self):
        asv_df, met_df, _ = generate_scenario("dual_co2_sensor_fault")
        res = compute_acpf(asv_df, met_df)

        # Hull geometry preserved at 6 ASVs across all hours
        assert (res.hourly["n_asvs_hull"] == 6).all()
        diffs = res.hourly["max_co2_sensor_diff_ppm"].to_numpy()
        assert np.any(np.isfinite(diffs) & (diffs > 4.0))  # +6 ppm spike
        assert np.any(np.isinf(diffs))  # Single-sensor null fallback

    def test_sharp_front_7th_interior_asv_improves_storage_cancellation(self):
        asv_df_7, met_df, meta = generate_scenario("sharp_front_7asv")
        assert meta.expected_cdrr_tco2_hr == 0.0

        res_7 = compute_acpf(asv_df_7, met_df)
        # Perimeter-only 6-ASV comparison (excluding interior Oshen7)
        res_6 = compute_acpf(asv_df_7.filter(pl.col("asv_id") != "Oshen7"), met_df)

        assert (res_7.hourly["n_asvs_total"] == 7).all()
        assert (res_7.hourly["n_asvs_hull"] == 6).all()
        oshen7_rows = res_7.aligned_asvs.filter(pl.col("asv_id") == "Oshen7")
        assert oshen7_rows.height == meta.n_hours
        assert not oshen7_rows["on_hull"].any()

        # Lateral perimeter flux is identical (uniform wind + identical 6-ASV hull)
        np.testing.assert_allclose(
            res_7.hourly["lateral_cdrr_tco2_hr"].to_numpy(),
            res_6.hourly["lateral_cdrr_tco2_hr"].to_numpy(),
            rtol=1e-10,
        )

        # Adding interior Oshen7 improves box-mean CO2 sampling and lowers peak residual |Phi_net|
        max_lat = float(np.nanmax(np.abs(res_7.hourly["lateral_cdrr_tco2_hr"].to_numpy())))
        max_net_7 = float(np.nanmax(np.abs(res_7.hourly["storage_adjusted_cdrr_tco2_hr"].to_numpy())))
        max_net_6 = float(np.nanmax(np.abs(res_6.hourly["storage_adjusted_cdrr_tco2_hr"].to_numpy())))
        assert max_lat > 100.0
        assert max_net_7 < max_net_6

    def test_degraded_geometry_includes_interior_asv_and_outages(self):
        asv_df, met_df, _ = generate_scenario("degraded_geometry_and_outages")
        res = compute_acpf(asv_df, met_df)

        # Exercise interior Oshen5: 6 total ASVs with 5 on hull
        interior_hours = res.hourly.filter(
            (pl.col("n_asvs_total") == 6) & (pl.col("n_asvs_hull") == 5)
        )
        assert interior_hours.height > 0
        assert (interior_hours["error_bar"] == ErrorBar.STABLE.value).all()
