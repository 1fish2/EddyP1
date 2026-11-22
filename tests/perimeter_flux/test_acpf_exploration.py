"""Pytest test suite for ACPF Altair chart builders and Marimo exploration notebook."""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import altair as alt
import polars as pl
import pytest

from perimeter_flux.charts import (
    ERROR_BAR_DOMAIN,
    build_co2_sensor_comparison_chart,
    build_hourly_cdrr_chart,
    build_polygon_flux_map,
    build_sensor_reliability_table,
    build_storage_cancellation_chart,
    compute_kpis,
)
from perimeter_flux.config import ACPFConfig, ERROR_BAR_DTYPE, ErrorBar
from perimeter_flux.pipeline import compute_acpf, summarize_window
from perimeter_flux.sample_data import IDL_EQUATOR_EDDY_VERTICES
from perimeter_flux.synthetic import SCENARIO_NAMES, generate_scenario


def _assert_valid_vegalite_json(chart: alt.TopLevelMixin) -> dict:
    """Validate Vega-Lite schema AND strict JSON compliance (no NaN or Infinity)."""
    spec = chart.to_dict(validate=True)
    # Strict JSON check rejects NaN / Infinity tokens that break browser rendering
    encoded = json.dumps(spec, allow_nan=False)
    assert len(encoded) > 0
    return spec


class TestAltairChartBuildersAcrossScenarios:
    """Validate all 4 chart builders across every synthetic scenario (including 7-ASV sharp front)."""

    @pytest.mark.parametrize("scenario_name", SCENARIO_NAMES)
    def test_all_charts_valid_vegalite_for_scenario(self, scenario_name: str):
        asv_df, met_df, meta = generate_scenario(scenario_name)
        cfg = ACPFConfig()
        res = compute_acpf(asv_df, met_df, config=cfg)
        win = summarize_window(res.hourly, timedelta(hours=6))

        # 1. Hourly CDRR & ErrorBar Overview
        hourly_chart = build_hourly_cdrr_chart(
            res.hourly, expected_cdrr_tco2_hr=meta.expected_cdrr_tco2_hr
        )
        hourly_spec = _assert_valid_vegalite_json(hourly_chart)
        assert "layer" in hourly_spec
        assert len(hourly_spec["layer"]) >= 3

        # 2. Selected-Hour Polygon, Normals & Per-Edge Flux Map
        mid_period = res.hourly["period_start"][res.hourly.height // 2]
        poly_chart = build_polygon_flux_map(
            res.edges, res.aligned_asvs, res.hourly, selected_period=mid_period
        )
        poly_spec = _assert_valid_vegalite_json(poly_chart)
        assert "layer" in poly_spec

        # 3. Storage-Term Cancellation Panel
        storage_chart = build_storage_cancellation_chart(res.hourly, win)
        storage_spec = _assert_valid_vegalite_json(storage_chart)
        assert "vconcat" in storage_spec
        assert len(storage_spec["vconcat"]) == 2

        # 4. CO2 Sensor Pair Comparison Panel
        co2_chart = build_co2_sensor_comparison_chart(
            res.aligned_asvs, max_co2_sensor_diff_ppm=cfg.max_co2_sensor_diff_ppm
        )
        co2_spec = _assert_valid_vegalite_json(co2_chart)
        assert "vconcat" in co2_spec
        assert len(co2_spec["vconcat"]) == 2

    def test_sharp_front_7asv_polygon_map_includes_interior_oshen7(self):
        asv_df, met_df, _ = generate_scenario("sharp_front_7asv")
        res = compute_acpf(asv_df, met_df)
        mid_period = res.hourly["period_start"][res.hourly.height // 2]

        poly_chart = build_polygon_flux_map(
            res.edges, res.aligned_asvs, res.hourly, selected_period=mid_period
        )
        spec = _assert_valid_vegalite_json(poly_chart)
        # Verify both hull and interior ASV roles are represented in the serialized datasets
        datasets_json = json.dumps(spec.get("datasets", {}))
        assert "Oshen1" in datasets_json
        assert "Oshen7" in datasets_json
        assert "Interior" in datasets_json
        assert "Hull" in datasets_json

    def test_idl_equator_scenario_renders_cleanly(self):
        asv_df, met_df, meta = generate_scenario(
            "steady_drawdown", vertices=IDL_EQUATOR_EDDY_VERTICES
        )
        res = compute_acpf(asv_df, met_df)
        mid_period = res.hourly["period_start"][0]
        _assert_valid_vegalite_json(
            build_hourly_cdrr_chart(res.hourly, expected_cdrr_tco2_hr=meta.expected_cdrr_tco2_hr)
        )
        _assert_valid_vegalite_json(
            build_polygon_flux_map(res.edges, res.aligned_asvs, res.hourly, selected_period=mid_period)
        )

    def test_degraded_nan_period_and_empty_frames_do_not_crash(self):
        # Test a 2-ASV outage hour (no hull edges, NaN flux) in degraded_geometry_and_outages
        asv_df, met_df, _ = generate_scenario("degraded_geometry_and_outages")
        res = compute_acpf(asv_df, met_df)
        outage_hours = res.hourly.filter(pl.col("error_bar") == ErrorBar.TWO_ASV_EXTRAPOLATED)
        assert outage_hours.height > 0
        outage_ts = outage_hours["period_start"][0]

        _assert_valid_vegalite_json(
            build_polygon_flux_map(res.edges, res.aligned_asvs, res.hourly, selected_period=outage_ts)
        )

        # Test completely empty DataFrames
        empty_res = compute_acpf(asv_df.clear(), met_df.clear())
        empty_win = summarize_window(empty_res.hourly, timedelta(hours=3))
        _assert_valid_vegalite_json(build_hourly_cdrr_chart(empty_res.hourly))
        _assert_valid_vegalite_json(
            build_polygon_flux_map(
                empty_res.edges,
                empty_res.aligned_asvs,
                empty_res.hourly,
                selected_period=datetime(2026, 6, 1, 0, 0, tzinfo=timezone.utc),
            )
        )
        _assert_valid_vegalite_json(build_storage_cancellation_chart(empty_res.hourly, empty_win))
        _assert_valid_vegalite_json(build_co2_sensor_comparison_chart(empty_res.aligned_asvs))


class TestSummaryTablesAndNotebookIntegrity:
    """Verify sensor reliability table, KPI summary, and Marimo notebook presence."""

    def test_sensor_reliability_table_and_kpis_on_dual_sensor_fault(self):
        asv_df, met_df, _ = generate_scenario("dual_co2_sensor_fault")
        res = compute_acpf(asv_df, met_df)

        table = build_sensor_reliability_table(res.aligned_asvs)
        assert table.height == 6
        oshen3 = table.filter(pl.col("asv_id") == "Oshen3")
        assert int(oshen3["n_flagged_periods"][0]) > 0
        assert int(oshen3["n_single_sensor_periods"][0]) > 0
        assert float(oshen3["max_finite_diff_ppm"][0]) > 5.0

        kpis = compute_kpis(res.hourly, expected_cdrr_tco2_hr=None)
        assert kpis["n_periods"] == res.hourly.height
        assert kpis["n_stable"] > 0
        assert math.isfinite(kpis["mean_stable_lateral_cdrr"])

    def test_error_bar_domain_matches_enum_order(self):
        assert ERROR_BAR_DOMAIN == [eb.value for eb in ErrorBar]

    def test_no_duplicate_legend_fields_in_vconcat_charts(self):
        """Prevent Vega/Marimo Duplicate signal name errors on legend_selection_<field>_tuple."""
        asv_df, met_df, _ = generate_scenario("combined_showcase")
        res = compute_acpf(asv_df, met_df)
        win = summarize_window(res.hourly, timedelta(hours=6))

        def _active_legend_fields(spec: dict) -> list[str]:
            fields: list[str] = []

            def _walk(node: object) -> None:
                if isinstance(node, dict):
                    enc = node.get("encoding")
                    if isinstance(enc, dict):
                        for ch_name in ("color", "strokeDash", "shape", "fill"):
                            ch = enc.get(ch_name)
                            if isinstance(ch, dict) and "field" in ch:
                                # Vega-Lite creates a legend unless legend is explicitly None/False
                                if ch.get("legend", True) is not None and ch.get("legend", True) is not False:
                                    fields.append(str(ch["field"]))
                    for v in node.values():
                        _walk(v)
                elif isinstance(node, list):
                    for item in node:
                        _walk(item)

            _walk(spec)
            return fields

        storage_fields = _active_legend_fields(
            _assert_valid_vegalite_json(build_storage_cancellation_chart(res.hourly, win))
        )
        assert len(storage_fields) == len(set(storage_fields)), f"Duplicate legends: {storage_fields}"

        co2_fields = _active_legend_fields(
            _assert_valid_vegalite_json(build_co2_sensor_comparison_chart(res.aligned_asvs))
        )
        assert len(co2_fields) == len(set(co2_fields)), f"Duplicate legends: {co2_fields}"

    def test_marimo_notebook_exists_and_is_valid_app(self):
        nb_path = Path(__file__).parent.parent.parent / "acpf_exploration.py"
        assert nb_path.exists()
        source = nb_path.read_text(encoding="utf-8")
        assert "marimo.App" in source
        assert "build_hourly_cdrr_chart" in source
        assert "build_polygon_flux_map" in source
        assert "build_storage_cancellation_chart" in source
        assert "build_co2_sensor_comparison_chart" in source
        # Every mo.ui.altair_chart call must disable auto legend_selection on composite charts
        assert source.count("mo.ui.altair_chart(") == source.count("legend_selection=False")
