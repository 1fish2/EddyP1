import marimo

__generated_with = "0.25.2"
app = marimo.App(
    width="medium",
    app_title="ACPF Explorer",
    sql_output="native",
)


@app.cell
def _():
    import math
    import sys
    from datetime import timedelta
    from pathlib import Path

    import marimo as mo
    import polars as pl

    _src_dir = str(Path(__file__).resolve().parent / "src")
    if _src_dir not in sys.path:
        sys.path.insert(0, _src_dir)

    from perimeter_flux.charts import (
        build_co2_sensor_comparison_chart,
        build_hourly_cdrr_chart,
        build_polygon_flux_map,
        build_sensor_reliability_table,
        build_storage_cancellation_chart,
        compute_kpis,
    )
    from perimeter_flux.config import (
        ACPFConfig,
        InterpolationMethod,
        WindMode,
    )
    from perimeter_flux.pipeline import compute_acpf, summarize_window
    from perimeter_flux.sample_data import (
        ACAPULCO_EDDY_VERTICES,
        IDL_EQUATOR_EDDY_VERTICES,
    )
    from perimeter_flux.synthetic import SCENARIO_NAMES, generate_scenario

    return (
        ACAPULCO_EDDY_VERTICES,
        ACPFConfig,
        IDL_EQUATOR_EDDY_VERTICES,
        InterpolationMethod,
        SCENARIO_NAMES,
        WindMode,
        build_co2_sensor_comparison_chart,
        build_hourly_cdrr_chart,
        build_polygon_flux_map,
        build_sensor_reliability_table,
        build_storage_cancellation_chart,
        compute_acpf,
        compute_kpis,
        generate_scenario,
        math,
        mo,
        pl,
        summarize_window,
        timedelta,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Atmospheric $\text{CO}_2$ Perimeter Flux (ACPF) Exploration

    Interactive diagnostic dashboard for the 3-stage ACPF pipeline (`resample.py` -> `flux.py` -> `pipeline.py`):
    - **Positive CDRR ($\Phi_{\text{lat}}, \Phi_{\text{net}} > 0$)** denotes **net ocean $\text{CO}_2$ uptake / removal** in $\text{tCO}_2/\text{hr}$.
    - **Storage-adjusted CDRR**: $\Phi_{\text{net}} = \Phi_{\text{lat}} + \Phi_{\text{stor}}$ cancels transient synoptic-front advection spikes.
    - **Ordered `ErrorBar` severity**: `stable < unstable_wind_or_mblh < co2_sensor_discrepancy < two_asv_extrapolated < insufficient_asvs`.
    """)
    return


@app.cell(hide_code=True)
def _(InterpolationMethod, SCENARIO_NAMES, WindMode, mo):
    scenario_dropdown = mo.ui.dropdown(
        options=list(SCENARIO_NAMES),
        value="combined_showcase",
        label="Scenario:",
    )
    eddy_location_dropdown = mo.ui.dropdown(
        options={
            "Acapulco Eddy (13.4°N, 101.7°W)": "acapulco",
            "Equator + IDL Eddy (0°N, ±180°)": "idl_equator",
        },
        value="Acapulco Eddy (13.4°N, 101.7°W)",
        label="Eddy Location:",
    )
    interp_dropdown = mo.ui.dropdown(
        options={m.value: m.value for m in InterpolationMethod},
        value=InterpolationMethod.LINEAR.value,
        label="Stage 1 Interpolator:",
    )
    wind_mode_dropdown = mo.ui.dropdown(
        options={
            "uniform_mean (zero divergence)": WindMode.UNIFORM_MEAN.value,
            "per_asv (local shear)": WindMode.PER_ASV.value,
        },
        value="uniform_mean (zero divergence)",
        label="Wind Mode:",
    )
    window_dropdown = mo.ui.dropdown(
        options={"3 hr": 3, "6 hr": 6, "12 hr": 12, "24 hr": 24},
        value="6 hr",
        label="Rolling Window τ_win:",
    )

    co2_diff_slider = mo.ui.slider(
        start=0.5,
        stop=6.0,
        step=0.25,
        value=2.0,
        label="Max ΔCO₂ Sensor Diff δ_CO2,max (ppm):",
    )
    constancy_slider = mo.ui.slider(
        start=0.50,
        stop=1.00,
        step=0.05,
        value=0.80,
        label="Min Wind Constancy r_const,min:",
    )
    wind_dir_slider = mo.ui.slider(
        start=10.0,
        stop=90.0,
        step=5.0,
        value=30.0,
        label="Max Wind Dir Shift Δθ_w,max (°):",
    )
    mblh_rel_slider = mo.ui.slider(
        start=0.10,
        stop=0.60,
        step=0.05,
        value=0.25,
        label="Max Rel MBLH Change η_H,max:",
    )

    mo.vstack([
        mo.md("### Controls & QA Thresholds"),
        mo.hstack(
            [
                scenario_dropdown,
                eddy_location_dropdown,
                interp_dropdown,
                wind_mode_dropdown,
                window_dropdown,
            ],
            justify="start",
            wrap=True,
        ),
        mo.hstack(
            [co2_diff_slider, constancy_slider, wind_dir_slider, mblh_rel_slider],
            justify="start",
            wrap=True,
        ),
    ])
    return (
        co2_diff_slider,
        constancy_slider,
        eddy_location_dropdown,
        interp_dropdown,
        mblh_rel_slider,
        scenario_dropdown,
        wind_dir_slider,
        wind_mode_dropdown,
        window_dropdown,
    )


@app.cell
def _(
    ACAPULCO_EDDY_VERTICES,
    ACPFConfig,
    IDL_EQUATOR_EDDY_VERTICES,
    InterpolationMethod,
    WindMode,
    co2_diff_slider,
    compute_acpf,
    constancy_slider,
    eddy_location_dropdown,
    generate_scenario,
    interp_dropdown,
    mblh_rel_slider,
    scenario_dropdown,
    summarize_window,
    timedelta,
    wind_dir_slider,
    wind_mode_dropdown,
    window_dropdown,
):
    vertices = (
        ACAPULCO_EDDY_VERTICES
        if eddy_location_dropdown.value == "acapulco"
        else IDL_EQUATOR_EDDY_VERTICES
    )
    asv_df, met_df, scenario_meta = generate_scenario(
        scenario_dropdown.value,
        seed=42,
        vertices=vertices,
    )
    config = ACPFConfig(
        interpolation_method=InterpolationMethod(interp_dropdown.value),
        wind_mode=WindMode(wind_mode_dropdown.value),
        max_co2_sensor_diff_ppm=float(co2_diff_slider.value),
        min_wind_constancy=float(constancy_slider.value),
        max_wind_dir_change_deg=float(wind_dir_slider.value),
        max_mblh_rel_change=float(mblh_rel_slider.value),
    )
    acpf_result = compute_acpf(asv_df, met_df, config=config)
    window_summary = summarize_window(
        acpf_result.hourly,
        timedelta(hours=int(window_dropdown.value)),
    )
    return acpf_result, config, scenario_meta, window_summary


@app.cell(hide_code=True)
def _(
    acpf_result,
    build_hourly_cdrr_chart,
    compute_kpis,
    math,
    mo,
    scenario_meta,
):
    kpis = compute_kpis(
        acpf_result.hourly,
        expected_cdrr_tco2_hr=scenario_meta.expected_cdrr_tco2_hr,
    )

    def _fmt(val):
        if val is None or (isinstance(val, float) and not math.isfinite(val)):
            return "n/a"
        return f"{val:.2f} tCO₂/hr"

    gt_line = (
        f" | **Ground Truth**: {_fmt(kpis['expected_cdrr_tco2_hr'])} "
        f"(Net residual: {_fmt(kpis['mean_net_error_tco2_hr'])})"
        if kpis["expected_cdrr_tco2_hr"] is not None
        else ""
    )
    hourly_chart = build_hourly_cdrr_chart(
        acpf_result.hourly,
        expected_cdrr_tco2_hr=scenario_meta.expected_cdrr_tco2_hr,
    )

    mo.vstack([
        mo.md(f"### 1. Hourly CDRR & `ErrorBar` Overview — `{scenario_meta.name}`"),
        mo.md(
            f"*{scenario_meta.description}*\n\n"
            f"- **Periods**: {kpis['n_periods']} total, **{kpis['n_stable']}** `stable`\n"
            f"- **Mean Stable Lateral CDRR (Φ_lat)**: {_fmt(kpis['mean_stable_lateral_cdrr'])} | "
            f"**Mean Stable Net CDRR (Φ_net)**: {_fmt(kpis['mean_stable_net_cdrr'])}{gt_line}"
        ),
        mo.ui.altair_chart(hourly_chart, chart_selection=False, legend_selection=False),
    ])
    return


@app.cell(hide_code=True)
def _(acpf_result, mo):
    max_idx = max(acpf_result.hourly.height - 1, 0)
    hour_slider = mo.ui.slider(
        start=0,
        stop=max_idx,
        step=1,
        value=min(max_idx // 2, max_idx),
        label="Selected Hour Index:",
    )
    mo.vstack([
        mo.md("### 2. Selected-Hour Polygon, Outward Normals & Per-Edge Flux Map"),
        hour_slider,
    ])
    return (hour_slider,)


@app.cell(hide_code=True)
def _(acpf_result, build_polygon_flux_map, hour_slider, mo, pl):
    if acpf_result.hourly.height == 0:
        _poly_output = mo.md("No periods available.")
    else:
        sel_period = acpf_result.hourly["period_start"][int(hour_slider.value)]
        poly_chart = build_polygon_flux_map(
            acpf_result.edges,
            acpf_result.aligned_asvs,
            acpf_result.hourly,
            selected_period=sel_period,
        )
        sel_edges = acpf_result.edges.filter(pl.col("period_start") == sel_period).select([
            "edge_idx",
            "start_asv_id",
            "end_asv_id",
            "edge_length_km",
            "edge_co2_ppm",
            "wind_dot_normal_m_s",
            "edge_outward_flux_tco2_hr",
        ])
        _poly_output = mo.vstack([
            mo.ui.altair_chart(poly_chart, chart_selection=False, legend_selection=False),
            mo.ui.table(sel_edges, pagination=False),
        ])
    _poly_output
    return


@app.cell(hide_code=True)
def _(acpf_result, build_storage_cancellation_chart, mo, window_summary):
    storage_chart = build_storage_cancellation_chart(acpf_result.hourly, window_summary)
    mo.vstack([
        mo.md("### 3. Storage-Term Cancellation Panel (Φ_lat, Φ_stor, Φ_net, Rolling τ_win)"),
        mo.ui.altair_chart(storage_chart, chart_selection=False, legend_selection=False),
    ])
    return


@app.cell(hide_code=True)
def _(
    acpf_result,
    build_co2_sensor_comparison_chart,
    build_sensor_reliability_table,
    config,
    mo,
):
    co2_chart = build_co2_sensor_comparison_chart(
        acpf_result.aligned_asvs,
        max_co2_sensor_diff_ppm=config.max_co2_sensor_diff_ppm,
    )
    reliability_table = build_sensor_reliability_table(acpf_result.aligned_asvs)
    mo.vstack([
        mo.md("### 4. Dual-Sensor CO₂ Comparison & Reliability Table"),
        mo.ui.altair_chart(co2_chart, chart_selection=False, legend_selection=False),
        mo.ui.table(reliability_table, pagination=False),
    ])
    return


if __name__ == "__main__":
    app.run()
