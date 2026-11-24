"""Reusable Altair visualization builders and summary helpers for ACPF exploration.

All chart builders accept Polars DataFrames produced by ``perimeter_flux.pipeline``
and sanitize ``NaN`` / ``±inf`` / ``pl.Enum`` columns before rendering so every
chart validates cleanly against the Vega-Lite JSON schema (``to_dict(validate=True)``)
with zero browser JSON errors.
"""

from __future__ import annotations

import math
from datetime import datetime

import altair as alt
import polars as pl

from perimeter_flux.config import ErrorBar

ERROR_BAR_DOMAIN: list[str] = [eb.value for eb in ErrorBar]
ERROR_BAR_COLORS: list[str] = [
    "#2ca02c",  # stable (green)
    "#ff7f0e",  # unstable_wind_or_mblh (amber)
    "#d62728",  # co2_sensor_discrepancy (red)
    "#9467bd",  # two_asv_extrapolated (purple)
    "#7f7f7f",  # insufficient_asvs (gray)
]
ERROR_BAR_SHAPES: list[str] = [
    "circle",
    "triangle-up",
    "diamond",
    "square",
    "cross",
]

_ARROW_SVG: str = "M0,-5 L3.5,4 L0,2 L-3.5,4 Z"


def _sanitize_for_altair(df: pl.DataFrame) -> pl.DataFrame:
    """Convert Polars Enum columns to String and map NaN/Inf floats to JSON-safe nulls."""
    exprs: list[pl.Expr] = []
    for name, dtype in df.schema.items():
        if isinstance(dtype, pl.Enum):
            exprs.append(pl.col(name).cast(pl.String).alias(name))
        elif dtype in (pl.Float64, pl.Float32):
            exprs.append(
                pl.when(pl.col(name).is_finite())
                .then(pl.col(name))
                .otherwise(None)
                .alias(name)
            )
    return df.with_columns(exprs) if exprs else df


def compute_kpis(
    hourly_df: pl.DataFrame,
    *,
    expected_cdrr_tco2_hr: float | None = None,
) -> dict[str, float | int | None]:
    """Compute high-level KPI metrics for the ACPF overview banner."""
    n_periods = hourly_df.height
    if n_periods == 0:
        return {
            "n_periods": 0,
            "n_stable": 0,
            "mean_stable_lateral_cdrr": float("nan"),
            "mean_stable_net_cdrr": float("nan"),
            "expected_cdrr_tco2_hr": expected_cdrr_tco2_hr,
            "mean_net_error_tco2_hr": None,
        }

    eb_str = hourly_df.with_columns(pl.col("error_bar").cast(pl.String))
    stable_df = eb_str.filter(pl.col("error_bar") == ErrorBar.STABLE.value)
    n_stable = stable_df.height

    lat_valid = stable_df.filter(pl.col("lateral_cdrr_tco2_hr").is_finite())["lateral_cdrr_tco2_hr"]
    net_valid = stable_df.filter(pl.col("storage_adjusted_cdrr_tco2_hr").is_finite())[
        "storage_adjusted_cdrr_tco2_hr"
    ]
    mean_lat = float(lat_valid.to_numpy().mean()) if lat_valid.len() > 0 else float("nan")
    mean_net = float(net_valid.to_numpy().mean()) if net_valid.len() > 0 else float("nan")

    err = (
        (mean_net - expected_cdrr_tco2_hr)
        if (expected_cdrr_tco2_hr is not None and math.isfinite(mean_net))
        else None
    )
    return {
        "n_periods": n_periods,
        "n_stable": n_stable,
        "mean_stable_lateral_cdrr": mean_lat,
        "mean_stable_net_cdrr": mean_net,
        "expected_cdrr_tco2_hr": expected_cdrr_tco2_hr,
        "mean_net_error_tco2_hr": err,
    }


def build_sensor_reliability_table(aligned_asvs_df: pl.DataFrame) -> pl.DataFrame:
    """Build a per-ASV dual-CO2 sensor reliability summary table."""
    schema = pl.Schema(
        {
            "asv_id": pl.String,
            "n_periods": pl.Int64,
            "n_flagged_periods": pl.Int64,
            "n_single_sensor_periods": pl.Int64,
            "max_finite_diff_ppm": pl.Float64,
            "mean_finite_diff_ppm": pl.Float64,
            "mean_wind_constancy": pl.Float64,
            "pct_on_hull": pl.Float64,
        }
    )
    if aligned_asvs_df.height == 0:
        return pl.DataFrame(schema=schema)

    finite_diff = pl.when(pl.col("co2_sensor_diff_ppm").is_finite()).then(
        pl.col("co2_sensor_diff_ppm")
    )
    finite_const = pl.when(pl.col("wind_constancy").is_finite()).then(pl.col("wind_constancy"))
    return (
        aligned_asvs_df.group_by("asv_id")
        .agg(
            pl.len().cast(pl.Int64).alias("n_periods"),
            pl.col("co2_sensor_flagged").sum().cast(pl.Int64).alias("n_flagged_periods"),
            pl.col("co2_sensor_diff_ppm")
            .is_infinite()
            .sum()
            .cast(pl.Int64)
            .alias("n_single_sensor_periods"),
            finite_diff.max().round(3).alias("max_finite_diff_ppm"),
            finite_diff.mean().round(3).alias("mean_finite_diff_ppm"),
            finite_const.mean().round(3).alias("mean_wind_constancy"),
            (pl.col("on_hull").mean() * 100.0).round(1).alias("pct_on_hull"),
        )
        .sort("asv_id")
    )


def build_hourly_cdrr_chart(
    hourly_df: pl.DataFrame,
    *,
    expected_cdrr_tco2_hr: float | None = None,
    title: str = "Hourly CDRR (Φ_lat & Φ_net) by Ordered ErrorBar Severity",
) -> alt.LayerChart | alt.FacetChart:
    """Build Panel 1: hourly lateral & storage-adjusted CDRR with ErrorBar color/shape encoding."""
    clean = _sanitize_for_altair(hourly_df)
    marker_df = clean.with_columns(
        pl.coalesce(
            pl.col("storage_adjusted_cdrr_tco2_hr"),
            pl.col("lateral_cdrr_tco2_hr"),
            pl.lit(0.0),
        ).alias("marker_y")
    )

    zero_rule = (
        alt.Chart(pl.DataFrame({"y": [0.0]}))
        .mark_rule(color="#888888", strokeDash=[3, 3], strokeWidth=1)
        .encode(y="y:Q")
    )

    lat_line = (
        alt.Chart(clean)
        .mark_line(strokeDash=[5, 3], color="#4c78a8", opacity=0.75, strokeWidth=1.8)
        .encode(
            x=alt.X("period_start:T", title="UTC Period Start"),
            y=alt.Y("lateral_cdrr_tco2_hr:Q", title="CDRR (tCO₂ / hr)"),
            tooltip=[
                alt.Tooltip("period_start:T", title="Period (UTC)"),
                alt.Tooltip("lateral_cdrr_tco2_hr:Q", format=".2f", title="Lateral Φ_lat"),
                alt.Tooltip("storage_adjusted_cdrr_tco2_hr:Q", format=".2f", title="Net Φ_net"),
                alt.Tooltip("error_bar:N", title="ErrorBar"),
            ],
        )
    )

    net_line = (
        alt.Chart(clean)
        .mark_line(color="#1f3b5c", strokeWidth=2.2)
        .encode(
            x=alt.X("period_start:T", title="UTC Period Start"),
            y=alt.Y("storage_adjusted_cdrr_tco2_hr:Q", title="CDRR (tCO₂ / hr)"),
        )
    )

    error_points = (
        alt.Chart(marker_df)
        .mark_point(filled=True, size=85, stroke="white", strokeWidth=0.6)
        .encode(
            x=alt.X("period_start:T", title="UTC Period Start"),
            y=alt.Y("marker_y:Q", title="CDRR (tCO₂ / hr)"),
            color=alt.Color(
                "error_bar:N",
                title="ErrorBar (Severity)",
                scale=alt.Scale(domain=ERROR_BAR_DOMAIN, range=ERROR_BAR_COLORS),
                sort=ERROR_BAR_DOMAIN,
            ),
            shape=alt.Shape(
                "error_bar:N",
                title="ErrorBar (Severity)",
                scale=alt.Scale(domain=ERROR_BAR_DOMAIN, range=ERROR_BAR_SHAPES),
                sort=ERROR_BAR_DOMAIN,
            ),
            tooltip=[
                alt.Tooltip("period_start:T", title="Period (UTC)"),
                alt.Tooltip("error_bar:N", title="ErrorBar"),
                alt.Tooltip("lateral_cdrr_tco2_hr:Q", format=".2f", title="Φ_lat (tCO₂/hr)"),
                alt.Tooltip("storage_flux_tco2_hr:Q", format=".2f", title="Φ_stor (tCO₂/hr)"),
                alt.Tooltip("storage_adjusted_cdrr_tco2_hr:Q", format=".2f", title="Φ_net (tCO₂/hr)"),
                alt.Tooltip("n_asvs_total:Q", title="ASVs Total"),
                alt.Tooltip("n_asvs_hull:Q", title="ASVs on Hull"),
                alt.Tooltip("wind_dir_delta_deg:Q", format=".1f", title="Δθ Wind (°)"),
                alt.Tooltip("max_co2_sensor_diff_ppm:Q", format=".2f", title="Max ΔCO₂ (ppm)"),
            ],
        )
    )

    layers: list[alt.Chart] = [zero_rule, lat_line, net_line, error_points]
    if expected_cdrr_tco2_hr is not None and math.isfinite(expected_cdrr_tco2_hr):
        gt_rule = (
            alt.Chart(pl.DataFrame({"y": [float(expected_cdrr_tco2_hr)]}))
            .mark_rule(color="#2ca02c", strokeDash=[6, 4], strokeWidth=1.8)
            .encode(y="y:Q")
        )
        layers.append(gt_rule)

    return alt.layer(*layers).properties(title=title, width=760, height=320)


def build_polygon_flux_map(
    edges_df: pl.DataFrame,
    aligned_asvs_df: pl.DataFrame,
    hourly_df: pl.DataFrame,
    selected_period: datetime,
) -> alt.LayerChart | alt.FacetChart:
    """Build Panel 2: local EN (km) CCW convex hull, outward normals, wind vectors, and per-edge flux."""
    edges_p = _sanitize_for_altair(edges_df.filter(pl.col("period_start") == selected_period))
    asvs_p = _sanitize_for_altair(aligned_asvs_df.filter(pl.col("period_start") == selected_period))
    hour_p = _sanitize_for_altair(hourly_df.filter(pl.col("period_start") == selected_period))

    if asvs_p.height == 0:
        empty_note = (
            alt.Chart(pl.DataFrame({"x": [0.0], "y": [0.0], "msg": ["No valid ASVs in selected period"]}))
            .mark_text(fontSize=13, color="#666666")
            .encode(
                x=alt.X("x:Q", title="East (km)"),
                y=alt.Y("y:Q", title="North (km)"),
                text="msg:N",
            )
        )
        return alt.layer(empty_note).properties(
            title=f"Selected Period Polygon & Edge Flux Map ({selected_period.strftime('%Y-%m-%d %H:%M UTC')})",
            width=540,
            height=460,
        )

    # Compute symmetric map bounds & vector arrow scale from ASV coordinates
    xs = asvs_p["x_km"].drop_nulls()
    ys = asvs_p["y_km"].drop_nulls()
    max_extent = max(
        float(xs.abs().to_numpy().max()) if xs.len() > 0 else 100.0,
        float(ys.abs().to_numpy().max()) if ys.len() > 0 else 100.0,
        10.0,
    )
    pad = max_extent * 1.38
    axis_domain = [-pad, pad]
    normal_len_km = max_extent * 0.22
    wind_scale_km_per_ms = max_extent * 0.032

    x_enc = alt.X("x_km:Q", title="East (km)", scale=alt.Scale(domain=axis_domain))
    y_enc = alt.Y("y_km:Q", title="North (km)", scale=alt.Scale(domain=axis_domain))

    layers: list[alt.Chart] = []

    if edges_p.height > 0:
        flux_vals = edges_p["edge_outward_flux_tco2_hr"].drop_nulls()
        max_flux = max(float(flux_vals.abs().to_numpy().max()) if flux_vals.len() > 0 else 1.0, 1.0)

        edges_enriched = edges_p.with_columns(
            (pl.col("mid_x_km") + pl.col("normal_x") * normal_len_km).alias("nx_end_km"),
            (pl.col("mid_y_km") + pl.col("normal_y") * normal_len_km).alias("ny_end_km"),
            (
                (pl.arctan2(pl.col("normal_x"), pl.col("normal_y")) * (180.0 / math.pi)) % 360.0
            ).alias("normal_angle_deg"),
            pl.format(
                "{} t/h",
                pl.col("edge_outward_flux_tco2_hr").round(1),
            ).alias("flux_label"),
        )

        # Hull perimeter segments colored by outward edge flux (blue = inflow, red = outflow)
        edge_rules = (
            alt.Chart(edges_enriched)
            .mark_rule(strokeWidth=4.5, strokeCap="round")
            .encode(
                x=alt.X("x1_km:Q", title="East (km)", scale=alt.Scale(domain=axis_domain)),
                y=alt.Y("y1_km:Q", title="North (km)", scale=alt.Scale(domain=axis_domain)),
                x2="x2_km:Q",
                y2="y2_km:Q",
                color=alt.Color(
                    "edge_outward_flux_tco2_hr:Q",
                    title="Outward Edge Flux (tCO₂/hr)",
                    scale=alt.Scale(
                        scheme="redblue",
                        reverse=True,
                        domain=[-max_flux, max_flux],
                    ),
                ),
                tooltip=[
                    alt.Tooltip("edge_idx:Q", title="Edge # (CCW)"),
                    alt.Tooltip("start_asv_id:N", title="Start ASV"),
                    alt.Tooltip("end_asv_id:N", title="End ASV"),
                    alt.Tooltip("edge_length_km:Q", format=".1f", title="Length (km)"),
                    alt.Tooltip("edge_co2_ppm:Q", format=".2f", title="Edge CO₂ (ppm)"),
                    alt.Tooltip("wind_dot_normal_m_s:Q", format=".2f", title="w · n (m/s)"),
                    alt.Tooltip(
                        "edge_outward_flux_tco2_hr:Q",
                        format=".2f",
                        title="Outward Flux (tCO₂/hr)",
                    ),
                ],
            )
        )

        # Outward unit normal shafts & arrowheads
        normal_shafts = (
            alt.Chart(edges_enriched)
            .mark_rule(color="#444444", strokeWidth=1.5, strokeDash=[3, 2])
            .encode(
                x="mid_x_km:Q",
                y="mid_y_km:Q",
                x2="nx_end_km:Q",
                y2="ny_end_km:Q",
            )
        )
        normal_heads = (
            alt.Chart(edges_enriched)
            .mark_point(shape=_ARROW_SVG, filled=True, size=60, color="#444444")
            .encode(
                x="nx_end_km:Q",
                y="ny_end_km:Q",
                angle=alt.Angle("normal_angle_deg:Q", scale=alt.Scale(domain=[0, 360], range=[0, 360])),
            )
        )
        edge_labels = (
            alt.Chart(edges_enriched)
            .mark_text(fontSize=10, fontWeight="bold", dy=-8, color="#222222")
            .encode(
                x="mid_x_km:Q",
                y="mid_y_km:Q",
                text="flux_label:N",
            )
        )
        layers.extend([edge_rules, normal_shafts, normal_heads, edge_labels])

    # Per-ASV wind arrows and ASV markers (distinguishing Hull vs Interior)
    asvs_enriched = asvs_p.with_columns(
        pl.when(pl.col("on_hull")).then(pl.lit("Hull")).otherwise(pl.lit("Interior")).alias("role"),
        (pl.col("x_km") + pl.col("u_wind") * wind_scale_km_per_ms).alias("wx_end_km"),
        (pl.col("y_km") + pl.col("v_wind") * wind_scale_km_per_ms).alias("wy_end_km"),
        ((pl.arctan2(pl.col("u_wind"), pl.col("v_wind")) * (180.0 / math.pi)) % 360.0).alias(
            "wind_angle_deg"
        ),
        pl.format("{} ({} ppm)", pl.col("asv_id"), pl.col("co2_ppm").round(2)).alias("asv_label"),
    )

    asv_wind_shafts = (
        alt.Chart(asvs_enriched)
        .mark_rule(color="#1f77b4", strokeWidth=2.0, opacity=0.8)
        .encode(x="x_km:Q", y="y_km:Q", x2="wx_end_km:Q", y2="wy_end_km:Q")
    )
    asv_wind_heads = (
        alt.Chart(asvs_enriched)
        .mark_point(shape=_ARROW_SVG, filled=True, size=75, color="#1f77b4", opacity=0.9)
        .encode(
            x="wx_end_km:Q",
            y="wy_end_km:Q",
            angle=alt.Angle("wind_angle_deg:Q", scale=alt.Scale(domain=[0, 360], range=[0, 360])),
        )
    )

    asv_points = (
        alt.Chart(asvs_enriched)
        .mark_point(filled=True, size=140, stroke="#111111", strokeWidth=1.2)
        .encode(
            x=x_enc,
            y=y_enc,
            shape=alt.Shape(
                "role:N",
                title="ASV Role",
                scale=alt.Scale(domain=["Hull", "Interior"], range=["circle", "diamond"]),
            ),
            fill=alt.Fill(
                "role:N",
                title="ASV Role",
                scale=alt.Scale(domain=["Hull", "Interior"], range=["#111111", "#e377c2"]),
            ),
            tooltip=[
                alt.Tooltip("asv_id:N", title="ASV"),
                alt.Tooltip("role:N", title="Role"),
                alt.Tooltip("co2_ppm:Q", format=".2f", title="CO₂ (ppm)"),
                alt.Tooltip("u_wind:Q", format=".2f", title="u Wind (m/s)"),
                alt.Tooltip("v_wind:Q", format=".2f", title="v Wind (m/s)"),
                alt.Tooltip("wind_constancy:Q", format=".3f", title="Wind Constancy"),
                alt.Tooltip("co2_sensor_flagged:N", title="CO₂ Flagged"),
            ],
        )
    )
    asv_labels = (
        alt.Chart(asvs_enriched)
        .mark_text(fontSize=11, fontWeight="bold", dy=-12, color="#111111")
        .encode(x="x_km:Q", y="y_km:Q", text="asv_label:N")
    )
    layers.extend([asv_wind_shafts, asv_wind_heads, asv_points, asv_labels])

    # Polygon-mean wind vector at origin (0, 0)
    if hour_p.height > 0:
        u_mean = hour_p["mean_wind_u"][0]
        v_mean = hour_p["mean_wind_v"][0]
        if u_mean is not None and v_mean is not None and math.hypot(u_mean, v_mean) > 1e-6:
            mean_df = pl.DataFrame(
                {
                    "x0": [0.0],
                    "y0": [0.0],
                    "x1": [u_mean * wind_scale_km_per_ms * 1.4],
                    "y1": [v_mean * wind_scale_km_per_ms * 1.4],
                    "angle": [(math.degrees(math.atan2(u_mean, v_mean))) % 360.0],
                    "label": [f"Mean w ({math.hypot(u_mean, v_mean):.1f} m/s)"],
                }
            )
            layers.append(
                alt.Chart(mean_df)
                .mark_rule(color="#009e73", strokeWidth=3.0)
                .encode(x="x0:Q", y="y0:Q", x2="x1:Q", y2="y1:Q")
            )
            layers.append(
                alt.Chart(mean_df)
                .mark_point(shape=_ARROW_SVG, filled=True, size=110, color="#009e73")
                .encode(
                    x="x1:Q",
                    y="y1:Q",
                    angle=alt.Angle("angle:Q", scale=alt.Scale(domain=[0, 360], range=[0, 360])),
                )
            )
            layers.append(
                alt.Chart(mean_df)
                .mark_text(fontSize=10, fontWeight="bold", dy=14, color="#009e73")
                .encode(x="x0:Q", y="y0:Q", text="label:N")
            )

    eb_str = str(hour_p["error_bar"][0]) if hour_p.height > 0 else "n/a"
    start_id = hour_p["hull_start_asv_id"][0] if hour_p.height > 0 else None
    subtitle = (
        f"ErrorBar: {eb_str} | Hull start: {start_id or 'none'} | "
        f"Blue = inflow, Red = outflow, Dashed = outward normal n_e"
    )
    return alt.layer(*layers).properties(
        title=alt.TitleParams(
            text=f"Polygon Perimeter & Edge Flux Map ({selected_period.strftime('%Y-%m-%d %H:%M UTC')})",
            subtitle=subtitle,
        ),
        width=540,
        height=460,
    )


def build_storage_cancellation_chart(
    hourly_df: pl.DataFrame,
    window_summary_df: pl.DataFrame,
) -> alt.VConcatChart:
    """Build Panel 3: overlaid Φ_lat, Φ_stor, Φ_net, rolling τ_win mean + synchronized box CO2."""
    h_clean = _sanitize_for_altair(hourly_df)
    w_clean = _sanitize_for_altair(window_summary_df)

    joined = h_clean.join(
        w_clean.select(["period_start", "mean_storage_adjusted_cdrr_tco2_hr"]),
        on="period_start",
        how="left",
    )

    flux_long = joined.select(
        [
            "period_start",
            pl.col("lateral_cdrr_tco2_hr").alias("Lateral Φ_lat"),
            pl.col("storage_flux_tco2_hr").alias("Storage Φ_stor"),
            pl.col("storage_adjusted_cdrr_tco2_hr").alias("Storage-Adjusted Φ_net"),
            pl.col("mean_storage_adjusted_cdrr_tco2_hr").alias("Rolling τ_win Mean Φ_net"),
        ]
    ).unpivot(index="period_start", variable_name="component", value_name="flux_tco2_hr")

    comp_domain = [
        "Lateral Φ_lat",
        "Storage Φ_stor",
        "Storage-Adjusted Φ_net",
        "Rolling τ_win Mean Φ_net",
    ]
    comp_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]

    zero_rule = (
        alt.Chart(pl.DataFrame({"y": [0.0]}))
        .mark_rule(color="#888888", strokeDash=[3, 3])
        .encode(y="y:Q")
    )
    flux_lines = (
        alt.Chart(flux_long)
        .mark_line(strokeWidth=2.2)
        .encode(
            x=alt.X("period_start:T", title=""),
            y=alt.Y("flux_tco2_hr:Q", title="Flux (tCO₂ / hr)"),
            color=alt.Color(
                "component:N",
                title="Flux Component",
                scale=alt.Scale(domain=comp_domain, range=comp_colors),
                sort=comp_domain,
            ),
            tooltip=[
                alt.Tooltip("period_start:T", title="Period (UTC)"),
                alt.Tooltip("component:N", title="Component"),
                alt.Tooltip("flux_tco2_hr:Q", format=".2f", title="Flux (tCO₂/hr)"),
            ],
        )
    )
    top_panel = (zero_rule + flux_lines).properties(
        title="Storage-Term Cancellation: Lateral vs. Storage vs. Net CDRR",
        width=760,
        height=260,
    )

    co2_long = h_clean.select(
        [
            "period_start",
            pl.col("box_mean_co2_ppm").alias("Box Mean c̄_box (all valid ASVs)"),
            pl.col("co2_ref_ppm").alias("Hull Ref c_ref (hull vertices)"),
        ]
    ).unpivot(index="period_start", variable_name="series", value_name="co2_ppm")

    bottom_panel = (
        alt.Chart(co2_long)
        .mark_line(strokeWidth=2.0)
        .encode(
            x=alt.X("period_start:T", title="UTC Period Start"),
            y=alt.Y("co2_ppm:Q", title="CO₂ (ppm)", scale=alt.Scale(zero=False)),
            color=alt.Color("series:N", title="CO₂ Series"),
            strokeDash=alt.StrokeDash("series:N", legend=None),
            tooltip=[
                alt.Tooltip("period_start:T", title="Period (UTC)"),
                alt.Tooltip("series:N", title="Series"),
                alt.Tooltip("co2_ppm:Q", format=".3f", title="CO₂ (ppm)"),
            ],
        )
        .properties(title="Box-Mean & Hull-Reference CO₂ Concentration", width=760, height=170)
    )

    return alt.vconcat(top_panel, bottom_panel).resolve_scale(color="independent")  # ty: ignore[invalid-explicit-return]


def build_co2_sensor_comparison_chart(
    aligned_asvs_df: pl.DataFrame,
    *,
    max_co2_sensor_diff_ppm: float = 2.0,
) -> alt.VConcatChart:
    """Build Panel 4: per-ASV co2_1 vs. co2_2 channels and |co2_1 - co2_2| discrepancy vs. threshold."""
    clean = _sanitize_for_altair(aligned_asvs_df)

    # Top panel: dual channels co2_1 (solid) vs co2_2 (dashed) per ASV
    channels_long = clean.select(["period_start", "asv_id", "co2_1", "co2_2"]).unpivot(
        index=["period_start", "asv_id"], variable_name="channel", value_name="co2_ppm"
    )
    top_panel = (
        alt.Chart(channels_long)
        .mark_line(strokeWidth=1.8)
        .encode(
            x=alt.X("period_start:T", title=""),
            y=alt.Y("co2_ppm:Q", title="CO₂ (ppm)", scale=alt.Scale(zero=False)),
            color=alt.Color("asv_id:N", title="ASV"),
            strokeDash=alt.StrokeDash(
                "channel:N",
                title="Sensor Channel",
                scale=alt.Scale(domain=["co2_1", "co2_2"], range=[[1, 0], [5, 3]]),
            ),
            tooltip=[
                alt.Tooltip("period_start:T", title="Period (UTC)"),
                alt.Tooltip("asv_id:N", title="ASV"),
                alt.Tooltip("channel:N", title="Channel"),
                alt.Tooltip("co2_ppm:Q", format=".2f", title="CO₂ (ppm)"),
            ],
        )
        .properties(title="Per-ASV Dual-Sensor CO₂ Channels (co2_1 solid, co2_2 dashed)", width=760, height=230)
    )

    # Bottom panel: finite |co2_1 - co2_2| plus single-sensor fallback markers
    # Separate finite diff from infinite (single-sensor missing) before sanitizing to avoid JSON Infinity
    finite_diffs = aligned_asvs_df.filter(pl.col("co2_sensor_diff_ppm").is_finite())
    max_finite = float(finite_diffs["co2_sensor_diff_ppm"].to_numpy().max()) if finite_diffs.height > 0 else 0.0
    fallback_y = max(max_co2_sensor_diff_ppm * 1.25, max_finite * 1.05, 2.5)

    single_sensor_df = _sanitize_for_altair(
        aligned_asvs_df.filter(pl.col("co2_sensor_diff_ppm").is_infinite()).with_columns(
            pl.lit(fallback_y).alias("fallback_y"),
            pl.lit("Single-sensor fallback (missing channel)").alias("status"),
        )
    )

    diff_line = (
        alt.Chart(clean)
        .mark_line(point=True, strokeWidth=1.8)
        .encode(
            x=alt.X("period_start:T", title="UTC Period Start"),
            y=alt.Y("co2_sensor_diff_ppm:Q", title="|co2_1 - co2_2| (ppm)"),
            color=alt.Color("asv_id:N", legend=None),
            tooltip=[
                alt.Tooltip("period_start:T", title="Period (UTC)"),
                alt.Tooltip("asv_id:N", title="ASV"),
                alt.Tooltip("co2_sensor_diff_ppm:Q", format=".3f", title="|ΔCO₂| (ppm)"),
                alt.Tooltip("co2_sensor_flagged:N", title="Flagged"),
            ],
        )
    )

    thresh_rule = (
        alt.Chart(pl.DataFrame({"y": [float(max_co2_sensor_diff_ppm)]}))
        .mark_rule(color="#d62728", strokeDash=[6, 3], strokeWidth=2.0)
        .encode(y="y:Q")
    )

    fallback_marks = (
        alt.Chart(single_sensor_df)
        .mark_point(shape="cross", size=120, strokeWidth=2.5)
        .encode(
            x=alt.X("period_start:T", title="UTC Period Start"),
            y=alt.Y("fallback_y:Q", title="|co2_1 - co2_2| (ppm)"),
            color=alt.Color("asv_id:N", legend=None),
            tooltip=[
                alt.Tooltip("period_start:T", title="Period (UTC)"),
                alt.Tooltip("asv_id:N", title="ASV"),
                alt.Tooltip("status:N", title="Status"),
            ],
        )
    )

    bottom_panel = (diff_line + thresh_rule + fallback_marks).properties(
        title=f"Dual-Sensor CO₂ Discrepancy vs. Threshold ({max_co2_sensor_diff_ppm:.1f} ppm; × = single-sensor fallback)",
        width=760,
        height=210,
    )

    return alt.vconcat(top_panel, bottom_panel)
