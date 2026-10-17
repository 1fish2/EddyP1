import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", sql_output="native")


@app.cell
def _():
    import math
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import polars as pl

    return Path, alt, math, mo, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 🌊 EddyP1: Satellite Wind Data Exploration

    This notebook explores satellite ocean wind vectors ($u, v$) and computes derived metrics:
    - **Wind Speed**: Vector magnitude $\sqrt{u^2 + v^2}$ in $\text{m/s}$.
    - **Meteorological Wind Direction**: Direction **from which** the wind is blowing, measured in degrees clockwise from true north ($0^\circ = \text{North}$, $90^\circ = \text{East}$, $180^\circ = \text{South}$, $270^\circ = \text{West}$):
      $$\text{dir}_{\text{from}} = \left(\operatorname{atan2}(-u, -v) \times \frac{180}{\pi}\right) \pmod{360}$$
    - **Vector Flow Heading**: Direction **toward which** the air mass is moving:
      $$\text{dir}_{\text{flow}} = \left(\operatorname{atan2}(u, v) \times \frac{180}{\pi}\right) \pmod{360}$$
    - **Compass Sectors**: 8-point cardinal bins ($\text{N, NE, E, SE, S, SW, W, NW}$).
    """)
    return


@app.cell(hide_code=True)
def _(Path, mo, pl):
    csv_file = Path("data/baja_2020_01_to_04_wind_geo.csv")

    if not csv_file.exists():
        mo.output.replace(
            mo.md(
                f"⚠️ **Dataset not found at `{csv_file}`.** Please run `python main.py` first to download the data."
            )
        )
        raw_df = pl.DataFrame()
    else:
        raw_df = pl.read_csv(csv_file)

    mo.md(
        f"**Source Data**: `{csv_file}` | **Total Rows**: {len(raw_df):,} | **Columns**: {', '.join(raw_df.columns)}"
    )
    return (raw_df,)


@app.cell
def _(math, pl, raw_df):
    # Filter out missing retrievals (e.g. land / dense cloud cover)
    # and compute derived wind metrics
    valid_df = raw_df.filter(
        pl.col("u_wind").is_not_null() & pl.col("v_wind").is_not_null()
    )

    enriched_df = (
        valid_df.with_columns(
            wind_speed=(pl.col("u_wind").pow(2) + pl.col("v_wind").pow(2)).sqrt(),
            wind_dir_deg=(
                (pl.arctan2(-pl.col("u_wind"), -pl.col("v_wind")) * (180.0 / math.pi))
                % 360
            ),
            wind_flow_deg=(
                (pl.arctan2(pl.col("u_wind"), pl.col("v_wind")) * (180.0 / math.pi))
                % 360
            ),
        )
        .with_columns(
            compass_sector=(
                pl.when(
                    pl.col("wind_dir_deg").is_between(337.5, 360)
                    | pl.col("wind_dir_deg").is_between(0, 22.5)
                )
                .then(pl.lit("N"))
                .when(pl.col("wind_dir_deg").is_between(22.5, 67.5))
                .then(pl.lit("NE"))
                .when(pl.col("wind_dir_deg").is_between(67.5, 112.5))
                .then(pl.lit("E"))
                .when(pl.col("wind_dir_deg").is_between(112.5, 157.5))
                .then(pl.lit("SE"))
                .when(pl.col("wind_dir_deg").is_between(157.5, 202.5))
                .then(pl.lit("S"))
                .when(pl.col("wind_dir_deg").is_between(202.5, 247.5))
                .then(pl.lit("SW"))
                .when(pl.col("wind_dir_deg").is_between(247.5, 292.5))
                .then(pl.lit("W"))
                .otherwise(pl.lit("NW"))
            )
        )
    )
    return (enriched_df,)


@app.cell(hide_code=True)
def _(enriched_df, mo):
    min_lat = float(enriched_df["latitude"].min()) if len(enriched_df) > 0 else 0.0
    max_lat = float(enriched_df["latitude"].max()) if len(enriched_df) > 0 else 40.0
    min_lon = float(enriched_df["longitude"].min()) if len(enriched_df) > 0 else -130.0
    max_lon = float(enriched_df["longitude"].max()) if len(enriched_df) > 0 else -80.0
    max_speed_val = (
        float(enriched_df["wind_speed"].max()) if len(enriched_df) > 0 else 25.0
    )

    speed_slider = mo.ui.slider(
        start=0.0,
        stop=round(max_speed_val, 1),
        step=0.5,
        value=0.0,
        label="Minimum Wind Speed (m/s):",
    )

    lat_slider = mo.ui.range_slider(
        start=round(min_lat, 1),
        stop=round(max_lat, 1),
        step=0.5,
        value=[round(min_lat, 1), round(max_lat, 1)],
        label="Latitude Range (°N):",
    )

    mo.vstack([
        mo.md("### 🎛️ Interactive Filters"),
        mo.hstack([speed_slider, lat_slider], justify="start"),
    ])
    return lat_slider, speed_slider


@app.cell
def _(enriched_df, lat_slider, pl, speed_slider):
    filtered_df = enriched_df.filter(
        (pl.col("wind_speed") >= speed_slider.value)
        & (pl.col("latitude") >= lat_slider.value[0])
        & (pl.col("latitude") <= lat_slider.value[1])
    )
    return (filtered_df,)


@app.cell(hide_code=True)
def _(filtered_df, mo):
    mo.md(f"""
    ### 📊 Filtered Dataset Summary
    - **Matching Observations**: {len(filtered_df):,}
    - **Mean Wind Speed**: {filtered_df['wind_speed'].mean():.2f} m/s = {filtered_df['wind_speed'].mean()*3.6:.2f} km/h
    - **Median Wind Speed**: {filtered_df['wind_speed'].median():.2f} m/s = {filtered_df['wind_speed'].median()*3.6:.2f} km/h
    - **Max Wind Speed**: {filtered_df['wind_speed'].max():.2f} m/s = {filtered_df['wind_speed'].max()*3.6:.2f} km/h
    """)
    return


@app.cell(hide_code=True)
def _(alt, filtered_df, mo):
    # Sector distribution (pure Polars into Altair)
    sector_counts = (
        filtered_df["compass_sector"]
        .value_counts()
        .sort("count", descending=True)
    )

    sector_chart = (
        alt.Chart(sector_counts)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X(
                "compass_sector:N",
                title="Wind Direction (From)",
                sort=["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
            ),
            y=alt.Y("count:Q", title="Observations"),
            color=alt.Color(
                "compass_sector:N",
                legend=None,
                scale=alt.Scale(scheme="category10"),
            ),
            tooltip=["compass_sector", "count"],
        )
        .properties(
            title="Wind Direction Distribution (Compass Rose Sectors)",
            width=350,
            height=280,
        )
    )

    # Geospatial downsampled scatter
    sample_size = min(len(filtered_df), 3000)
    sampled = (
        filtered_df.sample(n=sample_size, seed=42)
        if len(filtered_df) > sample_size
        else filtered_df
    )

    geo_chart = (
        alt.Chart(sampled)
        .mark_circle(size=18, opacity=0.6)
        .encode(
            x=alt.X("longitude:Q", title="Longitude (°W)", scale=alt.Scale(zero=False)),
            y=alt.Y("latitude:Q", title="Latitude (°N)", scale=alt.Scale(zero=False)),
            color=alt.Color(
                "wind_speed:Q",
                title="Speed (m/s)",
                scale=alt.Scale(scheme="viridis"),
            ),
            tooltip=[
                "latitude",
                "longitude",
                "wind_speed",
                "wind_dir_deg",
                "compass_sector",
            ],
        )
        .properties(
            title=f"Geospatial Wind Speed (Sample of {sample_size:,} points)",
            width=450,
            height=280,
        )
    )

    mo.hstack([sector_chart, geo_chart], justify="space-around")
    return


@app.cell(hide_code=True)
def _(mo):
    grid_res_selector = mo.ui.radio(
        options={
            "1.0° Grid (Clean Overview)": 1.0,
            "0.5° Grid (Fine Detail)": 0.5,
            "Raw Observations (Subsampled)": 0.0,
        },
        value="1.0° Grid (Clean Overview)",
        label="Spatial Aggregation:",
    )

    color_mode_selector = mo.ui.radio(
        options={
            "Compass Sector (Direction)": "compass_sector",
            "Wind Speed (m/s)": "wind_speed",
        },
        value="Compass Sector (Direction)",
        label="Color By:",
    )

    arrow_size_slider = mo.ui.slider(
        start=2,
        stop=20,
        step=1,
        value=8,
        label="Vector Arrow Size:",
    )

    mo.vstack([
        mo.md("### 🧭 Geospatial Wind Flow Direction & Vector Field"),
        mo.md(
            "Each arrow points in the direction of **wind travel (flow heading)**. "
            "Gridding computes vector averages ($\bar{u}, \bar{v}$) per spatial cell to eliminate track oversampling."
        ),
        mo.hstack([grid_res_selector, color_mode_selector, arrow_size_slider], justify="start"),
    ])
    return arrow_size_slider, color_mode_selector, grid_res_selector


@app.cell(hide_code=True)
def _(
    alt,
    arrow_size_slider,
    color_mode_selector,
    filtered_df,
    grid_res_selector,
    math,
    mo,
    pl,
):
    arrow_svg = "M0,-6 L4,5 L0,2.5 L-4,5 Z"

    res = grid_res_selector.value
    if res > 0.0:
        # Spatial vector binning: average u and v vectors within each grid cell
        scale_factor = 1.0 / res
        vector_data = (
            filtered_df.with_columns(
                lat_cell=(pl.col("latitude") * scale_factor).round() / scale_factor,
                lon_cell=(pl.col("longitude") * scale_factor).round() / scale_factor,
            )
            .group_by(["lat_cell", "lon_cell"])
            .agg(
                pl.len().alias("n_obs"),
                pl.col("u_wind").mean().alias("u_mean"),
                pl.col("v_wind").mean().alias("v_mean"),
            )
            .with_columns(
                latitude=pl.col("lat_cell"),
                longitude=pl.col("lon_cell"),
                wind_speed=(pl.col("u_mean").pow(2) + pl.col("v_mean").pow(2)).sqrt().round(2),
                wind_dir_deg=(
                    (pl.arctan2(-pl.col("u_mean"), -pl.col("v_mean")) * (180.0 / math.pi))
                    % 360
                ).round(1),
                wind_flow_deg=(
                    (pl.arctan2(pl.col("u_mean"), pl.col("v_mean")) * (180.0 / math.pi))
                    % 360
                ).round(1),
            )
            .with_columns(
                compass_sector=(
                    pl.when(
                        pl.col("wind_dir_deg").is_between(337.5, 360)
                        | pl.col("wind_dir_deg").is_between(0, 22.5)
                    )
                    .then(pl.lit("N"))
                    .when(pl.col("wind_dir_deg").is_between(22.5, 67.5))
                    .then(pl.lit("NE"))
                    .when(pl.col("wind_dir_deg").is_between(67.5, 112.5))
                    .then(pl.lit("E"))
                    .when(pl.col("wind_dir_deg").is_between(112.5, 157.5))
                    .then(pl.lit("SE"))
                    .when(pl.col("wind_dir_deg").is_between(157.5, 202.5))
                    .then(pl.lit("S"))
                    .when(pl.col("wind_dir_deg").is_between(202.5, 247.5))
                    .then(pl.lit("SW"))
                    .when(pl.col("wind_dir_deg").is_between(247.5, 292.5))
                    .then(pl.lit("W"))
                    .otherwise(pl.lit("NW"))
                )
            )
        )
    else:
        # Subsampled raw points
        raw_sample_size = min(len(filtered_df), 1500)
        vector_data = (
            filtered_df.sample(n=raw_sample_size, seed=42)
            if len(filtered_df) > raw_sample_size
            else filtered_df
        )

    # Color encoding configuration
    if color_mode_selector.value == "compass_sector":
        color_encoding = alt.Color(
            "compass_sector:N",
            title="Direction (From)",
            sort=["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
            scale=alt.Scale(scheme="category10"),
        )
    else:
        color_encoding = alt.Color(
            "wind_speed:Q",
            title="Speed (m/s)",
            scale=alt.Scale(scheme="viridis"),
        )

    quiver_chart = (
        alt.Chart(vector_data)
        .mark_point(shape=arrow_svg, filled=True, opacity=0.85)
        .encode(
            x=alt.X(
                "longitude:Q",
                title="Longitude (°W)",
                scale=alt.Scale(zero=False),
            ),
            y=alt.Y(
                "latitude:Q",
                title="Latitude (°N)",
                scale=alt.Scale(zero=False),
            ),
            angle=alt.Angle(
                "wind_flow_deg:Q",
                scale=alt.Scale(domain=[0, 360], range=[0, 360]),
            ),
            size=alt.value(arrow_size_slider.value),
            color=color_encoding,
            tooltip=[
                alt.Tooltip("latitude:Q", format=".2f", title="Latitude"),
                alt.Tooltip("longitude:Q", format=".2f", title="Longitude"),
                alt.Tooltip("wind_speed:Q", format=".2f", title="Speed (m/s)"),
                alt.Tooltip("wind_dir_deg:Q", format=".1f", title="Wind From (°)"),
                alt.Tooltip("wind_flow_deg:Q", format=".1f", title="Heading (°)"),
                alt.Tooltip("compass_sector:N", title="Sector"),
            ],
        )
        .properties(
            title=f"Geospatial Wind Flow Direction Map ({len(vector_data):,} vector cells)",
            width=760,
            height=460,
        )
    )

    mo.ui.altair_chart(quiver_chart)
    return


@app.cell
def _(filtered_df, mo, pl):
    mo.md("### 🗺️ High-Wind Spatial Hotspots (1° Grid Cells)")

    hotspots = (
        filtered_df.with_columns(
            lat_cell=pl.col("latitude").round(0),
            lon_cell=pl.col("longitude").round(0),
        )
        .group_by(["lat_cell", "lon_cell"])
        .agg(
            pl.len().alias("n_obs"),
            pl.col("wind_speed").mean().round(2).alias("mean_speed"),
            pl.col("wind_speed").max().round(2).alias("max_speed"),
        )
        .sort("mean_speed", descending=True)
        .head(10)
    )
    return (hotspots,)


@app.cell
def _(hotspots, mo):
    mo.ui.table(hotspots)
    return


@app.cell
def _(mo):
    mo.md("""
    ### 🔍 Raw Observations Table
    """)
    return


@app.cell
def _(filtered_df, mo):
    mo.ui.table(
        filtered_df.head(100),
        pagination=True,
        page_size=10,
    )
    return


if __name__ == "__main__":
    app.run()
