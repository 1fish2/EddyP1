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


@app.cell
def _(mo):
    mo.md(r"""
    # 🌊 EddyP1: Satellite Wind Data Exploration

    This notebook explores satellite ocean wind vectors ($u, v$) and computes derived metrics:
    - **Wind Speed**: Vector magnitude $\sqrt{u^2 + v^2}$ in $\text{m/s}$.
    - **Meteorological Wind Direction**: Direction **from which** the wind is blowing, measured in degrees clockwise from true north ($0^\circ = \text{North}$, $90^\circ = \text{East}$, $180^\circ = \text{South}$, $270^\circ = \text{West}$):
      $$\text{dir}_{\text{deg}} = \left(\operatorname{atan2}(-u, -v) \times \frac{180}{\pi}\right) \pmod{360}$$
    - **Compass Sectors**: 8-point cardinal bins ($\text{N, NE, E, SE, S, SW, W, NW}$).
    """)
    return


@app.cell
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


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(filtered_df, mo):
    mo.md(f"""
    ### 📊 Filtered Dataset Summary
    - **Matching Observations**: {len(filtered_df):,}
    - **Mean Wind Speed**: {filtered_df['wind_speed'].mean():.2f} m/s
    - **Median Wind Speed**: {filtered_df['wind_speed'].median():.2f} m/s
    - **Max Wind Speed**: {filtered_df['wind_speed'].max():.2f} m/s
    """)
    return


@app.cell
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
