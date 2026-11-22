# Implementation Plan: Core ACPF Flux Computation (Extended Milestone)

This plan covers the ground-up design and test-driven implementation of the **Atmospheric $\text{CO}_2$ Perimeter Flux (ACPF)** computation using **NumPy, SciPy (`scipy.spatial.ConvexHull`, `scipy.interpolate`), Polars, Marimo, Altair, and Pytest**, replacing the initial [geometry.py](file:///Users/jerry/dev/EddyP1/src/perimeter_flux/geometry.py) prototype (which remains untouched until you choose to delete it).

The persistent, collision-free mathematical specification and symbol glossary have already been created at [acpf_math.md](file:///Users/jerry/dev/EddyP1/doc/acpf_math.md).

---

## 1. Design Decisions Record

Every architectural choice agreed upon during the design interview:

1. **Terminology & PEP 8 Naming**: Replace `"Craft"` with `"ASV"` throughout. Follow PEP 8 acronym capitalization: `ASVObservation`, `ACPFConfig`, `PeriodFluxResult`, `InterpolationMethod`, `WindMode`, `ErrorBar` for classes; lowercase snake_case (`asv_id`, `co2_1`, `co2_2`, `u_wind`, `v_wind`, `mblh_m`) for variables and Polars columns.
2. **Spatial Integration Model**: Discrete closed-polygon line integral via 2D `scipy.spatial.ConvexHull` for $N_{\text{hull}} \ge 3$. Rotate the guaranteed-CCW hull vertex cycle so that the **alphabetically smallest `asv_id` on the hull** is vertex `0`. Defer the $N = 2$ extrapolation formula (returning `NaN` flux with `ErrorBar.TWO_ASV_EXTRAPOLATED` for now).
3. **Wind Cancellation for Bobbing/Tilting Sailboats**:
   - Within each period $\Delta t$, vector-average $(u, v)$ per ASV to cancel zero-mean turning oscillation, and compute intra-period wind vector constancy $r_{\text{const},i}$ to quantify turning/shift noise.
   - Across the hull, default `WindMode` to `UNIFORM_MEAN` (spatial vector mean $\bar{\mathbf{w}}_k$ across all valid ASVs, or `EXTERNAL` satellite wind when provided) so horizontal wind divergence is strictly zero and individual sailboat tilt/heading biases do not multiply the $\sim 420\text{ ppm}$ $\text{CO}_2$ background.
   - Always subtract hull-mean $c_{\text{ref}}$ before multiplying edge fluxes (eliminating catastrophic float cancellation, and suppressing $c_{\text{ref}} D_w$ error when running in `WindMode.PER_ASV` mode), while recording the raw local wind divergence $D_w$ as a diagnostic.
   - Rebuild the `ConvexHull` fresh each period so both fixed-coordinate and current-drifting ASVs work identically; track ASV horizontal drift speed $\mathbf{w}_{\text{asv}}$ as a diagnostic.
4. **Two-Stage Modular Pipeline**:
   - **Stage 1 (`resample.py`, Polars + SciPy)**: Aligns asynchronous sub-hourly/irregular ASV telemetry onto the regular $\Delta t$ grid (named constant `DEFAULT_TIME_INTERVAL = timedelta(hours=1)`).
   - **Stage 2 (`flux.py`, NumPy + SciPy)**: Stateless per-period geometric & thermodynamic flux kernel—directly callable on pre-aggregated hourly data for fast, clean TDD unit tests.
   - **Stage 3 (`pipeline.py`, Polars)**: Multi-period orchestrator computing inter-period wind/$H_{\text{MBL}}$ stability, Eulerian column storage term $\Phi_{\text{stor}}$, storage-adjusted CDRR $\Phi_{\text{net}}$, `ErrorBar` severity, and multi-period rolling window $\tau_{\text{win}}$ summaries.
5. **Asynchronous Resampling & Pluggable `InterpolationMethod`**:
   - Unified time-weighted interval integral $\bar{q}_{i,k} = \frac{1}{\Delta t}\int_{t_k}^{t_k+\Delta t} \hat{q}_i(t)\,dt$ over 3D ECEF position, $(u, v)$ wind, scalar wind speed $s$, and dual $\text{CO}_2$ channels.
   - `InterpolationMethod.LINEAR` (`make_interp_spline(..., k=1)`) as default; `InterpolationMethod.PCHIP` (`PchipInterpolator`, zero overshoot) and `InterpolationMethod.AKIMA` (`Akima1DInterpolator`) pluggable through the same `.integrate(a, b)` interface.
   - Strict no-extrapolation and maximum-gap guard ($\Delta t_{\text{gap,max}}$, default $3\,\Delta t$).
6. **Geodesy & Dependencies**: Pure NumPy WGS 84 `(lat, lon) ↔ ECEF` and IDL-safe `ECEF → local ENU` tangent plane ($<0.02\%$ horizontal distortion at $200\text{ km}$, zero seam bugs at $\pm 180^\circ$ or poles, no `pyproj`/`shapely` dependency needed). Add only `numpy` and `scipy` to [pyproject.toml](file:///Users/jerry/dev/EddyP1/pyproject.toml) via `uv`.
7. **Units, Sign Convention & Atmospheric Scaling**:
   - Output in **$\text{tCO}_2/\text{hr}$**, **positive = net ocean carbon removal / uptake** (CDRR = $-\Phi_{\text{outward}}$), plus areal density $\text{tCO}_2\,\text{km}^{-2}\,\text{hr}^{-1}$.
   - Require per-period $H_{\text{MBL}}$ (`mblh_m`); optional $T_{\text{air}}$ (`t_air_k`, default $293.15\text{ K}$) and $P_{\text{atm}}$ (`p_atm_pa`, default $101{,}325.0\text{ Pa}$) with hydrostatic barometric effective height $H_{\text{eff}} = H_{\text{scale}}(1 - e^{-H_{\text{MBL}}/H_{\text{scale}}})$.
8. **Non-Steady-State Transients**: Compute pure lateral perimeter flux $\Phi_{\text{lat}}$ in Stage 2; append column storage flux $\Phi_{\text{stor}}$ (centered difference $\Delta \bar{c}_{\text{box}} / \Delta t$ inside contiguous runs, one-sided at endpoints) and storage-adjusted CDRR $\Phi_{\text{net}} = \Phi_{\text{lat}} + \Phi_{\text{stor}}$ in Stage 3.
9. **Dual-Sensor $\text{CO}_2$ Reconciliation**: Average `(co2_1 + co2_2) / 2` (or fall back to the single finite sensor if one is null/NaN); keep the ASV in the hull; elevate `ErrorBar` to `CO2_SENSOR_DISCREPANCY` and record `max_co2_sensor_diff_ppm` + flagged ASV IDs whenever $|c_{i,1} - c_{i,2}| > \delta_{\text{CO}_2,\max}$ or one sensor is missing.
10. **Interior & Degenerate ASVs**: Integrate edges on `hull.vertices` only; include interior ASVs in $\bar{\mathbf{w}}_k$, $\bar{c}_{\text{box},k}$, and sensor QA diagnostics. Reject collinear or razor-thin hulls ($A_{\text{hull}} < A_{\min}$ or $Q_{\text{iso}} < Q_{\text{iso,min}}$, never using Qhull `"QJ"` joggle) and classify them as `ErrorBar.TWO_ASV_EXTRAPOLATED` with `NaN` flux.
11. **Ordered `ErrorBar` & Config**: Python `StrEnum` mapped directly to Polars `pl.Enum(ErrorBar)` in increasing severity order (`stable < unstable_wind_or_mblh < co2_sensor_discrepancy < two_asv_extrapolated < insufficient_asvs`), using worst-triggered precedence plus flat numeric diagnostic columns and a frozen `ACPFConfig` dataclass.
12. **Documentation & Visualization**: Persistent collision-free math spec at [acpf_math.md](file:///Users/jerry/dev/EddyP1/doc/acpf_math.md), inline equations & glossary definitions directly beside every algorithm stage in code, synthetic ASV scenario generator using `"Oshen1"`, `"Oshen2"`, ..., `"Oshen6"`, and a tested Marimo ACPF visualization notebook.
13. **Deferred Satellite Source Maturity**: This initial iteration uses ASV data (plus MBLH/met inputs) and keeps `ErrorBar` strictly 5-level. Future satellite providers (unavailable vs. low-accuracy near real-time vs. delayed high-accuracy science-grade) are deferred until upstream provenance metadata is defined; when added, prefer an orthogonal source-maturity/provenance column (following CF/QARTOD conventions) rather than mixing data latency into the 1D physical `ErrorBar` severity scale.

---

## 2. Proposed File & Module Changes

> [!NOTE]
> Per [GEMINI.md](file:///Users/jerry/dev/EddyP1/GEMINI.md): keep [__init__.py](file:///Users/jerry/dev/EddyP1/src/perimeter_flux/__init__.py) empty (import directly from submodules), leave [geometry.py](file:///Users/jerry/dev/EddyP1/src/perimeter_flux/geometry.py) and [test_geometry.py](file:///Users/jerry/dev/EddyP1/tests/perimeter_flux/test_geometry.py) untouched for now, and write tests before implementation code for every module.

### 2.1 Environment & Dependencies
#### [MODIFY] [pyproject.toml](file:///Users/jerry/dev/EddyP1/pyproject.toml)
- Add `numpy` and `scipy` via `uv add numpy scipy` (updates `pyproject.toml` and `uv.lock`).

### 2.2 Persistent Math Spec
#### [CREATED] [acpf_math.md](file:///Users/jerry/dev/EddyP1/doc/acpf_math.md)
- Already written with collision-free notation (`T_air`, `Δt`, `τ_win`, `H_MBL`, `H_scale`, `H_eff`, `R_gas`, `R_d`, `R_enu`, `r_const`, `ν(φ)`, `Φ_lat`, `Φ_stor`, `Φ_net`, `ψ_lat`, `ψ_net`), complete glossary tables, stage equations, and deferred small-effects list.

### 2.3 Five Lean Core Modules (`src/perimeter_flux/`)

#### [NEW] `src/perimeter_flux/config.py`
- Named constants: `DEFAULT_TIME_INTERVAL = timedelta(hours=1)`, WGS 84 constants (`WGS84_SEMI_MAJOR_M`, `WGS84_FLATTENING`, `WGS84_E2`), CODATA/NIST atmospheric constants (`UNIVERSAL_GAS_CONSTANT`, `MOLAR_MASS_CO2_KG`, `MOLAR_MASS_DRY_AIR_KG`, `DRY_AIR_GAS_CONSTANT`, `STANDARD_GRAVITY`, `KG_PER_S_TO_T_PER_HR`, `DEFAULT_T_AIR_K = 293.15`, `DEFAULT_P_ATM_PA = 101_325.0`), and default QA thresholds.
- Enums:
  - `InterpolationMethod(StrEnum)`: `LINEAR`, `PCHIP`, `AKIMA`.
  - `WindMode(StrEnum)`: `UNIFORM_MEAN`, `EXTERNAL`, `PER_ASV`.
  - `ErrorBar(StrEnum)`: ordered severity scale (`STABLE`, `UNSTABLE_WIND_OR_MBLH`, `CO2_SENSOR_DISCREPANCY`, `TWO_ASV_EXTRAPOLATED`, `INSUFFICIENT_ASVS`), plus `ERROR_BAR_DTYPE = pl.Enum(ErrorBar)`.
- Frozen dataclass `ACPFConfig` with `__post_init__` validation of positive intervals and threshold bounds.

#### [NEW] `src/perimeter_flux/geodesy.py`
- Pure NumPy vectorized geodetic transforms (inline equations + glossary block mirroring Section 3 of [acpf_math.md](file:///Users/jerry/dev/EddyP1/doc/acpf_math.md)):
  - `latlon_to_ecef(lat_deg, lon_deg) -> NDArray[np.float64]` (shape `(N, 3)` in meters).
  - `ecef_to_latlon(xyz_m) -> tuple[NDArray[np.float64], NDArray[np.float64]]` (Bowring / 2-step Newton-Raphson inverse, accurate to $< 10^{-12}\text{ deg}$).
  - `ecef_origin_from_points(xyz_m) -> tuple[float, float, NDArray[np.float64]]` (IDL-safe 3D centroid projected onto sea-level WGS 84 ellipsoid).
  - `ecef_to_enu(xyz_m, origin_lat_rad, origin_lon_rad, origin_xyz_m) -> NDArray[np.float64]` (returns local `(E, N, U)` in meters).
  - `project_latlon_to_local_en(lat_deg, lon_deg)` convenience wrapper returning `(xy_m, origin_lat_deg, origin_lon_deg)`.

#### [NEW] `src/perimeter_flux/resample.py`
- Stage 1 asynchronous telemetry alignment (inline equations + glossary block mirroring Section 4 of [acpf_math.md](file:///Users/jerry/dev/EddyP1/doc/acpf_math.md)):
  - `build_interpolator(t_sec, values, method: InterpolationMethod)` returning a SciPy piecewise polynomial/spline with `.integrate(a, b)` (`make_interp_spline(..., k=1)`, `PchipInterpolator(..., extrapolate=False)`, `Akima1DInterpolator(..., extrapolate=False)` with graceful fallback when $< 5$ points).
  - `interval_average_channel(t_obs_sec, values, period_starts_sec, dt_sec, method, max_gap_sec) -> NDArray[np.float64]` with duplicate-timestamp averaging, no-extrapolation mask, and gap guard.
  - `resample_asv_telemetry(asv_telemetry: pl.DataFrame, period_starts: pl.Series, config: ACPFConfig) -> pl.DataFrame` producing period-aligned per-ASV ECEF, lat/lon, `(u_wind, v_wind)`, `wind_constancy`, `co2_1`, `co2_2`, and inter-period `asv_drift_speed_m_s`.
  - `resample_met_series(met_df: pl.DataFrame, period_starts: pl.Series, config: ACPFConfig) -> pl.DataFrame` aligning `mblh_m`, optional `t_air_k`, `p_atm_pa`, `u_wind_ext`, `v_wind_ext`.

#### [NEW] `src/perimeter_flux/flux.py`
- Stage 2 stateless per-period flux kernel (inline equations + glossary block mirroring Section 5 of [acpf_math.md](file:///Users/jerry/dev/EddyP1/doc/acpf_math.md)):
  - Frozen dataclasses: `ASVObservation` (single-period per-ASV state for direct Stage 2 calls/tests), `EdgeFluxDetail` (per-edge geometry, outward unit normal, wind dot normal, mean $\text{CO}_2$, outward flux), and `PeriodFluxResult` (scalar flux, areal flux, geometry metrics, diagnostics, `ErrorBar`, and `edges: tuple[EdgeFluxDetail, ...]`).
  - `effective_mblh_m(mblh_m, t_air_k) -> float` and `co2_mass_per_ppm_kg_m3(t_air_k, p_atm_pa) -> float`.
  - `compute_period_flux(observations: Sequence[ASVObservation], mblh_m: float, *, config: ACPFConfig = ..., t_air_k: float | None = None, p_atm_pa: float | None = None, external_wind: tuple[float, float] | None = None) -> PeriodFluxResult`:
    - Reconciles dual $\text{CO}_2$ sensors (`co2_1`, `co2_2`) and flags discrepancies / single-sensor fallback.
    - Projects via `geodesy.py` to local EN meters; checks $N_{\text{tot}}$, rank, `ConvexHull`, $A_{\min}$, and $Q_{\text{iso,min}}$.
    - Rotates CCW `hull.vertices` so the alphabetically smallest hull `asv_id` is index `0`.
    - Computes outward unit normals, raw local wind divergence $D_w$, `WindMode` edge winds, $c_{\text{ref}}$-subtracted per-edge fluxes $\Phi_{\text{edge},e}$, net lateral CDRR $\Phi_{\text{lat}}$, and areal density $\psi_{\text{lat}}$.

#### [NEW] `src/perimeter_flux/pipeline.py`
- Stage 3 multi-period Polars orchestrator (inline equations + glossary block mirroring Section 6 of [acpf_math.md](file:///Users/jerry/dev/EddyP1/doc/acpf_math.md)):
  - Frozen dataclass `ACPFResult(hourly: pl.DataFrame, edges: pl.DataFrame, aligned_asvs: pl.DataFrame)` so flat Altair-ready hourly tables, per-edge polygon tables, and aligned per-ASV dual-sensor tables are all directly available to notebooks and downstream pipelines.
  - `compute_acpf(asv_telemetry: pl.DataFrame, met_series: pl.DataFrame, *, config: ACPFConfig = ..., period_starts: pl.Series | None = None) -> ACPFResult`:
    - Runs Stage 1 resampling (or accepts already-aligned hourly inputs cleanly via a direct helper `compute_acpf_from_aligned(...)`).
    - Evaluates Stage 2 per period.
    - Computes inter-period wind direction delta ($\Delta\theta_w$), relative wind speed delta, relative $H_{\text{MBL}}$ delta, combines with intra-hour `wind_constancy_min`, and applies ordered `ErrorBar` worst-severity precedence.
    - Computes centered/boundary $\Delta\bar{c}_{\text{box}}/\Delta t$, column storage flux $\Phi_{\text{stor}}$, storage-adjusted CDRR $\Phi_{\text{net}}$, and areal density $\psi_{\text{net}}$.
  - `summarize_window(hourly_df: pl.DataFrame, window_duration: timedelta, *, max_error_bar: ErrorBar = ErrorBar.UNSTABLE_WIND_OR_MBLH) -> pl.DataFrame` for multi-period $\tau_{\text{win}}$ rolling/fixed summaries.

### 2.4 Synthetic ASV Scenario Data Generator
#### [NEW] `src/perimeter_flux/synthetic.py`
- Generates realistic asynchronous and synchronous multi-ASV datasets named **`"Oshen1"`, `"Oshen2"`, ..., `"Oshen6"`** around an Acapulco-scale ocean eddy (~200 km diameter, ~13.4°N, 101.7°W, matching [6_points_around_acapulco_eddy.csv](file:///Users/jerry/dev/EddyP1/test_data/6_points_around_acapulco_eddy.csv)), returning `(asv_telemetry_df, met_df, ground_truth_metadata)`:
  1. `steady_drawdown`: Known analytical linear spatial $\text{CO}_2$ gradient + steady uniform wind + constant $H_{\text{MBL}}$ → exact ground-truth $\Phi_{\text{lat}} = \Phi_{\text{net}}$ (great for both TDD and visual baseline!).
  2. `async_sailboat_motion`: Asynchronous irregular timestamps (e.g., 5–12 min jitter per ASV), high-frequency Oshen sailboat turning/bobbing wind noise, and optional slow current-following orbital drift around the eddy.
  3. `synoptic_front_storage`: Zero ocean uptake + a $+2.0\text{ ppm}$ synoptic $\text{CO}_2$ front propagating across the 200 km eddy over 6–10 hours, explicitly demonstrating how $\Phi_{\text{stor}}$ cancels the transient fake spike in $\Phi_{\text{lat}}$.
  4. `unstable_wind_and_mblh`: Sudden wind-direction shift ($> 45^\circ$/hr) and rapid morning $H_{\text{MBL}}$ ramp triggering `ErrorBar.UNSTABLE_WIND_OR_MBLH`.
  5. `dual_co2_sensor_fault`: `Oshen3`'s `co2_2` drifts gradually then spikes by $+6\text{ ppm}$ and later drops to `null`, triggering `ErrorBar.CO2_SENSOR_DISCREPANCY` while preserving hull geometry.
  6. `degraded_geometry_and_outages`: Sequence transitioning through 6 ASVs → 1 interior ASV (`Oshen5` blown inside hull) → telemetry outage leaving $N = 2$ (`TWO_ASV_EXTRAPOLATED`) → $N = 1$ (`INSUFFICIENT_ASVS`).
  7. `combined_showcase`: A 48-hour multi-phase mission combining steady drawdown, a passing synoptic front, a wind/MBLH shift, an `Oshen3` dual-sensor drift event, and a brief 2-ASV outage for rich Marimo exploration.

### 2.5 Initial Tested Marimo ACPF Visualization Notebook
#### [NEW] `acpf_exploration.py`
- Interactive Marimo notebook (matching root-level [wind_exploration.py](file:///Users/jerry/dev/EddyP1/wind_exploration.py)) with pure, reusable Altair chart-builder functions (so visualization code is unit-tested via pytest per [GEMINI.md](file:///Users/jerry/dev/EddyP1/GEMINI.md) with zero logic duplication):
  1. **Interactive Controls**: Scenario selector (`combined_showcase`, `steady_drawdown`, `synoptic_front_storage`, `dual_co2_sensor_fault`, `async_sailboat_motion`, etc.), `InterpolationMethod` (`linear` / `pchip` / `akima`), `WindMode` (`uniform_mean` / `per_asv`), `ACPFConfig` QA sliders ($\delta_{\text{CO}_2,\max}$, $r_{\text{const,min}}$, $\Delta\theta_{w,\max}$, $\eta_{H,\max}$), rolling window $\tau_{\text{win}}$ selector, and selected-hour slider.
  2. **Hourly CDRR & `ErrorBar` Overview**: Time-series chart of $\Phi_{\text{lat}}$ and $\Phi_{\text{net}}$ with color/shape encoding for the ordered `ErrorBar` enum, plus summary KPIs (mean CDRR across stable periods, ground-truth comparison when available).
  3. **Selected-Hour Polygon, Outward Normals & Per-Edge Flux Map**: Local EN (km) geospatial plot of the CCW `ConvexHull` starting from the alphabetically first hull ASV (`Oshen1`), showing interior ASVs if any, outward unit normal vectors $\hat{\mathbf{n}}_e$, wind arrows $\mathbf{w}_i$ and $\bar{\mathbf{w}}_k$, and edges colored/labeled by outward edge flux $\Phi_{\text{edge},e}$ (inflow vs. outflow).
  4. **Storage-Term Cancellation Panel**: Side-by-side / overlaid plot of $\Phi_{\text{lat}}$ (lateral), $\Phi_{\text{stor}}$ (storage), $\Phi_{\text{net}}$ (storage-adjusted), and rolling $\tau_{\text{win}}$ mean, alongside $\bar{c}_{\text{box}}(t)$ so front-passage cancellation is visually unmistakable.
  5. **$\text{CO}_2$ Sensor Pair Comparison Panel**: Per-ASV (`Oshen1`–`Oshen6`) comparison of `co2_1` vs. `co2_2`, discrepancy time series $\delta c_i = |c_{i,1} - c_{i,2}|$ with the $\delta_{\text{CO}_2,\max}$ threshold rule, and a sortable sensor-reliability summary table.

---

## 3. Polars Data Contracts

### 3.1 Input Schemas
- **`asv_telemetry` (`pl.DataFrame`)**:
  `asv_id: pl.String`, `timestamp: pl.Datetime("us", "UTC")`, `latitude: pl.Float64`, `longitude: pl.Float64`, `co2_1: pl.Float64` (nullable), `co2_2: pl.Float64` (nullable), `u_wind: pl.Float64`, `v_wind: pl.Float64`.
- **`met_series` (`pl.DataFrame`)**:
  `timestamp: pl.Datetime("us", "UTC")`, `mblh_m: pl.Float64`, plus optional `t_air_k: pl.Float64`, `p_atm_pa: pl.Float64`, `u_wind_ext: pl.Float64`, `v_wind_ext: pl.Float64`.

### 3.2 Output Schemas (`ACPFResult`)
- **`hourly` (`pl.DataFrame`, flat & Altair-ready)**:
  - Time & Flux: `period_start`, `lateral_cdrr_tco2_hr`, `storage_flux_tco2_hr`, `storage_adjusted_cdrr_tco2_hr`, `lateral_cdrr_tco2_km2_hr`, `storage_adjusted_cdrr_tco2_km2_hr`.
  - Quality: `error_bar: pl.Enum(ErrorBar)`.
  - Geometry & Counts: `n_asvs_total`, `n_asvs_hull`, `hull_start_asv_id`, `hull_area_km2`, `hull_perimeter_km`, `hull_compactness`.
  - Met & Atmospheric State: `mblh_m`, `effective_mblh_m`, `t_air_k`, `p_atm_pa`, `box_mean_co2_ppm`, `co2_ref_ppm`.
  - Diagnostics: `mean_wind_u`, `mean_wind_v`, `mean_wind_speed_m_s`, `wind_divergence_m2_s`, `wind_constancy_min`, `wind_dir_delta_deg`, `wind_speed_rel_delta`, `mblh_rel_delta`, `max_co2_sensor_diff_ppm`, `n_co2_flagged_asvs`, `max_asv_drift_speed_m_s`.
- **`edges` (`pl.DataFrame`, per-edge polygon visualization)**:
  `period_start`, `edge_idx`, `start_asv_id`, `end_asv_id`, `x1_km`, `y1_km`, `x2_km`, `y2_km`, `mid_x_km`, `mid_y_km`, `normal_x`, `normal_y`, `edge_length_km`, `edge_co2_ppm`, `edge_co2_anomaly_ppm`, `edge_wind_u`, `edge_wind_v`, `wind_dot_normal_m_s`, `edge_outward_flux_tco2_hr`.
- **`aligned_asvs` (`pl.DataFrame`, dual-sensor & ASV reliability inspection)**:
  `period_start`, `asv_id`, `latitude`, `longitude`, `x_km`, `y_km`, `on_hull`, `co2_1`, `co2_2`, `co2_ppm`, `co2_sensor_diff_ppm`, `co2_sensor_flagged`, `u_wind`, `v_wind`, `wind_constancy`, `asv_drift_speed_m_s`.

---

## 4. Test-Driven Development (TDD) Execution Sequence

Each phase writes the `pytest` tests **first** (verifying they fail for the right reason), then implements the module until `uv run pytest` passes cleanly:

1. **Phase 1 — Dependencies & `config.py`**:
   - Run `uv add numpy scipy` (verify Python 3.15 prebuilt wheels).
   - Write `tests/perimeter_flux/test_config.py` (default `DEFAULT_TIME_INTERVAL == 1 hr`, physical constant sanity, `ErrorBar` ordering in both Python and Polars `pl.Enum` comparisons/sorting, `InterpolationMethod` & `WindMode` values, `ACPFConfig` validation errors) → implement `src/perimeter_flux/config.py`.
2. **Phase 2 — `geodesy.py`**:
   - Write `tests/perimeter_flux/test_geodesy.py` (equator/pole/mid-latitude WGS 84 ECEF exact values, round-trip `latlon -> ecef -> latlon` $< 10^{-11}\text{ deg}$, ENU orthonormality $\mathbf{R}_{\text{enu}}^\top \mathbf{R}_{\text{enu}} = \mathbf{I}$, East/North distance accuracy across Acapulco eddy, International Date Line $\pm 179.5^\circ$ continuity with zero seam jump) → implement `src/perimeter_flux/geodesy.py`.
3. **Phase 3 — `resample.py` (Stage 1)**:
   - Write `tests/perimeter_flux/test_resample.py` (exact reproduction of constants and linear ramps $a + b(t_k + \Delta t/2)$ for irregular async timestamps under `LINEAR` and `PCHIP`; exact trapezoidal cancellation of symmetric sailboat wind oscillation + `wind_constancy < 1`; PCHIP zero-overshoot & total-integral conservation; ECEF interpolation across $\pm 180^\circ$; strict no-extrapolation and `max_interp_gap` dropout; duplicate timestamp handling; ASV drift velocity accuracy) → implement `src/perimeter_flux/resample.py`.
4. **Phase 4 — `flux.py` (Stage 2)**:
   - Write `tests/perimeter_flux/test_flux.py`:
     - Analytical box-flux check on a known square/hexagon with linear $\text{CO}_2$ gradient and uniform wind (exact match to $\gamma_{\text{unit}} \kappa_{\text{CO}_2} H_{\text{eff}} (-\nabla c \cdot \mathbf{w}) A_{\text{hull}}$ to `< 1e-10` relative error!).
     - Background-invariance test: adding $+1000\text{ ppm}$ to all sensors leaves $\Phi_{\text{lat}}$ unchanged to `< 1e-10`.
     - Canonical hull start test: shuffled `["Oshen4", "Oshen1", "Oshen6", ...]` always starts CCW edges at `"Oshen1"`; when `"Oshen1"` is interior, hull starts at `"Oshen2"`.
     - Closure identity $\sum L_e \hat{\mathbf{n}}_e = \mathbf{0}$ and positive outward normals.
     - Hydrostatic $H_{\text{eff}} < H_{\text{MBL}}$ vs. small-$H_{\text{MBL}}$ linear limit.
     - `WindMode.UNIFORM_MEAN`, `WindMode.EXTERNAL`, and `WindMode.PER_ASV` + non-zero `wind_divergence_m2_s` diagnostic on biased sailboat anemometer.
     - Dual-sensor reconciliation (agree, disagree → `CO2_SENSOR_DISCREPANCY`, one `NaN` → single-sensor fallback + flag, both `NaN` → ASV omitted).
     - Interior ASV included in mean wind & $\bar{c}_{\text{box}}$ but excluded from hull edges.
     - Collinear/sliver $N \ge 3$ and $N = 2$ → `NaN` flux + `TWO_ASV_EXTRAPOLATED`; $N < 2$ or invalid $H_{\text{MBL}}$ → `NaN` flux + `INSUFFICIENT_ASVS`.
   - Implement `src/perimeter_flux/flux.py`.
5. **Phase 5 — `pipeline.py` (Stage 3)**:
   - Write `tests/perimeter_flux/test_pipeline.py` (end-to-end async multi-hour pipeline, Polars `pl.Enum` ordering/filtering, `ErrorBar` worst-case precedence when wind shift + $\text{CO}_2$ discrepancy coincide, circular wrap-safe wind shift across $350^\circ \to 10^\circ$, storage-term cancellation on a propagating synoptic front with zero surface uptake, zero-wind pure storage drawdown, contiguous vs. gapped run boundary derivatives, `summarize_window` over $\tau_{\text{win}}$) → implement `src/perimeter_flux/pipeline.py`.
6. **Phase 6 — `synthetic.py`**:
   - Write `tests/perimeter_flux/test_synthetic.py` (verifying `"Oshen1"`..`"Oshen6"` naming, schema compliance, deterministic seeded reproducibility, `steady_drawdown` recovering analytical ground-truth CDRR, `synoptic_front_storage` showing $|\Phi_{\text{net}}| \ll |\Phi_{\text{lat}}|$, and every scenario triggering its target `ErrorBar` category) → implement `src/perimeter_flux/synthetic.py`.
7. **Phase 7 — `acpf_exploration.py` (Marimo Notebook) & Visualization Tests**:
   - Write `tests/perimeter_flux/test_acpf_exploration.py` validating all Altair chart builders (`chart.to_dict(validate=True)` / structural encoding checks for hourly CDRR + `ErrorBar`, polygon + normals + per-edge flux map, storage-term cancellation chart, and `Oshen1`–`Oshen6` $\text{CO}_2$ sensor pair comparison chart across all scenarios, including empty/`NaN` periods) → implement `acpf_exploration.py`.

---

## 5. Verification Plan

### Automated Verification
- Run `uv run pytest` and verify all existing tests in [test_geometry.py](file:///Users/jerry/dev/EddyP1/tests/perimeter_flux/test_geometry.py) (49 tests) plus all new test suites (`test_config.py`, `test_geodesy.py`, `test_resample.py`, `test_flux.py`, `test_pipeline.py`, `test_synthetic.py`, `test_acpf_exploration.py`) pass with zero warnings or skipped tests.
- Run `uv run marimo check acpf_exploration.py` to verify Marimo cell graph integrity and zero undefined/circular cell references.

### Manual Interactive Verification
- Launch `uv run marimo run acpf_exploration.py` (or `marimo edit`) to visually inspect:
  - Polygon vertex order starting at `"Oshen1"`, CCW outward normals pointing away from the centroid, and inflow/outflow edge color symmetry.
  - Synoptic front cancellation where storage flux offsets the transient lateral spike.
  - `Oshen3` dual-sensor drift/spike highlighting and threshold slider reactivity.
