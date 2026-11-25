<!--suppress ALL -->
# Atmospheric $\text{CO} _ 2$ Perimeter Flux (ACPF) Mathematical Specification

This document is the persistent mathematical reference for the EddyP1 Atmospheric $\text{CO} _ 2$ Perimeter Flux (ACPF) computation. All symbols are globally unique across sections (zero symbol collisions). Each implementation module mirrors the relevant equations and glossary entries inline next to its algorithm stage.

Reference methodology: *Measuring large-scale mCDR using a novel Atmospheric* $\text{CO} _ 2$ *Perimeter Flux (ACPF) method*, Frontiers in Climate (2026), DOI `10.3389/fclim.2026.1941538`.

---

## 1. Pipeline Overview

```mermaid
flowchart LR
    A["Raw Async ASV Telemetry\n(Polars DataFrame)"] --> B["Stage 1: Temporal Alignment\n(ECEF + u,v + CO2 Interval Integral)"]
    M["MBLH & Met Series\n(H_MBL, optional T_air, P_atm, w_ext)"] --> B
    B --> C["Stage 2: Per-Period Kernel\n(WGS84 ENU + ConvexHull + Edge Flux)"]
    C --> D["Stage 3: Multi-Period Orchestration\n(Storage Term, ErrorBar QA, Rolling τ_win)"]
    D --> E["Hourly Polars Table +\nPer-Edge Polars Table"]
```

- **Stage 1 (`resample.py`)**: Aligns asynchronous ASV telemetry onto a uniform UTC time grid with period step $\Delta t$ (default $1\text{ hr}$) using time-weighted interval integration of a pluggable per-ASV interpolant (`InterpolationMethod`), computing sailboat wind constancy and ASV drift diagnostics.
- **Stage 2 (`flux.py` + `geodesy.py`)**: Projects each period's valid ASVs to a local WGS 84 East-North tangent plane, constructs the CCW `scipy.spatial.ConvexHull` anchored at the alphabetically smallest hull `asv_id`, reconciles dual $\text{CO} _ 2$ sensors, and integrates lateral $\text{CO} _ 2$ perimeter flux with hydrostatic column scaling.
- **Stage 3 (`pipeline.py`)**: Combines per-period results across time to evaluate inter-period wind and $H _ {\text{MBL}}$ stability, append the Eulerian column storage term $\Phi _ {\text{stor}}$ and storage-adjusted CDRR $\Phi _ {\text{net}}$, assign the ordered `ErrorBar` classification, and support multi-period window aggregation over $\tau _ {\text{win}}$.

---

## 2. Symbol Glossary & Constants

### 2.1 Time, Indexing, and Counts

| Symbol | Code Identifier | Units | Definition |
| :--- | :--- | :--- | :--- |
| $t$ | `t` | $\text{s}$ | Continuous UTC time (epoch seconds as `float64`). |
| $t _ k$ | `period_start` | $\text{s}$ | Start timestamp of discrete period $k$; period $k$ covers $[t _ k,\thinspace t _ k + \Delta t)$. |
| $\Delta t$ | `time_interval` | $\text{s}$ | Duration of one aggregation step (named constant `DEFAULT_TIME_INTERVAL = timedelta(hours=1)`, i.e. $3600\text{ s}$). |
| $\Delta t _ {\text{gap,max}}$ | `max_interp_gap` | $\text{s}$ | Maximum silence gap between consecutive raw observations of an ASV that may be bridged by interpolation (default $3\thinspace \Delta t$). |
| $\tau _ {\text{win}}$ | `window_duration` | $\text{s}$ (or $\text{hr}$ / $\text{d}$) | Multi-period aggregation or rolling-average window length $(\tau _ {\text{win}} = K _ {\text{win}}\thinspace \Delta t \ge \Delta t)$. |
| $K _ {\text{win}}$ | `window_periods` | count | Integer number of $\Delta t$ aggregation steps in the multi-period window $\tau _ {\text{win}}$. |
| $\mathcal{K} _ {\text{valid}}$ | `valid_period_indices` | set | Set of period indices $k$ within $\tau _ {\text{win}}$ whose `ErrorBar` meets the selected severity filter. |
| $i$ | `asv_idx` | — | Index of a valid ASV in period $k$, $i \in \lbrace 0, \dots, N _ {\text{tot}} - 1 \rbrace$. |
| $N _ {\text{tot}}$ | `n_asvs_total` | count | Number of valid ASVs with usable position, wind, and $\text{CO} _ 2$ in period $k$. |
| $N _ {\text{hull}}$ | `n_asvs_hull` | count | Number of ASVs forming the outer CCW convex-hull boundary $(N _ {\text{hull}} \le N _ {\text{tot}})$. |
| $e$ | `edge_idx` | — | Index of a CCW convex-hull edge from vertex $e$ to $(e + 1) \bmod N _ {\text{hull}}$, where vertex $0$ has the alphabetically smallest `asv_id` on the hull. |

### 2.2 Geodesy & Polygon Geometry

| Symbol | Code Identifier | Units | Value / Definition |
| :--- | :--- | :--- | :--- |
| $a _ {\text{wgs}}$ | `WGS84_SEMI_MAJOR_M` | $\text{m}$ | $6{,}378{,}137.0\text{ m}$ (exact WGS 84 equatorial radius). |
| $f _ {\text{wgs}}$ | `WGS84_FLATTENING` | $1$ | $1 / 298.257223563$ (exact WGS 84 flattening). |
| $e _ {\text{wgs}}^2$ | `WGS84_E2` | $1$ | $2 f _ {\text{wgs}} - f _ {\text{wgs}}^2 \approx 0.00669437999014132$ (first eccentricity squared). |
| $\phi,\thinspace \lambda$ | `latitude`, `longitude` | $\text{rad}$ (inputs in $\text{deg}$) | Geodetic latitude (North positive) and longitude (East positive). |
| $\phi _ 0,\thinspace \lambda _ 0$ | `origin_lat`, `origin_lon` | $\text{rad}$ | Reference origin of the local ENU tangent plane, computed from the normalized 3D ECEF centroid. |
| $\nu(\phi)$ | `prime_vertical_radius` | $\text{m}$ | Prime vertical radius of curvature $a _ {\text{wgs}} / \sqrt{1 - e _ {\text{wgs}}^2 \sin^2\phi}$. |
| $\mathbf{X} _ i = (X _ i, Y _ i, Z _ i)^\top$ | `ecef_xyz` | $\text{m}$ | 3D Earth-Centered Earth-Fixed (ECEF) position of ASV $i$ on the WGS 84 sea-level surface $(h _ {\text{ellip}} = 0)$. |
| $\mathbf{R} _ {\text{enu}}(\phi _ 0, \lambda _ 0)$ | `enu_rotation` | $1$ | $3 \times 3$ orthonormal rotation matrix from ECEF displacements to local East-North-Up (ENU). |
| $\mathbf{p} _ i = (x _ i, y _ i)^\top$ | `xy_m` | $\text{m}$ | 2D local metric coordinates of ASV $i$ (where $x$ = East, $y$ = North). |
| $\mathbf{P}$ | `xy_m_matrix` | $\text{m}$ | $N _ {\text{tot}} \times 2$ matrix whose rows are the local EN coordinate vectors $\mathbf{p} _ i^\top$ for all valid ASVs in period $k$. |
| $\bar{\mathbf{p}}$ | `xy_centroid_m` | $\text{m}$ | 2D arithmetic centroid $\frac{1}{N _ {\text{tot}}}\sum _ i \mathbf{p} _ i$ of the projected local EN coordinates. |
| $\tilde{\mathbf{P}}$ | `xy_centered_m` | $\text{m}$ | Mean-centered $N _ {\text{tot}} \times 2$ coordinate matrix $\mathbf{P} - \mathbf{1}\bar{\mathbf{p}}^\top$ used for the collinearity rank test. |
| $L _ e$ | `edge_length_m` | $\text{m}$ | Length of CCW hull edge $e$: $\lVert \mathbf{p} _ {(e+1)\bmod N _ {\text{hull}}} - \mathbf{p} _ e \rVert _ 2$. |
| $L _ {\text{perim}}$ | `hull_perimeter_m` | $\text{m}$ | Total perimeter length $\sum _ e L _ e$ (`hull.area` in 2D `ConvexHull`). |
| $A _ {\text{hull}}$ | `hull_area_m2` | $\text{m}^2$ | Enclosed 2D polygon area (`hull.volume` in 2D `ConvexHull`). |
| $Q _ {\text{iso}}$ | `hull_compactness` | $1$ | Isoperimetric quotient (Polsby-Popper score) $4\pi A _ {\text{hull}} / L _ {\text{perim}}^2 \in (0, 1]$. |
| $\hat{\mathbf{n}} _ e = (n _ {e,x}, n _ {e,y})^\top$ | `edge_normal_xy` | $1$ | Outward-pointing unit normal to CCW edge $e$. |

### 2.3 Wind, Motion, and $\text{CO} _ 2$ Measurements

| Symbol | Code Identifier | Units | Definition |
| :--- | :--- | :--- | :--- |
| $\mathbf{w} _ i = (u _ i, v _ i)^\top$ | `(u_wind, v_wind)` | $\text{m}\thinspace \text{s}^{-1}$ | Period-averaged horizontal wind velocity vector (flow *toward* direction: $u$ Eastward, $v$ Northward) at ASV $i$. |
| $\bar{\mathbf{w}} _ k = (\bar{u} _ k, \bar{v} _ k)^\top$ | `(mean_wind_u, mean_wind_v)` | $\text{m}\thinspace \text{s}^{-1}$ | Spatial vector mean wind across all $N _ {\text{tot}}$ valid ASVs in period $k$. |
| $\mathbf{w} _ {\text{ext},k}$ | `(u_wind_ext, v_wind_ext)` | $\text{m}\thinspace \text{s}^{-1}$ | Optional external (e.g., satellite/reanalysis) uniform horizontal wind vector for period $k$. |
| $\mathbf{w} _ e$ | `edge_wind_xy` | $\text{m}\thinspace \text{s}^{-1}$ | Effective horizontal wind vector used on hull edge $e$ (determined by `WindMode`). |
| $\mathbf{w} _ {\text{asv},i}$ | `(asv_drift_u, asv_drift_v)` | $\text{m}\thinspace \text{s}^{-1}$ | Horizontal ground/drift velocity of ASV $i$ between consecutive periods (moving-boundary diagnostic). |
| $s _ i$ | `wind_speed` | $\text{m}\thinspace \text{s}^{-1}$ | Scalar wind speed $\lVert \mathbf{w} _ i \rVert _ 2$ at ASV $i$. |
| $s _ {\text{poly},k}$ | `mean_wind_speed_m_s` | $\text{m}\thinspace \text{s}^{-1}$ | Magnitude of the polygon-mean wind vector $\lVert \bar{\mathbf{w}} _ k \rVert _ 2$ in period $k$. |
| $\theta _ {w,k}$ | `wind_flow_rad` | $\text{rad}$ | Wind flow heading angle $\mathrm{atan2}(\bar{v} _ k, \bar{u} _ k) \in (-\pi, \pi]$ (CCW from East). |
| $r _ {\text{const},i}$ | `wind_constancy` | $1$ | Intra-period wind vector constancy $\lVert \bar{\mathbf{w}} _ {i,k} \rVert _ 2 / \overline{\lVert \mathbf{w} _ i \rVert} _ {k} \in [0, 1]$ (1 = perfectly steady direction; 0 = complete reversal/turning cancellation). |
| $D _ w$ | `wind_divergence_m2_s` | $\text{m}^2\thinspace \text{s}^{-1}$ | Raw discrete horizontal wind divergence around the hull using local per-ASV edge winds. |
| $c _ {i,1},\thinspace c _ {i,2}$ | `co2_1`, `co2_2` | $\text{ppm}$ | Period-averaged dry-air $\text{CO} _ 2$ mole fractions $(\mu\text{mol}\thinspace \text{mol}^{-1})$ from sensors 1 and 2 on ASV $i$. |
| $c _ i$ | `co2_ppm` | $\text{ppm}$ | Reconciled dry-air $\text{CO} _ 2$ mole fraction for ASV $i$. |
| $\delta c _ i$ | `co2_sensor_diff_ppm` | $\text{ppm}$ | Absolute dual-sensor discrepancy $\lvert c _ {i,1} - c _ {i,2} \rvert$ on ASV $i$. |
| $\bar{c} _ e$ | `edge_co2_ppm` | $\text{ppm}$ | Mean dry-air $\text{CO} _ 2$ mole fraction along hull edge $e$. |
| $c _ {\text{ref}}$ | `co2_ref_ppm` | $\text{ppm}$ | Reference background $\text{CO} _ 2$ mole fraction subtracted before edge flux multiplication. |
| $\bar{c} _ {\text{box},k}$ | `box_mean_co2_ppm` | $\text{ppm}$ | Spatial mean $\text{CO} _ 2$ mole fraction across all $N _ {\text{tot}}$ valid ASVs in period $k$. |

### 2.4 Atmospheric Thermodynamics & Flux Outputs

| Symbol | Code Identifier | Units | Value / Definition |
| :--- | :--- | :--- | :--- |
| $R _ {\text{gas}}$ | `UNIVERSAL_GAS_CONSTANT` | $\text{J}\thinspace \text{mol}^{-1}\thinspace \text{K}^{-1}$ | $8.314462618$ (CODATA 2018/2022 exact SI value). |
| $M _ {\text{CO} _ 2}$ | `MOLAR_MASS_CO2_KG` | $\text{kg}\thinspace \text{mol}^{-1}$ | $0.0440095$ (IUPAC/NIST molar mass of $\text{CO} _ 2$). |
| $M _ {\text{air}}$ | `MOLAR_MASS_DRY_AIR_KG` | $\text{kg}\thinspace \text{mol}^{-1}$ | $0.0289644$ (U.S. Standard Atmosphere 1976 dry air). |
| $R _ d$ | `DRY_AIR_GAS_CONSTANT` | $\text{J}\thinspace \text{kg}^{-1}\thinspace \text{K}^{-1}$ | $R _ {\text{gas}} / M _ {\text{air}} \approx 287.05796$. |
| $g _ 0$ | `STANDARD_GRAVITY` | $\text{m}\thinspace \text{s}^{-2}$ | $9.80665$ (exact standard gravity). |
| $H _ {\text{MBL}}$ | `mblh_m` | $\text{m}$ | Marine Boundary Layer Height for period $k$ (required positive input). |
| $T _ {\text{air}}$ | `t_air_k` | $\text{K}$ | Near-surface air temperature (optional; default $293.15\text{ K} = 20^\circ\text{C}$). |
| $P _ {\text{atm}}$ | `p_atm_pa` | $\text{Pa}$ | Surface barometric pressure (optional; default $101{,}325.0\text{ Pa}$). |
| $H _ {\text{scale}}$ | `scale_height_m` | $\text{m}$ | Hydrostatic dry-air scale height $R _ d T _ {\text{air}} / g _ 0$. |
| $H _ {\text{eff}}$ | `effective_mblh_m` | $\text{m}$ | Density-weighted effective MBL column height. |
| $\kappa _ {\text{CO} _ 2}$ | `co2_mass_per_ppm_kg_m3` | $\text{kg}\thinspace \text{m}^{-3}\thinspace \text{ppm}^{-1}$ | Surface mass concentration of $\text{CO} _ 2$ per $1\text{ ppm}$ dry-air mole fraction. |
| $\gamma _ {\text{unit}}$ | `KG_PER_S_TO_T_PER_HR` | $(\text{tCO} _ 2\thinspace \text{hr}^{-1})/(\text{kg}\thinspace \text{s}^{-1})$ | $3600 / 1000 = 3.6$ (exact). |
| $\Phi _ {\text{edge},e}$ | `edge_outward_flux_tco2_hr` | $\text{tCO} _ 2\thinspace \text{hr}^{-1}$ | Outward lateral $\text{CO} _ 2$ mass flux across edge $e$ (positive = leaving polygon). |
| $\Phi _ {\text{lat}}$ | `lateral_cdrr_tco2_hr` | $\text{tCO} _ 2\thinspace \text{hr}^{-1}$ | Net Carbon Dioxide Removal Rate estimated from horizontal (lateral) perimeter flux alone, before adding column storage (positive = net ocean uptake/removal). |
| $\Phi _ {\text{stor}}$ | `storage_flux_tco2_hr` | $\text{tCO} _ 2\thinspace \text{hr}^{-1}$ | Eulerian column storage contribution to surface CDRR. |
| $\Phi _ {\text{net}}$ | `storage_adjusted_cdrr_tco2_hr` | $\text{tCO} _ 2\thinspace \text{hr}^{-1}$ | Storage-adjusted surface CDRR $(\Phi _ {\text{lat}} + \Phi _ {\text{stor}})$. |
| $\psi _ {\text{lat}},\thinspace \psi _ {\text{net}}$ | `lateral_cdrr_tco2_km2_hr`, `storage_adjusted_cdrr_tco2_km2_hr` | $\text{tCO} _ 2\thinspace \text{km}^{-2}\thinspace \text{hr}^{-1}$ | Areal removal flux densities (removal rate per unit enclosed ocean surface area, i.e., $\Phi$ divided by $A _ {\text{hull}}$ in $\text{km}^2$). |

### 2.5 Configuration & Quality-Assurance Threshold Constants (`ACPFConfig`)

| Symbol | Code Identifier | Units | Default / Definition |
| :--- | :--- | :--- | :--- |
| $\epsilon _ w$ | `MIN_WIND_SPEED_EPS` | $\text{m}\thinspace \text{s}^{-1}$ | $10^{-9}\text{ m}\thinspace \text{s}^{-1}$ zero-wind guard for constancy denominator. |
| $A _ {\min}$ | `min_hull_area_m2` | $\text{m}^2$ | $10^{6}\text{ m}^2$ $(1\text{ km}^2)$ minimum enclosing hull area to reject degenerate hulls. |
| $Q _ {\text{iso,min}}$ | `min_hull_compactness` | $1$ | $0.05$ minimum Polsby-Popper compactness to reject nearly collinear slivers. |
| $\delta _ {\text{CO} _ 2,\max}$ | `max_co2_sensor_diff_ppm` | $\text{ppm}$ | $2.0\text{ ppm}$ dual-sensor discrepancy warning threshold. |
| $r _ {\text{const,min}}$ | `min_wind_constancy` | $1$ | $0.80$ minimum intra-period wind vector constancy before flagging unstable wind. |
| $\Delta \theta _ {w,\max}$ | `max_wind_dir_change_deg` | $\text{deg}$ (math: $\text{rad}$) | $30.0^\circ$ maximum inter-period wind direction shift before flagging unstable wind. |
| $\eta _ {w,\max}$ | `max_wind_speed_rel_change` | $1$ | $0.35$ (35%) maximum relative inter-period wind speed change. |
| $\eta _ {H,\max}$ | `max_mblh_rel_change` | $1$ | $0.25$ (25%) maximum relative inter-period $H _ {\text{MBL}}$ change. |

---

## 3. Geodetic Coordinate Transformations (`geodesy.py`)

### 3.1 Geodetic $(\phi, \lambda)$ to 3D WGS 84 ECEF

For sea-surface ASVs $(h _ {\text{ellip}} = 0)$, geodetic latitude $\phi$ and longitude $\lambda$ map to 3D ECEF coordinates $\mathbf{X} = (X, Y, Z)^\top$ via:

$$
\nu(\phi) = \frac{a _ {\text{wgs}}}{\sqrt{1 - e _ {\text{wgs}}^2 \sin^2\phi}}
$$

$$
X = \nu(\phi)\cos\phi\cos\lambda, \qquad Y = \nu(\phi)\cos\phi\sin\lambda, \qquad Z = \nu(\phi)(1 - e _ {\text{wgs}}^2)\sin\phi
$$

Interpolating ASV positions in 3D ECEF (Stage 1) and computing the centroid in 3D ECEF (Stage 2) avoids all $\pm 180^\circ$ International Date Line and polar discontinuities.

### 3.2 Inverse ECEF to Geodetic Origin $(\phi _ 0, \lambda _ 0)$

Given $N _ {\text{tot}}$ ASV ECEF vectors $\lbrace \mathbf{X} _ i \rbrace$, form the mean ECEF vector $\bar{\mathbf{X}} = \frac{1}{N _ {\text{tot}}}\sum _ i \mathbf{X} _ i$ and convert it to reference geodetic coordinates $(\phi _ 0, \lambda _ 0)$ using Bowring's exact closed-form / iterative ellipsoidal projection (2 Newton-Raphson iterations from $\phi^{(0)} = \mathrm{atan2}(\bar{Z},\thinspace (1 - e _ {\text{wgs}}^2)\sqrt{\bar{X}^2 + \bar{Y}^2})$ converge to $< 10^{-15}\text{ rad}$), and project the origin onto the sea-level ellipsoid $\mathbf{X} _ 0 = \mathbf{X}(\phi _ 0, \lambda _ 0)$.

### 3.3 ECEF to Local East-North-Up (ENU) Tangent Plane

Each ASV displacement $\Delta \mathbf{X} _ i = \mathbf{X} _ i - \mathbf{X} _ 0$ is rotated into the local ENU frame:

$$
\begin{bmatrix}
x _ i \\
y _ i \\
z _ {\text{enu},i}
\end{bmatrix}
= \mathbf{R} _ {\text{enu}}(\phi _ 0, \lambda _ 0)\thinspace \Delta \mathbf{X} _ i =
\begin{bmatrix}
-\sin\lambda _ 0 & \cos\lambda _ 0 & 0 \\
-\sin\phi _ 0\cos\lambda _ 0 & -\sin\phi _ 0\sin\lambda _ 0 & \cos\phi _ 0 \\
\cos\phi _ 0\cos\lambda _ 0 & \cos\phi _ 0\sin\lambda _ 0 & \sin\phi _ 0
\end{bmatrix}
\begin{bmatrix}
X _ i - X _ 0 \\
Y _ i - Y _ 0 \\
Z _ i - Z _ 0
\end{bmatrix}
$$

We take the horizontal components $\mathbf{p} _ i = (x _ i, y _ i)^\top$ in meters for 2D convex-hull geometry. Over a $200\text{ km}$ eddy radius, flat tangent-plane horizontal distortion is $< 0.02$% (well below wind and $H _ {\text{MBL}}$ uncertainties).

---

## 4. Stage 1: Asynchronous Telemetry Alignment & Resampling (`resample.py`)

### 4.1 Pluggable Time-Weighted Interval Integral

ASVs transmit asynchronously and at irregular intervals. For each ASV $i$ and each scalar channel $q(t) \in \lbrace X(t), Y(t), Z(t), u(t), v(t), c _ 1(t), c _ 2(t) \rbrace$ sorted by strictly increasing timestamp $t$:

1. **Duplicate timestamps**: Any exact duplicate timestamps for the same ASV are averaged before fitting the interpolant.
2. **Pluggable interpolant $\hat{q}(t)$** selected by `InterpolationMethod`:
   - `LINEAR` (default): piecewise-linear spline via `scipy.interpolate.make_interp_spline(t_obs, q_obs, k=1)`.
   - `PCHIP`: shape-preserving monotonicity-guaranteed cubic Hermite spline via `scipy.interpolate.PchipInterpolator(t_obs, q_obs, extrapolate=False)` (strictly zero overshoot).
   - `AKIMA`: outlier-resistant $C^1$ local cubic spline via `scipy.interpolate.Akima1DInterpolator(t_obs, q_obs, extrapolate=False)` (requires $\ge 5$ samples; falls back to `PCHIP` or `LINEAR` when fewer samples exist).
3. **Time-weighted period average** over period $k = [t _ k,\thinspace t _ k + \Delta t)$:

$$
\bar{q} _ {i,k} = \frac{1}{\Delta t} \int _ {t _ k}^{t _ k + \Delta t} \hat{q} _ i(t)\thinspace dt
$$

This unified integral simultaneously:
- Time-weight-averages dense sub-hourly observations (canceling high-frequency sailboat turning/bobbing oscillations without clustered-sample bias).
- Smoothly bridges asynchronous or slightly sparser observations via the exact same `.integrate(t_k, t_k + Δt)` interface.
- Reproduces constants $q(t) = a$ and linear ramps $q(t) = a + b t$ exactly—yielding $a + b(t _ k + \Delta t / 2)$—for both `LINEAR` and `PCHIP`.

### 4.2 No-Extrapolation and Maximum Gap Guard

A period $[t _ k,\thinspace t _ k + \Delta t)$ is valid for channel $q$ on ASV $i$ **if and only if**:
1. The period lies within the ASV's observed span: $t _ {\text{obs},0} \le t _ k$ and $t _ k + \Delta t \le t _ {\text{obs},\text{last}}$ (never extrapolate).
2. Every consecutive raw sample interval $[t _ m, t _ {m+1}]$ that overlaps $[t _ k, t _ k + \Delta t)$ satisfies $t _ {m+1} - t _ m \le \Delta t _ {\text{gap,max}}$.

If a period fails either check for position or wind (or for *both* `co2_1` and `co2_2`), that ASV is omitted from period $k$ rather than fabricating data across a telemetry outage. (If only one of `co2_1` or `co2_2` passes, the ASV remains valid and triggers the dual-sensor discrepancy flag in Stage 2.)

### 4.3 Intra-Period Sailboat Wind Constancy & Drift Velocity Diagnostics

- **Intra-period wind vector constancy** (measuring sub-hourly sailboat turning/yaw noise and intra-hour wind shifts):
  For the raw wind speed channel $s _ i(t) = \sqrt{u _ i(t)^2 + v _ i(t)^2}$, compute the time-weighted scalar speed mean $\bar{s} _ {i,k} = \frac{1}{\Delta t}\int _ {t _ k}^{t _ k + \Delta t} \hat{s} _ i(t)\thinspace dt$ and compare it to the magnitude of the period-averaged vector wind $\bar{\mathbf{w}} _ {i,k} = (\bar{u} _ {i,k}, \bar{v} _ {i,k})^\top$:

$$
r _ {\text{const},i,k} = \begin{cases}
\dfrac{\lVert \bar{\mathbf{w}} _ {i,k} \rVert _ 2}{\bar{s} _ {i,k}}, & \bar{s} _ {i,k} > \epsilon _ w \\
1.0, & \bar{s} _ {i,k} \le \epsilon _ w
\end{cases}
$$

  where $r _ {\text{const},i,k} \in [0, 1]$ (with $\epsilon _ w = 10^{-9}\text{ m}\thinspace \text{s}^{-1}$). Values significantly below $1$ indicate high-frequency turning oscillation or rapid wind-direction shifts within the hour.
- **ASV drift velocity** (for moving vs. station-keeping sailboats):
  Between adjacent periods $k-1$ and $k$ (separated by $\Delta t$), the local EN displacement $\Delta \mathbf{p} _ {i,k} = \mathbf{p} _ {i,k} - \mathbf{p} _ {i,k-1}$ gives the ASV's horizontal velocity $\mathbf{w} _ {\text{asv},i,k} = \Delta \mathbf{p} _ {i,k} / \Delta t$, reported as a diagnostic (`max_asv_drift_speed_m_s`) to monitor whether moving-boundary Reynolds corrections $(\mathbf{w} - \mathbf{w} _ {\text{asv}})$ become significant relative to wind speed.

### 4.4 MBLH & Meteorological Alignment

The meteorological table (`timestamp`, `mblh_m`, optional `t_air_k`, `p_atm_pa`, `u_wind_ext`, `v_wind_ext`) is aligned onto the same $[t _ k,\thinspace t _ k + \Delta t)$ grid using the same interval-integral resampler (with gap guard), or matched directly when already provided on the $\Delta t$ grid.

---

## 5. Stage 2: Per-Period Convex-Hull Perimeter Flux (`flux.py`)

### 5.1 Dual-Sensor $\text{CO} _ 2$ Reconciliation

For each valid ASV $i$ in period $k$:

$$
c _ i = \begin{cases}
\dfrac{c _ {i,1} + c _ {i,2}}{2}, & \text{both } c _ {i,1}, c _ {i,2} \text{ finite} \\
c _ {i,1}, & c _ {i,2} \text{ missing/NaN} \\
c _ {i,2}, & c _ {i,1} \text{ missing/NaN}
\end{cases}
$$

$$
\delta c _ i = \begin{cases}
\lvert c _ {i,1} - c _ {i,2} \rvert, & \text{both finite} \\
+\infty \text{ (flagged as single-sensor)}, & \text{one sensor missing}
\end{cases}
$$

An ASV is flagged for sensor discrepancy if one sensor is missing or $\delta c _ i > \delta _ {\text{CO} _ 2,\max}$. Flagged ASVs remain in the hull so geometry is preserved, while elevating the period's `ErrorBar` classification.

### 5.2 Convex Hull Construction, Canonical Ordering, and Degeneracy Checks

Given $N _ {\text{tot}}$ valid ASVs projected to local EN coordinates $\lbrace \mathbf{p} _ i \rbrace$:

1. **Count checks**:
   - If $N _ {\text{tot}} < 2$ or $H _ {\text{MBL}} \le 0$ or $H _ {\text{MBL}}$ is NaN: return `NaN` flux with `ErrorBar.INSUFFICIENT_ASVS`.
   - If $N _ {\text{tot}} = 2$: return `NaN` flux with `ErrorBar.TWO_ASV_EXTRAPOLATED` (extrapolation for $N = 2$ deferred).
2. **Degeneracy check** (for $N _ {\text{tot}} \ge 3$):
   - Compute centered points $\tilde{\mathbf{P}} = \mathbf{P} - \bar{\mathbf{p}}$. If $\mathrm{rank}(\tilde{\mathbf{P}}) < 2$ or `scipy.spatial.ConvexHull(P)` raises `QhullError` (never use `"QJ"` random joggle), classify as `ErrorBar.TWO_ASV_EXTRAPOLATED` with `NaN` flux.
   - Extract `hull = ConvexHull(P)`:
     - $A _ {\text{hull}} = \text{hull.volume}$ $(\text{m}^2)$
     - $L _ {\text{perim}} = \text{hull.area}$ $(\text{m})$
     - $Q _ {\text{iso}} = \dfrac{4\pi A _ {\text{hull}}}{L _ {\text{perim}}^2}$
   - If $A _ {\text{hull}} < A _ {\min}$ or $Q _ {\text{iso}} < Q _ {\text{iso,min}}$, the array forms a degenerate sliver without an enclosing 2D perimeter; classify as `ErrorBar.TWO_ASV_EXTRAPOLATED` with `NaN` flux.
3. **Canonical CCW rotation**:
   - `hull.vertices` in 2D SciPy are guaranteed counterclockwise (CCW).
   - Find index $m$ in `hull.vertices` whose corresponding `asv_id` is **alphabetically smallest** (e.g., `"Oshen1"` before `"Oshen2"`), and cyclically shift `hull.vertices` so that vertex $m$ is at position $0$.
4. **Interior ASVs** $(N _ {\text{hull}} < N _ {\text{tot}})$:
   - Only the $N _ {\text{hull}}$ boundary vertices form edges for the line integral.
   - All $N _ {\text{tot}}$ valid ASVs still contribute to $\bar{\mathbf{w}} _ k$, $\bar{c} _ {\text{box},k}$, and dual-sensor QA diagnostics.

### 5.3 Edge Lengths and Outward Unit Normals

For each CCW edge $e \in \lbrace 0, \dots, N _ {\text{hull}} - 1 \rbrace$ from $\mathbf{p} _ {e _ 0}$ to $\mathbf{p} _ {e _ 1}$ (where $e _ 1 = (e + 1) \bmod N _ {\text{hull}}$):

$$
\Delta x _ e = x _ {e _ 1} - x _ {e _ 0}, \qquad \Delta y _ e = y _ {e _ 1} - y _ {e _ 0}, \qquad L _ e = \sqrt{\Delta x _ e^2 + \Delta y _ e^2}
$$

For a CCW polygon, rotating the tangent vector $(\Delta x _ e, \Delta y _ e)^\top$ by $-90^\circ$ (to the right) yields the **outward-pointing unit normal**:

$$
\hat{\mathbf{n}} _ e = \frac{1}{L _ e}\begin{bmatrix}
\Delta y _ e \\
-\Delta x _ e
\end{bmatrix}
$$

Geometric closure identity (exact to machine precision for any closed polygon):

$$
\sum _ {e=0}^{N _ {\text{hull}}-1} L _ e\thinspace \hat{\mathbf{n}} _ e = \mathbf{0}
$$

### 5.4 Wind Mode, Background Subtraction, and Divergence Diagnostic

Let the local per-ASV edge wind be $\mathbf{w} _ {e,\text{local}} = \frac{1}{2}(\mathbf{w} _ {e _ 0} + \mathbf{w} _ {e _ 1})$. The **raw discrete horizontal wind divergence diagnostic** is always computed (even in uniform wind mode) to detect individual sailboat anemometer tilt or heading biases:

$$
D _ w = \sum _ {e=0}^{N _ {\text{hull}}-1} (\mathbf{w} _ {e,\text{local}} \cdot \hat{\mathbf{n}} _ e)\thinspace L _ e \qquad [\text{m}^2\thinspace \text{s}^{-1}]
$$

The effective edge wind $\mathbf{w} _ e$ used in the flux integral depends on `WindMode`:
- `UNIFORM_MEAN` (default): $\mathbf{w} _ e = \bar{\mathbf{w}} _ k = \frac{1}{N _ {\text{tot}}}\sum _ {i=0}^{N _ {\text{tot}}-1} \mathbf{w} _ i$. By the geometric closure identity, $\sum _ e (\bar{\mathbf{w}} _ k \cdot \hat{\mathbf{n}} _ e) L _ e = 0$ identically, completely eliminating anemometer divergence error on the ~420 ppm background.
- `EXTERNAL`: $\mathbf{w} _ e = \mathbf{w} _ {\text{ext},k}$ (e.g., satellite or reanalysis wind; also strictly zero divergence).
- `PER_ASV`: $\mathbf{w} _ e = \mathbf{w} _ {e,\text{local}}$ (retains spatial wind shear across the eddy; relies on $c _ {\text{ref}}$ subtraction below to suppress the $c _ {\text{ref}} D _ w$ background-divergence error).

For every edge $e$, compute edge-mean $\text{CO} _ 2$ and subtract the hull-mean reference background $c _ {\text{ref}}$:

$$
\bar{c} _ e = \frac{c _ {e _ 0} + c _ {e _ 1}}{2}, \qquad c _ {\text{ref}} = \frac{1}{N _ {\text{hull}}}\sum _ {j=0}^{N _ {\text{hull}}-1} c _ j
$$

Under uniform wind $(\mathbf{w} _ e = \bar{\mathbf{w}} _ k)$, subtracting $c _ {\text{ref}}$ leaves the exact integral unchanged analytically while eliminating catastrophic floating-point cancellation on $\sim 420\text{ ppm}$ numbers. Under `PER_ASV` wind, subtracting $c _ {\text{ref}}$ removes the spurious $c _ {\text{ref}} D _ w$ mass-imbalance term (implicitly assuming compensating vertical entrainment/subsidence carries concentration $c _ {\text{ref}}$).

### 5.5 Hydrostatic Column Density Scaling & Flux Conversion

1. **Surface $\text{CO} _ 2$ mass concentration per $1\text{ ppm}$ dry-air mole fraction** (ideal gas law):

$$
\kappa _ {\text{CO} _ 2} = 10^{-6} \cdot \frac{P _ {\text{atm}}\thinspace M _ {\text{CO} _ 2}}{R _ {\text{gas}}\thinspace T _ {\text{air}}} \qquad [\text{kg}\thinspace \text{m}^{-3}\thinspace \text{ppm}^{-1}]
$$

2. **Hydrostatic barometric effective column height** across $H _ {\text{MBL}}$ (assuming a well-mixed dry-air mole fraction $c$ vertically through the MBL and hydrostatic exponential density decay with scale height $H _ {\text{scale}}$):

$$
H _ {\text{scale}} = \frac{R _ d\thinspace T _ {\text{air}}}{g _ 0}, \qquad H _ {\text{eff}} = \int _ 0^{H _ {\text{MBL}}} e^{-z / H _ {\text{scale}}}\thinspace dz = H _ {\text{scale}}\left(1 - \exp\negthinspace\left(-\frac{H _ {\text{MBL}}}{H _ {\text{scale}}}\right)\right)
$$

   Note: At $T _ {\text{air}} = 293.15\text{ K}$ and $H _ {\text{MBL}} = 1000\text{ m}$, $H _ {\text{scale}} \approx 8581\text{ m}$ and $H _ {\text{eff}} \approx 943.9\text{ m}$, correcting the $\approx 5.6$% constant-density overestimate.

3. **Per-edge outward $\text{CO} _ 2$ mass flux** (in $\text{tCO} _ 2\thinspace \text{hr}^{-1}$, positive = leaving polygon):

$$
\Phi _ {\text{edge},e} = \gamma _ {\text{unit}} \cdot \kappa _ {\text{CO} _ 2} \cdot H _ {\text{eff}} \cdot (\bar{c} _ e - c _ {\text{ref}}) \cdot (\mathbf{w} _ e \cdot \hat{\mathbf{n}} _ e) \cdot L _ e
$$

4. **Net lateral Carbon Dioxide Removal Rate (CDRR)** (in $\text{tCO} _ 2\thinspace \text{hr}^{-1}$, **positive = net ocean removal/uptake**)—the net removal rate estimated solely from horizontal (lateral) flow across the perimeter walls, before adding the Stage 3 column storage correction $\Phi _ {\text{stor}}$:

$$
\Phi _ {\text{lat}} = -\sum _ {e=0}^{N _ {\text{hull}}-1} \Phi _ {\text{edge},e}
$$

5. **Areal removal flux density** $(\text{tCO} _ 2\thinspace \text{km}^{-2}\thinspace \text{hr}^{-1})$—the removal rate per unit of enclosed ocean surface area (normalizing $\Phi _ {\text{lat}}$ by $A _ {\text{hull}}$ in $\text{km}^2$ so removal intensity is comparable across different eddy sizes or drifting hull areas):

$$
\psi _ {\text{lat}} = \frac{\Phi _ {\text{lat}}}{A _ {\text{hull}} \times 10^{-6}}
$$

---

## 6. Stage 3: Multi-Period Continuity, Storage Term & Quality Classification (`pipeline.py`)

### 6.1 Column Storage Term & Storage-Adjusted CDRR

Across a ~200 km eddy at 5–10 m/s wind, an air mass takes 6–13 hours to transit from the upwind to downwind perimeter. Ignoring top-of-MBL entrainment, integral mass conservation over effective volume $V _ {\text{eff},k} = A _ {\text{hull},k} H _ {\text{eff},k}$ gives:

$$
\Phi _ {\text{net},k} = \Phi _ {\text{lat},k} + \Phi _ {\text{stor},k}
$$

$$
\Phi _ {\text{stor},k} = -\gamma _ {\text{unit}} \cdot \kappa _ {\text{CO} _ 2,k} \cdot A _ {\text{hull},k} \cdot H _ {\text{eff},k} \cdot \left(\frac{\Delta \bar{c} _ {\text{box}}}{\Delta t}\right) _ k
$$

where $\bar{c} _ {\text{box},k} = \frac{1}{N _ {\text{tot},k}}\sum _ {i=0}^{N _ {\text{tot},k}-1} c _ {i,k}$ and the discrete time derivative uses a second-order centered difference within contiguous valid period runs (with one-sided differences at run boundaries, and `NaN` for isolated single periods):

$$
\left(\frac{\Delta \bar{c} _ {\text{box}}}{\Delta t}\right) _ k = \begin{cases}
\dfrac{\bar{c} _ {\text{box},k+1} - \bar{c} _ {\text{box},k-1}}{2\thinspace \Delta t}, & \text{periods } k-1 \text{ and } k+1 \text{ adjacent and valid} \\
\dfrac{\bar{c} _ {\text{box},k+1} - \bar{c} _ {\text{box},k}}{\Delta t}, & \text{start of contiguous run} \\
\dfrac{\bar{c} _ {\text{box},k} - \bar{c} _ {\text{box},k-1}}{\Delta t}, & \text{end of contiguous run} \\
\text{NaN}, & \text{isolated period}
\end{cases}
$$

When a synoptic front raises background $\text{CO} _ 2$ entering the upwind boundary (yielding $\Phi _ {\text{lat}} > 0$ transiently), $\Delta \bar{c} _ {\text{box}}/\Delta t > 0$ yields $\Phi _ {\text{stor}} < 0$, canceling the transient lateral spike.

### 6.2 Multi-Period Window Aggregation $(\tau _ {\text{win}})$

Over a rolling or fixed multi-period window of duration $\tau _ {\text{win}} = K _ {\text{win}}\thinspace \Delta t$ (consisting of $K _ {\text{win}}$ aggregation steps, e.g., $K _ {\text{win}} = 24$ for 24 hr or $168$ for 7 days), mean removal rates are computed over the set $\mathcal{K} _ {\text{valid}}$ of valid periods meeting a chosen maximum `ErrorBar` severity filter:

$$
\overline{\Phi} _ {\text{net}}(\tau _ {\text{win}}) = \frac{1}{\lvert \mathcal{K} _ {\text{valid}} \rvert} \sum _ {k \in \mathcal{K} _ {\text{valid}}} \Phi _ {\text{net},k}
$$

Over long windows $\tau _ {\text{win}} \gg \Delta t$, the storage contribution telescopes and decays as $O(1/\tau _ {\text{win}})$.

### 6.3 Ordered `ErrorBar` Enumeration & Precedence

Each period receives a single ordered `ErrorBar` category (`StrEnum` backed by Polars `pl.Enum`), ordered from lowest to highest uncertainty. When multiple conditions trigger in period $k$, the **highest-severity** level wins while all underlying numeric diagnostics remain in flat columns:

| Severity | Enum Member | Trigger Condition | Flux Output |
| :---: | :--- | :--- | :--- |
| 0 | `stable` | $N _ {\text{hull}} \ge 3$, valid geometry, dual $\text{CO} _ 2$ sensors agree, and wind & $H _ {\text{MBL}}$ meet all stability thresholds. | Computed |
| 1 | `unstable_wind_or_mblh` | Any of: $\min _ i r _ {\text{const},i,k} < r _ {\text{const,min}}$; inter-period wind direction shift $\lvert \Delta \theta _ {w,k} \rvert > \Delta \theta _ {w,\max}$; relative wind speed change $\lvert \Delta s _ {\text{poly},k} \rvert / s _ {\text{poly},k} > \eta _ {w,\max}$ (where $\Delta s _ {\text{poly},k} = s _ {\text{poly},k} - s _ {\text{poly},k-1}$); or relative MBLH change $\lvert \Delta H _ {\text{MBL},k} \rvert / H _ {\text{MBL},k} > \eta _ {H,\max}$. | Computed |
| 2 | `co2_sensor_discrepancy` | Any valid ASV has one missing $\text{CO} _ 2$ sensor or $\delta c _ i > \delta _ {\text{CO} _ 2,\max}$. | Computed |
| 3 | `two_asv_extrapolated` | $N _ {\text{tot}} = 2$, or $N _ {\text{tot}} \ge 3$ with collinear / degenerate hull (where $A _ {\text{hull}} < A _ {\min}$ or $Q _ {\text{iso}} < Q _ {\text{iso,min}}$). *(Extrapolation algorithm deferred.)* | `NaN` |
| 4 | `insufficient_asvs` | $N _ {\text{tot}} < 2$, or missing / non-positive $H _ {\text{MBL}}$. | `NaN` |

Inter-period wind direction change uses circular wrap-safe angle difference:

$$
\Delta \theta _ {w,k} = \mathrm{atan2}\negthinspace\left(\sin(\theta _ {w,k} - \theta _ {w,k-1}),\thinspace \cos(\theta _ {w,k} - \theta _ {w,k-1})\right) \in (-\pi, \pi]
$$

---

## 7. Known Simplifications & Deferred Small Effects

Tracked explicitly for future phases (aligning with [README.md](../README.md)):
1. $N = 2$ **baseline extrapolation**: Reserved as `ErrorBar.TWO_ASV_EXTRAPOLATED`; requires eddy circle diameter / cross-wind width assumption and minimum wind-to-chord alignment cutoff.
2. **Water-vapor dilution (WPL / humidity correction)**: Currently assumes dry-air mole fraction $x _ {\text{CO} _ 2}$ and dry-air column density; wet-air adjustment $(1 - e _ v / P _ {\text{atm}})$ is ~1–3% in warm marine boundary layers.
3. **Top-of-MBL entrainment flux** $(F _ {\text{top}} \propto \frac{dH _ {\text{MBL}}}{dt}(c _ {\text{free}} - \bar{c} _ {\text{box}}))$: Currently flagged via `unstable_wind_or_mblh` when $H _ {\text{MBL}}$ changes rapidly rather than parameterized.
4. **Moving-boundary Reynolds velocity subtraction** (using $(\mathbf{w} - \mathbf{w} _ {\text{asv}})\cdot \hat{\mathbf{n}} _ e$): Currently tracked as a diagnostic (`max_asv_drift_speed_m_s`) until station-keeping vs. current-drifting ASV trajectories are finalized.
5. **Passing-ship / point-plume spike filtering**: To be added as a pre-Stage-1 QC layer once real/simulated plume signatures are characterized.
6. **Quantitative $\pm\text{tCO} _ 2/\text{hr}$ error propagation**: Deferred until empirical Oshen ASV anemometer tilt/yaw and NDIR drift uncertainty budgets are calibrated.
7. **Satellite data sources & data-maturity tiers**: This initial implementation uses ASV telemetry (plus aligned MBLH/met inputs). A future iteration adding satellite providers (where each source may be unavailable, near real-time / lower accuracy such as ERA5T or NRT OCO-2/ASCAT, or delayed science-grade) should evaluate tracking source maturity as an orthogonal provenance column (following CF/QARTOD conventions) or mapping it into `ErrorBar` once upstream provider metadata is known.
