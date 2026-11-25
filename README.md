# Eddy Carbon Phase 1

## Work-in-progress

Building tools to analyze bouy/ASV data plus satellite data to compute
Carbon Dioxide Removal Rate (CDDR) using the Atmospheric CO₂ Perimeter Flux (ACPF)
method described in:

* [Measuring large-scale mCDR using a novel Atmospheric CO₂ Perimeter Flux (ACPF) method](https://www.frontiersin.org/journals/climate/articles/10.3389/fclim.2026.1941538/full#fig1)

## Initial Components

* Core implementation of the ACPF pipeline. See [ACPF math](doc/acpf_math.md) and [ACPF implementation plan](doc/acpf_implementation_plan.md).
* Marimo notebooks to visualize wind flow and ACPF analysis.

![a wind flow direction visualization](doc/visualization.png "initial wind flow direction visualization")

## Instructions

* Run `uv sync` in the project directory to create or update a Python virtual environment in the `.venv` folder and
  install/update all the library and tool dependencies listed in `pyproject.toml`.  
  (`uv sync --no-dev` for a production-only environment without dev dependencies.)  
  Launching PyCharm should notice the virtual environment and offer to use it. If PyCharm is already running, click the Python Interpreter selector in the bottom-right status bar. Built-in PyCharm terminal windows will automatically activate the virtual environment.
* Run all the unit tests: `uv run pytest`, or `uv run pytest -v` with verbose output.
* Run the `ty` type checker: `uv run ty check`, or  `uv run ty check src` to check just the `src/` directory.
* Run the `ruff` lint checker: `uv run ruff check .`, or `uv run ruff check --fix .` to auto-fix safe issues,
  `uv run ruff format --check .` to check formatting without modifying files, or `uv run ruff format .` to
  check and fix formatting.
* Run `uv run marimo check notebooks` to check the notebooks.
* Marimo notebooks can run in PyCharm, VS Code, and Antigravity IDE; best done with the Marimo plugin.

## Components TODO

* xxx.
* Figure out the algorithm to fit a circle to an eddy current in the ocean current data.
* Design visualizations to make it clear how this method measures CO₂ removal rate.
* Compare data from 2 sensors per buoy to look for sensor measurement problems. Experiment with how to handle discrepancies.
* Plan to discard or interpolate across problematic time intervals like a nearby passing diesel ship or rapid wind direction change.
* Simulate sensor data for testing the visualization and the computation, including handling bad sensor values and temporary situations.
* Data of various types from various satellites become available with different delay times. Some sources provide less accurate "near real time" data within days, then provide more accurate data after more days or weeks.
* Combine or switch between satellite vs. ASV wind data.
* Measure or compute the factors that limit accuracy.
* Keep a list of small effects to consider later (also tracked in `doc/acpf_math.md` Section 7):
  * **Local ENU tangent-plane vs. ellipsoidal geodesic geometry**: At ~200 km eddy scale, flat WGS 84 ENU horizontal chord-vs-arc distortion is < 0.02% (~8 m) and fractional area error scales as `(L / 2R)²`, negligible relative to wind and MBLH uncertainties. If deployments ever approach ~1000 km scale, consider switching to Karney ellipsoidal geodesic lengths/areas.
  * **ASV drift velocity frame & moving-boundary Reynolds correction**: Computing hourly ASV drift in a shared midpoint ENU frame avoids origin-shift drift when the hull centroid moves and provides `(u, v)` components for future `(w - w_asv) · n` moving-boundary corrections; for scalar speed alone, the 3D ECEF chord norm `‖X_k - X_{k-1}‖ / Δt` agrees to `< 1e-6` over hourly ASV displacements.
