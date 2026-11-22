# Eddy Carbon Phase 1

## Work-in-progress

Building tools to analyze bouy/ASV data plus satellite data to compute
Carbon Dioxide Removal Rate (CDDR) using the Atmospheric CO₂ Perimeter Flux (ACPF)
method described in:

* [Measuring large-scale mCDR using a novel Atmospheric CO₂ Perimeter Flux (ACPF) method](https://www.frontiersin.org/journals/climate/articles/10.3389/fclim.2026.1941538/full#fig1)


## Initial components

* Download initial data from F4CR's satellite data digestion pipeline. More to do.
* Load it into a Polars dataframe.
* Start to explore it with a Marimo interactive notebook.

![a wind flow direction visualization](doc/visualization.png "initial wind flow direction visualization")


## Components TODO

* Design the algorithm to approximate CO₂ flux measurements over the eddy circle with a variable number of CO₂-measuring ASVs. The plan is for 6 ASVs around the eddy circumference. The bare minimum is two ASVs, approximately across the eddy from each other. There's no useful data when the wind is perpendicular to the vector between two ASVs.
* Figure out the algorithm to fit a circle to an eddy current in the ocean current data.
* Design visualizations to make it clear how this method measures CO₂ removal rate.
* Compare data from 2 sensors per buoy to look for sensor measurement problems. Experiment with how to handle discrepancies.
* Plan to discard or interpolate across problematic time intervals like a nearby passing diesel ship or rapid wind direction change.
* Simulate sensor data for testing the visualization and the computation, including handling bad sensor values and temporary situations.
* Data of various types from various satellites become available with different delay times. Some sources provide less accurate "near real time" data within days, then provide more accurate data after more days or weeks.
* Combine or switch between satellite vs. ASV wind data.
* Measure or compute the factors that limit accuracy.
* Keep a list of small effects to consider later.
