"""Shared static test fixtures and sample coordinates for perimeter_flux tests."""

from __future__ import annotations

# Sample ASV locations around an eddy near Acapulco
# (matching test_data/6_points_around_acapulco_eddy.csv)
ACAPULCO_EDDY_VERTICES: tuple[tuple[float, float], ...] = (
    (14.29, -101.64),
    (13.87, -102.88),
    (12.93, -102.82),
    (12.45, -101.67),
    (12.97, -100.53),
    (13.79, -100.59),
)
