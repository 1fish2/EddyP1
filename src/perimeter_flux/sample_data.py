"""Shared sample ocean-eddy ASV coordinates for synthetic scenarios and tests."""

from __future__ import annotations

# 6-ASV array (~200 km diameter) around an anticyclonic eddy near Acapulco
# (matching test_data/6_points_around_acapulco_eddy.csv), in CCW order.
ACAPULCO_EDDY_VERTICES: tuple[tuple[float, float], ...] = (
    (14.29, -101.64),
    (13.87, -102.88),
    (12.93, -102.82),
    (12.45, -101.67),
    (12.97, -100.53),
    (13.79, -100.59),
)

# 6-ASV array (~200 km diameter) straddling BOTH the Equator (0 deg) and the
# International Date Line (+/-180 deg), in CCW order. Useful for testing
# hemisphere and antimeridian continuity end-to-end.
IDL_EQUATOR_EDDY_VERTICES: tuple[tuple[float, float], ...] = (
    (0.92, -179.95),
    (0.48, 178.92),
    (-0.46, 178.96),
    (-0.92, -179.98),
    (-0.44, -178.90),
    (0.46, -178.94),
)
