"""Pytest test suite for perimeter_flux ocean eddy geometric computations and polygon validation."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from perimeter_flux.geometry import (
    ACAPULCO_EDDY_SAMPLE_VERTICES,
    OceanEddyGeometry,
    sort_eddy_vertices,
)

SAMPLE_CSV_PATH = Path(__file__).parent.parent.parent / "test_data" / "6_points_around_acapulco_eddy.csv"


class TestSampleDataLoading:
    """Tests for loading and verifying the sample Acapulco eddy buoy dataset."""

    def test_load_sample_file_from_csv(self):
        """Test reading the 6 buoy coordinates from the CSV file."""
        vertices = OceanEddyGeometry.load_csv(SAMPLE_CSV_PATH)
        assert len(vertices) == 6
        assert vertices == list(ACAPULCO_EDDY_SAMPLE_VERTICES)

    def test_static_sample_object(self):
        """Test that the static Python object contains the exact 6 buoy coordinates."""
        assert len(ACAPULCO_EDDY_SAMPLE_VERTICES) == 6
        expected_points = [
            (14.29, -101.64),
            (13.87, -102.88),
            (12.93, -102.82),
            (12.45, -101.67),
            (12.97, -100.53),
            (13.79, -100.59),
        ]
        assert ACAPULCO_EDDY_SAMPLE_VERTICES == tuple(expected_points)

    def test_sample_file_meets_all_sanity_checks(self):
        """Verify that the sample Acapulco eddy buoys pass all sanity checks."""
        cleaned = OceanEddyGeometry.validate_vertices(ACAPULCO_EDDY_SAMPLE_VERTICES)
        assert len(cleaned) == 6

        # Check latitude span <= 10 degrees
        lats = [lat for lat, _ in cleaned]
        lat_span = max(lats) - min(lats)
        assert lat_span <= 10.0
        assert math.isclose(lat_span, 14.29 - 12.45, abs_tol=1e-5)

        # Check longitude span <= 10 degrees
        lons = [lon for _, lon in cleaned]
        lon_span = max(lons) - min(lons)
        assert lon_span <= 10.0
        assert math.isclose(lon_span, -100.53 - (-102.88), abs_tol=1e-5)

    def test_sample_file_is_already_convex_and_ccw(self):
        """Verify that the sample buoy locations are already in counterclockwise convex order."""
        pts = ACAPULCO_EDDY_SAMPLE_VERTICES
        assert OceanEddyGeometry.is_counterclockwise(pts) is True
        assert OceanEddyGeometry.is_convex(pts, ccw_only=True) is True

    def test_sort_sample_file_preserves_order(self):
        """Verify that sorting the sample file returns the exact same counterclockwise order."""
        sorted_pts = sort_eddy_vertices(ACAPULCO_EDDY_SAMPLE_VERTICES)
        assert sorted_pts == list(ACAPULCO_EDDY_SAMPLE_VERTICES)


class TestVertexSorting:
    """Tests for counterclockwise sorting of convex polygon vertices."""

    def test_sort_shuffled_vertices_restores_ccw_order(self):
        """Shuffling the vertices should sort back to the same counterclockwise convex polygon."""
        original = list(ACAPULCO_EDDY_SAMPLE_VERTICES)
        # Permute vertices in arbitrary order
        shuffled = [original[3], original[0], original[4], original[1], original[5], original[2]]

        # Sorting with start anchored to original[0] must match original order exactly
        sorted_anchored = sort_eddy_vertices(shuffled, start=original[0])
        assert sorted_anchored == original

        # Default sort should produce a cyclically equivalent counterclockwise polygon
        sorted_default = sort_eddy_vertices(shuffled)
        assert OceanEddyGeometry.are_polygons_equivalent(sorted_default, original)
        assert OceanEddyGeometry.is_counterclockwise(sorted_default)
        assert OceanEddyGeometry.is_convex(sorted_default)

    def test_sort_reversed_clockwise_vertices(self):
        """Reversing vertices (clockwise order) should be sorted back into counterclockwise order."""
        original = list(ACAPULCO_EDDY_SAMPLE_VERTICES)
        reversed_pts = list(reversed(original))
        assert OceanEddyGeometry.is_counterclockwise(reversed_pts) is False

        sorted_pts = sort_eddy_vertices(reversed_pts, start=original[0])
        assert sorted_pts == original
        assert OceanEddyGeometry.is_counterclockwise(sorted_pts) is True

    def test_sort_start_options(self):
        """Test different start options ('min_lat', 'min_lon', 'first')."""
        pts = list(ACAPULCO_EDDY_SAMPLE_VERTICES)

        # Min latitude should place (12.45, -101.67) first
        sorted_min_lat = sort_eddy_vertices(pts, start="min_lat")
        assert sorted_min_lat[0] == (12.45, -101.67)
        assert OceanEddyGeometry.are_polygons_equivalent(sorted_min_lat, pts)

        # Min longitude should place (13.87, -102.88) first
        sorted_min_lon = sort_eddy_vertices(pts, start="min_lon")
        assert sorted_min_lon[0] == (13.87, -102.88)
        assert OceanEddyGeometry.are_polygons_equivalent(sorted_min_lon, pts)

    def test_convex_hull_calculation(self):
        """Test Jarvis March convex hull returns the 6 buoy boundary points in CCW order."""
        shuffled = [
            ACAPULCO_EDDY_SAMPLE_VERTICES[2],
            ACAPULCO_EDDY_SAMPLE_VERTICES[5],
            ACAPULCO_EDDY_SAMPLE_VERTICES[1],
            ACAPULCO_EDDY_SAMPLE_VERTICES[4],
            ACAPULCO_EDDY_SAMPLE_VERTICES[0],
            ACAPULCO_EDDY_SAMPLE_VERTICES[3],
        ]
        hull = OceanEddyGeometry.convex_hull(shuffled)
        assert len(hull) == 6
        assert OceanEddyGeometry.is_counterclockwise(hull)
        assert OceanEddyGeometry.are_polygons_equivalent(hull, ACAPULCO_EDDY_SAMPLE_VERTICES)


class TestSanityChecks:
    """Tests for sanity check validations."""

    def test_international_date_line_spanning_rejected(self):
        """Points spanning across the International Date Line (180° / -180°) must be rejected."""
        # Small eddy right across the IDL (179°E to -179°W = 181°E)
        idl_crossing_eddy = [
            (10.0, 179.0),
            (10.5, 179.5),
            (10.0, -179.5),
            (9.5, -179.0),
        ]
        with pytest.raises(ValueError, match="International Date Line"):
            OceanEddyGeometry.validate_vertices(idl_crossing_eddy)

    def test_international_date_line_touching_rejected(self):
        """Points located directly on the 180° meridian must be rejected."""
        idl_touching = [
            (10.0, 180.0),
            (10.5, 179.5),
            (9.5, 179.2),
        ]
        with pytest.raises(ValueError, match="International Date Line"):
            OceanEddyGeometry.validate_vertices(idl_touching)

    def test_prime_meridian_allowed(self):
        """Points crossing the Prime Meridian (0°) are NOT spanning the IDL and must be allowed."""
        greenwich_eddy = [
            (50.0, -0.5),
            (50.5, 0.0),
            (50.3, 0.6),
            (49.8, 0.2),
        ]
        # Should pass validation without error
        cleaned = OceanEddyGeometry.validate_vertices(greenwich_eddy)
        assert len(cleaned) == 4

    def test_north_pole_spanning_rejected(self):
        """Points reaching or spanning the North Pole (+90°) must be rejected."""
        north_pole_pts = [
            (90.0, 0.0),
            (85.0, 5.0),
            (85.0, -5.0),
        ]
        with pytest.raises(ValueError, match="pole"):
            OceanEddyGeometry.validate_vertices(north_pole_pts)

    def test_south_pole_spanning_rejected(self):
        """Points reaching or spanning the South Pole (-90°) must be rejected."""
        south_pole_pts = [
            (-90.0, 0.0),
            (-85.0, 5.0),
            (-85.0, -5.0),
        ]
        with pytest.raises(ValueError, match="pole"):
            OceanEddyGeometry.validate_vertices(south_pole_pts)

    def test_enclosing_pole_rejected(self):
        """Vertices surrounding a pole must be rejected."""
        surrounding_pole = [
            (85.0, 0.0),
            (85.0, 120.0),
            (85.0, -120.0),
        ]
        with pytest.raises(ValueError, match="pole"):
            OceanEddyGeometry.validate_vertices(surrounding_pole)

    def test_latitude_span_exceeding_10_degrees_rejected(self):
        """Vertices spanning more than 10 degrees latitude must be rejected."""
        wide_lat_pts = [
            (10.0, -100.0),
            (20.5, -100.0),  # Span = 10.5 degrees
            (15.0, -95.0),
        ]
        with pytest.raises(ValueError, match="latitude, exceeding the maximum allowed limit of 10 degrees"):
            OceanEddyGeometry.validate_vertices(wide_lat_pts)

    def test_latitude_span_boundary_10_degrees_allowed(self):
        """Vertices spanning exactly 10 degrees latitude must be allowed."""
        boundary_lat_pts = [
            (10.0, -100.0),
            (20.0, -100.0),  # Span = exactly 10.0 degrees
            (15.0, -95.0),
        ]
        cleaned = OceanEddyGeometry.validate_vertices(boundary_lat_pts)
        assert len(cleaned) == 3

    def test_longitude_span_exceeding_10_degrees_rejected(self):
        """Vertices spanning more than 10 degrees longitude must be rejected."""
        wide_lon_pts = [
            (14.0, -105.0),
            (14.0, -94.5),  # Span = 10.5 degrees
            (18.0, -100.0),
        ]
        with pytest.raises(ValueError, match="longitude, exceeding the maximum allowed limit of 10 degrees"):
            OceanEddyGeometry.validate_vertices(wide_lon_pts)

    def test_longitude_span_boundary_10_degrees_allowed(self):
        """Vertices spanning exactly 10 degrees longitude must be allowed."""
        boundary_lon_pts = [
            (14.0, -105.0),
            (14.0, -95.0),  # Span = exactly 10.0 degrees
            (18.0, -100.0),
        ]
        cleaned = OceanEddyGeometry.validate_vertices(boundary_lon_pts)
        assert len(cleaned) == 3

    def test_minimum_vertices_requirement(self):
        """Polygon must have at least 3 vertices."""
        with pytest.raises(ValueError, match="requires at least 3 vertices"):
            OceanEddyGeometry.validate_vertices([(10.0, -100.0), (11.0, -101.0)])

    def test_invalid_latitude_range(self):
        """Latitudes outside [-90, 90] must raise ValueError."""
        with pytest.raises(ValueError, match="outside valid"):
            OceanEddyGeometry.validate_vertices([(95.0, -100.0), (10.0, -100.0), (12.0, -101.0)])

    def test_collinear_vertices_rejected(self):
        """Collinear vertices cannot form a valid polygon."""
        collinear_pts = [
            (10.0, -100.0),
            (11.0, -100.0),
            (12.0, -100.0),
        ]
        with pytest.raises(ValueError, match="collinear or have zero area"):
            OceanEddyGeometry.validate_vertices(collinear_pts)

    def test_concave_vertices_rejected_when_enforcing_convex(self):
        """Concave vertices must raise ValueError when enforce_convex=True."""
        # Square (0,0), (2,0), (2,2), (0,2) with an indented/interior point (1, 1.5)
        concave_pts = [
            (0.0, 0.0),
            (0.0, 2.0),
            (2.0, 2.0),
            (1.5, 1.0),
            (2.0, 0.0),
        ]
        with pytest.raises(ValueError, match="strictly convex"):
            sort_eddy_vertices(concave_pts, enforce_convex=True)


class TestOceanEddyGeometryClass:
    """Tests for the OceanEddyGeometry class instantiation and properties."""

    def test_instance_properties(self):
        """Test properties of an instantiated OceanEddyGeometry."""
        eddy = OceanEddyGeometry(ACAPULCO_EDDY_SAMPLE_VERTICES)
        assert len(eddy.vertices) == 6
        assert eddy.lat_span < 2.0
        assert eddy.lon_span < 2.5
        assert eddy.area_deg2 > 3.0
        assert eddy.area_km2 > 30000.0  # ~37,000 km^2
        c_lat, c_lon = eddy.centroid_lat_lon
        assert 13.0 < c_lat < 14.0
        assert -102.0 < c_lon < -101.0
