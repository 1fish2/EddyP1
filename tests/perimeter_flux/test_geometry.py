"""Pytest test suite for perimeter_flux ocean eddy geometric computations and polygon validation."""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

from perimeter_flux.geometry import (
    ACAPULCO_EDDY_SAMPLE_VERTICES,
    CraftSample,
    Vector,
    SensorEdge,
    SensorPerimeter,
    sort_eddy_vertices,
)

SAMPLE_CSV_PATH = Path(__file__).parent.parent.parent / "test_data" / "6_points_around_acapulco_eddy.csv"


class TestSampleDataLoading:
    """Tests for loading and verifying the sample Acapulco eddy buoy dataset."""

    def test_load_sample_file_from_csv(self):
        """Test reading the 6 buoy coordinates from the CSV file."""
        vertices = SensorPerimeter.load_csv(SAMPLE_CSV_PATH)
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
        cleaned = SensorPerimeter.validate_vertices(ACAPULCO_EDDY_SAMPLE_VERTICES)
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
        assert SensorPerimeter.is_counterclockwise(pts) is True
        assert SensorPerimeter.is_convex(pts, ccw_only=True) is True

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
        assert SensorPerimeter.are_polygons_equivalent(sorted_default, original)
        assert SensorPerimeter.is_counterclockwise(sorted_default)
        assert SensorPerimeter.is_convex(sorted_default)

    def test_sort_reversed_clockwise_vertices(self):
        """Reversing vertices (clockwise order) should be sorted back into counterclockwise order."""
        original = list(ACAPULCO_EDDY_SAMPLE_VERTICES)
        reversed_pts = list(reversed(original))
        assert SensorPerimeter.is_counterclockwise(reversed_pts) is False

        sorted_pts = sort_eddy_vertices(reversed_pts, start=original[0])
        assert sorted_pts == original
        assert SensorPerimeter.is_counterclockwise(sorted_pts) is True

    def test_sort_start_options(self):
        """Test different start options ('min_lat', 'min_lon', 'first')."""
        pts = list(ACAPULCO_EDDY_SAMPLE_VERTICES)

        # Min latitude should place (12.45, -101.67) first
        sorted_min_lat = sort_eddy_vertices(pts, start="min_lat")
        assert sorted_min_lat[0] == (12.45, -101.67)
        assert SensorPerimeter.are_polygons_equivalent(sorted_min_lat, pts)

        # Min longitude should place (13.87, -102.88) first
        sorted_min_lon = sort_eddy_vertices(pts, start="min_lon")
        assert sorted_min_lon[0] == (13.87, -102.88)
        assert SensorPerimeter.are_polygons_equivalent(sorted_min_lon, pts)

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
        hull = SensorPerimeter.convex_hull(shuffled)
        assert len(hull) == 6
        assert SensorPerimeter.is_counterclockwise(hull)
        assert SensorPerimeter.are_polygons_equivalent(hull, ACAPULCO_EDDY_SAMPLE_VERTICES)


class TestSanityChecks:
    """Tests for sanity check validations."""

    def test_international_date_line_spanning_rejected(self):
        """Points spanning across the International Date Line (180° / -180°) must be rejected."""
        idl_crossing_eddy = [
            (10.0, 179.0),
            (10.5, 179.5),
            (10.0, -179.5),
            (9.5, -179.0),
        ]
        with pytest.raises(ValueError, match="International Date Line"):
            SensorPerimeter.validate_vertices(idl_crossing_eddy)

    def test_international_date_line_touching_rejected(self):
        """Points located directly on the 180° meridian must be rejected."""
        idl_touching = [
            (10.0, 180.0),
            (10.5, 179.5),
            (9.5, 179.2),
        ]
        with pytest.raises(ValueError, match="International Date Line"):
            SensorPerimeter.validate_vertices(idl_touching)

    def test_prime_meridian_allowed(self):
        """Points crossing the Prime Meridian (0°) are NOT spanning the IDL and must be allowed."""
        greenwich_eddy = [
            (50.0, -0.5),
            (50.5, 0.0),
            (50.3, 0.6),
            (49.8, 0.2),
        ]
        cleaned = SensorPerimeter.validate_vertices(greenwich_eddy)
        assert len(cleaned) == 4

    def test_north_pole_spanning_rejected(self):
        """Points reaching or spanning the North Pole (+90°) must be rejected."""
        north_pole_pts = [
            (90.0, 0.0),
            (85.0, 5.0),
            (85.0, -5.0),
        ]
        with pytest.raises(ValueError, match="pole"):
            SensorPerimeter.validate_vertices(north_pole_pts)

    def test_south_pole_spanning_rejected(self):
        """Points reaching or spanning the South Pole (-90°) must be rejected."""
        south_pole_pts = [
            (-90.0, 0.0),
            (-85.0, 5.0),
            (-85.0, -5.0),
        ]
        with pytest.raises(ValueError, match="pole"):
            SensorPerimeter.validate_vertices(south_pole_pts)

    def test_enclosing_pole_rejected(self):
        """Vertices surrounding a pole must be rejected."""
        surrounding_pole = [
            (85.0, 0.0),
            (85.0, 120.0),
            (85.0, -120.0),
        ]
        with pytest.raises(ValueError, match="pole"):
            SensorPerimeter.validate_vertices(surrounding_pole)

    def test_latitude_span_exceeding_10_degrees_rejected(self):
        """Vertices spanning more than 10 degrees latitude must be rejected."""
        wide_lat_pts = [
            (10.0, -100.0),
            (20.5, -100.0),  # Span = 10.5 degrees
            (15.0, -95.0),
        ]
        with pytest.raises(ValueError, match="latitude, exceeding the maximum allowed limit of 10 degrees"):
            SensorPerimeter.validate_vertices(wide_lat_pts)

    def test_latitude_span_boundary_10_degrees_allowed(self):
        """Vertices spanning exactly 10 degrees latitude must be allowed."""
        boundary_lat_pts = [
            (10.0, -100.0),
            (20.0, -100.0),  # Span = exactly 10.0 degrees
            (15.0, -95.0),
        ]
        cleaned = SensorPerimeter.validate_vertices(boundary_lat_pts)
        assert len(cleaned) == 3

    def test_longitude_span_exceeding_10_degrees_rejected(self):
        """Vertices spanning more than 10 degrees longitude must be rejected."""
        wide_lon_pts = [
            (14.0, -105.0),
            (14.0, -94.5),  # Span = 10.5 degrees
            (18.0, -100.0),
        ]
        with pytest.raises(ValueError, match="longitude, exceeding the maximum allowed limit of 10 degrees"):
            SensorPerimeter.validate_vertices(wide_lon_pts)

    def test_longitude_span_boundary_10_degrees_allowed(self):
        """Vertices spanning exactly 10 degrees longitude must be allowed."""
        boundary_lon_pts = [
            (14.0, -105.0),
            (14.0, -95.0),  # Span = exactly 10.0 degrees
            (18.0, -100.0),
        ]
        cleaned = SensorPerimeter.validate_vertices(boundary_lon_pts)
        assert len(cleaned) == 3

    def test_minimum_vertices_requirement(self):
        """Polygon must have at least 3 vertices."""
        with pytest.raises(ValueError, match="requires at least 3 vertices"):
            SensorPerimeter.validate_vertices([(10.0, -100.0), (11.0, -101.0)])

    def test_invalid_type_error_message(self):
        """Passing an invalid non-sequence type must display the type name in the error message."""
        with pytest.raises(ValueError, match="got int"):
            SensorPerimeter.validate_vertices(12345)  # type: ignore[arg-type]

    def test_invalid_latitude_range(self):
        """Latitudes outside [-90, 90] must raise ValueError."""
        with pytest.raises(ValueError, match="outside valid"):
            SensorPerimeter.validate_vertices([(95.0, -100.0), (10.0, -100.0), (12.0, -101.0)])

    def test_collinear_vertices_rejected(self):
        """Collinear vertices cannot form a valid polygon."""
        collinear_pts = [
            (10.0, -100.0),
            (11.0, -100.0),
            (12.0, -100.0),
        ]
        with pytest.raises(ValueError, match="collinear or have zero area"):
            SensorPerimeter.validate_vertices(collinear_pts)

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


class TestSensorPerimeterAndEdges:
    """Tests for SensorPerimeter immutability and SensorEdge normal calculations."""

    def test_vertices_is_immutable_tuple(self):
        """Test that SensorPerimeter.vertices is an immutable tuple of tuples."""
        perimeter = SensorPerimeter(ACAPULCO_EDDY_SAMPLE_VERTICES)
        assert isinstance(perimeter.vertices, tuple)
        assert len(perimeter.vertices) == 6
        assert all(isinstance(v, tuple) and len(v) == 2 for v in perimeter.vertices)
        assert perimeter.vertices == ACAPULCO_EDDY_SAMPLE_VERTICES

    def test_edges_property(self):
        """Test that perimeter.edges returns a tuple of SensorEdge instances in CCW order."""
        perimeter = SensorPerimeter(ACAPULCO_EDDY_SAMPLE_VERTICES)
        edges = perimeter.edges
        assert isinstance(edges, tuple)
        assert len(edges) == 6

        verts = perimeter.vertices
        for i, edge in enumerate(edges):
            assert isinstance(edge, SensorEdge)
            assert edge.v1 == verts[i]
            assert edge.v2 == verts[(i + 1) % 6]
            assert edge.vertices == (verts[i], verts[(i + 1) % 6])

    def test_edges_form_closed_loop(self):
        """Test that the end vertex of each edge is the start vertex of the next edge."""
        perimeter = SensorPerimeter(ACAPULCO_EDDY_SAMPLE_VERTICES)
        edges = perimeter.edges
        n = len(edges)
        for i in range(n):
            assert edges[i].v2 == edges[(i + 1) % n].v1

    def test_sensor_edge_properties_and_midpoint(self):
        """Test SensorEdge v1, v2, midpoint, repr, and equality."""
        p1 = (14.0, -102.0)
        p2 = (14.5, -101.0)
        edge = SensorEdge(p1, p2)
        assert edge.v1 == p1
        assert edge.v2 == p2
        assert edge.vertices == (p1, p2)
        assert edge.midpoint == (14.25, -101.5)
        assert edge.length_km() > 0.0
        assert repr(edge) == "SensorEdge(v1=(14.0, -102.0), v2=(14.5, -101.0))"
        assert edge == SensorEdge(p1, p2)
        assert edge != SensorEdge(p2, p1)

    def test_vector_properties(self):
        """Test Vector components, aliases, polar coordinates, and sequence behavior."""
        v = Vector(u=3.0, v=4.0)
        assert v.u == 3.0
        assert v.v == 4.0
        assert v.x == 3.0
        assert v.y == 4.0
        assert v.east == 3.0
        assert v.north == 4.0
        assert v[0] == 3.0
        assert v[1] == 4.0
        u_val, v_val = v
        assert (u_val, v_val) == (3.0, 4.0)

        # Magnitude and angle (polar coordinates)
        assert math.isclose(v.magnitude, 5.0, abs_tol=1e-12)
        assert math.isclose(v.angle, math.atan2(4.0, 3.0), abs_tol=1e-12)

        # Cardinal directions angle
        assert math.isclose(Vector(1.0, 0.0).angle, 0.0, abs_tol=1e-12)
        assert math.isclose(Vector(0.0, 1.0).angle, math.pi / 2, abs_tol=1e-12)
        assert math.isclose(Vector(-1.0, 0.0).angle, math.pi, abs_tol=1e-12)
        assert math.isclose(Vector(0.0, -1.0).angle, -math.pi / 2, abs_tol=1e-12)

    def test_vector_dot_product(self):
        """Test Vector.dot method and @ operator."""
        v1 = Vector(u=0.6, v=0.8)
        v2 = Vector(u=0.8, v=-0.6)
        v3 = Vector(u=0.6, v=0.8)
        v_opposite = Vector(u=-0.6, v=-0.8)

        # Orthogonal vectors -> dot product is 0.0
        assert math.isclose(v1.dot(v2), 0.0, abs_tol=1e-12)
        assert math.isclose(v1 @ v2, 0.0, abs_tol=1e-12)

        # Parallel unit vectors -> dot product is 1.0
        assert math.isclose(v1.dot(v3), 1.0, abs_tol=1e-12)
        assert math.isclose(v1 @ v3, 1.0, abs_tol=1e-12)

        # Anti-parallel unit vectors -> dot product is -1.0
        assert math.isclose(v1.dot(v_opposite), -1.0, abs_tol=1e-12)

        # Non-unit vector dot product
        v_a = Vector(u=2.0, v=3.0)
        v_b = Vector(u=4.0, v=-1.0)
        assert math.isclose(v_a @ v_b, 5.0, abs_tol=1e-12)

        # Interoperability with sequences / tuples
        assert math.isclose(v1.dot((0.8, -0.6)), 0.0, abs_tol=1e-12)
        assert math.isclose(v1 @ [0.6, 0.8], 1.0, abs_tol=1e-12)

        # Invalid type should raise TypeError
        with pytest.raises(TypeError, match="Cannot compute dot product"):
            v1.dot(42)  # type: ignore[arg-type]

    def test_vector_normalize(self):
        """Test Vector.normalize static method with two float arguments."""
        n = Vector.normalize(3.0, 4.0)
        assert isinstance(n, Vector)
        assert math.isclose(n.u, 0.6, abs_tol=1e-12)
        assert math.isclose(n.v, 0.8, abs_tol=1e-12)
        assert math.isclose(n.magnitude, 1.0, abs_tol=1e-12)

        # Caller can unpack sequence with *
        pair = (0.0, -10.0)
        n_unpacked = Vector.normalize(*pair)
        assert math.isclose(n_unpacked.u, 0.0, abs_tol=1e-12)
        assert math.isclose(n_unpacked.v, -1.0, abs_tol=1e-12)

        # Zero length vector raises ValueError
        with pytest.raises(ValueError, match="Cannot normalize a zero-length vector"):
            Vector.normalize(0.0, 0.0)

        # Missing argument raises TypeError
        with pytest.raises(TypeError):
            Vector.normalize(42)  # type: ignore[call-arg]

    def test_downwind_vertex_square(self):
        """Test downwind vertex and index on a canonical CCW square."""
        # Square corners: SW 0:(0,0), SE 1:(0,2), NE 2:(2,2), NW 3:(2,0)
        square_pts = [(0.0, 0.0), (0.0, 2.0), (2.0, 2.0), (2.0, 0.0)]
        perimeter = SensorPerimeter(square_pts)

        # Wind blowing North-East: downwind is NE (2, 2) at index 2
        wind_ne = Vector.normalize(1.0, 1.0)
        assert perimeter.downwind_vertex(wind_ne) == ((2.0, 2.0), 2)

        # Wind blowing South-West: downwind is SW (0, 0) at index 0
        wind_sw = Vector.normalize(-1.0, -1.0)
        assert perimeter.downwind_vertex(wind_sw) == ((0.0, 0.0), 0)

        # Wind blowing North-West: downwind is NW (2, 0) at index 3
        wind_nw = Vector.normalize(-1.0, 1.0)
        assert perimeter.downwind_vertex(wind_nw) == ((2.0, 0.0), 3)

        # Wind blowing South-East: downwind is SE (0, 2) at index 1
        wind_se = Vector.normalize(1.0, -1.0)
        assert perimeter.downwind_vertex(wind_se) == ((0.0, 2.0), 1)

        # Tie-breaking test: Wind East (1, 0) is perpendicular to edge (0,2) -> (2,2)
        # Both index 1 (0,2) and index 2 (2,2) have maximum Eastward projection.
        # Downwind_vertex returns the first vertex in CCW traversal order (index 1).
        wind_east = Vector(1.0, 0.0)
        assert perimeter.downwind_vertex(wind_east) == ((0.0, 2.0), 1)

    def test_downwind_vertex_acapulco_eddy(self):
        """Test downwind vertex and index on the Acapulco eddy buoy perimeter."""
        perimeter = SensorPerimeter(ACAPULCO_EDDY_SAMPLE_VERTICES)

        # 0: (14.29, -101.64) - furthest North
        # 1: (13.87, -102.88) - furthest West
        # 2: (12.93, -102.82)
        # 3: (12.45, -101.67) - furthest South
        # 4: (12.97, -100.53) - furthest East
        # 5: (13.79, -100.59)

        # Wind blowing North (u=0, v=1) -> furthest North vertex at index 0
        wind_north = Vector(u=0.0, v=1.0)
        assert perimeter.downwind_vertex(wind_north) == ((14.29, -101.64), 0)

        # Wind blowing South (u=0, v=-1) -> furthest South vertex at index 3
        wind_south = Vector(u=0.0, v=-1.0)
        assert perimeter.downwind_vertex(wind_south) == ((12.45, -101.67), 3)

        # Wind blowing West (u=-1, v=0) -> furthest West vertex at index 1
        wind_west = Vector(u=-1.0, v=0.0)
        assert perimeter.downwind_vertex(wind_west) == ((13.87, -102.88), 1)

        # Wind blowing East (u=1, v=0) -> furthest East vertex at index 4
        wind_east = Vector(u=1.0, v=0.0)
        assert perimeter.downwind_vertex(wind_east) == ((12.97, -100.53), 4)

        # Non-unit wind vector works identically
        wind_north_scaled = Vector(u=0.0, v=45.0)
        assert perimeter.downwind_vertex(wind_north_scaled) == ((14.29, -101.64), 0)

        # Invalid wind vector errors
        with pytest.raises(ValueError, match="non-zero magnitude"):
            perimeter.downwind_vertex(Vector(0.0, 0.0))

        # wind must be a Vector instance
        with pytest.raises(TypeError, match="wind must be a Vector"):
            perimeter.downwind_vertex([0.0, 50.0])  # type: ignore[arg-type]

    def test_perimeter_km_property(self):
        """Test perimeter_km equals the sum of edge lengths."""
        perimeter = SensorPerimeter(ACAPULCO_EDDY_SAMPLE_VERTICES)
        expected_total = sum(edge.length_km() for edge in perimeter.edges)
        assert math.isclose(perimeter.perimeter_km, expected_total, rel_tol=1e-9)
        assert 700.0 < perimeter.perimeter_km < 800.0  # Acapulco eddy perimeter ~735 km

    def test_sensor_edge_normal_unit_magnitude(self):
        """Verify that every edge normal has magnitude 1.0."""
        perimeter = SensorPerimeter(ACAPULCO_EDDY_SAMPLE_VERTICES)
        for edge in perimeter.edges:
            n = edge.normal()
            assert isinstance(n, Vector)
            magnitude = math.hypot(n.u, n.v)
            assert math.isclose(magnitude, 1.0, rel_tol=1e-9)

    def test_sensor_edge_normal_is_orthogonal_to_edge(self):
        """Verify that the normal is orthogonal to the edge tangent vector."""
        perimeter = SensorPerimeter(ACAPULCO_EDDY_SAMPLE_VERTICES)
        for edge in perimeter.edges:
            n = edge.normal(metric=True)
            mid_lat = (edge.v1[0] + edge.v2[0]) / 2.0
            dx = (edge.v2[1] - edge.v1[1]) * math.cos(math.radians(mid_lat))
            dy = edge.v2[0] - edge.v1[0]
            # Dot product of tangent and normal should be 0
            dot = dx * n.u + dy * n.v
            assert math.isclose(dot, 0.0, abs_tol=1e-12)

    def test_sensor_edge_unprojected_normal(self):
        """Verify unprojected (degree space) normal calculation."""
        edge = SensorEdge((10.0, -100.0), (12.0, -98.0))
        n_deg = edge.normal(metric=False)
        dx = edge.v2[1] - edge.v1[1]  # 2.0
        dy = edge.v2[0] - edge.v1[0]  # 2.0
        # Dot product with tangent in unprojected space must be 0
        dot = dx * n_deg.u + dy * n_deg.v
        assert math.isclose(dot, 0.0, abs_tol=1e-12)
        assert math.isclose(math.hypot(n_deg.u, n_deg.v), 1.0, rel_tol=1e-9)

    def test_sensor_edge_normal_points_outward(self):
        """Verify that the normal points away from the centroid of the perimeter."""
        perimeter = SensorPerimeter(ACAPULCO_EDDY_SAMPLE_VERTICES)
        c_lat, c_lon = perimeter.centroid_lat_lon

        for edge in perimeter.edges:
            n = edge.normal(metric=True)
            mid_lat, mid_lon = edge.midpoint

            # Ray from centroid to edge midpoint
            ray_x = (mid_lon - c_lon) * math.cos(math.radians(mid_lat))
            ray_y = mid_lat - c_lat

            # Dot product with outward normal must be positive
            outward_dot = ray_x * n.u + ray_y * n.v
            assert outward_dot > 0.0

    def test_cardinal_direction_normals(self):
        """Test outward normals for canonical cardinal directions on a CCW square."""
        # CCW square: (0,0) -> (0,1) -> (1,1) -> (1,0) -> (0,0)
        # 1. Bottom edge (0,0) -> (0,1) heading East: normal should point South (u=0, v=-1)
        edge_south = SensorEdge((0.0, 0.0), (0.0, 1.0))
        n_south = edge_south.normal()
        assert math.isclose(n_south.u, 0.0, abs_tol=1e-9)
        assert math.isclose(n_south.v, -1.0, abs_tol=1e-9)

        # 2. Right edge (0,1) -> (1,1) heading North: normal should point East (u=1, v=0)
        edge_east = SensorEdge((0.0, 1.0), (1.0, 1.0))
        n_east = edge_east.normal()
        assert math.isclose(n_east.u, 1.0, abs_tol=1e-9)
        assert math.isclose(n_east.v, 0.0, abs_tol=1e-9)

        # 3. Top edge (1,1) -> (1,0) heading West: normal should point North (u=0, v=1)
        edge_north = SensorEdge((1.0, 1.0), (1.0, 0.0))
        n_north = edge_north.normal()
        assert math.isclose(n_north.u, 0.0, abs_tol=1e-9)
        assert math.isclose(n_north.v, 1.0, abs_tol=1e-9)

        # 4. Left edge (1,0) -> (0,0) heading South: normal should point West (u=-1, v=0)
        edge_west = SensorEdge((1.0, 0.0), (0.0, 0.0))
        n_west = edge_west.normal()
        assert math.isclose(n_west.u, -1.0, abs_tol=1e-9)
        assert math.isclose(n_west.v, 0.0, abs_tol=1e-9)

    def test_zero_length_edge_raises_error(self):
        """Test that calling normal on a degenerate zero-length edge raises ValueError."""
        degenerate_edge = SensorEdge((10.0, -100.0), (10.0, -100.0))
        with pytest.raises(ValueError, match="zero-length edge"):
            degenerate_edge.normal()


class TestCraftSample:
    """Tests for CraftSample frozen dataclass with slots."""

    def test_craft_sample_creation(self):
        """Test creating a CraftSample with standard fields."""
        dt = datetime(2026, 10, 26, 12, 0, 0, tzinfo=timezone.utc)
        sample = CraftSample(
            craft_id="ASV-01",
            timestamp=dt,
            position=(14.29, -101.64),
            co2_1=415.5,
            co2_2=416.1,
            wind=Vector(u=3.2, v=-1.8),
        )

        assert sample.craft_id == "ASV-01"
        assert sample.timestamp == dt
        assert sample.position == (14.29, -101.64)
        assert sample.co2_1 == 415.5
        assert sample.co2_2 == 416.1
        assert sample.wind == Vector(u=3.2, v=-1.8)

    def test_craft_sample_frozen_immutability(self):
        """Test that modifying any attribute of CraftSample raises FrozenInstanceError."""
        dt = datetime(2026, 10, 26, 12, 0, 0, tzinfo=timezone.utc)
        sample = CraftSample(
            craft_id="ASV-01",
            timestamp=dt,
            position=(14.29, -101.64),
            co2_1=415.5,
            co2_2=416.1,
            wind=Vector(u=3.2, v=-1.8),
        )

        with pytest.raises(FrozenInstanceError):
            sample.craft_id = "ASV-02"  # type: ignore[misc]

        with pytest.raises(FrozenInstanceError):
            sample.co2_1 = 420.0  # type: ignore[misc]

        with pytest.raises(FrozenInstanceError):
            sample.position = (10.0, -100.0)  # type: ignore[misc]

    def test_craft_sample_slots(self):
        """Test that CraftSample uses slots (has no __dict__) and forbids arbitrary new attributes."""
        dt = datetime(2026, 10, 26, 12, 0, 0, tzinfo=timezone.utc)
        sample = CraftSample(
            craft_id="ASV-01",
            timestamp=dt,
            position=(14.29, -101.64),
            co2_1=415.5,
            co2_2=416.1,
            wind=Vector(u=3.2, v=-1.8),
        )

        assert not hasattr(sample, "__dict__")
        assert set(sample.__slots__) == {
            "craft_id",
            "timestamp",
            "position",
            "co2_1",
            "co2_2",
            "wind",
        }

        with pytest.raises((FrozenInstanceError, AttributeError)):
            sample.arbitrary_attribute = "disallowed"  # type: ignore[attr-defined]

    def test_craft_sample_properties(self):
        """Test retained convenience properties on CraftSample."""
        dt = datetime(2026, 10, 26, 12, 0, 0, tzinfo=timezone.utc)
        sample = CraftSample(
            craft_id="ASV-01",
            timestamp=dt,
            position=(14.29, -101.64),
            co2_1=415.0,
            co2_2=417.0,
            wind=Vector(u=3.5, v=-2.5),
        )

        # Coordinate accessors
        assert sample.latitude == 14.29
        assert sample.longitude == -101.64

        # Mean CO2 accessor
        assert math.isclose(sample.mean_co2, 416.0, abs_tol=1e-9)

    def test_craft_sample_timestamp_coercions(self):
        """Test timezone handling for naive datetimes, timezone-aware datetimes, and ISO strings."""
        # 1. Naive datetime is interpreted as GMT/UTC
        naive_dt = datetime(2026, 10, 26, 15, 30, 0)
        sample_naive = CraftSample(
            craft_id="BUOY-01",
            timestamp=naive_dt,
            position=(12.0, -100.0),
            co2_1=412.0,
            co2_2=412.5,
            wind=Vector(1.0, 2.0),
        )
        assert sample_naive.timestamp.tzinfo == timezone.utc
        assert sample_naive.timestamp.hour == 15

        # 2. Aware datetime with non-UTC offset converted to UTC
        pst = timezone(timedelta(hours=-7))
        aware_dt = datetime(2026, 10, 26, 8, 30, 0, tzinfo=pst)
        sample_aware = CraftSample(
            craft_id="BUOY-01",
            timestamp=aware_dt,
            position=(12.0, -100.0),
            co2_1=412.0,
            co2_2=412.5,
            wind=Vector(1.0, 2.0),
        )
        assert sample_aware.timestamp.tzinfo == timezone.utc
        assert sample_aware.timestamp.hour == 15

        # 3. ISO format string parsed directly
        sample_iso = CraftSample(
            craft_id="BUOY-01",
            timestamp="2026-10-26T15:30:00Z",
            position=(12.0, -100.0),
            co2_1=412.0,
            co2_2=412.5,
            wind=Vector(1.0, 2.0),
        )
        assert sample_iso.timestamp.tzinfo == timezone.utc
        assert sample_iso.timestamp.hour == 15

        # 4. Invalid timestamp type
        with pytest.raises(TypeError, match="timestamp must be a datetime"):
            CraftSample(
                craft_id="BUOY-01",
                timestamp=123456789,  # type: ignore[arg-type]
                position=(12.0, -100.0),
                co2_1=412.0,
                co2_2=412.5,
                wind=Vector(1.0, 2.0),
            )

    def test_craft_sample_wind_coercions(self):
        """Test accepting wind as a Vector instance or 2-element sequence."""
        dt = datetime(2026, 10, 26, 12, 0, 0, tzinfo=timezone.utc)

        # Vector instance
        sample1 = CraftSample(
            craft_id="ASV-1",
            timestamp=dt,
            position=(13.0, -101.0),
            co2_1=410.0,
            co2_2=410.2,
            wind=Vector(2.0, -4.0),
        )
        assert isinstance(sample1.wind, Vector)
        assert sample1.wind.u == 2.0 and sample1.wind.v == -4.0

        # Tuple (u, v) sequence
        sample2 = CraftSample(
            craft_id="ASV-1",
            timestamp=dt,
            position=(13.0, -101.0),
            co2_1=410.0,
            co2_2=410.2,
            wind=(2.0, -4.0),  # type: ignore[arg-type]
        )
        assert isinstance(sample2.wind, Vector)
        assert sample2.wind == Vector(2.0, -4.0)

        # Invalid wind type
        with pytest.raises(TypeError, match="wind must be a Vector"):
            CraftSample(
                craft_id="ASV-1",
                timestamp=dt,
                position=(13.0, -101.0),
                co2_1=410.0,
                co2_2=410.2,
                wind=100.0,  # type: ignore[arg-type]
            )

    def test_craft_sample_position_validation(self):
        """Test position sanity checks and conversion."""
        dt = datetime(2026, 10, 26, 12, 0, 0, tzinfo=timezone.utc)

        # Latitude out of range
        with pytest.raises(ValueError, match="latitude 95.0° outside valid"):
            CraftSample(
                craft_id="ASV-1",
                timestamp=dt,
                position=(95.0, -100.0),
                co2_1=410.0,
                co2_2=410.0,
                wind=Vector(1.0, 1.0),
            )

        # Longitude out of range
        with pytest.raises(ValueError, match="longitude 400.0° outside valid"):
            CraftSample(
                craft_id="ASV-1",
                timestamp=dt,
                position=(10.0, 400.0),
                co2_1=410.0,
                co2_2=410.0,
                wind=Vector(1.0, 1.0),
            )

        # Non-numeric coordinate
        with pytest.raises(ValueError, match="position coordinates must be numeric"):
            CraftSample(
                craft_id="ASV-1",
                timestamp=dt,
                position=("not_a_num", -100.0),  # type: ignore[arg-type]
                co2_1=410.0,
                co2_2=410.0,
                wind=Vector(1.0, 1.0),
            )

        # NaN / Inf coordinate
        with pytest.raises(ValueError, match="finite real numbers"):
            CraftSample(
                craft_id="ASV-1",
                timestamp=dt,
                position=(float("nan"), -100.0),
                co2_1=410.0,
                co2_2=410.0,
                wind=Vector(1.0, 1.0),
            )

        # Non-pair position
        with pytest.raises(ValueError, match="must be a \\(latitude, longitude\\) pair"):
            CraftSample(
                craft_id="ASV-1",
                timestamp=dt,
                position=(10.0,),  # type: ignore[arg-type]
                co2_1=410.0,
                co2_2=410.0,
                wind=Vector(1.0, 1.0),
            )

    def test_craft_sample_co2_validation(self):
        """Test CO2 concentration parsing and validation."""
        dt = datetime(2026, 10, 26, 12, 0, 0, tzinfo=timezone.utc)

        # Integer or numeric string CO2 readings coerce to float
        sample = CraftSample(
            craft_id="ASV-1",
            timestamp=dt,
            position=(10.0, -100.0),
            co2_1=415,  # type: ignore[arg-type]
            co2_2="416.5",  # type: ignore[arg-type]
            wind=Vector(1.0, 1.0),
        )
        assert sample.co2_1 == 415.0
        assert sample.co2_2 == 416.5

        # NaN / inf CO2 readings rejected
        with pytest.raises(ValueError, match="finite real numbers"):
            CraftSample(
                craft_id="ASV-1",
                timestamp=dt,
                position=(10.0, -100.0),
                co2_1=float("nan"),
                co2_2=416.0,
                wind=Vector(1.0, 1.0),
            )

    def test_craft_sample_equality_and_hashing(self):
        """Test that identical CraftSample instances are equal and can be hashed in sets/dicts."""
        dt = datetime(2026, 10, 26, 12, 0, 0, tzinfo=timezone.utc)
        sample1 = CraftSample(
            craft_id="ASV-01",
            timestamp=dt,
            position=(14.29, -101.64),
            co2_1=415.0,
            co2_2=417.0,
            wind=Vector(u=3.5, v=-2.5),
        )
        sample2 = CraftSample(
            craft_id="ASV-01",
            timestamp=dt,
            position=(14.29, -101.64),
            co2_1=415.0,
            co2_2=417.0,
            wind=Vector(u=3.5, v=-2.5),
        )
        sample3 = CraftSample(
            craft_id="ASV-02",
            timestamp=dt,
            position=(14.29, -101.64),
            co2_1=415.0,
            co2_2=417.0,
            wind=Vector(u=3.5, v=-2.5),
        )

        assert sample1 == sample2
        assert sample1 != sample3
        assert hash(sample1) == hash(sample2)

        sample_set = {sample1, sample2, sample3}
        assert len(sample_set) == 2
        assert sample1 in sample_set
