"""Pytest test suite for WGS 84 ECEF and local ENU geodetic transformations."""

from __future__ import annotations

import math

import numpy as np
import pytest

from perimeter_flux.config import WGS84_E2, WGS84_FLATTENING, WGS84_SEMI_MAJOR_M
from perimeter_flux.geodesy import (
    ecef_origin_from_points,
    ecef_to_enu,
    ecef_to_latlon,
    enu_rotation_matrix,
    latlon_to_ecef,
    project_latlon_to_local_en,
)
from perimeter_flux.sample_data import ACAPULCO_EDDY_VERTICES

WGS84_SEMI_MINOR_M = WGS84_SEMI_MAJOR_M * (1.0 - WGS84_FLATTENING)


def _wgs84_meridional_radius_m(lat_rad: float) -> float:
    """Meridional radius of curvature M(phi) = a(1 - e^2) / (1 - e^2 sin^2 phi)^(3/2)."""
    denom = 1.0 - WGS84_E2 * math.sin(lat_rad) ** 2
    return WGS84_SEMI_MAJOR_M * (1.0 - WGS84_E2) / (denom**1.5)


def _wgs84_prime_vertical_radius_m(lat_rad: float) -> float:
    """Prime vertical radius of curvature nu(phi) = a / sqrt(1 - e^2 sin^2 phi)."""
    return WGS84_SEMI_MAJOR_M / math.sqrt(1.0 - WGS84_E2 * math.sin(lat_rad) ** 2)


class TestLatLonToECEF:
    """Exact WGS 84 forward geodetic-to-ECEF checks."""

    def test_equator_cardinal_meridians_and_poles(self):
        lats = [0.0, 0.0, 0.0, 0.0, 90.0, -90.0]
        lons = [0.0, 90.0, 180.0, -180.0, 0.0, 0.0]
        xyz = latlon_to_ecef(lats, lons)

        assert xyz.shape == (6, 3)
        assert xyz.dtype == np.float64

        expected = np.array(
            [
                [WGS84_SEMI_MAJOR_M, 0.0, 0.0],
                [0.0, WGS84_SEMI_MAJOR_M, 0.0],
                [-WGS84_SEMI_MAJOR_M, 0.0, 0.0],
                [-WGS84_SEMI_MAJOR_M, 0.0, 0.0],
                [0.0, 0.0, WGS84_SEMI_MINOR_M],
                [0.0, 0.0, -WGS84_SEMI_MINOR_M],
            ],
            dtype=np.float64,
        )
        np.testing.assert_allclose(xyz, expected, rtol=0.0, atol=1e-9)

    def test_mid_latitude_exact_analytical_value(self):
        lat_deg = 45.0
        lon_deg = -101.7
        phi = math.radians(lat_deg)
        lam = math.radians(lon_deg)
        nu = _wgs84_prime_vertical_radius_m(phi)

        expected = np.array(
            [
                [
                    nu * math.cos(phi) * math.cos(lam),
                    nu * math.cos(phi) * math.sin(lam),
                    nu * (1.0 - WGS84_E2) * math.sin(phi),
                ]
            ],
            dtype=np.float64,
        )
        xyz = latlon_to_ecef([lat_deg], [lon_deg])
        np.testing.assert_allclose(xyz, expected, rtol=1e-15, atol=1e-9)

    def test_points_lie_on_wgs84_ellipsoid(self):
        lats = np.linspace(-90.0, 90.0, 37)
        lons = np.linspace(-180.0, 180.0, 37)
        xyz = latlon_to_ecef(lats, lons)

        ellipsoid_eq = (
            (xyz[:, 0] ** 2 + xyz[:, 1] ** 2) / (WGS84_SEMI_MAJOR_M**2)
            + (xyz[:, 2] ** 2) / (WGS84_SEMI_MINOR_M**2)
        )
        np.testing.assert_allclose(ellipsoid_eq, 1.0, rtol=0.0, atol=1e-14)

    def test_empty_and_invalid_inputs(self):
        empty = latlon_to_ecef([], [])
        assert empty.shape == (0, 3)
        assert empty.dtype == np.float64

        with pytest.raises(ValueError, match="same 1D shape"):
            latlon_to_ecef([0.0, 10.0], [0.0])

        with pytest.raises(ValueError, match="1D"):
            latlon_to_ecef([[0.0, 10.0]], [[0.0, 10.0]])


class TestECEFToLatLon:
    """Inverse ECEF-to-geodetic precision across globe, poles, and interior centroids."""

    def test_round_trip_precision_across_globe(self):
        lat_grid = np.array(
            [-90.0, -89.999, -75.0, -45.0, -13.4, -0.001, 0.0, 0.001, 13.4, 45.0, 75.0, 89.999, 90.0]
        )
        lon_grid = np.array([-179.9, -120.5, -101.7, -0.5, 0.0, 0.5, 90.0, 179.9])
        lat_mesh, lon_mesh = np.meshgrid(lat_grid, lon_grid, indexing="ij")
        lats = lat_mesh.ravel()
        lons = lon_mesh.ravel()

        xyz = latlon_to_ecef(lats, lons)
        lat_out, lon_out = ecef_to_latlon(xyz)

        assert lat_out.shape == lats.shape
        assert lon_out.shape == lons.shape
        np.testing.assert_allclose(lat_out, lats, rtol=0.0, atol=1e-12)

        non_pole = np.abs(lats) < 90.0
        dlon = (lon_out[non_pole] - lons[non_pole] + 180.0) % 360.0 - 180.0
        np.testing.assert_allclose(dlon, 0.0, rtol=0.0, atol=1e-12)

    def test_exact_polar_xyz_vectors(self):
        poles = np.array(
            [
                [0.0, 0.0, WGS84_SEMI_MINOR_M],
                [0.0, 0.0, -WGS84_SEMI_MINOR_M],
            ],
            dtype=np.float64,
        )
        lat_out, lon_out = ecef_to_latlon(poles)
        np.testing.assert_allclose(lat_out, [90.0, -90.0], rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(lon_out, [0.0, 0.0], rtol=0.0, atol=1e-12)

    def test_interior_chord_centroid_recovers_geodetic_latitude(self):
        # Simulate a point ~3.14 km below the ellipsoid surface (200 km chord drop)
        lats = np.array([-60.0, -13.4, 0.0, 13.4, 45.0, 75.0])
        lons = np.array([-179.5, -101.7, 0.0, 101.7, 179.5, -45.0])
        phi = np.radians(lats)
        lam = np.radians(lons)
        nu = WGS84_SEMI_MAJOR_M / np.sqrt(1.0 - WGS84_E2 * np.sin(phi) ** 2)
        h = -3140.0
        xyz_interior = np.column_stack(
            (
                (nu + h) * np.cos(phi) * np.cos(lam),
                (nu + h) * np.cos(phi) * np.sin(lam),
                (nu * (1.0 - WGS84_E2) + h) * np.sin(phi),
            )
        )

        lat_out, lon_out = ecef_to_latlon(xyz_interior)
        np.testing.assert_allclose(lat_out, lats, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(lon_out, lons, rtol=0.0, atol=1e-12)

    def test_single_point_empty_and_invalid_shape(self):
        single = latlon_to_ecef([13.4], [-101.7])[0]
        lat_out, lon_out = ecef_to_latlon(single)
        assert lat_out.shape == (1,)
        assert lon_out.shape == (1,)
        assert math.isclose(lat_out[0], 13.4, abs_tol=1e-12)
        assert math.isclose(lon_out[0], -101.7, abs_tol=1e-12)

        empty_lat, empty_lon = ecef_to_latlon(np.empty((0, 3), dtype=np.float64))
        assert empty_lat.shape == (0,)
        assert empty_lon.shape == (0,)

        with pytest.raises(ValueError, match="shape"):
            ecef_to_latlon(np.zeros((4, 2)))


class TestENURotationAndTransform:
    """Verify ENU rotation matrix orthonormality, handedness, and local metric accuracy."""

    @pytest.mark.parametrize(
        ("lat_deg", "lon_deg"),
        [
            (0.0, 0.0),
            (13.4, -101.7),
            (-33.9, 151.2),
            (65.0, 179.9),
            (65.0, -179.9),
            (90.0, 0.0),
            (-90.0, 45.0),
        ],
    )
    def test_rotation_matrix_is_proper_orthonormal(self, lat_deg: float, lon_deg: float):
        r = enu_rotation_matrix(math.radians(lat_deg), math.radians(lon_deg))
        assert r.shape == (3, 3)
        np.testing.assert_allclose(r.T @ r, np.eye(3), rtol=0.0, atol=1e-15)
        np.testing.assert_allclose(r @ r.T, np.eye(3), rtol=0.0, atol=1e-15)
        assert math.isclose(float(np.linalg.det(r)), 1.0, abs_tol=1e-15)

    def test_local_cardinal_displacements_and_curvature_drop(self):
        lat0_deg, lon0_deg = 13.4, -101.7
        lat0_rad, lon0_rad = math.radians(lat0_deg), math.radians(lon0_deg)
        origin_xyz = latlon_to_ecef([lat0_deg], [lon0_deg])[0]

        ddeg = 0.01  # ~1.1 km step
        drad = math.radians(ddeg)
        pts_xyz = latlon_to_ecef(
            [lat0_deg, lat0_deg + ddeg, lat0_deg - ddeg, lat0_deg, lat0_deg],
            [lon0_deg, lon0_deg, lon0_deg, lon0_deg + ddeg, lon0_deg - ddeg],
        )
        enu = ecef_to_enu(pts_xyz, lat0_rad, lon0_rad, origin_xyz)
        assert enu.shape == (5, 3)

        # Origin maps to (0, 0, 0)
        np.testing.assert_allclose(enu[0], [0.0, 0.0, 0.0], atol=1e-12)

        expected_north_m = _wgs84_meridional_radius_m(lat0_rad) * drad
        expected_east_m = _wgs84_prime_vertical_radius_m(lat0_rad) * math.cos(lat0_rad) * drad

        # North & South displacements
        assert math.isclose(enu[1, 0], 0.0, abs_tol=1e-6)
        assert math.isclose(enu[1, 1], expected_north_m, rel_tol=1e-5)
        assert enu[1, 2] < 0.0  # Earth curves downward away from tangent plane
        assert math.isclose(enu[2, 1], -expected_north_m, rel_tol=1e-5)

        # East & West displacements
        assert math.isclose(enu[3, 0], expected_east_m, rel_tol=1e-5)
        assert abs(enu[3, 1]) < 0.1  # Tiny second-order meridian convergence term
        assert enu[3, 2] < 0.0
        assert math.isclose(enu[4, 0], -expected_east_m, rel_tol=1e-5)


class TestECEFOriginFromPoints:
    """Verify IDL-safe 3D centroid origin projected onto the sea-level ellipsoid."""

    def test_origin_projects_onto_sea_level_ellipsoid(self):
        lats = [p[0] for p in ACAPULCO_EDDY_VERTICES]
        lons = [p[1] for p in ACAPULCO_EDDY_VERTICES]
        xyz = latlon_to_ecef(lats, lons)

        origin_lat_rad, origin_lon_rad, origin_xyz = ecef_origin_from_points(xyz)
        assert origin_xyz.shape == (3,)
        assert 13.0 < math.degrees(origin_lat_rad) < 13.8
        assert -102.0 < math.degrees(origin_lon_rad) < -101.4

        # Projected origin must lie exactly on the sea-level WGS 84 ellipsoid
        ellipsoid_eq = (
            (origin_xyz[0] ** 2 + origin_xyz[1] ** 2) / (WGS84_SEMI_MAJOR_M**2)
            + (origin_xyz[2] ** 2) / (WGS84_SEMI_MINOR_M**2)
        )
        assert math.isclose(float(ellipsoid_eq), 1.0, abs_tol=1e-14)

    def test_empty_or_antipodal_points_raise_value_error(self):
        with pytest.raises(ValueError, match="at least one point"):
            ecef_origin_from_points(np.empty((0, 3), dtype=np.float64))

        antipodal = latlon_to_ecef([0.0, 0.0], [0.0, 180.0])
        with pytest.raises(ValueError, match="geocenter"):
            ecef_origin_from_points(antipodal)


class TestProjectLatLonToLocalEN:
    """Acapulco eddy distance accuracy and International Date Line continuity."""

    def test_acapulco_eddy_orientation_and_metric_distances(self):
        lats = np.array([p[0] for p in ACAPULCO_EDDY_VERTICES], dtype=np.float64)
        lons = np.array([p[1] for p in ACAPULCO_EDDY_VERTICES], dtype=np.float64)

        xy_m, origin_lat_deg, origin_lon_deg = project_latlon_to_local_en(lats, lons)
        assert xy_m.shape == (6, 2)
        assert xy_m.dtype == np.float64

        # Cardinal extremes match Acapulco eddy vertices
        assert int(np.argmax(xy_m[:, 1])) == 0  # 14.29 N (furthest North)
        assert int(np.argmin(xy_m[:, 0])) == 1  # -102.88 E (furthest West)
        assert int(np.argmin(xy_m[:, 1])) == 3  # 12.45 N (furthest South)
        assert int(np.argmax(xy_m[:, 0])) == 4  # -100.53 E (furthest East)

        # North-South span between vertex 0 (14.29) and vertex 3 (12.45)
        origin_lat_rad = math.radians(origin_lat_deg)
        expected_ns_m = _wgs84_meridional_radius_m(origin_lat_rad) * math.radians(14.29 - 12.45)
        actual_ns_m = float(xy_m[0, 1] - xy_m[3, 1])
        assert math.isclose(actual_ns_m, expected_ns_m, rel_tol=5e-4)

        # East-West span between vertex 1 (-102.88) and vertex 4 (-100.53)
        mid_ew_lat_rad = math.radians((13.87 + 12.97) / 2.0)
        expected_ew_m = (
            _wgs84_prime_vertical_radius_m(mid_ew_lat_rad)
            * math.cos(mid_ew_lat_rad)
            * math.radians(-100.53 - (-102.88))
        )
        actual_ew_m = float(xy_m[4, 0] - xy_m[1, 0])
        assert math.isclose(actual_ew_m, expected_ew_m, rel_tol=5e-4)

        # Horizontal tangent-plane chord vs 3D ECEF chord distortion across ~200 km eddy < 0.02%
        xyz = latlon_to_ecef(lats, lons)
        dist_3d = float(np.linalg.norm(xyz[4] - xyz[1]))
        dist_2d = float(np.linalg.norm(xy_m[4] - xy_m[1]))
        assert abs(dist_2d - dist_3d) / dist_3d < 2e-4

        assert 13.0 < origin_lat_deg < 13.8
        assert -102.0 < origin_lon_deg < -101.4

    def test_international_date_line_continuity_and_shift_invariance(self):
        # Rotate Acapulco eddy around Earth's polar axis to straddle +/-180 deg
        lats = np.array([p[0] for p in ACAPULCO_EDDY_VERTICES], dtype=np.float64)
        lons_acapulco = np.array([p[1] for p in ACAPULCO_EDDY_VERTICES], dtype=np.float64)

        # Shift so mean longitude sits directly on 180 deg, wrapped into [-180, 180]
        mean_lon = float(np.mean(lons_acapulco))
        lons_idl = ((lons_acapulco - mean_lon + 180.0 + 180.0) % 360.0) - 180.0
        # Confirm polygon genuinely straddles both sides of +/-180 deg
        assert np.any(lons_idl > 178.0)
        assert np.any(lons_idl < -178.0)

        xy_acapulco, lat0_a, _ = project_latlon_to_local_en(lats, lons_acapulco)
        xy_idl, lat0_idl, lon0_idl = project_latlon_to_local_en(lats, lons_idl)

        # Rotational symmetry of WGS 84 guarantees identical local EN geometry (zero seam jump)
        np.testing.assert_allclose(xy_idl, xy_acapulco, rtol=0.0, atol=1e-7)
        assert math.isclose(lat0_idl, lat0_a, abs_tol=1e-11)
        assert abs(abs(lon0_idl) - 180.0) < 0.1

    def test_idl_box_matches_prime_meridian_box_and_0_360_convention(self):
        lats = np.array([13.0, 13.0, 14.0, 14.0], dtype=np.float64)
        lons_prime = np.array([-0.5, 0.5, 0.5, -0.5], dtype=np.float64)
        lons_idl_signed = np.array([179.5, -179.5, -179.5, 179.5], dtype=np.float64)
        lons_idl_360 = np.array([179.5, 180.5, 180.5, 179.5], dtype=np.float64)

        xy_prime, _, _ = project_latlon_to_local_en(lats, lons_prime)
        xy_idl_signed, _, _ = project_latlon_to_local_en(lats, lons_idl_signed)
        xy_idl_360, _, _ = project_latlon_to_local_en(lats, lons_idl_360)

        np.testing.assert_allclose(xy_idl_signed, xy_prime, rtol=0.0, atol=1e-7)
        np.testing.assert_allclose(xy_idl_360, xy_prime, rtol=0.0, atol=1e-7)
