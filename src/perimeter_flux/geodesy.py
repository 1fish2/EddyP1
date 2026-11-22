"""Pure NumPy WGS 84 geodetic, ECEF, and local ENU coordinate transformations.

Mirrors Section 3 of ``doc/acpf_math.md``:

Symbol Glossary (Section 2.2)
-----------------------------
- ``a_wgs`` (``WGS84_SEMI_MAJOR_M``): 6,378,137.0 m exact WGS 84 equatorial radius.
- ``f_wgs`` (``WGS84_FLATTENING``): 1 / 298.257223563 exact WGS 84 flattening.
- ``e_wgs^2`` (``WGS84_E2``): 2 f_wgs - f_wgs^2 first eccentricity squared.
- ``phi, lam``: geodetic latitude (North+) and longitude (East+) [rad] (API inputs in deg).
- ``nu(phi)``: prime vertical radius of curvature ``a_wgs / sqrt(1 - e_wgs^2 sin^2 phi)`` [m].
- ``X = (X, Y, Z)^T``: 3D sea-level (h_ellip = 0) WGS 84 ECEF coordinates [m].
- ``R_enu(phi_0, lam_0)``: 3x3 orthonormal ECEF-to-local-ENU rotation matrix.
- ``p_i = (x_i, y_i)^T``: 2D local metric East-North coordinates [m].
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from perimeter_flux.config import WGS84_E2, WGS84_FLATTENING, WGS84_SEMI_MAJOR_M

_WGS84_SEMI_MINOR_M: float = WGS84_SEMI_MAJOR_M * (1.0 - WGS84_FLATTENING)
_WGS84_EP2: float = WGS84_E2 / (1.0 - WGS84_E2)


def _as_ecef_2d(xyz_m: ArrayLike) -> NDArray[np.float64]:
    """Validate and coerce an ECEF coordinate array to shape (N, 3) float64."""
    arr = np.asarray(xyz_m, dtype=np.float64)
    if arr.ndim == 1:
        if arr.shape != (3,):
            raise ValueError(f"1D ECEF vector must have shape (3,), got {arr.shape}.")
        return arr.reshape(1, 3)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"ECEF array must have shape (N, 3) or (3,), got {arr.shape}.")
    return arr


def latlon_to_ecef(lat_deg: ArrayLike, lon_deg: ArrayLike) -> NDArray[np.float64]:
    """Convert sea-level geodetic (lat, lon) in degrees to 3D WGS 84 ECEF meters.

    Equations (doc/acpf_math.md Section 3.1, h_ellip = 0)::

        nu(phi) = a_wgs / sqrt(1 - e_wgs^2 * sin^2(phi))
        X = nu(phi) * cos(phi) * cos(lam)
        Y = nu(phi) * cos(phi) * sin(lam)
        Z = nu(phi) * (1 - e_wgs^2) * sin(phi)

    Returns an array of shape ``(N, 3)`` in meters.
    """
    lat_arr = np.atleast_1d(np.asarray(lat_deg, dtype=np.float64))
    lon_arr = np.atleast_1d(np.asarray(lon_deg, dtype=np.float64))
    if lat_arr.ndim != 1 or lon_arr.ndim != 1:
        raise ValueError(
            f"lat_deg and lon_deg must be 1D sequences, got shapes {lat_arr.shape} and {lon_arr.shape}."
        )
    if lat_arr.shape != lon_arr.shape:
        raise ValueError(
            f"lat_deg and lon_deg must have the same 1D shape, got {lat_arr.shape} and {lon_arr.shape}."
        )
    if lat_arr.size == 0:
        return np.empty((0, 3), dtype=np.float64)

    phi = np.radians(lat_arr)
    lam = np.radians(lon_arr)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    nu = WGS84_SEMI_MAJOR_M / np.sqrt(1.0 - WGS84_E2 * sin_phi * sin_phi)

    x = nu * cos_phi * np.cos(lam)
    y = nu * cos_phi * np.sin(lam)
    z = nu * (1.0 - WGS84_E2) * sin_phi
    return np.column_stack((x, y, z))


def ecef_to_latlon(xyz_m: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Convert 3D WGS 84 ECEF meters to geodetic (lat_deg, lon_deg).

    Uses Bowring's closed-form ellipsoidal starter followed by two exact
    ellipsoidal fixed-point refinements (doc/acpf_math.md Section 3.2),
    achieving < 1e-12 deg accuracy across sea-level and shallow-interior
    centroid points (including poles and the International Date Line)::

        p     = sqrt(X^2 + Y^2)
        theta = atan2(a_wgs * Z, b_wgs * p)
        phi_0 = atan2(Z + e'^2 * b_wgs * sin^3(theta), p - e_wgs^2 * a_wgs * cos^3(theta))
        phi   <- atan2(Z + e_wgs^2 * nu(phi) * sin(phi), p)   (2 iterations)
        lam   = atan2(Y, X)
    """
    xyz = _as_ecef_2d(xyz_m)
    if xyz.shape[0] == 0:
        empty = np.empty((0,), dtype=np.float64)
        return empty, empty.copy()

    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    p = np.hypot(x, y)

    theta = np.arctan2(WGS84_SEMI_MAJOR_M * z, _WGS84_SEMI_MINOR_M * p)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    phi = np.arctan2(
        z + _WGS84_EP2 * _WGS84_SEMI_MINOR_M * sin_theta**3,
        p - WGS84_E2 * WGS84_SEMI_MAJOR_M * cos_theta**3,
    )

    for _ in range(2):
        sin_phi = np.sin(phi)
        nu = WGS84_SEMI_MAJOR_M / np.sqrt(1.0 - WGS84_E2 * sin_phi * sin_phi)
        phi = np.arctan2(z + WGS84_E2 * nu * sin_phi, p)

    lam = np.arctan2(y, x)
    return np.degrees(phi), np.degrees(lam)


def enu_rotation_matrix(origin_lat_rad: float, origin_lon_rad: float) -> NDArray[np.float64]:
    """Return the 3x3 orthonormal rotation matrix R_enu(phi_0, lam_0).

    Equation (doc/acpf_math.md Section 3.3)::

        R_enu = [[-sin(lam_0),            cos(lam_0),           0         ],
                 [-sin(phi_0)*cos(lam_0), -sin(phi_0)*sin(lam_0), cos(phi_0)],
                 [ cos(phi_0)*cos(lam_0),  cos(phi_0)*sin(lam_0), sin(phi_0)]]
    """
    sin_phi = math.sin(origin_lat_rad)
    cos_phi = math.cos(origin_lat_rad)
    sin_lam = math.sin(origin_lon_rad)
    cos_lam = math.cos(origin_lon_rad)
    return np.array(
        [
            [-sin_lam, cos_lam, 0.0],
            [-sin_phi * cos_lam, -sin_phi * sin_lam, cos_phi],
            [cos_phi * cos_lam, cos_phi * sin_lam, sin_phi],
        ],
        dtype=np.float64,
    )


def ecef_origin_from_points(xyz_m: ArrayLike) -> tuple[float, float, NDArray[np.float64]]:
    """Compute an IDL-safe local tangent-plane origin from 3D ECEF points.

    Forms the 3D arithmetic centroid ``X_bar = mean(X_i)``, converts it to
    geodetic ``(phi_0, lam_0)`` in radians, and projects the origin onto the
    sea-level WGS 84 ellipsoid ``X_0 = X(phi_0, lam_0)`` (doc/acpf_math.md Section 3.2).

    Returns ``(origin_lat_rad, origin_lon_rad, origin_xyz_m)``.
    """
    xyz = _as_ecef_2d(xyz_m)
    if xyz.shape[0] == 0:
        raise ValueError("ecef_origin_from_points requires at least one point.")

    mean_xyz = np.mean(xyz, axis=0)
    if float(np.linalg.norm(mean_xyz)) < 1.0:
        raise ValueError(
            "3D ECEF centroid lies near the Earth geocenter (e.g. antipodal points); "
            "cannot define a local tangent-plane origin."
        )

    origin_lat_deg_arr, origin_lon_deg_arr = ecef_to_latlon(mean_xyz)
    origin_lat_deg = float(origin_lat_deg_arr[0])
    origin_lon_deg = float(origin_lon_deg_arr[0])
    origin_xyz_m = latlon_to_ecef([origin_lat_deg], [origin_lon_deg])[0]
    return math.radians(origin_lat_deg), math.radians(origin_lon_deg), origin_xyz_m


def ecef_to_enu(
    xyz_m: ArrayLike,
    origin_lat_rad: float,
    origin_lon_rad: float,
    origin_xyz_m: ArrayLike,
) -> NDArray[np.float64]:
    """Rotate ECEF displacements into the local East-North-Up (ENU) frame in meters.

    Equation (doc/acpf_math.md Section 3.3)::

        [x_i, y_i, z_enu_i]^T = R_enu(phi_0, lam_0) @ (X_i - X_0)

    Returns an array of shape ``(N, 3)`` with columns ``(East, North, Up)`` in meters.
    """
    xyz = _as_ecef_2d(xyz_m)
    origin = np.asarray(origin_xyz_m, dtype=np.float64)
    if origin.shape != (3,):
        raise ValueError(f"origin_xyz_m must have shape (3,), got {origin.shape}.")
    if xyz.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float64)

    rotation = enu_rotation_matrix(origin_lat_rad, origin_lon_rad)
    return (xyz - origin) @ rotation.T


def project_latlon_to_local_en(
    lat_deg: ArrayLike,
    lon_deg: ArrayLike,
) -> tuple[NDArray[np.float64], float, float]:
    """Project geodetic (lat, lon) points to 2D local East-North meters.

    Computes the IDL-safe sea-level WGS 84 origin from the 3D ECEF centroid and
    returns ``(xy_m, origin_lat_deg, origin_lon_deg)``, where ``xy_m`` is a
    contiguous ``(N, 2)`` float64 array of ``(East, North)`` meters.

    Accuracy notes:
        - Tangent-plane distortion: Over a ~200 km eddy, horizontal chord-vs-arc
          distortion is < 0.02% (~8 m over 200 km) and fractional area distortion
          scales roughly as (L / (2 R))^2, well below wind and MBLH uncertainties.
          If deployments ever grow toward ~1000 km scale, switch polygon area and
          edge lengths to an ellipsoidal geodesic solver (e.g., Karney / GeographicLib).
        - Inter-period ASV drift speed (Stage 1): Compute displacement either from
          consecutive ECEF positions rotated into a shared midpoint ENU frame via
          ``ecef_to_enu`` (which avoids origin-shift artifacts and gives (u, v) drift
          components) or via the 3D ECEF chord norm ||X_k - X_{k-1}|| / dt (equivalent
          scalar speed to < 1e-6 for hourly ASV motion).
    """
    xyz_m = latlon_to_ecef(lat_deg, lon_deg)
    origin_lat_rad, origin_lon_rad, origin_xyz_m = ecef_origin_from_points(xyz_m)
    enu_m = ecef_to_enu(xyz_m, origin_lat_rad, origin_lon_rad, origin_xyz_m)
    return (
        np.ascontiguousarray(enu_m[:, :2]),
        math.degrees(origin_lat_rad),
        math.degrees(origin_lon_rad),
    )
