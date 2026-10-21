"""Ocean eddy geometric computations and polygon validation utility for perimeter flux.

Provides geometric validation and counterclockwise polygon sorting for ocean eddies,
such as buoy arrays, sensor networks, and vortex boundaries.
Part of the Atmospheric CO₂ Perimeter Flux (ACPF / perimeter_flux) package.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

# Static copy of sample buoy locations around an eddy near Acapulco
# (from test_data/6_points_around_acapulco_eddy.csv)
ACAPULCO_EDDY_SAMPLE_VERTICES: tuple[tuple[float, float], ...] = (
    (14.29, -101.64),
    (13.87, -102.88),
    (12.93, -102.82),
    (12.45, -101.67),
    (12.97, -100.53),
    (13.79, -100.59),
)

# Mean Earth radius in kilometers (WGS84 spherical approximation)
EARTH_RADIUS_KM = 6371.0088
KM_PER_DEG_LAT = 111.139


class NormalVector(NamedTuple):
    """2D unit normal vector (u, v) where u is Eastward and v is Northward."""

    u: float  # Eastward / zonal component (dx)
    v: float  # Northward / meridional component (dy)

    @property
    def x(self) -> float:
        """Alias for Eastward (u) component."""
        return self.u

    @property
    def y(self) -> float:
        """Alias for Northward (v) component."""
        return self.v

    @property
    def east(self) -> float:
        """Eastward component."""
        return self.u

    @property
    def north(self) -> float:
        """Northward component."""
        return self.v


class SensorEdge:
    """An edge between two sensor vertices along the perimeter.

    Knows its two vertices (v1, v2) and provides an outward-pointing unit normal vector.
    """

    def __init__(
        self,
        v1: tuple[float, float],
        v2: tuple[float, float],
    ) -> None:
        self._v1: tuple[float, float] = (float(v1[0]), float(v1[1]))
        self._v2: tuple[float, float] = (float(v2[0]), float(v2[1]))

    @property
    def v1(self) -> tuple[float, float]:
        """First vertex (start of edge) as (latitude, longitude)."""
        return self._v1

    @property
    def v2(self) -> tuple[float, float]:
        """Second vertex (end of edge) as (latitude, longitude)."""
        return self._v2

    @property
    def vertices(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Pair of vertices (v1, v2) forming the edge."""
        return self._v1, self._v2

    @property
    def midpoint(self) -> tuple[float, float]:
        """Midpoint coordinate (latitude, longitude) of the edge."""
        return (self._v1[0] + self._v2[0]) / 2.0, (self._v1[1] + self._v2[1]) / 2.0

    def length_km(self) -> float:
        """Approximate physical length of the edge in kilometers."""
        mid_lat = (self._v1[0] + self._v2[0]) / 2.0
        dx_km = (self._v2[1] - self._v1[1]) * math.cos(math.radians(mid_lat)) * KM_PER_DEG_LAT
        dy_km = (self._v2[0] - self._v1[0]) * KM_PER_DEG_LAT
        return math.hypot(dx_km, dy_km)

    def normal(self, metric: bool = True) -> NormalVector:
        """Return the outward-pointing 2D unit vector normal to this edge.

        For a counterclockwise-traversed perimeter edge from v1 to v2,
        the outward normal points to the right of the tangent vector.

        Args:
            metric: If True (default), accounts for the convergence of meridians
                on a spherical Earth by scaling longitude delta by cos(mean_latitude).
                If False, computes the normal in unprojected (degree) Cartesian space.

        Returns:
            NormalVector(u, v) with magnitude 1.0, where u is Eastward and v is Northward.

        Raises:
            ValueError: If the edge has zero length (v1 == v2).
        """
        mid_lat = (self._v1[0] + self._v2[0]) / 2.0
        if metric:
            dx = (self._v2[1] - self._v1[1]) * math.cos(math.radians(mid_lat))
        else:
            dx = self._v2[1] - self._v1[1]
        dy = self._v2[0] - self._v1[0]

        # Right-turn normal in (x, y): (dy, -dx)
        nx = dy
        ny = -dx
        length = math.hypot(nx, ny)
        if length < 1e-15:
            raise ValueError("Cannot compute normal of a zero-length edge.")

        return NormalVector(u=nx / length, v=ny / length)

    def __repr__(self) -> str:
        return f"SensorEdge(v1={self._v1}, v2={self._v2})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SensorEdge):
            return NotImplemented
        return self._v1 == other._v1 and self._v2 == other._v2


class SensorPerimeter:
    """The polygon perimeter for a set of ocean sensor buoys/ASVs."""

    def __init__(
        self,
        vertices: Sequence[tuple[float, float]],
        start: str | tuple[float, float] = "first",
    ) -> None:
        """Initialize a SensorPerimeter instance with a list of (latitude, longitude) vertices.

        Args:
            vertices: Sequence of (lat, lon) coordinates.
            start: Reference starting vertex convention ("first", "min_lat", "min_lon",
                or a specific (lat, lon) coordinate).
        """
        validated = self.validate_vertices(vertices)
        sorted_pts = self.sort_vertices(validated, start=start, enforce_convex=True)

        # Maintain immutable tuple of immutable vertex tuples
        self._vertices: tuple[tuple[float, float], ...] = tuple(
            (float(lat), float(lon)) for lat, lon in sorted_pts
        )

        lats = [lat for lat, _ in self._vertices]
        lons = [lon for _, lon in self._vertices]
        self._lat_span = max(lats) - min(lats)
        self._lon_span = max(lons) - min(lons)
        self._centroid = self.centroid(self._vertices)

    @property
    def vertices(self) -> tuple[tuple[float, float], ...]:
        """Return the polygon vertices as an immutable tuple of (lat, lon) tuples."""
        return self._vertices

    @property
    def edges(self) -> tuple[SensorEdge, ...]:
        """Return the sequence of polygon edges, each an instance of SensorEdge."""
        n = len(self._vertices)
        return tuple(
            SensorEdge(self._vertices[i], self._vertices[(i + 1) % n])
            for i in range(n)
        )

    @property
    def centroid_lat_lon(self) -> tuple[float, float]:
        """Return the centroid coordinate (latitude, longitude)."""
        return self._centroid

    @property
    def lat_span(self) -> float:
        """Return the latitude span in degrees."""
        return self._lat_span

    @property
    def lon_span(self) -> float:
        """Return the longitude span in degrees."""
        return self._lon_span

    @property
    def area_deg2(self) -> float:
        """Return the planar polygon area in square degrees."""
        return self.polygon_area_deg2(self._vertices)

    @property
    def area_km2(self) -> float:
        """Return the approximate physical area in square kilometers."""
        return self.polygon_area_km2(self._vertices)

    @property
    def perimeter_km(self) -> float:
        """Return the total perimeter length in kilometers."""
        return sum(edge.length_km() for edge in self.edges)

    @staticmethod
    def _spans_international_date_line(lons: Sequence[float]) -> bool:
        """Check whether longitudes touch or span across the International Date Line (180° meridian)."""
        # 1. Any point directly on ±180°
        for lon in lons:
            if abs(abs(lon) - 180.0) < 1e-9:
                return True

        # 2. Coordinates given in [-180, 180] range crossing antimeridian:
        # If there are positive longitudes and negative longitudes, and their
        # distance across ±180° is shorter than across 0° (or <= 180°):
        pos_lons = [lon for lon in lons if lon > 0]
        neg_lons = [lon for lon in lons if lon < 0]
        if pos_lons and neg_lons:
            # Circular span across ±180°:
            # (180 - min_pos) + (max_neg - (-180)) = 360 - (min_pos - max_neg)
            span_across_180 = (180.0 - min(pos_lons)) + (max(neg_lons) - (-180.0))
            if span_across_180 <= 180.0:
                return True

        # 3. Coordinates given in [0, 360] range crossing 180°:
        min_lon = min(lons)
        max_lon = max(lons)
        if min_lon < 180.0 < max_lon and (max_lon - min_lon) <= 180.0:
            return True

        return False

    @staticmethod
    def _spans_pole(vertices: Sequence[tuple[float, float]]) -> bool:
        """Check whether vertices reach, cross, or enclose either the North or South Pole."""
        # 1. Reaching or exceeding ±90° latitude
        if any(abs(lat) >= 90.0 for lat, _ in vertices):
            return True

        # 2. Enclosing a pole (winding number around pole != 0)
        # Sum of signed delta longitude across vertices:
        total_dlon = 0.0
        n = len(vertices)
        for i in range(n):
            lon1 = vertices[i][1]
            lon2 = vertices[(i + 1) % n][1]
            dlon = (lon2 - lon1 + 180.0) % 360.0 - 180.0
            total_dlon += dlon

        if abs(total_dlon) > 180.0:
            return True

        return False

    @classmethod
    def validate_vertices(
        cls,
        vertices: Sequence[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        """Perform sanity checks on a sequence of (latitude, longitude) vertices.

        Sanity checks performed:
          1. At least 3 vertices are provided.
          2. All vertices are valid numerical pairs without NaNs or Infinities.
          3. Latitudes are within the valid [-90, 90] range.
          4. Vertices do not reach or span a pole.
          5. Vertices do not span or touch the International Date Line (180° meridian).
          6. Vertices do not span more than 10 degrees in latitude.
          7. Vertices do not span more than 10 degrees in longitude.
          8. Vertices are not degenerate (all coincident or collinear).

        Returns:
            A sanitized list of (lat, lon) float tuples.

        Raises:
            ValueError: If any sanity check fails.
        """
        if not isinstance(vertices, Sequence) or len(vertices) < 3:
            actual_type = type(vertices).__name__
            count = len(vertices) if isinstance(vertices, Sequence) else actual_type
            raise ValueError(f"A polygon requires at least 3 vertices, got {count}.")

        clean_pts: list[tuple[float, float]] = []
        for idx, pt in enumerate(vertices):
            if not (isinstance(pt, (tuple, list)) and len(pt) == 2):
                raise ValueError(
                    f"Vertex at index {idx} must be a (latitude, longitude) pair, got: {pt!r}"
                )
            try:
                lat = float(pt[0])
                lon = float(pt[1])
            except (ValueError, TypeError) as err:
                raise ValueError(
                    f"Vertex at index {idx} contains non-numeric coordinates: {pt!r}"
                ) from err

            if math.isnan(lat) or math.isnan(lon) or math.isinf(lat) or math.isinf(lon):
                raise ValueError(
                    f"Vertex at index {idx} contains NaN or Infinite coordinate: ({lat}, {lon})"
                )

            if lat < -90.0 or lat > 90.0:
                raise ValueError(
                    f"Vertex at index {idx} has latitude {lat}° outside valid [-90, 90] range."
                )

            # Accept longitudes in standard [-180, 180] or [0, 360]
            if lon < -180.0 or lon > 360.0:
                raise ValueError(
                    f"Vertex at index {idx} has longitude {lon}° outside valid [-180, 360] range."
                )

            clean_pts.append((lat, lon))

        # Check unique vertices
        unique_pts = set(clean_pts)
        if len(unique_pts) < 3:
            raise ValueError(
                f"Polygon has only {len(unique_pts)} unique vertices; at least 3 distinct vertices are required."
            )

        # 1. Pole sanity check
        if cls._spans_pole(clean_pts):
            raise ValueError("Vertices reach, span, or enclose a pole (latitude ±90°).")

        # 2. International Date Line sanity check
        lons = [lon for _, lon in clean_pts]
        if cls._spans_international_date_line(lons):
            raise ValueError(
                "Vertices span or touch the International Date Line (180° meridian)."
            )

        # 3. Maximum latitude span check (<= 10 degrees)
        lats = [lat for lat, _ in clean_pts]
        lat_span = max(lats) - min(lats)
        if lat_span > 10.0:
            raise ValueError(
                f"Vertices span {lat_span:.4f}° latitude, exceeding the maximum allowed limit of 10 degrees."
            )

        # 4. Maximum longitude span check (<= 10 degrees)
        lon_span = max(lons) - min(lons)
        if lon_span > 10.0:
            raise ValueError(
                f"Vertices span {lon_span:.4f}° longitude, exceeding the maximum allowed limit of 10 degrees."
            )

        # 5. Non-collinearity check (area must be non-zero)
        area = cls.polygon_area_deg2(clean_pts)
        if abs(area) < 1e-12:
            raise ValueError("Vertices are collinear or have zero area; cannot form a polygon.")

        return clean_pts

    @staticmethod
    def centroid(vertices: Sequence[tuple[float, float]]) -> tuple[float, float]:
        """Compute the centroid (mean latitude, mean longitude) of vertices."""
        n = len(vertices)
        if n == 0:
            raise ValueError("Cannot compute centroid of an empty vertex list.")
        c_lat = sum(p[0] for p in vertices) / n
        c_lon = sum(p[1] for p in vertices) / n
        return c_lat, c_lon

    @classmethod
    def polygon_area_deg2(cls, vertices: Sequence[tuple[float, float]]) -> float:
        """Calculate signed planar polygon area in square degrees using the Shoelace formula.

        x = longitude (East), y = latitude (North).
        Returns positive value for counterclockwise ordering, negative for clockwise.
        """
        n = len(vertices)
        if n < 3:
            return 0.0
        area2 = 0.0
        for i in range(n):
            lat1, lon1 = vertices[i]
            lat2, lon2 = vertices[(i + 1) % n]
            # (x1 * y2 - x2 * y1) where x = lon, y = lat
            area2 += lon1 * lat2 - lon2 * lat1
        return area2 / 2.0

    @classmethod
    def polygon_area_km2(cls, vertices: Sequence[tuple[float, float]]) -> float:
        """Calculate approximate physical area in square kilometers using equirectangular projection."""
        area_deg = abs(cls.polygon_area_deg2(vertices))
        c_lat, _ = cls.centroid(vertices)
        km_per_deg_lon = KM_PER_DEG_LAT * math.cos(math.radians(c_lat))
        return area_deg * KM_PER_DEG_LAT * km_per_deg_lon

    @classmethod
    def is_counterclockwise(cls, vertices: Sequence[tuple[float, float]]) -> bool:
        """Return True if vertices are oriented in counterclockwise order (x=lon, y=lat)."""
        return cls.polygon_area_deg2(vertices) > 1e-12

    @staticmethod
    def _cross_product_turn(
        p0: tuple[float, float],
        p1: tuple[float, float],
        p2: tuple[float, float],
    ) -> float:
        """2D cross product of vectors (p1 - p0) and (p2 - p1) with x=lon, y=lat.

        Positive indicates a left turn (counterclockwise).
        """
        dx1 = p1[1] - p0[1]
        dy1 = p1[0] - p0[0]
        dx2 = p2[1] - p1[1]
        dy2 = p2[0] - p1[0]
        return dx1 * dy2 - dy1 * dx2

    @classmethod
    def is_convex(
        cls,
        vertices: Sequence[tuple[float, float]],
        ccw_only: bool = True,
    ) -> bool:
        """Check if vertices form a strictly convex polygon.

        Args:
            vertices: Sequence of (lat, lon) pairs.
            ccw_only: If True, only accepts counterclockwise strictly convex polygons.
        """
        n = len(vertices)
        if n < 3:
            return False

        cross_products = []
        for i in range(n):
            cp = cls._cross_product_turn(
                vertices[i],
                vertices[(i + 1) % n],
                vertices[(i + 2) % n],
            )
            if abs(cp) <= 1e-12:
                # Collinear consecutive edges
                return False
            cross_products.append(cp)

        if ccw_only:
            return all(cp > 0 for cp in cross_products)

        all_pos = all(cp > 0 for cp in cross_products)
        all_neg = all(cp < 0 for cp in cross_products)
        return all_pos or all_neg

    @classmethod
    def sort_vertices(
        cls,
        vertices: Sequence[tuple[float, float]],
        start: str | tuple[float, float] = "first",
        enforce_convex: bool = True,
    ) -> list[tuple[float, float]]:
        """Sort vertices into a convex polygon in counterclockwise order.

        Performs full sanity checks, sorts vertices counterclockwise around their centroid,
        and optionally verifies that the resulting polygon is strictly convex.

        Args:
            vertices: Sequence of (latitude, longitude) pairs.
            start: Starting vertex reference:
                - "first": Preserves `vertices[0]` as the starting vertex.
                - "min_lat" or "canonical": Starts with vertex having minimum latitude.
                - "min_lon": Starts with vertex having minimum longitude.
                - (lat, lon) tuple: Starts with the specified vertex.
            enforce_convex: If True, raises ValueError if sorted vertices do not form
                a strictly convex polygon.

        Returns:
            List of (lat, lon) vertices in counterclockwise order.

        Raises:
            ValueError: If sanity checks fail or vertices do not form a strictly convex polygon.
        """
        pts = cls.validate_vertices(vertices)
        c_lat, c_lon = cls.centroid(pts)

        # Determine reference starting vertex
        if start == "first":
            ref_vertex = pts[0]
        elif start in ("min_lat", "canonical"):
            # Lowest latitude; tie-break by lowest longitude
            ref_vertex = min(pts, key=lambda p: (p[0], p[1]))
        elif start == "min_lon":
            # Lowest longitude; tie-break by lowest latitude
            ref_vertex = min(pts, key=lambda p: (p[1], p[0]))
        elif isinstance(start, (tuple, list)) and len(start) == 2:
            ref_coord = (float(start[0]), float(start[1]))
            # Find closest matching vertex
            ref_vertex = min(
                pts,
                key=lambda p: (p[0] - ref_coord[0]) ** 2 + (p[1] - ref_coord[1]) ** 2,
            )
        else:
            raise ValueError(f"Unknown start option: {start!r}")

        # Compute reference angle from centroid to reference vertex
        # x = lon, y = lat
        ref_dx = ref_vertex[1] - c_lon
        ref_dy = ref_vertex[0] - c_lat
        base_angle = math.atan2(ref_dy, ref_dx)

        # Sort vertices by counterclockwise relative angle [0, 2*pi)
        def polar_key(p: tuple[float, float]) -> tuple[float, float]:
            dx = p[1] - c_lon
            dy = p[0] - c_lat
            angle = math.atan2(dy, dx)
            rel_angle = (angle - base_angle) % (2.0 * math.pi)
            dist_sq = dx**2 + dy**2
            return rel_angle, dist_sq

        sorted_pts = sorted(pts, key=polar_key)

        # Convexity verification
        if enforce_convex and not cls.is_convex(sorted_pts, ccw_only=True):
            raise ValueError(
                "Vertices cannot be arranged into a strictly convex counterclockwise polygon."
            )

        return sorted_pts

    @classmethod
    def convex_hull(
        cls,
        vertices: Sequence[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        """Compute the convex hull of vertices in counterclockwise order using Jarvis March.

        Time complexity: O(n * h) <= O(n^2), where h is the number of hull vertices.
        """
        pts = cls.validate_vertices(vertices)
        n = len(pts)
        if n < 3:
            return pts

        # Start with bottom-most point (min lat, tie-break min lon)
        start_pt = min(pts, key=lambda p: (p[0], p[1]))
        hull: list[tuple[float, float]] = []
        curr = start_pt

        while True:
            hull.append(curr)
            next_cand = pts[0]
            for candidate in pts[1:]:
                if candidate == curr:
                    continue
                if next_cand == curr:
                    next_cand = candidate
                    continue

                # Cross product: (next_cand - curr) x (candidate - curr)
                # with x = lon, y = lat:
                cp = (next_cand[1] - curr[1]) * (candidate[0] - curr[0]) - (
                    next_cand[0] - curr[0]
                ) * (candidate[1] - curr[1])
                if cp < -1e-12:
                    # candidate is to the left of curr -> next_cand (counterclockwise turn)
                    next_cand = candidate
                elif abs(cp) <= 1e-12:
                    # Collinear: choose the point further away
                    d_next = (next_cand[1] - curr[1]) ** 2 + (next_cand[0] - curr[0]) ** 2
                    d_cand = (candidate[1] - curr[1]) ** 2 + (candidate[0] - curr[0]) ** 2
                    if d_cand > d_next:
                        next_cand = candidate

            curr = next_cand
            if curr == start_pt:
                break

        return hull

    @staticmethod
    def are_polygons_equivalent(
        poly1: Sequence[tuple[float, float]],
        poly2: Sequence[tuple[float, float]],
        tol: float = 1e-7,
    ) -> bool:
        """Check if two polygons contain the same vertices in the same cyclic order."""
        if len(poly1) != len(poly2):
            return False
        n = len(poly1)
        doubled = list(poly1) + list(poly1)

        def pt_equal(p1: tuple[float, float], p2: tuple[float, float]) -> bool:
            return abs(p1[0] - p2[0]) < tol and abs(p1[1] - p2[1]) < tol

        for i in range(n):
            if all(pt_equal(doubled[i + j], poly2[j]) for j in range(n)):
                return True
        return False

    @classmethod
    def load_csv(cls, file_path: str | Path) -> list[tuple[float, float]]:
        """Load (latitude, longitude) vertices from a CSV file.

        Supports standard CSV headers like 'latitude'/'lat' and 'longitude'/'lon'.
        """
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"CSV file not found: {path}")

        vertices: list[tuple[float, float]] = []
        with open(path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"CSV file {path} has no header row.")

            # Identify latitude and longitude column names
            lat_col = next(
                (col for col in reader.fieldnames if col.strip().lower() in ("latitude", "lat")),
                None,
            )
            lon_col = next(
                (col for col in reader.fieldnames if col.strip().lower() in ("longitude", "lon")),
                None,
            )

            if lat_col is None or lon_col is None:
                raise ValueError(
                    f"CSV file {path} must contain latitude and longitude columns, found: {reader.fieldnames}"
                )

            for row in reader:
                lat_val = row[lat_col]
                lon_val = row[lon_col]
                if lat_val is not None and lon_val is not None:
                    vertices.append((float(lat_val.strip()), float(lon_val.strip())))

        return vertices


def sort_eddy_vertices(
    vertices: Sequence[tuple[float, float]],
    start: str | tuple[float, float] = "first",
    enforce_convex: bool = True,
) -> list[tuple[float, float]]:
    """Convenience function to validate and sort ocean eddy vertices into a counterclockwise convex polygon.

    Args:
        vertices: Sequence of (latitude, longitude) pairs.
        start: Starting vertex reference ("first", "min_lat", "min_lon", or (lat, lon)).
        enforce_convex: If True, raises ValueError if the vertices do not form a strictly convex polygon.

    Returns:
        Sorted list of (latitude, longitude) pairs in counterclockwise order.
    """
    return SensorPerimeter.sort_vertices(
        vertices=vertices,
        start=start,
        enforce_convex=enforce_convex,
    )
