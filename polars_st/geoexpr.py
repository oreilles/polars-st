from __future__ import annotations

from functools import wraps
from inspect import signature
from pathlib import Path
from typing import TYPE_CHECKING, Literal, ParamSpec, cast

import polars as pl
from polars._utils.parse import parse_into_expression
from polars._utils.wrap import wrap_expr
from polars.api import register_expr_namespace
from polars.plugins import register_plugin_function

from polars_st import _lib
from polars_st.geometry import GeometryType, PolarsGeometryType
from polars_st.utils.internal import is_empty_method

if TYPE_CHECKING:
    from collections.abc import Sequence

    from polars_st.typing import (
        IntoExprColumn,
        IntoGeoExprColumn,
        IntoIntegerExpr,
        IntoNumericExpr,
    )

    P = ParamSpec("P")

__all__ = [
    "GeoExpr",
    "GeoExprNameSpace",
]


def register_plugin(is_aggregation: bool = False):  # noqa: ANN202
    def decorator(func):  # noqa: ANN001, ANN202
        assert is_empty_method(func)  # noqa: S101

        func_name = func.__name__
        sig = signature(func)
        params = sig.parameters
        expr_args = dict.fromkeys(k for k, v in params.items() if "Expr" in str(v.annotation))
        other_args = dict.fromkeys(k for k, v in params.items() if k not in expr_args)

        @wraps(func)
        def wrapper(self: GeoExprNameSpace, *args: P.args, **kwargs: P.kwargs):  # noqa: ANN202
            bound = sig.bind(self._expr, *args, **kwargs)
            bound.apply_defaults()
            return register_plugin_function(
                plugin_path=Path(__file__).parent,
                function_name=func_name,
                args=[self._expr, *[bound.arguments[k] for k in expr_args]],
                kwargs={k: bound.arguments[k] for k in other_args},
                is_elementwise=not is_aggregation,
                returns_scalar=is_aggregation,
            )

        return wrapper

    return decorator


class GeoExpr(pl.Expr):
    """`GeoExpr` is used as an alias for [`polars.Expr`](https://docs.pola.rs/api/python/stable/reference/expressions/index.html) with type annotations added for the `st` namespace."""  # noqa: E501

    @property
    def st(self) -> GeoExprNameSpace:
        return GeoExprNameSpace(self)

    def __new__(cls) -> GeoExpr:  # noqa: PYI034
        return cast("GeoExpr", pl.Expr())


@register_expr_namespace("st")
class GeoExprNameSpace:
    def __init__(self, expr: pl.Expr) -> None:
        self._expr = cast("GeoExpr", expr)

    def geometry_type(self) -> pl.Expr:
        """Return the type of each geometry.

        Examples:
            >>> gdf = st.GeoDataFrame([
            ...     "POINT(0 0)",
            ...     "LINESTRING(0 0, 1 2)",
            ...     "POLYGON((0 0, 1 1, 1 0, 0 0))"
            ... ])
            >>> gdf.select(st.geom().st.geometry_type())
            shape: (3, 1)
            ┌────────────┐
            │ geometry   │
            │ ---        │
            │ enum       │
            ╞════════════╡
            │ Point      │
            │ LineString │
            │ Polygon    │
            └────────────┘
        """
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="geometry_type",
            args=[self._expr],
            is_elementwise=True,
        ).map_batches(lambda s: s.cast(PolarsGeometryType), PolarsGeometryType)
        # Needed because pola-rs/polars#22125, pola-rs/pyo3-polars#131
        # Cannot use cast directly, see comments in pola-rs/polars#6106

    @register_plugin()
    def dimensions(self) -> pl.Expr:
        """Return the inherent dimensionality of each geometry.

        The inherent dimension is 0 for points, 1 for linestrings and linearrings,
            and 2 for polygons. For geometrycollections it is the max of the containing
            elements.

        Examples:
            >>> gdf = st.GeoDataFrame([
            ...     "POINT(0 0)",
            ...     "LINESTRING(0 0, 1 2)",
            ...     "POLYGON((0 0, 1 1, 1 0, 0 0))"
            ... ])
            >>> gdf.select(st.geom().st.dimensions())
            shape: (3, 1)
            ┌──────────┐
            │ geometry │
            │ ---      │
            │ i32      │
            ╞══════════╡
            │ 0        │
            │ 1        │
            │ 2        │
            └──────────┘
        """
        ...

    @register_plugin()
    def coordinate_dimension(self) -> pl.Expr:
        """Return the coordinate dimension (2, 3 or 4) of each geometry."""
        ...

    @register_plugin()
    def area(self) -> pl.Expr:
        """Return the area of each geometry."""
        ...

    @register_plugin()
    def bounds(self) -> pl.Expr:
        """Return the bounds of each geometry."""
        ...

    @register_plugin()
    def length(self) -> pl.Expr:
        """Return the length of each geometry."""
        ...

    @register_plugin()
    def minimum_clearance(self) -> pl.Expr:
        """Return the geometry minimum clearance."""
        ...

    @register_plugin()
    def x(self) -> pl.Expr:
        """Return the `x` value of Point geometries."""
        ...

    @register_plugin()
    def y(self) -> pl.Expr:
        """Return the `y` value of Point geometries."""
        ...

    @register_plugin()
    def z(self) -> pl.Expr:
        """Return the `z` value of Point geometries."""
        ...

    @register_plugin()
    def m(self) -> pl.Expr:
        """Return the `m` value of Point geometries."""
        ...

    @register_plugin()
    def count_coordinates(self) -> pl.Expr:
        """Return the number of coordinates in each geometry."""
        ...

    @register_plugin()
    def coordinates(self, output_dimension: Literal[2, 3] | None = None) -> pl.Expr:
        """Return the coordinates of each geometry."""
        ...

    @register_plugin()
    def exterior_ring(self) -> GeoExpr:
        """Return the exterior ring of Polygon geometries."""
        ...

    @register_plugin()
    def interior_rings(self) -> pl.Expr:
        """Return the list of interior rings for Polygon geometries."""
        ...

    @register_plugin()
    def count_interior_rings(self) -> pl.Expr:
        """Return the number of interior rings in Polygon geometries."""
        ...

    @register_plugin()
    def get_interior_ring(self, index: IntoIntegerExpr) -> GeoExpr:
        """Return the nth ring of Polygon geometries."""
        ...

    @register_plugin()
    def count_geometries(self) -> pl.Expr:
        """Return the number of parts in multipart geometries."""
        ...

    @register_plugin()
    def get_geometry(self, index: IntoIntegerExpr) -> GeoExpr:
        """Return the nth part of multipart geometries."""
        ...

    @register_plugin()
    def count_points(self) -> pl.Expr:
        """Return the number of points in LineString geometries."""
        ...

    @register_plugin()
    def get_point(self, index: IntoIntegerExpr) -> GeoExpr:
        """Return the nth point of LineString geometries."""
        ...

    @register_plugin()
    def parts(self) -> pl.Expr:
        """Return the list of parts for multipart geometries."""
        ...

    @register_plugin()
    def precision(self) -> pl.Expr:
        """Return the precision of each geometry."""
        ...

    @register_plugin()
    def set_precision(
        self,
        grid_size: IntoNumericExpr,
        mode: Literal["valid_output", "no_topo", "keep_collapsed"] = "valid_output",
    ) -> GeoExpr:
        """Set the precision of each geometry to a certain grid size."""
        ...

    @register_plugin()
    def distance(self, other: IntoGeoExprColumn) -> pl.Expr:
        """Return the distance from each geometry to other."""
        ...

    @register_plugin()
    def hausdorff_distance(
        self,
        other: IntoGeoExprColumn,
        densify: float | None = None,
    ) -> pl.Expr:
        """Return the hausdorff distance from each geometry to other."""
        ...

    @register_plugin()
    def frechet_distance(
        self,
        other: IntoGeoExprColumn,
        densify: float | None = None,
    ) -> pl.Expr:
        """Return the frechet distance from each geometry to other."""
        ...

    # Projection operations

    @register_plugin()
    def srid(self) -> pl.Expr:
        """Return the geometry SRID."""
        ...

    @register_plugin()
    def set_srid(self, srid: IntoIntegerExpr) -> GeoExpr:
        """Set the SRID of each geometry to a given value.

        Args:
            srid: The geometry new SRID
        """
        ...

    @register_plugin()
    def to_srid(self, srid: IntoIntegerExpr) -> GeoExpr:
        """Transform the coordinates of each geometry into a new CRS.

        Args:
            srid: The srid code of the new CRS
        """
        ...

    # Serialization

    @register_plugin()
    def to_wkt(
        self,
        rounding_precision: int | None = 6,
        trim: bool = True,
        output_dimension: Literal[2, 3, 4] = 3,
        old_3d: bool = False,
    ) -> pl.Expr:
        """Serialize each geometry as WKT (Well-Known Text).

        Args:
            rounding_precision: The rounding precision when writing the WKT string.
                Set to None to indicate the full precision.
            trim: If True, trim unnecessary decimals (trailing zeros).
            output_dimension: The output dimension for the WKT string. Specifying 3
                means that up to 3 dimensions will be written but 2D geometries will
                still be represented as 2D in the WKT string.
            old_3d (bool, optional): Enable old style 3D/4D WKT generation. By default,
                new style 3D/4D WKT (ie. “POINT Z (10 20 30)”) is returned, but with
                `old_3d=True` the WKT will be formatted in the style “POINT (10 20 30)”.
        """
        ...

    @register_plugin()
    def to_ewkt(
        self,
        rounding_precision: int | None = 6,
        trim: bool = True,
        output_dimension: Literal[2, 3, 4] = 3,
        old_3d: bool = False,
    ) -> pl.Expr:
        """Serialize each geometry as EWKT (Extended Well-Known Text).

        Args:
            rounding_precision: The rounding precision when writing the WKT string.
                Set to None to indicate the full precision.
            trim: If True, trim unnecessary decimals (trailing zeros).
            output_dimension: The output dimension for the WKT string. Specifying 3
                means that up to 3 dimensions will be written but 2D geometries will
                still be represented as 2D in the WKT string.
            old_3d (bool, optional): Enable old style 3D/4D WKT generation. By default,
                new style 3D/4D WKT (ie. “POINT Z (10 20 30)”) is returned, but with
                `old_3d=True` the WKT will be formatted in the style “POINT (10 20 30)”.
        """
        ...

    @register_plugin()
    def to_wkb(
        self,
        output_dimension: Literal[2, 3, 4] = 3,
        byte_order: Literal[0, 1] | None = None,
        include_srid: bool = False,
    ) -> pl.Expr:
        """Serialize each geometry as WKB (Well-Known Binary).

        Args:
            output_dimension:
                The output dimension for the WKB. Specifying 3 means that up to 3 dimensions
                will be written but 2D geometries will still be represented as 2D in the WKB
                representation.
            byte_order:
                Defaults to native machine byte order (`None`). Use 0 to force big endian
                and 1 for little endian.
            include_srid:
                If True, the SRID is be included in WKB (this is an extension
                to the OGC WKB specification).
        """
        ...

    @register_plugin()
    def to_geojson(self, indent: int | None = None) -> pl.Expr:
        """Serialize each geometry as GeoJSON.

        Args:
            indent:
                If indent is not `None`, then GeoJSON will be pretty-printed.
                An indent level of 0 will only insert newlines. `None` (the default)
                outputs the most compact representation.
        """
        ...

    def to_shapely(self) -> pl.Expr:
        """Convert each geometry to a Shapely object."""
        import shapely

        return self._expr.map_batches(
            lambda s: pl.Series(s.name, shapely.from_wkb(s), dtype=pl.Object()),
            return_dtype=pl.Object(),
            is_elementwise=True,
        )

    def to_dict(self) -> pl.Expr:
        """Convert each geometry to a GeoJSON-like Python [`dict`][] object."""
        return self._expr.map_batches(
            lambda s: pl.Series(s.name, _lib.to_python_dict(s._s), dtype=pl.Object),  # noqa: SLF001
            return_dtype=pl.Object(),
            is_elementwise=True,
        )

    def cast(self, into: IntoExprColumn) -> pl.Expr:
        """Cast each geometry into a different compatible geometry type.

        Valid casts are:

        | Source          | Destination |
        |-----------------|-------------|
        | Point           | MultiPoint  |
        | MultiPoint      | LineString, CircularString |
        | LineString      | MultiPoint, CircularString, MultiLineString, MultiCurve |
        | CircularString  | MultiPoint, LineString, MultiLineString, MultiCurve |
        | MultiLineString | Polygon |
        | Polygon         | MultiPolygon, MultiSurface |
        | CurvePolygon    | MultiSurface |
        | Any             | GeometryCollection |
        """
        into = wrap_expr(parse_into_expression(into, str_as_lit=True))
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="cast",
            args=[self._expr, into],
            is_elementwise=True,
        )

    @register_plugin()
    def multi(self) -> pl.Expr:
        """Cast each geometry into their multipart equivalent."""
        ...

    # Unary predicates

    @register_plugin()
    def has_z(self) -> pl.Expr:
        """Return `True` for each geometry with `z` coordinate values."""
        ...

    @register_plugin()
    def has_m(self) -> pl.Expr:
        """Return `True` for each geometry with `m` coordinate values."""
        ...

    @register_plugin()
    def is_ccw(self) -> pl.Expr:
        """Return `True` for linear geometries with counter-clockwise coord sequence."""
        ...

    @register_plugin()
    def is_closed(self) -> pl.Expr:
        """Return `True` for closed linear geometries."""
        ...

    @register_plugin()
    def is_empty(self) -> pl.Expr:
        """Return `True` for empty geometries."""
        ...

    @register_plugin()
    def is_ring(self) -> pl.Expr:
        """Return `True` for ring geometries."""
        ...

    @register_plugin()
    def is_simple(self) -> pl.Expr:
        """Return `True` for simple geometries."""
        ...

    @register_plugin()
    def is_valid(self) -> pl.Expr:
        """Return `True` for valid geometries."""
        ...

    @register_plugin()
    def is_valid_reason(self) -> pl.Expr:
        """Return an explanation string for the invalidity of each geometry."""
        ...

    # Binary predicates

    @register_plugin()
    def crosses(self, other: IntoGeoExprColumn) -> pl.Expr:
        """Return `True` when each geometry crosses other."""
        ...

    @register_plugin()
    def contains(self, other: IntoGeoExprColumn) -> pl.Expr:
        """Return `True` when each geometry contains other."""
        ...

    @register_plugin()
    def contains_properly(self, other: IntoGeoExprColumn) -> pl.Expr:
        """Return `True` when each geometry properly contains other."""
        ...

    @register_plugin()
    def covered_by(self, other: IntoGeoExprColumn) -> pl.Expr:
        """Return `True` when each geometry is covered by other."""
        ...

    @register_plugin()
    def covers(self, other: IntoGeoExprColumn) -> pl.Expr:
        """Return `True` when each geometry covers other."""
        ...

    @register_plugin()
    def disjoint(self, other: IntoGeoExprColumn) -> pl.Expr:
        """Return `True` when each geometry is disjoint from other."""
        ...

    @register_plugin()
    def dwithin(self, other: IntoGeoExprColumn, distance: IntoNumericExpr) -> pl.Expr:
        """Return `True` when each geometry is within given distance to other."""
        ...

    @register_plugin()
    def intersects(self, other: IntoGeoExprColumn) -> pl.Expr:
        """Return `True` when each geometry intersects other."""
        ...

    @register_plugin()
    def overlaps(self, other: IntoGeoExprColumn) -> pl.Expr:
        """Return `True` when each geometry overlaps other."""
        ...

    @register_plugin()
    def touches(self, other: IntoGeoExprColumn) -> pl.Expr:
        """Return `True` when each geometry touches other."""
        ...

    @register_plugin()
    def within(self, other: IntoGeoExprColumn) -> pl.Expr:
        """Return `True` when each geometry is within other."""
        ...

    @register_plugin()
    def equals(self, other: IntoGeoExprColumn) -> pl.Expr:
        """Return `True` when each geometry is equal to other."""
        ...

    @register_plugin()
    def equals_exact(
        self,
        other: IntoGeoExprColumn,
        tolerance: float = 0.0,
    ) -> pl.Expr:
        """Return `True` when each geometry is equal to other."""
        ...

    @register_plugin()
    def equals_identical(self, other: IntoGeoExprColumn) -> pl.Expr:
        """Return `True` when each geometry is equal to other."""
        ...

    @register_plugin()
    def relate(self, other: IntoGeoExprColumn) -> pl.Expr:
        """Return the DE-9IM intersection matrix of each geometry with other."""
        ...

    @register_plugin()
    def relate_pattern(
        self,
        other: IntoGeoExprColumn,
        pattern: str,
    ) -> pl.Expr:
        """Return `True` when the DE-9IM intersection matrix of geometry with other matches a given pattern."""  # noqa: E501
        ...

    # Set operations

    @register_plugin()
    def union(self, other: IntoGeoExprColumn, grid_size: float | None = None) -> GeoExpr:
        """Return the union of each geometry with other."""
        ...

    @register_plugin()
    def unary_union(self, grid_size: float | None = None) -> GeoExpr:
        """Return the unary union of each geometry."""
        ...

    @register_plugin()
    def coverage_union(self) -> GeoExpr:
        """Return the coverage union of each geometry with other."""
        ...

    @register_plugin()
    def intersection(
        self,
        other: IntoGeoExprColumn,
        grid_size: float | None = None,
    ) -> GeoExpr:
        """Return the intersection of each geometry with other."""
        ...

    @register_plugin()
    def difference(
        self,
        other: IntoGeoExprColumn,
        grid_size: float | None = None,
    ) -> GeoExpr:
        """Return the difference of each geometry with other."""
        ...

    @register_plugin()
    def symmetric_difference(
        self,
        other: IntoGeoExprColumn,
        grid_size: float | None = None,
    ) -> GeoExpr:
        """Return the symmetric difference of each geometry with other."""
        ...

    # Constructive operations

    @register_plugin()
    def boundary(self) -> GeoExpr:
        """Return the topological boundary of each geometry."""
        ...

    @register_plugin()
    def buffer(
        self,
        distance: IntoNumericExpr,
        quad_segs: int = 8,
        cap_style: Literal["round", "square", "flat"] = "round",
        join_style: Literal["round", "mitre", "bevel"] = "round",
        mitre_limit: float = 5.0,
        single_sided: bool = False,
    ) -> GeoExpr:
        """Return a buffer around each geometry."""
        ...

    @register_plugin()
    def offset_curve(
        self,
        distance: IntoNumericExpr,
        quad_segs: int = 8,
        join_style: Literal["round", "mitre", "bevel"] = "round",
        mitre_limit: float = 5.0,
    ) -> GeoExpr:
        """Return a line at a given distance of each geometry."""
        ...

    @register_plugin()
    def centroid(self) -> GeoExpr:
        """Return the centroid of each geometry."""
        ...

    @register_plugin()
    def center(self) -> GeoExpr:
        """Return the bounding box center of each geometry."""
        ...

    @register_plugin()
    def clip_by_rect(self, bounds: IntoExprColumn) -> GeoExpr:
        """Clips each geometry by a bounding rectangle."""
        ...

    @register_plugin()
    def convex_hull(self) -> GeoExpr:
        """Return the convex hull of each geometry."""
        ...

    @register_plugin()
    def concave_hull(self, ratio: float = 0.0, allow_holes: bool = False) -> GeoExpr:
        """Return the concave hull of each geometry."""
        ...

    @register_plugin()
    def segmentize(self, max_segment_length: IntoNumericExpr) -> GeoExpr: ...

    @register_plugin()
    def envelope(self) -> GeoExpr:
        """Return the envelope of each geometry."""
        ...

    @register_plugin()
    def extract_unique_points(self) -> GeoExpr: ...

    @register_plugin()
    def build_area(self) -> GeoExpr: ...

    @register_plugin()
    def make_valid(self) -> GeoExpr: ...

    @register_plugin()
    def normalize(self) -> GeoExpr: ...

    @register_plugin()
    def node(self) -> GeoExpr: ...

    @register_plugin()
    def point_on_surface(self) -> GeoExpr:
        """Return a point that intersects of each geometry."""
        ...

    @register_plugin()
    def remove_repeated_points(self, tolerance: IntoNumericExpr = 0.0) -> GeoExpr:
        """Remove the repeated points for each geometry."""
        ...

    @register_plugin()
    def reverse(self) -> GeoExpr:
        """Reverse the coordinates order of each geometry."""
        ...

    @register_plugin()
    def simplify(
        self,
        tolerance: IntoNumericExpr,
        preserve_topology: bool = True,
    ) -> GeoExpr:
        """Simplify each geometry with a given tolerance."""
        ...

    @register_plugin()
    def force_2d(self) -> GeoExpr:
        """Force the dimensionality of a geometry to 2D."""
        ...

    @register_plugin()
    def force_3d(self, z: IntoNumericExpr = 0.0) -> GeoExpr:
        """Force the dimensionality of a geometry to 3D."""
        ...

    @register_plugin()
    def flip_coordinates(self) -> GeoExpr:
        """Flip the x and y coordinates of each geometry."""
        ...

    @register_plugin()
    def minimum_rotated_rectangle(self) -> GeoExpr: ...

    @register_plugin()
    def snap(
        self,
        other: IntoGeoExprColumn,
        tolerance: IntoNumericExpr,
    ) -> GeoExpr: ...

    @register_plugin()
    def shortest_line(self, other: IntoGeoExprColumn) -> GeoExpr:
        """Return the shortest line between each geometry and other."""
        ...

    # Affine tranforms

    def affine_transform(self, matrix: IntoExprColumn | Sequence[float]) -> GeoExpr:
        """Apply a 2D or 3D transformation matrix to the coordinates of each geometry.

        Args:
            matrix:
                The transformation matrix to apply to coordinates. Should contains 6
                elements for a 2D transform or 12 for a 3D transform. The matrix elements
                order should be, in order:
                - `m11`, `m12`, `m21`, `m22`, `tx`, `ty` for 2D transformations
                - `m11`, `m12`, `m13`, `m21`, `m22`, `m23`, `m31`, `m32`, `m33`, `tx`, `ty`, `tz`
                    for 3D transformations

        """
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="affine_transform",
            args=[
                self._expr,
                matrix
                if isinstance(matrix, pl.Expr | pl.Series | str)
                else pl.lit(matrix, dtype=pl.Array(pl.Float64, len(matrix))),
            ],
            is_elementwise=True,
        ).pipe(lambda e: cast("GeoExpr", e))

    def translate(
        self,
        x: IntoNumericExpr = 0.0,
        y: IntoNumericExpr = 0.0,
        z: IntoNumericExpr = 0.0,
    ) -> GeoExpr:
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="translate",
            args=[self._expr, pl.concat_list(x, y, z)],
            is_elementwise=True,
        ).pipe(lambda e: cast("GeoExpr", e))

    @register_plugin()
    def rotate(
        self,
        angle: IntoNumericExpr,
        origin: Literal["center", "centroid"] | Sequence[float] = "center",
    ) -> GeoExpr: ...

    def scale(
        self,
        x: IntoNumericExpr = 1.0,
        y: IntoNumericExpr = 1.0,
        z: IntoNumericExpr = 1.0,
        origin: Literal["center", "centroid"] | Sequence[float] = "center",
    ) -> GeoExpr:
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="scale",
            args=[self._expr, pl.concat_list(x, y, z)],
            kwargs={"origin": origin},
            is_elementwise=True,
        ).pipe(lambda e: cast("GeoExpr", e))

    def skew(
        self,
        x: IntoNumericExpr = 0.0,
        y: IntoNumericExpr = 0.0,
        z: IntoNumericExpr = 0.0,
        origin: Literal["center", "centroid"] | Sequence[float] = "center",
    ) -> GeoExpr:
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="skew",
            args=[self._expr, pl.concat_list(x, y, z)],
            kwargs={"origin": origin},
            is_elementwise=True,
        ).pipe(lambda e: cast("GeoExpr", e))

    # Linestring operations

    @register_plugin()
    def interpolate(
        self,
        distance: IntoNumericExpr,
        normalized: bool = False,
    ) -> GeoExpr: ...

    @register_plugin()
    def project(
        self,
        other: IntoGeoExprColumn,
        normalized: bool = False,
    ) -> pl.Expr: ...

    @register_plugin()
    def substring(self, start: IntoNumericExpr, end: IntoNumericExpr) -> GeoExpr:
        """Returns the substring of each line starting and ending at the given fractional locations."""  # noqa: E501
        ...

    @register_plugin()
    def line_merge(self, directed: bool = False) -> GeoExpr: ...

    @register_plugin()
    def shared_paths(self, other: IntoGeoExprColumn) -> GeoExpr: ...

    # Aggregations

    @register_plugin(is_aggregation=True)
    def total_bounds(self) -> pl.Expr:
        """Return the total bounds of all geometries."""
        ...

    @register_plugin(is_aggregation=True)
    def collect(self, into: GeometryType | None = None) -> GeoExpr:
        """Aggregate geometries into a single collection."""
        ...

    @register_plugin(is_aggregation=True)
    def union_all(self, grid_size: float | None = None) -> GeoExpr:
        """Return the union of all geometries."""
        ...

    @register_plugin(is_aggregation=True)
    def coverage_union_all(self) -> GeoExpr:
        """Return the coverage union of all geometries."""
        ...

    @register_plugin(is_aggregation=True)
    def intersection_all(self, grid_size: float | None = None) -> GeoExpr:
        """Return the intersection of all geometries."""
        ...

    @register_plugin(is_aggregation=True)
    def difference_all(self, grid_size: float | None = None) -> GeoExpr:
        """Return the difference of all geometries."""
        ...

    @register_plugin(is_aggregation=True)
    def symmetric_difference_all(self, grid_size: float | None = None) -> GeoExpr:
        """Return the symmetric difference of all geometries."""
        ...

    @register_plugin(is_aggregation=True)
    def polygonize(self) -> GeoExpr: ...

    @register_plugin(is_aggregation=True)
    def voronoi_polygons(
        self,
        tolerance: float = 0.0,
        extend_to: bytes | None = None,
        only_edges: bool = False,
    ) -> GeoExpr:
        """Return a Voronoi diagram of all geometries vertices."""
        ...

    @register_plugin(is_aggregation=True)
    def delaunay_triangles(
        self,
        tolerance: float = 0.0,
        only_edges: bool = False,
    ) -> GeoExpr:
        """Return a Delaunay triangulation of all geometries vertices."""
        ...
