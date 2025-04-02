from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import polars as pl
from polars._utils.parse import parse_into_expression
from polars._utils.wrap import wrap_expr
from polars.api import register_expr_namespace
from polars.plugins import register_plugin_function

from polars_st import _lib

if TYPE_CHECKING:
    from collections.abc import Sequence

    from polars_st.typing import (
        IntoDecimalExpr,
        IntoExprColumn,
        IntoGeoExprColumn,
        IntoIntegerExpr,
    )

__all__ = [
    "GeoExpr",
    "GeoExprNameSpace",
]


class GeoExpr(pl.Expr):
    """A `GeoExpr` is a regular [`polars.Expr`](https://docs.pola.rs/api/python/stable/reference/expressions/index.html) with type annotations added for the `st` namespace."""  # noqa: E501

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
        """Return the type ID of each geometry.

        -  **0** = Unknown
        -  **1** = Point
        -  **2** = LineString
        -  **3** = Polygon
        -  **4** = MultiPoint
        -  **5** = MultiLineString
        -  **6** = MultiPolygon
        -  **7** = GeometryCollection
        -  **8** = CircularString
        -  **9** = CompoundCurve
        - **10** = CurvePolygon
        - **11** = MultiCurve
        - **12** = MultiSurface
        - **13** = Curve
        - **14** = Surface
        - **15** = PolyhedralSurface
        - **16** = Tin
        - **17** = Triangle

        Examples:
            >>> gdf = st.GeoDataFrame([
            ...     "POINT(0 0)",
            ...     "LINESTRING(0 0, 1 2)",
            ...     "POLYGON((0 0, 1 1, 1 0, 0 0))"
            ... ])
            >>> gdf.select(st.geom().st.geometry_type())
            shape: (3, 1)
            ┌──────────┐
            │ geometry │
            │ ---      │
            │ u32      │
            ╞══════════╡
            │ 1        │
            │ 2        │
            │ 3        │
            └──────────┘
        """
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="geometry_type",
            args=self._expr,
            is_elementwise=True,
        )

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
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="dimensions",
            args=self._expr,
            is_elementwise=True,
        )

    def coordinate_dimension(self) -> pl.Expr:
        """Return the coordinate dimension (2, 3 or 4) of each geometry."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="coordinate_dimension",
            args=self._expr,
            is_elementwise=True,
        )

    def area(self) -> pl.Expr:
        """Return the area of each geometry."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="area",
            args=self._expr,
            is_elementwise=True,
        )

    def bounds(self) -> pl.Expr:
        """Return the bounds of each geometry."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="bounds",
            args=[self._expr],
            is_elementwise=True,
        )

    def length(self) -> pl.Expr:
        """Return the length of each geometry."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="length",
            args=self._expr,
            is_elementwise=True,
        )

    def minimum_clearance(self) -> pl.Expr:
        """Return the geometry minimum clearance."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="minimum_clearance",
            args=[self._expr],
            is_elementwise=True,
        )

    def x(self) -> pl.Expr:
        """Return the `x` value of Point geometries."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="x",
            args=self._expr,
            is_elementwise=True,
        )

    def y(self) -> pl.Expr:
        """Return the `y` value of Point geometries."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="y",
            args=self._expr,
            is_elementwise=True,
        )

    def z(self) -> pl.Expr:
        """Return the `z` value of Point geometries."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="z",
            args=self._expr,
            is_elementwise=True,
        )

    def m(self) -> pl.Expr:
        """Return the `m` value of Point geometries."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="m",
            args=self._expr,
            is_elementwise=True,
        )

    def count_coordinates(self) -> pl.Expr:
        """Return the number of coordinates in each geometry."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="count_coordinates",
            args=[self._expr],
            is_elementwise=True,
        )

    def coordinates(self, output_dimension: Literal[2, 3] = 2) -> pl.Expr:
        """Return the coordinates of each geometry."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="coordinates",
            args=[self._expr],
            kwargs={"output_dimension": output_dimension},
            is_elementwise=True,
        )

    def exterior_ring(self) -> GeoExpr:
        """Return the exterior ring of Polygon geometries."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="exterior_ring",
            args=self._expr,
            is_elementwise=True,
        ).pipe(lambda e: cast("GeoExpr", e))

    def interior_rings(self) -> pl.Expr:
        """Return the list of interior rings for Polygon geometries."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="interior_rings",
            args=self._expr,
            is_elementwise=True,
        )

    def count_interior_rings(self) -> pl.Expr:
        """Return the number of interior rings in Polygon geometries."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="count_interior_rings",
            args=self._expr,
            is_elementwise=True,
        )

    def get_interior_ring(self, index: IntoIntegerExpr) -> GeoExpr:
        """Return the nth ring of Polygon geometries."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="get_interior_ring",
            args=[self._expr, index],
            is_elementwise=True,
        ).pipe(lambda e: cast("GeoExpr", e))

    def count_geometries(self) -> pl.Expr:
        """Return the number of parts in multipart geometries."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="count_geometries",
            args=self._expr,
            is_elementwise=True,
        )

    def get_geometry(self, index: IntoIntegerExpr) -> GeoExpr:
        """Return the nth part of multipart geometries."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="get_geometry",
            args=[self._expr, index],
            is_elementwise=True,
        ).pipe(lambda e: cast("GeoExpr", e))

    def count_points(self) -> pl.Expr:
        """Return the number of points in LineString geometries."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="count_points",
            args=self._expr,
            is_elementwise=True,
        )

    def get_point(self, index: IntoIntegerExpr) -> GeoExpr:
        """Return the nth point of LineString geometries."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="get_point",
            args=[self._expr, index],
            is_elementwise=True,
        ).pipe(lambda e: cast("GeoExpr", e))

    def parts(self) -> pl.Expr:
        """Return the list of parts for multipart geometries."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="parts",
            args=self._expr,
            is_elementwise=True,
        )

    def precision(self) -> pl.Expr:
        """Return the precision of each geometry."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="precision",
            args=self._expr,
            is_elementwise=True,
        )

    def set_precision(
        self,
        grid_size: IntoDecimalExpr,
        mode: Literal["valid_output", "no_topo", "keep_collapsed"] = "valid_output",
    ) -> GeoExpr:
        """Set the precision of each geometry to a certain grid size."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="set_precision",
            args=[self._expr, grid_size],
            kwargs={"mode": mode},
            is_elementwise=True,
        ).pipe(lambda e: cast("GeoExpr", e))

    def distance(self, other: IntoGeoExprColumn) -> pl.Expr:
        """Return the distance from each geometry to other."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="distance",
            args=[self._expr, other],
            is_elementwise=True,
        )

    def hausdorff_distance(
        self,
        other: IntoGeoExprColumn,
        densify: float | None = None,
    ) -> pl.Expr:
        """Return the hausdorff distance from each geometry to other."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="hausdorff_distance",
            args=[self._expr, other],
            kwargs={"densify": densify},
            is_elementwise=True,
        )

    def frechet_distance(
        self,
        other: IntoGeoExprColumn,
        densify: float | None = None,
    ) -> pl.Expr:
        """Return the frechet distance from each geometry to other."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="frechet_distance",
            args=[self._expr, other],
            kwargs={"densify": densify},
            is_elementwise=True,
        )

    # Projection operations

    def srid(self) -> pl.Expr:
        """Return the geometry SRID."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="srid",
            args=self._expr,
            is_elementwise=True,
        )

    def set_srid(self, srid: IntoIntegerExpr) -> GeoExpr:
        """Set the SRID of each geometry to a given value.

        Args:
            srid: The geometry new SRID
        """
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="set_srid",
            args=[self._expr, srid],
            is_elementwise=True,
        ).pipe(lambda s: cast("GeoExpr", s))

    def to_srid(self, srid: int) -> GeoExpr:
        """Transform the coordinates of each geometry into a new CRS.

        Args:
            srid: The srid code of the new CRS
        """
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="to_srid",
            args=[self._expr, srid],
            is_elementwise=True,
        ).pipe(lambda e: cast("GeoExpr", e))

    # Serialization

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
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="to_wkt",
            args=self._expr,
            kwargs={
                "rounding_precision": rounding_precision,
                "trim": trim,
                "output_dimension": output_dimension,
                "old_3d": old_3d,
            },
            is_elementwise=True,
        )

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
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="to_ewkt",
            args=self._expr,
            kwargs={
                "rounding_precision": rounding_precision,
                "trim": trim,
                "output_dimension": output_dimension,
                "old_3d": old_3d,
            },
            is_elementwise=True,
        )

    def to_wkb(
        self,
        output_dimension: Literal[2, 3, 4] = 3,
        byte_order: Literal[0, 1] | None = None,
        include_srid: bool = False,
    ) -> pl.Expr:
        """Serialize each geometry as WKB (Well-Known Binary).

        Args:
            output_dimension :
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
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="to_wkb",
            args=self._expr,
            kwargs={
                "output_dimension": output_dimension,
                "byte_order": byte_order,
                "include_srid": include_srid,
            },
            is_elementwise=True,
        )

    def to_geojson(self, indent: int | None = None) -> pl.Expr:
        """Serialize each geometry as GeoJSON.

        Args:
            indent:
                If indent is not `None`, then GeoJSON will be pretty-printed.
                An indent level of 0 will only insert newlines. `None` (the default)
                outputs the most compact representation.
        """
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="to_geojson",
            args=self._expr,
            kwargs={"indent": indent},
            is_elementwise=True,
        )

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
            lambda s: pl.Series(s.name, _lib.to_python_dict(s), dtype=pl.Object),
            is_elementwise=True,
        )

    # Unary predicates

    def has_z(self) -> pl.Expr:
        """Return `True` for each geometry with `z` coordinate values."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="has_z",
            args=[self._expr],
            is_elementwise=True,
        )

    def has_m(self) -> pl.Expr:
        """Return `True` for each geometry with `m` coordinate values."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="has_m",
            args=[self._expr],
            is_elementwise=True,
        )

    def is_ccw(self) -> pl.Expr:
        """Return `True` for linear geometries with counter-clockwise coord sequence."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="is_ccw",
            args=[self._expr],
            is_elementwise=True,
        )

    def is_closed(self) -> pl.Expr:
        """Return `True` for closed linear geometries."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="is_closed",
            args=[self._expr],
            is_elementwise=True,
        )

    def is_empty(self) -> pl.Expr:
        """Return `True` for empty geometries."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="is_empty",
            args=[self._expr],
            is_elementwise=True,
        )

    def is_ring(self) -> pl.Expr:
        """Return `True` for ring geometries."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="is_ring",
            args=[self._expr],
            is_elementwise=True,
        )

    def is_simple(self) -> pl.Expr:
        """Return `True` for simple geometries."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="is_simple",
            args=[self._expr],
            is_elementwise=True,
        )

    def is_valid(self) -> pl.Expr:
        """Return `True` for valid geometries."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="is_valid",
            args=[self._expr],
            is_elementwise=True,
        )

    def is_valid_reason(self) -> pl.Expr:
        """Return an explanation string for the invalidity of each geometry."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="is_valid_reason",
            args=[self._expr],
            is_elementwise=True,
        )

    # Binary predicates

    def crosses(self, other: IntoGeoExprColumn) -> pl.Expr:
        """Return `True` when each geometry crosses other."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="crosses",
            args=[self._expr, other],
            is_elementwise=True,
        )

    def contains(self, other: IntoGeoExprColumn) -> pl.Expr:
        """Return `True` when each geometry contains other."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="crosses",
            args=[self._expr, other],
            is_elementwise=True,
        )

    def contains_properly(self, other: IntoGeoExprColumn) -> pl.Expr:
        """Return `True` when each geometry properly contains other."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="contains_properly",
            args=[self._expr, other],
            is_elementwise=True,
        )

    def covered_by(self, other: IntoGeoExprColumn) -> pl.Expr:
        """Return `True` when each geometry is covered by other."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="covered_by",
            args=[self._expr, other],
            is_elementwise=True,
        )

    def covers(self, other: IntoGeoExprColumn) -> pl.Expr:
        """Return `True` when each geometry covers other."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="covers",
            args=[self._expr, other],
            is_elementwise=True,
        )

    def disjoint(self, other: IntoGeoExprColumn) -> pl.Expr:
        """Return `True` when each geometry is disjoint from other."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="disjoint",
            args=[self._expr, other],
            is_elementwise=True,
        )

    def dwithin(self, other: IntoGeoExprColumn, distance: float) -> pl.Expr:
        """Return `True` when each geometry is within given distance to other."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="dwithin",
            args=[self._expr, other],
            kwargs={"distance": distance},
            is_elementwise=True,
        )

    def intersects(self, other: IntoGeoExprColumn) -> pl.Expr:
        """Return `True` when each geometry intersects other."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="intersects",
            args=[self._expr, other],
            is_elementwise=True,
        )

    def overlaps(self, other: IntoGeoExprColumn) -> pl.Expr:
        """Return `True` when each geometry overlaps other."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="overlaps",
            args=[self._expr, other],
            is_elementwise=True,
        )

    def touches(self, other: IntoGeoExprColumn) -> pl.Expr:
        """Return `True` when each geometry touches other."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="touches",
            args=[self._expr, other],
            is_elementwise=True,
        )

    def within(self, other: IntoGeoExprColumn) -> pl.Expr:
        """Return `True` when each geometry is within other."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="within",
            args=[self._expr, other],
            is_elementwise=True,
        )

    def equals(self, other: IntoGeoExprColumn) -> pl.Expr:
        """Return `True` when each geometry is equal to other."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="equals",
            args=[self._expr, other],
            is_elementwise=True,
        )

    def equals_exact(
        self,
        other: IntoGeoExprColumn,
        tolerance: float = 0.0,
    ) -> pl.Expr:
        """Return `True` when each geometry is equal to other."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="equals_exact",
            args=[self._expr, other],
            kwargs={"tolerance": tolerance},
            is_elementwise=True,
        )

    def equals_identical(self, other: IntoGeoExprColumn) -> pl.Expr:
        """Return `True` when each geometry is equal to other."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="equals_identical",
            args=[self._expr, other],
            is_elementwise=True,
        )

    def relate(self, other: IntoGeoExprColumn) -> pl.Expr:
        """Return the DE-9IM intersection matrix of each geometry with other."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="relate",
            args=[self._expr, other],
            is_elementwise=True,
        )

    def relate_pattern(
        self,
        other: IntoGeoExprColumn,
        pattern: str,
    ) -> pl.Expr:
        """Return `True` when the DE-9IM intersection matrix of geometry with other matches a given pattern."""  # noqa: E501
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="relate_pattern",
            args=[self._expr, other],
            kwargs={"pattern": pattern},
            is_elementwise=True,
        )

    # Set operations

    def union(self, other: IntoGeoExprColumn, grid_size: float | None = None) -> GeoExpr:
        """Return the union of each geometry with other."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="union",
            args=[self._expr, other],
            kwargs={"grid_size": grid_size},
            is_elementwise=True,
        ).pipe(lambda e: cast("GeoExpr", e))

    def unary_union(self, grid_size: float | None = None) -> GeoExpr:
        """Return the unary union of each geometry."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="unary_union",
            args=[self._expr],
            kwargs={"grid_size": grid_size},
            is_elementwise=True,
        ).pipe(lambda e: cast("GeoExpr", e))

    def coverage_union(self) -> GeoExpr:
        """Return the coverage union of each geometry with other."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="coverage_union",
            args=[self._expr],
            is_elementwise=True,
        ).pipe(lambda e: cast("GeoExpr", e))

    def intersection(
        self,
        other: IntoGeoExprColumn,
        grid_size: float | None = None,
    ) -> GeoExpr:
        """Return the intersection of each geometry with other."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="intersection",
            args=[self._expr, other],
            kwargs={"grid_size": grid_size},
            is_elementwise=True,
        ).pipe(lambda e: cast("GeoExpr", e))

    def difference(
        self,
        other: IntoGeoExprColumn,
        grid_size: float | None = None,
    ) -> GeoExpr:
        """Return the difference of each geometry with other."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="difference",
            args=[self._expr, other],
            kwargs={"grid_size": grid_size},
            is_elementwise=True,
        ).pipe(lambda e: cast("GeoExpr", e))

    def symmetric_difference(
        self,
        other: IntoGeoExprColumn,
        grid_size: float | None = None,
    ) -> GeoExpr:
        """Return the symmetric difference of each geometry with other."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="symmetric_difference",
            args=[self._expr, other],
            kwargs={"grid_size": grid_size},
            is_elementwise=True,
        ).pipe(lambda e: cast("GeoExpr", e))

    # Constructive operations

    def boundary(self) -> GeoExpr:
        """Return the topological boundary of each geometry."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="boundary",
            args=[self._expr],
            is_elementwise=True,
        ).pipe(lambda e: cast("GeoExpr", e))

    def buffer(
        self,
        distance: IntoDecimalExpr,
        quad_segs: int = 8,
        cap_style: Literal["round", "square", "flat"] = "round",
        join_style: Literal["round", "mitre", "bevel"] = "round",
        mitre_limit: float = 5.0,
        single_sided: bool = False,
    ) -> GeoExpr:
        """Return a buffer around each geometry."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="buffer",
            args=[self._expr, distance],
            kwargs={
                "quad_segs": quad_segs,
                "cap_style": cap_style,
                "join_style": join_style,
                "mitre_limit": mitre_limit,
                "single_sided": single_sided,
            },
            is_elementwise=True,
        ).pipe(lambda e: cast("GeoExpr", e))

    def offset_curve(
        self,
        distance: IntoDecimalExpr,
        quad_segs: int = 8,
        join_style: Literal["round", "mitre", "bevel"] = "round",
        mitre_limit: float = 5.0,
    ) -> GeoExpr:
        """Return a line at a given distance of each geometry."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="offset_curve",
            args=[self._expr, distance],
            kwargs={
                "quad_segs": quad_segs,
                "join_style": join_style,
                "mitre_limit": mitre_limit,
            },
            is_elementwise=True,
        ).pipe(lambda e: cast("GeoExpr", e))

    def centroid(self) -> GeoExpr:
        """Return the centroid of each geometry."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="centroid",
            args=[self._expr],
            is_elementwise=True,
        ).pipe(lambda e: cast("GeoExpr", e))

    def center(self) -> GeoExpr:
        """Return the bounding box center of each geometry."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="center",
            args=[self._expr],
            is_elementwise=True,
        ).pipe(lambda e: cast("GeoExpr", e))

    def clip_by_rect(
        self,
        xmin: float,
        ymin: float,
        xmax: float,
        ymax: float,
    ) -> GeoExpr:
        """Clip each geometry by a bounding rectangle."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="clip_by_rect",
            args=[self._expr],
            kwargs={
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
            },
            is_elementwise=True,
        ).pipe(lambda e: cast("GeoExpr", e))

    def convex_hull(self) -> GeoExpr:
        """Return the convex hull of each geometry."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="convex_hull",
            args=[self._expr],
            is_elementwise=True,
        ).pipe(lambda e: cast("GeoExpr", e))

    def concave_hull(self, ratio: float = 0.0, allow_holes: bool = False) -> GeoExpr:
        """Return the concave hull of each geometry."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="concave_hull",
            args=[self._expr],
            kwargs={"ratio": ratio, "allow_holes": allow_holes},
            is_elementwise=True,
        ).pipe(lambda e: cast("GeoExpr", e))

    def segmentize(self, max_segment_length: IntoDecimalExpr) -> GeoExpr:
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="segmentize",
            args=[self._expr, max_segment_length],
            is_elementwise=True,
        ).pipe(lambda e: cast("GeoExpr", e))

    def envelope(self) -> GeoExpr:
        """Return the envelope of each geometry."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="envelope",
            args=[self._expr],
            is_elementwise=True,
        ).pipe(lambda e: cast("GeoExpr", e))

    def extract_unique_points(self) -> GeoExpr:
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="extract_unique_points",
            args=[self._expr],
            is_elementwise=True,
        ).pipe(lambda e: cast("GeoExpr", e))

    def build_area(self) -> GeoExpr:
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="build_area",
            args=[self._expr],
            is_elementwise=True,
        ).pipe(lambda e: cast("GeoExpr", e))

    def make_valid(self) -> GeoExpr:
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="make_valid",
            args=[self._expr],
            is_elementwise=True,
        ).pipe(lambda e: cast("GeoExpr", e))

    def normalize(self) -> GeoExpr:
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="normalize",
            args=[self._expr],
            is_elementwise=True,
        ).pipe(lambda e: cast("GeoExpr", e))

    def node(self) -> GeoExpr:
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="node",
            args=[self._expr],
            is_elementwise=True,
        ).pipe(lambda e: cast("GeoExpr", e))

    def point_on_surface(self) -> GeoExpr:
        """Return a point that intersects of each geometry."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="point_on_surface",
            args=[self._expr],
            is_elementwise=True,
        ).pipe(lambda e: cast("GeoExpr", e))

    def remove_repeated_points(self, tolerance: IntoDecimalExpr = 0.0) -> GeoExpr:
        """Remove the repeated points for each geometry."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="remove_repeated_points",
            args=[self._expr, tolerance],
            is_elementwise=True,
        ).pipe(lambda e: cast("GeoExpr", e))

    def reverse(self) -> GeoExpr:
        """Reverse the coordinates order of each geometry."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="reverse",
            args=[self._expr],
            is_elementwise=True,
        ).pipe(lambda e: cast("GeoExpr", e))

    def simplify(
        self,
        tolerance: IntoDecimalExpr,
        preserve_topology: bool = True,
    ) -> GeoExpr:
        """Simplify each geometry with a given tolerance."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="simplify",
            args=[self._expr, tolerance],
            kwargs={"preserve_topology": preserve_topology},
            is_elementwise=True,
        ).pipe(lambda e: cast("GeoExpr", e))

    def force_2d(self) -> GeoExpr:
        """Force the dimensionality of a geometry to 2D."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="force_2d",
            args=[self._expr],
            is_elementwise=True,
        ).pipe(lambda e: cast("GeoExpr", e))

    def force_3d(self, z: IntoDecimalExpr = 0.0) -> GeoExpr:
        """Force the dimensionality of a geometry to 3D."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="force_3d",
            args=[self._expr, z],
            is_elementwise=True,
        ).pipe(lambda e: cast("GeoExpr", e))

    def flip_coordinates(self) -> GeoExpr:
        """Flip the x and y coordinates of each geometry."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="flip_coordinates",
            args=[self._expr],
            is_elementwise=True,
        ).pipe(lambda e: cast("GeoExpr", e))

    def minimum_rotated_rectangle(self) -> GeoExpr:
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="minimum_rotated_rectangle",
            args=[self._expr],
            is_elementwise=True,
        ).pipe(lambda e: cast("GeoExpr", e))

    def snap(
        self,
        other: IntoGeoExprColumn,
        tolerance: IntoDecimalExpr,
    ) -> GeoExpr:
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="snap",
            args=[self._expr, other, tolerance],
            is_elementwise=True,
        ).pipe(lambda e: cast("GeoExpr", e))

    def shortest_line(self, other: IntoGeoExprColumn) -> GeoExpr:
        """Return the shortest line between each geometry and other."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="shortest_line",
            args=[self._expr, other],
            is_elementwise=True,
        ).pipe(lambda e: cast("GeoExpr", e))

    # Affine tranforms

    def affine_transform(self, matrix: IntoExprColumn | Sequence[float]) -> GeoExpr:
        """Apply a 2D or 3D transformation matrix to the coordinates of each geometry.

        Args:
            matrix:
                The transformation matrix to apply to coordinates. Should contains 6
                elements for a 2D transform or 12 for a 3D transform. The matrix elements
                order should be:
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
            returns_scalar=True,
        ).pipe(lambda e: cast("GeoExpr", e))

    def translate(
        self,
        x: IntoDecimalExpr = 0.0,
        y: IntoDecimalExpr = 0.0,
        z: IntoDecimalExpr = 0.0,
    ) -> GeoExpr:
        matrix = pl.concat_list(
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
            x, y, z,
        ).cast(pl.Array(pl.Float64, 12))  # fmt: off
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="affine_transform",
            args=[self._expr, matrix],
            returns_scalar=True,
        ).pipe(lambda e: cast("GeoExpr", e))

    def rotate(
        self,
        angle: IntoDecimalExpr,
        origin: Literal["center", "centroid"] | Sequence[float] | pl.Expr | pl.Series = "center",
    ) -> GeoExpr:
        _origin = _interpret_origin(self._expr, origin)
        x0 = _origin.list.get(0)
        y0 = _origin.list.get(1)

        angle = wrap_expr(parse_into_expression(angle)).radians()
        cosp = angle.cos()
        sinp = angle.sin()
        matrix = pl.concat_list(
            cosp, -sinp, 0.0,
            sinp, cosp, 0.0,
            0.0, 0.0, 1.0,
            x0 - x0 * cosp + y0 * sinp, y0 - x0 * sinp - y0 * cosp, 0.0,
        ).cast(pl.Array(pl.Float64, 12))  # fmt: off
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="affine_transform",
            args=[self._expr, matrix],
            returns_scalar=True,
        ).pipe(lambda e: cast("GeoExpr", e))

    def scale(
        self,
        x: IntoDecimalExpr = 1.0,
        y: IntoDecimalExpr = 1.0,
        z: IntoDecimalExpr = 1.0,
        origin: Literal["center", "centroid"] | Sequence[float] | pl.Expr | pl.Series = "center",
    ) -> GeoExpr:
        _origin = _interpret_origin(self._expr, origin)
        x0 = _origin.list.get(0)
        y0 = _origin.list.get(1)
        z0 = pl.when(_origin.list.len().ge(3)).then(_origin.list.get(2)).otherwise(0)

        matrix = pl.concat_list(
            x, 0.0, 0.0,
            0.0, y, 0.0,
            0.0, 0.0, z,
            x0 - x0 * x, y0 - y0 * y, z0 - z0 * z,
        ).cast(pl.Array(pl.Float64, 12))  # fmt: off
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="affine_transform",
            args=[self._expr, matrix],
            returns_scalar=True,
        ).pipe(lambda e: cast("GeoExpr", e))

    # Linestring operations

    def interpolate(
        self,
        distance: IntoDecimalExpr,
        normalized: bool = False,
    ) -> GeoExpr:
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="interpolate",
            args=[self._expr, distance],
            kwargs={"normalized": normalized},
            is_elementwise=True,
        ).pipe(lambda e: cast("GeoExpr", e))

    def project(
        self,
        other: IntoGeoExprColumn,
        normalized: bool = False,
    ) -> pl.Expr:
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="project",
            args=[self._expr, other],
            kwargs={"normalized": normalized},
            is_elementwise=True,
        )

    def line_merge(self, directed: bool = False) -> GeoExpr:
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="line_merge",
            args=[self._expr],
            kwargs={"directed": directed},
            is_elementwise=True,
        ).pipe(lambda e: cast("GeoExpr", e))

    def shared_paths(self, other: IntoGeoExprColumn) -> GeoExpr:
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="shared_paths",
            args=[self._expr, other],
            is_elementwise=True,
        ).pipe(lambda e: cast("GeoExpr", e))

    # Aggregations

    def total_bounds(self) -> pl.Expr:
        """Return the total bounds of all geometries."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="total_bounds",
            args=[self._expr],
            returns_scalar=True,
        )

    def multipoint(self) -> GeoExpr:
        """Aggregate Point geometries into a single MultiPoint."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="multipoint",
            args=[self._expr],
            returns_scalar=True,
        ).pipe(lambda e: cast("GeoExpr", e))

    def multilinestring(self) -> GeoExpr:
        """Aggregate LineString geometries into a single MultiLineString."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="multilinestring",
            args=[self._expr],
            returns_scalar=True,
        ).pipe(lambda e: cast("GeoExpr", e))

    def multipolygon(self) -> GeoExpr:
        """Aggregate Polygon geometries into a single MultiPolygon."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="multipolygon",
            args=[self._expr],
            returns_scalar=True,
        ).pipe(lambda e: cast("GeoExpr", e))

    def geometrycollection(self) -> GeoExpr:
        """Aggregate geometries into a single GeometryCollection."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="geometrycollection",
            args=[self._expr],
            returns_scalar=True,
        ).pipe(lambda e: cast("GeoExpr", e))

    def collect(self) -> GeoExpr:
        """Aggregate geometries into a single MultiPart geometry or GeometryCollection."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="collect",
            args=[self._expr],
            returns_scalar=True,
        ).pipe(lambda e: cast("GeoExpr", e))

    def union_all(self, grid_size: float | None = None) -> GeoExpr:
        """Return the union of all geometries."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="union_all",
            args=[self._expr],
            kwargs={"grid_size": grid_size},
            returns_scalar=True,
        ).pipe(lambda e: cast("GeoExpr", e))

    def coverage_union_all(self) -> GeoExpr:
        """Return the coverage union of all geometries."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="coverage_union_all",
            args=[self._expr],
            returns_scalar=True,
        ).pipe(lambda e: cast("GeoExpr", e))

    def intersection_all(self, grid_size: float | None = None) -> GeoExpr:
        """Return the intersection of all geometries."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="intersection_all",
            args=[self._expr],
            kwargs={"grid_size": grid_size},
            returns_scalar=True,
        ).pipe(lambda e: cast("GeoExpr", e))

    def difference_all(self, grid_size: float | None = None) -> GeoExpr:
        """Return the difference of all geometries."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="difference_all",
            args=[self._expr],
            kwargs={"grid_size": grid_size},
            returns_scalar=True,
        ).pipe(lambda e: cast("GeoExpr", e))

    def symmetric_difference_all(self, grid_size: float | None = None) -> GeoExpr:
        """Return the symmetric difference of all geometries."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="symmetric_difference_all",
            args=[self._expr],
            kwargs={"grid_size": grid_size},
            returns_scalar=True,
        ).pipe(lambda e: cast("GeoExpr", e))

    def polygonize(self) -> GeoExpr:
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="polygonize",
            args=[self._expr],
            returns_scalar=True,
        ).pipe(lambda e: cast("GeoExpr", e))

    def voronoi_polygons(
        self,
        tolerance: float = 0.0,
        extend_to: bytes | None = None,
        only_edges: bool = False,
    ) -> GeoExpr:
        """Return a Voronoi diagram of all geometries vertices."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="voronoi_polygons",
            args=[self._expr],
            kwargs={
                "tolerance": tolerance,
                "extend_to": extend_to,
                "only_edges": only_edges,
            },
            returns_scalar=True,
        ).pipe(lambda e: cast("GeoExpr", e))

    def delaunay_triangles(
        self,
        tolerance: float = 0.0,
        only_edges: bool = False,
    ) -> GeoExpr:
        """Return a Delaunay triangulation of all geometries vertices."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="delaunay_triangles",
            args=[self._expr],
            kwargs={
                "tolerance": tolerance,
                "only_edges": only_edges,
            },
            returns_scalar=True,
        ).pipe(lambda e: cast("GeoExpr", e))


def _interpret_origin(
    geom: GeoExpr,
    origin: Literal["center", "centroid"] | Sequence[float] | pl.Expr | pl.Series,
) -> pl.Expr:
    match origin:
        case "center":
            coord = geom.st.center().st.coordinates().list.get(0)
            return coord.cast(pl.List(pl.Float64))
        case "centroid":
            coord = geom.st.centroid().st.coordinates().list.get(0)
            return coord.cast(pl.List(pl.Float64))
        case origin if isinstance(origin, pl.Expr | pl.Series):
            return wrap_expr(parse_into_expression(origin)).cast(pl.List(pl.Float64))
        case _:
            return pl.lit(origin, dtype=pl.List(pl.Float64))
