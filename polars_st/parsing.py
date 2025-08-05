from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

import polars as pl
from polars._utils.parse import parse_into_expression
from polars._utils.wrap import wrap_expr
from polars.plugins import register_plugin_function

if TYPE_CHECKING:
    from polars._typing import IntoExprColumn

    from polars_st.geoexpr import GeoExpr
    from polars_st.typing import IntoIntegerExpr


__all__ = [
    "circularstring",
    "circularstring",
    "from_ewkt",
    "from_geojson",
    "from_shapely",
    "from_wkb",
    "from_wkt",
    "linestring",
    "multilinestring",
    "multipoint",
    "point",
    "polygon",
    "rectangle",
]


def point(coords: IntoExprColumn, srid: IntoIntegerExpr = 0) -> GeoExpr:
    """Create Point geometries from coordinates.

    Examples:
        >>> df = pl.DataFrame({
        ...     "coords": [
        ...          [0, 1],
        ...          [2, 3],
        ...     ]
        ... })
        >>> df = df.select(geometry=st.point("coords"))
        >>> df.st.to_wkt()
        shape: (2, 1)
        ┌─────────────┐
        │ geometry    │
        │ ---         │
        │ str         │
        ╞═════════════╡
        │ POINT (0 1) │
        │ POINT (2 3) │
        └─────────────┘

        >>> df = pl.DataFrame({
        ...     "x": [0, 1],
        ...     "y": [0, 2],
        ...     "z": [0, 3],
        ... })
        >>> df = df.select(geometry=st.point(pl.concat_arr("x", "y", "z")))
        >>> df.st.to_wkt()
        shape: (2, 1)
        ┌─────────────────┐
        │ geometry        │
        │ ---             │
        │ str             │
        ╞═════════════════╡
        │ POINT Z (0 0 0) │
        │ POINT Z (1 2 3) │
        └─────────────────┘
    """
    return register_plugin_function(
        plugin_path=Path(__file__).parent,
        function_name="point",
        args=[coords, srid],
        is_elementwise=True,
    ).pipe(lambda e: cast("GeoExpr", e))


def multipoint(coords: IntoExprColumn, srid: IntoIntegerExpr = 0) -> GeoExpr:
    """Create MultiPoint geometries from list of coordinates.

    Examples:
        >>> df = pl.DataFrame({
        ...     "coords": [
        ...          [[0, 1],[2, 3]],
        ...          [[4, 5],[6, 7]],
        ...     ]
        ... })
        >>> df = df.select(geometry=st.multipoint("coords"))
        >>> df.st.to_wkt()
        shape: (2, 1)
        ┌───────────────────────────┐
        │ geometry                  │
        │ ---                       │
        │ str                       │
        ╞═══════════════════════════╡
        │ MULTIPOINT ((0 1), (2 3)) │
        │ MULTIPOINT ((4 5), (6 7)) │
        └───────────────────────────┘
    """
    return register_plugin_function(
        plugin_path=Path(__file__).parent,
        function_name="multipoint",
        args=[coords, srid],
        is_elementwise=True,
    ).pipe(lambda e: cast("GeoExpr", e))


def linestring(coords: IntoExprColumn, srid: IntoIntegerExpr = 0) -> GeoExpr:
    """Create LineString geometries from lists of coordinates.

    Examples:
        >>> df = pl.DataFrame({
        ...     "coords": [
        ...         [[0, 1], [2, 3], [4, 5]]
        ...     ],
        ... })
        >>> df = df.select(geometry=st.linestring("coords"))
        >>> df.st.to_wkt()
        shape: (1, 1)
        ┌────────────────────────────┐
        │ geometry                   │
        │ ---                        │
        │ str                        │
        ╞════════════════════════════╡
        │ LINESTRING (0 1, 2 3, 4 5) │
        └────────────────────────────┘

        >>> df = pl.DataFrame({
        ...     "idx": [0, 0, 1, 1],
        ...     "x": [0, 1, 3, 5],
        ...     "y": [0, 2, 4, 6],
        ... })
        >>> df = df.group_by("idx").agg(coords=pl.concat_list("x", "y"))
        >>> df = df.select("idx", geometry=st.linestring("coords"))
        >>> df.sort("idx").st.to_wkt()
        shape: (2, 2)
        ┌─────┬───────────────────────┐
        │ idx ┆ geometry              │
        │ --- ┆ ---                   │
        │ i64 ┆ str                   │
        ╞═════╪═══════════════════════╡
        │ 0   ┆ LINESTRING (0 0, 1 2) │
        │ 1   ┆ LINESTRING (3 4, 5 6) │
        └─────┴───────────────────────┘
    """
    return register_plugin_function(
        plugin_path=Path(__file__).parent,
        function_name="linestring",
        args=[coords, srid],
        is_elementwise=True,
    ).pipe(lambda e: cast("GeoExpr", e))


def circularstring(coords: IntoExprColumn, srid: IntoIntegerExpr = 0) -> GeoExpr:
    """Create CircularString geometries from lists of coordinates.

    Examples:
        >>> df = pl.DataFrame({
        ...     "coords": [
        ...         [[0, 1], [2, 3], [4, 5]]
        ...     ],
        ... })
        >>> df = df.select(geometry=st.circularstring("coords"))
        >>> df.st.to_wkt()
        shape: (1, 1)
        ┌────────────────────────────────┐
        │ geometry                       │
        │ ---                            │
        │ str                            │
        ╞════════════════════════════════╡
        │ CIRCULARSTRING (0 1, 2 3, 4 5) │
        └────────────────────────────────┘
    """
    return register_plugin_function(
        plugin_path=Path(__file__).parent,
        function_name="circularstring",
        args=[coords, srid],
        is_elementwise=True,
    ).pipe(lambda e: cast("GeoExpr", e))


def multilinestring(coords: IntoExprColumn, srid: IntoIntegerExpr = 0) -> GeoExpr:
    """Create MultiLineString geometries from lists of lists of coordinates.

    Examples:
        >>> df = pl.DataFrame({
        ...     "coords": [
        ...         [[[1, 2], [3, 4]],[[5, 6], [7, 8]]]
        ...     ]
        ... })
        >>> df = df.select(geometry=st.multilinestring("coords"))
        >>> df.st.to_wkt()
        shape: (1, 1)
        ┌─────────────────────────────────┐
        │ geometry                        │
        │ ---                             │
        │ str                             │
        ╞═════════════════════════════════╡
        │ MULTILINESTRING ((1 2, 3 4), (… │
        └─────────────────────────────────┘
    """
    return register_plugin_function(
        plugin_path=Path(__file__).parent,
        function_name="multilinestring",
        args=[coords, srid],
        is_elementwise=True,
    ).pipe(lambda e: cast("GeoExpr", e))


def polygon(coords: IntoExprColumn, srid: IntoIntegerExpr = 0) -> GeoExpr:
    """Create Polygon geometries from lists of lists of coordinates.

    Examples:
        >>> df = pl.DataFrame({
        ...     "coords": [
        ...         [[[0, 0], [2, 4], [4, 0], [0, 0]]]
        ...     ]
        ... })
        >>> df = df.select(geometry=st.polygon("coords"))
        >>> df.st.to_wkt()
        shape: (1, 1)
        ┌────────────────────────────────┐
        │ geometry                       │
        │ ---                            │
        │ str                            │
        ╞════════════════════════════════╡
        │ POLYGON ((0 0, 2 4, 4 0, 0 0)) │
        └────────────────────────────────┘
    """
    return register_plugin_function(
        plugin_path=Path(__file__).parent,
        function_name="polygon",
        args=[coords, srid],
        is_elementwise=True,
    ).pipe(lambda e: cast("GeoExpr", e))


def rectangle(bounds: IntoExprColumn, srid: IntoIntegerExpr = 0) -> GeoExpr:
    """Create Polygon geometries from bounds.

    Examples:
        >>> df = pl.DataFrame({
        ...     "bounds": [
        ...         [0.0, 0.0, 1.0, 2.0],
        ...         [5.0, 6.0, 7.0, 8.0],
        ...     ]
        ... })
        >>> df = df.select(geometry=st.rectangle("bounds"))
        >>> df.st.to_wkt()
        shape: (2, 1)
        ┌─────────────────────────────────┐
        │ geometry                        │
        │ ---                             │
        │ str                             │
        ╞═════════════════════════════════╡
        │ POLYGON ((0 0, 1 0, 1 2, 0 2, … │
        │ POLYGON ((5 6, 7 6, 7 8, 5 8, … │
        └─────────────────────────────────┘
    """
    return register_plugin_function(
        plugin_path=Path(__file__).parent,
        function_name="rectangle",
        args=[bounds, srid],
        is_elementwise=True,
    ).pipe(lambda e: cast("GeoExpr", e))


def from_wkb(expr: IntoExprColumn) -> GeoExpr:
    """Parse geometries from Well-Known Binary (WKB) representation.

    Examples:
        >>> df = pl.read_database(
        ...     query="SELECT ST_AsEWKB(geom) AS geometry FROM test_data",
        ...     connection=user_conn,
        ... ) # doctest: +SKIP
        >>> gdf = df.select(st.from_wkb("geometry")) # doctest: +SKIP
    """
    return register_plugin_function(
        plugin_path=Path(__file__).parent,
        function_name="from_wkb",
        args=[expr],
        is_elementwise=True,
    ).pipe(lambda e: cast("GeoExpr", e))


def from_wkt(expr: IntoExprColumn) -> GeoExpr:
    """Parse geometries from Well-Known Text (WKT) representation.

    Examples:
        >>> df = pl.Series("geometry", [
        ...     "POINT(0 0)",
        ...     "POINT(1 2)",
        ... ]).to_frame()
        >>> gdf = df.select(st.from_wkt("geometry"))
    """
    return register_plugin_function(
        plugin_path=Path(__file__).parent,
        function_name="from_wkt",
        args=[expr],
        is_elementwise=True,
    ).pipe(lambda e: cast("GeoExpr", e))


def from_ewkt(expr: IntoExprColumn) -> GeoExpr:
    """Parse geometries from Extended Well-Known Text (EWKT) representation.

    Examples:
        >>> df = pl.Series("geometry", [
        ...     "SRID=4326;POINT(0 0)",
        ...     "SRID=3857;POINT(1 2)",
        ... ]).to_frame()
        >>> gdf = df.select(st.from_ewkt("geometry"))
    """
    return register_plugin_function(
        plugin_path=Path(__file__).parent,
        function_name="from_ewkt",
        args=[expr],
        is_elementwise=True,
    ).pipe(lambda e: cast("GeoExpr", e))


def from_geojson(expr: IntoExprColumn) -> GeoExpr:
    """Parse geometries from GeoJSON representation.

    Examples:
        >>> df = pl.Series("geometry", [
        ...     '{"type": "Point", "coordinates": [0, 0]}',
        ...     '{"type": "Point", "coordinates": [1, 2]}',
        ... ]).to_frame()
        >>> gdf = df.select(st.from_geojson("geometry"))
        >>> gdf.st.to_wkt()
        shape: (2, 1)
        ┌─────────────┐
        │ geometry    │
        │ ---         │
        │ str         │
        ╞═════════════╡
        │ POINT (0 0) │
        │ POINT (1 2) │
        └─────────────┘
    """
    return register_plugin_function(
        plugin_path=Path(__file__).parent,
        function_name="from_geojson",
        args=[expr],
        is_elementwise=True,
    ).pipe(lambda e: cast("GeoExpr", e))


def from_shapely(expr: IntoExprColumn) -> GeoExpr:
    """Parse geometries from shapely objects.

    Examples:
        >>> import shapely
        >>> df = pl.Series("geometry", [
        ...     shapely.Point(0, 0),
        ...     shapely.Point(1, 2),
        ... ], dtype=pl.Object).to_frame()
        >>> df.select(st.from_shapely("geometry")).st.to_wkt()
        shape: (2, 1)
        ┌─────────────┐
        │ geometry    │
        │ ---         │
        │ str         │
        ╞═════════════╡
        │ POINT (0 0) │
        │ POINT (1 2) │
        └─────────────┘
    """
    import shapely

    expr = wrap_expr(parse_into_expression(expr))
    res = expr.map_batches(
        lambda s: pl.Series(
            s.name,
            list(shapely.to_wkb(s.to_numpy(), include_srid=True)),
            pl.Binary,
        ),
        return_dtype=pl.Binary,
        is_elementwise=True,
    )
    return cast("GeoExpr", res)
