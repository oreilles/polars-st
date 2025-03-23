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
    from polars_st.typing import IntoDecimalExpr


__all__ = [
    "from_ewkt",
    "from_geojson",
    "from_shapely",
    "from_wkb",
    "from_wkt",
    "from_xy",
]


def from_wkb(expr: IntoExprColumn) -> GeoExpr:
    """Parse geometries from Well-Known Binary (WKB) representation.

    Examples:
        >>> df = pl.read_database(
        ...     query="SELECT ST_AsEWKB(geom) AS geometry FROM test_data",
        ...     connection=user_conn,
        ... ) # doctest: +SKIP
        >>> gdf = df.select(st.from_wkb("geometry")) # doctest: +SKIP
    """
    return cast("GeoExpr", wrap_expr(parse_into_expression(expr)))


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
    expr = wrap_expr(parse_into_expression(expr))
    s = expr.str.extract_groups(r"^(SRID=(.*);)?(.+)$")
    wkt = s.struct[2]
    srid = s.struct[1].replace(dict.fromkeys((None, ""), "0"))
    return from_wkt(wkt).st.set_srid(srid)


def from_geojson(expr: IntoExprColumn) -> GeoExpr:
    """Parse geometries from GeoJSON representation.

    Examples:
        >>> df = pl.Series("geometry", [
        ...     '{"type": "Point", "coordinates": [0, 0]}',
        ...     '{"type": "Point", "coordinates": [1, 2]}',
        ... ]).to_frame()
        >>> gdf = df.select(st.from_geojson("geometry"))
    """
    return register_plugin_function(
        plugin_path=Path(__file__).parent,
        function_name="from_geojson",
        args=[expr],
        is_elementwise=True,
    ).pipe(lambda e: cast("GeoExpr", e))


def from_xy(
    x: IntoDecimalExpr,
    y: IntoDecimalExpr,
    z: IntoDecimalExpr | None = None,
) -> GeoExpr:
    """Create points from x, y (, z) coordinates.

    Examples:
        >>> df = pl.DataFrame({
        ...     "x": [0, 1],
        ...     "y": [0, 2],
        ... })
        >>> gdf = df.select(st.from_xy("x", "y"))
    """
    return register_plugin_function(
        plugin_path=Path(__file__).parent,
        function_name="from_xy",
        args=pl.struct(x=x, y=y) if z is None else pl.struct(x=x, y=y, z=z),
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
        >>> gdf = df.select(st.from_shapely("geometry"))
    """
    import shapely

    expr = wrap_expr(parse_into_expression(expr))
    res = expr.map_batches(
        lambda s: pl.Series(s.name, list(shapely.to_wkb(s.to_numpy(), include_srid=True))),
        return_dtype=pl.Binary,
        is_elementwise=True,
    )
    return cast("GeoExpr", res)
