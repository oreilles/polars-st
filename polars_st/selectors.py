from __future__ import annotations

from typing import TYPE_CHECKING, cast

import polars as pl

from polars_st.config import Config

if TYPE_CHECKING:
    from collections.abc import Iterable

    from polars_st.geoexpr import GeoExpr


__all__ = [
    "element",
    "geom",
]


def geom(name: str | Iterable[str] | None = None, *more_names: str) -> GeoExpr:
    """Create a geometry column expression.

    Alias for
        [`polars.col`](https://docs.pola.rs/api/python/stable/reference/expressions/col.html).
        If no name is given, the default from [`st.Config`][polars_st.Config] will be used.

    Args:
        name: The name or datatype of the geometry column(s) to represent. Accepts regular
            expression input. Regular expressions should start with `^` and end with `$`.
        more_names: Additional names or datatypes of columns to represent, specified as positional
            arguments.

    Examples:
        Pass a single column name to represent that column:

        >>> gdf = st.GeoSeries("my_geom", ["POINT(0 0)"]).to_frame()
        >>> gdf.select(st.geom("my_geom").st.to_wkt())
        shape: (1, 1)
        ┌─────────────┐
        │ my_geom     │
        │ ---         │
        │ str         │
        ╞═════════════╡
        │ POINT (0 0) │
        └─────────────┘

        Call `geom` without a column name to use the default:

        >>> gdf = st.GeoDataFrame([
        ...     "POINT(0 0)",
        ...     "POINT(1 2)",
        ... ])
        >>> gdf.select(st.geom().st.union_all().st.to_wkt())
        shape: (1, 1)
        ┌───────────────────────────┐
        │ geometry                  │
        │ ---                       │
        │ str                       │
        ╞═══════════════════════════╡
        │ MULTIPOINT ((0 0), (1 2)) │
        └───────────────────────────┘
    """
    columns = pl.col(Config.get_geometry_column()) if name is None else pl.col(name, *more_names)
    return cast("GeoExpr", columns)


def element() -> GeoExpr:
    """Alias for [`polars.element`](https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.element.html).

    Examples:
        >>> gdf = st.GeoDataFrame([
        ...     "MULTIPOINT ((0 0), (1 2))"
        ... ])
        >>> gdf.select(st.parts().list.eval(st.element().st.to_wkt()))
        shape: (1, 1)
        ┌────────────────────────────────┐
        │ geometry                       │
        │ ---                            │
        │ list[str]                      │
        ╞════════════════════════════════╡
        │ ["POINT (0 0)", "POINT (1 2)"] │
        └────────────────────────────────┘
    """
    return cast("GeoExpr", pl.element())
