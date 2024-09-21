from __future__ import annotations

from typing import TYPE_CHECKING, overload

if TYPE_CHECKING:
    from polars import DataFrame, Expr, LazyFrame, Series

    from polars_st.geodataframe import GeoDataFrameNameSpace
    from polars_st.geoexpr import GeoExprNameSpace
    from polars_st.geolazyframe import GeoLazyFrameNameSpace
    from polars_st.geoseries import GeoSeriesNameSpace


@overload
def st(d: Expr) -> GeoExprNameSpace: ...


@overload
def st(d: Series) -> GeoSeriesNameSpace: ...


@overload
def st(d: DataFrame) -> GeoDataFrameNameSpace: ...


@overload
def st(d: LazyFrame) -> GeoLazyFrameNameSpace: ...


def st(
    d: Expr | Series | DataFrame | LazyFrame,
) -> GeoExprNameSpace | GeoSeriesNameSpace | GeoDataFrameNameSpace | GeoLazyFrameNameSpace:
    return d.st  # type: ignore  # noqa: PGH003
