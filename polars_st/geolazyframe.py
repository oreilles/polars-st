from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import polars as pl
from polars import Expr, LazyFrame
from polars._utils.parse import parse_into_expression
from polars.api import register_lazyframe_namespace
from polars.datatypes import N_INFER_DEFAULT
from polars.plugins import register_plugin_function

from polars_st.config import Config
from polars_st.geodataframe import GeoDataFrame

if TYPE_CHECKING:
    from polars._typing import (
        FrameInitTypes,
        JoinStrategy,
        JoinValidation,
        Orientation,
        SchemaDefinition,
        SchemaDict,
    )


__all__ = [
    "GeoLazyFrame",
    "GeoLazyFrameNameSpace",
]


class GeoLazyFrame(LazyFrame):
    @property
    def st(self) -> GeoLazyFrameNameSpace: ...

    def __new__(  # noqa: PYI034
        cls,
        data: FrameInitTypes | None = None,
        schema: SchemaDefinition | None = None,
        *,
        schema_overrides: SchemaDict | None = None,
        strict: bool = True,
        orient: Orientation | None = None,
        infer_schema_length: int | None = N_INFER_DEFAULT,
        nan_to_null: bool = False,
    ) -> GeoLazyFrame:
        ldf = GeoDataFrame(
            data=data,
            schema=schema,
            schema_overrides=schema_overrides,
            strict=strict,
            orient=orient,
            infer_schema_length=infer_schema_length,
            nan_to_null=nan_to_null,
        ).lazy()
        return cast(GeoLazyFrame, ldf)


@register_lazyframe_namespace("st")
class GeoLazyFrameNameSpace:
    def __init__(self, lf: LazyFrame) -> None:
        self._lf = lf

    def sjoin(
        self,
        other: LazyFrame,
        on: str | Expr | None = None,
        how: JoinStrategy = "inner",
        predicate: Literal[
            "intersects_bbox",
            "intersects",
            "within",
            "contains",
            "overlaps",
            "crosses",
            "touches",
            "covers",
            "covered_by",
            "contains_properly",
        ] = "intersects",
        *,
        left_on: str | Expr | None = None,
        right_on: str | Expr | None = None,
        suffix: str = "_right",
        validate: JoinValidation = "m:m",
        join_nulls: bool = False,
        coalesce: bool | None = None,
        allow_parallel: bool = True,
        force_parallel: bool = False,
    ) -> LazyFrame:
        """Perform a spatial join operation with another LazyFrame."""
        if not isinstance(other, LazyFrame):
            msg = f"expected `other` join table to be a LazyFrame, not a {type(other).__name__!r}"
            raise TypeError(msg)

        if how == "cross":
            msg = """Use of `how="cross" not supported on sjoin.`"""
            self._lf.join(
                other=other,
                on=on,
                how=how,
                left_on=left_on,
                right_on=right_on,
                suffix=suffix,
                validate=validate,
                join_nulls=join_nulls,
                coalesce=coalesce,
                allow_parallel=allow_parallel,
                force_parallel=force_parallel,
            )
            raise ValueError(msg)

        if on is not None:
            left_expr = on
            right_expr = on
        elif left_on is not None and right_on is not None:
            left_expr = left_on
            right_expr = right_on
        else:
            left_expr = Config.get_geometry_column()
            right_expr = Config.get_geometry_column()

        if (
            parse_into_expression(left_expr).meta_has_multiple_outputs()
            or parse_into_expression(right_expr).meta_has_multiple_outputs()
        ):
            msg = "spatial join expressions should not return multiple output"
            raise ValueError(msg)

        sjoin_index = (
            pl.concat(
                [
                    self._lf.select(_sjoin_geom_left=left_expr),
                    other.select(_sjoin_geom_right=right_expr),
                ],
                how="horizontal",
            )
            .select(
                register_plugin_function(
                    plugin_path=Path(__file__).parent,
                    function_name="sjoin",
                    args=["_sjoin_geom_left", "_sjoin_geom_right"],
                    kwargs={"predicate": predicate},
                    is_elementwise=True,
                ),
            )
            .select(
                _sjoin_index_left=pl.nth(0).struct[0],
                _sjoin_index_right=pl.nth(0).struct[1],
            )
        )

        return (
            sjoin_index.join(
                self._lf.with_row_index("_sjoin_index_left"),
                on="_sjoin_index_left",
                how="full",
                suffix=suffix,
                coalesce=coalesce,
                allow_parallel=allow_parallel,
                force_parallel=force_parallel,
            )
            .join(
                other.with_row_index("_sjoin_index_right"),
                on="_sjoin_index_right",
                how=how,
                validate=validate,
                join_nulls=join_nulls,
                coalesce=coalesce,
                allow_parallel=allow_parallel,
                force_parallel=force_parallel,
            )
            .drop(
                "_sjoin_index_left",
                "_sjoin_index_right",
                "_sjoin_index_left_right",
            )
        )
