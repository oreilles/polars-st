from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

import polars as pl

IntoExprColumn: TypeAlias = pl.Expr | pl.Series | str
IntoGeoExprColumn: TypeAlias = IntoExprColumn
IntoIntegerExpr: TypeAlias = IntoExprColumn | int
IntoDecimalExpr: TypeAlias = IntoExprColumn | int | float

CoordinatesApply: TypeAlias = Callable[
    [pl.Series, pl.Series, pl.Series | None],
    tuple[pl.Series, pl.Series, pl.Series | None],
]
