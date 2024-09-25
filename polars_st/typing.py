from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeAlias, Union

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import numpy as np
    import pandas as pd
    import polars as pl
    import pyarrow as pa
    from polars._typing import (
        ArrowArrayExportable,
        ArrowStreamExportable,
    )

    ArrayLike = Union[  # noqa: UP007
        Sequence[Any],
        pl.Series,
        pa.Array,
        pa.ChunkedArray,
        np.ndarray[Any, Any],
        pd.Series[Any],
        pd.DatetimeIndex,
        ArrowArrayExportable,
        ArrowStreamExportable,
    ]

    IntoExprColumn: TypeAlias = pl.Expr | pl.Series | str
    IntoGeoExprColumn: TypeAlias = IntoExprColumn
    IntoIntegerExpr: TypeAlias = IntoExprColumn | int
    IntoDecimalExpr: TypeAlias = IntoExprColumn | int | float

    CoordinatesApply: TypeAlias = Callable[
        [pl.Series, pl.Series, pl.Series | None],
        tuple[pl.Series, pl.Series, pl.Series | None],
    ]
