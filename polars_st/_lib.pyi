from collections.abc import Callable

import polars as pl

__version__: str

def apply_coordinates(
    series: pl.Series,
    transform: Callable[
        [pl.Series, pl.Series, pl.Series | None],
        tuple[pl.Series, pl.Series, pl.Series | None],
    ],
) -> tuple[str, str]: ...
