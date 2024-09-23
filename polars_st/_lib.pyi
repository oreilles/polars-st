from collections.abc import Callable

import polars as pl

__version__: str

def apply_coordinates(
    series: pl.Series,
    transform: Callable[[float, float, float | None], tuple[float, float, float | None]],
) -> tuple[str, str]: ...
