import polars as pl

from polars_st.geoseries import GeoSeries
from polars_st.typing import CoordinatesApply

__version__: str

def apply_coordinates(series: pl.Series, transform: CoordinatesApply) -> GeoSeries: ...
