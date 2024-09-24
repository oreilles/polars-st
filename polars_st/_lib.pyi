import polars as pl

from polars_st.geoseries import GeoSeries
from polars_st.typing import CoordinatesApply

__version__: str

def get_crs_auth_code(definition: str) -> tuple[str, str]: ...
def apply_coordinates(series: pl.Series, transform: CoordinatesApply) -> GeoSeries: ...
