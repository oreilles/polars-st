import polars as pl

from polars_st.geoseries import GeoSeries
from polars_st.typing import CoordinatesApply

__version__: str

def get_crs_from_code(srid: int) -> str | None: ...
def get_crs_authority(definition: str) -> tuple[str, str] | None: ...
def apply_coordinates(series: pl.Series, transform: CoordinatesApply) -> GeoSeries: ...
