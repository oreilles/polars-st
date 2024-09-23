from __future__ import annotations

import contextlib
from typing import cast

from pydantic import BaseModel, ConfigDict
from typing_extensions import Self

__all__ = ["Config"]


class ConfigValues(BaseModel):
    geometry_column: str = "geometry"

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
    )


CONFIG = ConfigValues()


class Config(contextlib.ContextDecorator):
    @classmethod
    def set_geometry_column(cls, name: str | None = None) -> type[Self]:
        if name is not None:
            CONFIG.geometry_column = name
        else:
            CONFIG.geometry_column = ConfigValues.model_fields["geometry_column"].default
        return cls

    @classmethod
    def get_geometry_column(cls) -> str:
        return CONFIG.geometry_column

    def __init__(
        self,
        geometry_column: str | None = None,
    ) -> None:
        """Configuration object for `polars-st`.

        Args:
            geometry_column: Default geometry column name. Using `None` will reset this value
                to the built-in default `"geometry"`.

        Examples:
            Use as a context manager:

            >>> gdf = st.GeoSeries("my_geometry", ["POINT(1 2)"]).to_frame()
            >>> with st.Config(geometry_column="my_geometry"):
            ...     x = gdf.select(st.x())
            >>> x.schema
            Schema({'my_geometry': Float64})

            Use as a context decorator:

            >>> @st.Config(geometry_column="my_geometry")
            ... def get_my_bounds(gdf):
            ...     return gdf.select(st.bounds())
            >>> gdf = st.GeoSeries("my_geometry", ["POINT(1 2)"]).to_frame()
            >>> bounds = get_my_bounds(gdf)
            >>> bounds.schema
            Schema({'my_geometry': Array(Float64, shape=(4,))})

            Use as a global configuration object:

            >>> st.Config.set_geometry_column("my_geometry")
            <class 'polars_st.config.Config'>
            >>> gdf = st.GeoSeries(["POINT(1 2)"]).to_frame()
            >>> gdf.schema
            Schema({'my_geometry': Binary})
        """
        if geometry_column is None:
            geometry_column = ConfigValues.model_fields["geometry_column"].default

        self._config = ConfigValues(geometry_column=cast(str, geometry_column))

    def __enter__(self) -> Self:
        """Support setting temporary Config options that are reset on scope exit."""
        self._original_config = CONFIG.model_copy(deep=True)
        CONFIG.__dict__.update(self._config.model_dump())
        return self

    def __exit__(self, *exc) -> None:  # noqa: ANN002
        """Reset any Config options that were set within the scope."""
        CONFIG.__dict__.update(self._original_config.model_dump())
