from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast, overload

import polars as pl
import polars.selectors as cs
from polars import DataFrame, Expr
from polars.api import register_dataframe_namespace
from polars.datatypes import N_INFER_DEFAULT
from pyogrio import write_arrow

from polars_st._lib import get_crs_from_code
from polars_st.casting import st
from polars_st.config import Config
from polars_st.geometry import GeometryType
from polars_st.geoseries import GeoSeries
from polars_st.selectors import geom

if TYPE_CHECKING:
    from io import BytesIO, IOBase
    from pathlib import Path

    import altair as alt
    import geopandas as gpd
    from altair.vegalite.v5.schema._config import MarkConfigKwds
    from polars._typing import (
        FrameInitTypes,
        JoinStrategy,
        JoinValidation,
        Orientation,
        SchemaDefinition,
        SchemaDict,
    )
    from typing_extensions import Unpack

__all__ = [
    "GeoDataFrame",
    "GeoDataFrameNameSpace",
]


class GeoDataFrame(DataFrame):
    @property
    def st(self) -> GeoDataFrameNameSpace: ...

    def __new__(  # noqa: PYI034
        cls,
        data: FrameInitTypes | None = None,
        schema: SchemaDefinition | None = None,
        *,
        schema_overrides: SchemaDict | None = None,
        strict: bool = True,
        orient: Literal["col", "row"] | None = None,
        infer_schema_length: int | None = N_INFER_DEFAULT,
        nan_to_null: bool = False,
    ) -> GeoDataFrame:
        df = pl.DataFrame(
            data,
            schema,
            schema_overrides=schema_overrides,
            strict=strict,
            orient=orient,
            infer_schema_length=infer_schema_length,
            nan_to_null=nan_to_null,
        )
        geometry_name = Config.get_geometry_column()
        if df.columns == ["column_0"]:
            df = df.rename({"column_0": geometry_name})
        if geometry_name in df.columns:
            df = df.with_columns(GeoSeries(df.get_column(geometry_name)))
        return cast("GeoDataFrame", df)

    def __init__(
        self,
        data: FrameInitTypes | None = None,
        schema: SchemaDefinition | None = None,
        *,
        schema_overrides: SchemaDict | None = None,
        strict: bool = True,
        orient: Orientation | None = None,
        infer_schema_length: int | None = N_INFER_DEFAULT,
        nan_to_null: bool = False,
    ) -> None:
        """Create a new GeoDataFrame.

        A GeoDataFrame is a regular [`polars.DataFrame`](https://docs.pola.rs/api/python/stable/reference/dataframe/index.html)
        with type annotations added for the `st` namespace.

        If a GeoDataFrame is created with a column matching the [`Configuration`][polars_st.Config]
            default geometry column name, that column will be parsed into a GeoSeries.

        Examples:
            >>> gdf = st.GeoDataFrame({
            ...     "geometry": [
            ...         "POINT(0 0)",
            ...         "POINT(1 2)",
            ...     ]
            ... })
            >>> gdf.schema
            Schema({'geometry': Binary})

            >>> gdf = st.GeoDataFrame([
            ...     "POINT(0 0)",
            ...     "POINT(1 2)",
            ... ])
            >>> gdf.schema
            Schema({'geometry': Binary})
        """
        ...


@register_dataframe_namespace("st")
class GeoDataFrameNameSpace:
    def __init__(self, df: DataFrame) -> None:
        self._df = cast("GeoDataFrame", df)

    def sjoin(
        self,
        other: DataFrame,
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
        coalesce: bool | None = None,
    ) -> GeoDataFrame:
        """Perform a spatial join operation with another DataFrame."""
        if not isinstance(other, DataFrame):
            msg = f"expected `other` join table to be a DataFrame, got {type(other).__name__!r}"
            raise TypeError(msg)

        return (
            self._df.lazy()
            .pipe(st)
            .sjoin(
                other=other.lazy(),
                left_on=left_on,
                right_on=right_on,
                on=on,
                how=how,
                predicate=predicate,
                suffix=suffix,
                validate=validate,
                coalesce=coalesce,
            )
            .collect(_eager=True)
            .pipe(lambda df: cast("GeoDataFrame", df))
        )

    def to_wkt(
        self,
        rounding_precision: int | None = 6,
        trim: bool = True,
        output_dimension: Literal[2, 3, 4] = 3,
        old_3d: bool = False,
    ) -> DataFrame:
        """Serialize the DataFrame geometry column as WKT.

        See [`GeoExprNameSpace.to_wkt`][polars_st.GeoExprNameSpace.to_wkt].
        """
        return self._df.with_columns(
            geom().st.to_wkt(
                rounding_precision,
                trim,
                output_dimension,
                old_3d,
            ),
        )

    def to_ewkt(
        self,
        rounding_precision: int | None = 6,
        trim: bool = True,
        output_dimension: Literal[2, 3, 4] = 3,
        old_3d: bool = False,
    ) -> DataFrame:
        """Serialize the DataFrame geometry column as EWKT.

        See [`GeoExprNameSpace.to_ewkt`][polars_st.GeoExprNameSpace.to_ewkt].
        """
        return self._df.with_columns(
            geom().st.to_ewkt(
                rounding_precision,
                trim,
                output_dimension,
                old_3d,
            ),
        )

    def to_wkb(
        self,
        output_dimension: Literal[2, 3, 4] = 3,
        byte_order: Literal[0, 1] | None = None,
        include_srid: bool = False,
    ) -> DataFrame:
        """Serialize the DataFrame geometry column as WKB.

        See [`GeoExprNameSpace.to_wkb`][polars_st.GeoExprNameSpace.to_wkb].
        """
        return self._df.with_columns(
            geom().st.to_wkb(
                output_dimension,
                byte_order,
                include_srid,
            ),
        )

    def to_geojson(self, indent: int | None = None) -> DataFrame:
        """Serialize the DataFrame geometry column as GeoJSON.

        See [`GeoExprNameSpace.to_geojson`][polars_st.GeoExprNameSpace.to_geojson].
        """
        return self._df.with_columns(geom().st.to_geojson(indent))

    def to_shapely(self) -> DataFrame:
        """Convert the DataFrame geometry column to a shapely representation.

        See [`GeoExprNameSpace.to_shapely`][polars_st.GeoExprNameSpace.to_shapely].
        """
        return self._df.with_columns(geom().st.to_shapely())

    def to_dict(self) -> DataFrame:
        """Convert the DataFrame geometry column to a GeoJSON-like Python [`dict`][] representation.

        See [`GeoExprNameSpace.to_dict`][polars_st.GeoExprNameSpace.to_dict].
        """
        return self._df.with_columns(geom().st.to_dict())

    def to_dicts(self) -> list[dict[str, Any]]:
        """Convert every row to a Python dictionary representation of a GeoJSON Feature."""
        return self._df.select(
            type=pl.lit("Feature"),
            geometry=geom().st.to_dict(),
            properties=pl.struct(cs.exclude(geom())) if len(self._df.columns) > 1 else None,
        ).to_dicts()

    def to_geopandas(
        self,
        *,
        use_pyarrow_extension_array: bool = False,
        **kwargs: Any,
    ) -> gpd.GeoDataFrame:
        """Convert this DataFrame to a geopandas GeoDataFrame."""
        import geopandas as gpd

        return gpd.GeoDataFrame(
            self.to_shapely().to_pandas(
                use_pyarrow_extension_array=use_pyarrow_extension_array,
                **kwargs,
            ),
        )

    @property
    def __geo_interface__(self) -> dict:
        """Return a GeoJSON FeatureCollection [`dict`][] representation of the DataFrame.

        Examples:
            >>> gdf = st.GeoDataFrame({
            ...     "geometry": ["POINT(0 0)", "POINT(1 2)"],
            ...     "name": ["Alice", "Bob"]
            ... })
            >>> interface = gdf.st.__geo_interface__
            >>> pprint.pp(interface)
            {'type': 'FeatureCollection',
             'features': [{'type': 'Feature',
                           'geometry': {'type': 'Point', 'coordinates': [0.0, 0.0]},
                           'properties': {'name': 'Alice'}},
                          {'type': 'Feature',
                           'geometry': {'type': 'Point', 'coordinates': [1.0, 2.0]},
                           'properties': {'name': 'Bob'}}]}
        """
        return {
            "type": "FeatureCollection",
            "features": self.to_dicts(),
        }

    def write_file(
        self,
        path: str | BytesIO,
        layer: str | None = None,
        driver: str | None = None,
        encoding: str | None = None,
        append: bool = False,
        dataset_metadata: dict | None = None,
        layer_metadata: dict | None = None,
        metadata: dict | None = None,
        dataset_options: dict | None = None,
        layer_options: dict | None = None,
        **kwargs: dict[str, Any],
    ) -> None:
        """Write the GeoDataFrame to an OGR supported file format.

        Args:
            path:
                path to output file on writeable file system or an io.BytesIO object to
                allow writing to memory
                NOTE: support for writing to memory is limited to specific drivers.
            layer:
                layer name to create.  If writing to memory and layer name is not
                provided, it layer name will be set to a UUID4 value.
            driver:
                The OGR format driver used to write the vector file. By default attempts
                to infer driver from path.  Must be provided to write to memory.
                The available drivers can be listed by calling:
                ```py
                >>> import pyogrio
                >>> pyogrio.list_drivers() # doctest: +SKIP
                {..., 'GeoJSON': 'rw', 'GeoJSONSeq': 'rw',...}
                ```

            encoding:
                Only used for the .dbf file of ESRI Shapefiles. If not specified,
                uses the default locale.
            append:
                If True, the data source specified by path already exists, and the
                driver supports appending to an existing data source, will cause the
                data to be appended to the existing records in the data source.  Not
                supported for writing to in-memory files.
                NOTE: append support is limited to specific drivers and GDAL versions.
            dataset_metadata:
                Metadata to be stored at the dataset level in the output file; limited
                to drivers that support writing metadata, such as GPKG, and silently
                ignored otherwise. Keys and values must be strings.
            layer_metadata:
                Metadata to be stored at the layer level in the output file; limited to
                drivers that support writing metadata, such as GPKG, and silently
                ignored otherwise. Keys and values must be strings.
            metadata:
                alias of layer_metadata
            dataset_options:
                Dataset creation options (format specific) passed to OGR. Specify as
                a key-value dictionary.
            layer_options:
                Layer creation options (format specific) passed to OGR. Specify as
                a key-value dictionary.
            **kwargs:
                Additional driver-specific dataset or layer creation options passed
                to OGR. pyogrio will attempt to automatically pass those keywords
                either as dataset or as layer creation option based on the known
                options for the specific driver. Alternatively, you can use the
                explicit `dataset_options` or `layer_options` keywords to manually
                do this (for example if an option exists as both dataset and layer
                option).
        """
        geometry_type = self._df.select(geom().st.geometry_type().unique().drop_nulls()).to_series()
        geometry_type = (
            GeometryType(geometry_type[0]).name if len(geometry_type) == 1 else "Unknown"
        )

        srids = self._df.select(geom().st.srid().unique().drop_nulls())
        crs = None
        if len(srids) == 1 and (srid := srids[0, 0]) != 0:
            crs = get_crs_from_code(srid)
            if crs is None:
                msg = f"Couldn't find CRS information for SRID {srid}"
                raise ValueError(msg)
        elif len(srids) > 1:
            msg = "DataFrame with mixed SRIDs aren't supported"
            raise ValueError(msg)
        else:
            crs = None

        write_arrow(
            self._df.to_arrow(),
            path=path,
            layer=layer,
            driver=driver,
            geometry_name=Config.get_geometry_column(),
            geometry_type=geometry_type,
            crs=crs,
            encoding=encoding,
            append=append,
            dataset_metadata=dataset_metadata,
            layer_metadata=layer_metadata,
            metadata=metadata,
            dataset_options=dataset_options,
            layer_options=layer_options,
            **kwargs,
        )

    @overload
    def write_geojson(self, file: None = None) -> str: ...

    @overload
    def write_geojson(self, file: IOBase | str | Path) -> None: ...

    def write_geojson(self, file: IOBase | str | Path | None = None) -> str | None:
        r"""Serialize to GeoJSON FeatureCollection representation.

        The result will be invalid if the geometry column contains different geometry types.

        Examples:
            >>> gdf = st.GeoDataFrame({
            ...     "geometry": ["POINT(0 0)", "POINT(1 2)"],
            ...     "name": ["Alice", "Bob"]
            ... })
            >>> geojson = gdf.st.write_geojson()
            >>> print(geojson)
            {"type":"FeatureCollection","features":[{"properties":{"name":"Alice"},"geometry":{"type":"Point","coordinates":[0.0,0.0]}},{"properties":{"name":"Bob"},"geometry":{"type":"Point","coordinates":[1.0,2.0]}}]}
            <BLANKLINE>
        """
        return (
            self._df.select(
                properties=pl.struct(cs.exclude(geom())) if len(self._df.columns) > 1 else None,
                geometry=geom().st.to_geojson().str.json_decode(),
            )
            .group_by(0)
            .agg(
                type=pl.lit("FeatureCollection"),
                features=pl.struct("properties", "geometry"),
            )
            .select("type", "features")
            .write_ndjson(file)
        )

    @overload
    def write_ndgeojson(self, file: None = None) -> str: ...

    @overload
    def write_ndgeojson(self, file: IOBase | str | Path) -> None: ...

    def write_ndgeojson(self, file: IOBase | str | Path | None = None) -> str | None:
        """Serialize to newline-delimited GeoJSON representation.

        The result will be invalid if the geometry column contains different geometry types.

        Examples:
            >>> gdf = st.GeoDataFrame({
            ...     "geometry": ["POINT(0 0)", "POINT(1 2)"],
            ...     "name": ["Alice", "Bob"]
            ... })
            >>> ndgeojson = gdf.st.write_ndgeojson()
            >>> print(ndgeojson)
            {"properties":{"name":"Alice"},"geometry":{"type":"Point","coordinates":[0.0,0.0]}}
            {"properties":{"name":"Bob"},"geometry":{"type":"Point","coordinates":[1.0,2.0]}}
            <BLANKLINE>
        """
        return self._df.select(
            properties=pl.struct(cs.exclude(geom())) if len(self._df.columns) > 1 else None,
            geometry=geom().st.to_geojson().str.json_decode(),
        ).write_ndjson(file)

    def plot(self, **kwargs: Unpack[MarkConfigKwds]) -> alt.Chart:
        """Draw map plot.

        Polars does not implement plotting logic itself but instead defers to
        [`Altair`](https://altair-viz.github.io/).

        `df.st.plot(**kwargs)` is shorthand for
        `alt.Chart({"values": df.st.to_dicts()}).mark_geoshape(**kwargs).interactive()`. Please read
        Altair [GeoShape](https://altair-viz.github.io/user_guide/marks/geoshape.html) documentation
        for available options.

        Please note that the dataframe will be converted to a GeoJSON FeatureCollection, so
        columns will need to be prefixed with `properties.` for access in Altair functions.

        Examples:
            >>> url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
            >>> plot = (
            ...     st.read_file(url)
            ...     .with_columns(st.simplify(tolerance=1))
            ...     .st.plot()
            ...     .encode(color="properties.CONTINENT:N")
            ...     .configure_legend(title=None)
            ...     .properties(height=150)
            ... )

            >>> import altair as alt
            >>> df = st.GeoDataFrame({
            ... "color": ["red","yellow", "blue"],
            ... "geometry": [
            ...     "POLYGON((0 0, 0 2, 2 2, 2 0, 0 0))",
            ...     "POLYGON((0 0, 1 2, 2 0, 0 0))",
            ...     "POINT(2 1)"
            ... ]})
            >>> plot = (
            ...     df.st.plot(blend="difference")
            ...     .encode(fill=alt.Color("properties.color:N", scale=None))
            ...     .project("identity", reflectY=True, pointRadius=100)
            ...     .properties(height=200)
            ... )
        """
        import altair as alt

        chart = alt.Chart({"values": self.to_dicts()})
        return chart.mark_geoshape(**kwargs).interactive()
