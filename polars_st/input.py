from __future__ import annotations

from typing import TYPE_CHECKING, cast, overload

import polars as pl
from polars.dependencies import pandas as pd
from pyogrio import read_arrow

from polars_st.casting import st
from polars_st.config import Config
from polars_st.selectors import geom
from polars_st.utils.srid import get_crs_srid_or_warn

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import geopandas as gpd
    from polars._typing import SchemaDict

    from polars_st.geodataframe import GeoDataFrame
    from polars_st.geoseries import GeoSeries


__all__ = [
    "from_geopandas",
    "read_file",
]


def read_file(
    path_or_buffer: Path | str | bytes,
    /,
    layer: int | str | None = None,
    encoding: str | None = None,
    columns: Sequence[str] | None = None,
    read_geometry: bool = True,
    force_2d: bool = False,
    skip_features: int = 0,
    max_features: int | None = None,
    where: str | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    fids: Sequence[int] | None = None,
    sql: str | None = None,
    sql_dialect: str | None = None,
    return_fids: bool = False,
) -> GeoDataFrame:
    """Read OGR data source into a GeoDataFrame.

    IMPORTANT: non-linear geometry types (e.g., MultiSurface) are converted
    to their linear approximations.

    Args:
        path_or_buffer: A dataset path or URI, or raw buffer.
        layer:
            If an integer is provided, it corresponds to the index of the layer
            with the data source.  If a string is provided, it must match the name
            of the layer in the data source.  Defaults to first layer in data source.
        encoding:
            If present, will be used as the encoding for reading string values from
            the data source, unless encoding can be inferred directly from the data
            source.
        columns:
            List of column names to import from the data source.  Column names must
            exactly match the names in the data source, and will be returned in
            the order they occur in the data source.  To avoid reading any columns,
            pass an empty list-like.
        read_geometry:
            If True, will read geometry into WKB. If False, geometry will be None.
            Defaults to `True`.
        force_2d:
            If the geometry has Z values, setting this to True will cause those to
            be ignored and 2D geometries to be returned. Defaults to `False`.
        skip_features:
            Number of features to skip from the beginning of the file before returning
            features.  Must be less than the total number of features in the file.
        max_features:
            Number of features to read from the file.  Must be less than the total
            number of features in the file minus skip_features (if used).
        where:
            Where clause to filter features in layer by attribute values.  Uses a
            restricted form of SQL WHERE clause, defined [here](http://ogdi.sourceforge.net/prop/6.2.CapabilitiesMetadata.html).
            For examples:

            - `"ISO_A3 = 'CAN'"`
            - `"POP_EST > 10000000 AND POP_EST < 100000000"`
        bbox:
            If present, will be used to filter records whose geometry intersects this
            box.  This must be in the same CRS as the dataset.  If GEOS is present
            and used by GDAL, only geometries that intersect this bbox will be
            returned; if GEOS is not available or not used by GDAL, all geometries
            with bounding boxes that intersect this bbox will be returned.
        fids:
            Array of integer feature id (FID) values to select. Cannot be combined
            with other keywords to select a subset (`skip_features`, `max_features`,
            `where` or `bbox`). Note that the starting index is driver and file
            specific (e.g. typically 0 for Shapefile and 1 for GeoPackage, but can
            still depend on the specific file). The performance of reading a large
            number of features usings FIDs is also driver specific.
        sql:
            The SQL statement to execute. Look at the sql_dialect parameter for more
            information on the syntax to use for the query. When combined with other
            keywords like `columns`, `skip_features`, `max_features`,
            `where`, `bbox`, or `mask`, those are applied after the SQL query.
            Be aware that this can have an impact on performance, (e.g. filtering
            with the `bbox` or `mask` keywords may not use spatial indexes).
            Cannot be combined with the `layer` or `fids` keywords.
        sql_dialect:
            The SQL dialect the SQL statement is written in. Possible values:

            - `None`: if the data source natively supports SQL, its specific SQL dialect
                will be used by default (eg. SQLite and Geopackage: `SQLITE`_, PostgreSQL).
                If the data source doesn't natively support SQL, the `OGRSQL`_ dialect is
                the default.
            - `OGRSQL`: can be used on any data source. Performance can suffer
                when used on data sources with native support for SQL.
            - `SQLITE`: can be used on any data source. All spatialite_
                functions can be used. Performance can suffer on data sources with
                native support for SQL, except for Geopackage and SQLite as this is
                their native SQL dialect.
        return_fids:
            If True, will return the FIDs of the feature that were read.
    """
    metadata, table = read_arrow(
        path_or_buffer,
        layer=layer,
        encoding=encoding,
        columns=columns,
        read_geometry=read_geometry,
        force_2d=force_2d,
        skip_features=skip_features,
        max_features=max_features,
        where=where,
        bbox=bbox,
        fids=fids,
        sql=sql,
        sql_dialect=sql_dialect,
        return_fids=return_fids,
    )

    import pyarrow as pa

    geometry_name = metadata["geometry_name"] or "wkb_geometry"
    if geometry_name in table.column_names:
        geometry_index = table.schema.get_field_index(geometry_name)
        table = table.cast(table.schema.set(geometry_index, pa.field(geometry_name, pa.binary())))
        res = cast("pl.DataFrame", pl.from_arrow(table))
        if (crs := metadata["crs"]) and (srid := get_crs_srid_or_warn(crs)):
            res = res.with_columns(geom(geometry_name).st.set_srid(srid))
        if not metadata["geometry_name"]:
            res = res.rename({"wkb_geometry": Config.get_geometry_column()})
    else:
        res = cast("pl.DataFrame", pl.from_arrow(table))
    return st(res)._df  # noqa: SLF001


@overload
def from_geopandas(
    data: gpd.GeoDataFrame,
    *,
    schema_overrides: SchemaDict | None = None,
    rechunk: bool = True,
    nan_to_null: bool = True,
    include_index: bool = False,
) -> GeoDataFrame: ...


@overload
def from_geopandas(
    data: gpd.GeoSeries,
    *,
    schema_overrides: SchemaDict | None = None,
    rechunk: bool = True,
    nan_to_null: bool = True,
    include_index: bool = False,
) -> GeoSeries: ...


def from_geopandas(
    data: gpd.GeoDataFrame | gpd.GeoSeries,
    *,
    schema_overrides: SchemaDict | None = None,
    rechunk: bool = True,
    nan_to_null: bool = True,
    include_index: bool = False,
) -> GeoDataFrame | GeoSeries:
    """Create DataFrame or Series from Geopandas GeoDataFrame or GeoSeries.

    Examples:
        >>> import shapely
        >>> import geopandas as gpd
        >>> pd_gdf = gpd.GeoDataFrame({
        ...     "geometry": [shapely.Point(0, 0), shapely.Point(1, 2)]
        ... }, crs="EPSG:4326")
        >>> gdf = st.from_geopandas(pd_gdf)
    """
    res = pl.from_pandas(
        data.to_wkb(),
        schema_overrides=schema_overrides,
        rechunk=rechunk,
        nan_to_null=nan_to_null,
        include_index=include_index,
    )

    if isinstance(data, pd.Series):
        res = pl.from_pandas(
            data.to_wkb(),
            schema_overrides=schema_overrides,
            rechunk=rechunk,
            nan_to_null=nan_to_null,
            include_index=include_index,
        )
        res = cast("pl.Series", res)
        if (crs := data.crs) and (srid := get_crs_srid_or_warn(str(crs))):
            res = st(res).set_srid(srid)
        return st(res)._series  # noqa: SLF001

    res = cast("pl.DataFrame", res).with_columns(
        geom(str(col)).st.set_srid(srid)
        for col in data.dtypes.index[data.dtypes == "geometry"]
        if (crs := data[col].crs) and (srid := get_crs_srid_or_warn(str(crs)))
    )
    return st(res)._df  # noqa: SLF001
