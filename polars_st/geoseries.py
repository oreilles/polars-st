from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING, Any, Literal, ParamSpec, cast

import polars as pl
from polars.api import register_series_namespace

from polars_st.casting import st
from polars_st.parsing import (
    circularstring,
    from_ewkt,
    from_geojson,
    from_shapely,
    from_wkt,
    linestring,
    multilinestring,
    multipoint,
    point,
    polygon,
)
from polars_st.utils.internal import is_empty_method

if TYPE_CHECKING:
    from collections.abc import Sequence

    import altair as alt
    import geopandas as gpd
    from altair.vegalite.v5.schema._config import MarkConfigKwds
    from lonboard import Map
    from lonboard.types.layer import (
        PathLayerKwargs,
        PolygonLayerKwargs,
        ScatterplotLayerKwargs,
    )
    from lonboard.types.map import MapKwargs
    from polars._typing import PolarsDataType
    from typing_extensions import Unpack

    from polars_st.geometry import GeometryType
    from polars_st.typing import (
        ArrayLike,
        IntoExprColumn,
        IntoGeoExprColumn,
        IntoIntegerExpr,
        IntoNumericExpr,
    )

    P = ParamSpec("P")


__all__ = [
    "GeoSeries",
    "GeoSeriesNameSpace",
]


class GeoSeriesMeta(type):
    def __instancecheck__(cls, instance: Any) -> bool:
        # The GeoSeries constructor doesn't return an instance of GeoSeries but an
        # instance of pl.DataFrame. This design decision is made because Polars doesn't
        # support subclassing of its code datatypes. In order to prevent misuse,
        # instance checks are forbidden.
        msg = "instance check on abstract class GeoSeries is not allowed"
        raise TypeError(msg)


class GeoSeries(pl.Series, metaclass=GeoSeriesMeta):
    @property
    def st(self) -> GeoSeriesNameSpace:
        return GeoSeriesNameSpace(self)

    def __new__(  # noqa: C901, PLR0912, PLR0915, PYI034
        cls,
        name: str | ArrayLike | None = None,
        values: ArrayLike | None = None,
        dtype: PolarsDataType | None = None,
        *,
        strict: bool = True,
        nan_to_null: bool = False,
        geometry_format: Literal[
            "wkb",
            "wkt",
            "ewkt",
            "geojson",
            "shapely",
            "point",
            "multipoint",
            "linestring",
            "circularstring",
            "multilinestring",
            "polygon",
        ]
        | None = None,
    ) -> GeoSeries:
        s = pl.Series(name, values, dtype, strict=strict, nan_to_null=nan_to_null)
        if s.name == "" and not (isinstance(name, str) and name == ""):
            s = s.rename("geometry")
        if len(s) == 0 or s.dtype == pl.Null:
            return cast("GeoSeries", s.cast(pl.Binary))
        if geometry_format is None:
            match s.dtype:
                case pl.Binary:
                    geometry_format = "wkb"
                case pl.String:
                    first_value: str | None = s[cast("int", s.is_not_null().arg_max())]
                    if first_value is None:
                        return cast("GeoSeries", s.cast(pl.Binary))
                    if first_value.startswith("{"):
                        geometry_format = "geojson"
                    elif first_value.startswith("SRID="):
                        geometry_format = "ewkt"
                    else:
                        geometry_format = "wkt"
                case pl.Object:
                    geometry_format = "shapely"
                case pl.List | pl.Array:
                    inner = s.dtype.inner
                    if inner.is_numeric():
                        geometry_format = "point"
                    elif inner in (pl.List, pl.Array):
                        inner = inner.inner
                        if inner.is_numeric():
                            geometry_format = "linestring"
                        elif inner in (pl.List, pl.Array):
                            inner = inner.inner
                            if inner.is_numeric():
                                geometry_format = "polygon"

        match geometry_format:
            case None:
                msg = f"Couldn't infer geometry format from dtype {s.dtype}"
                raise ValueError(msg)
            case "wkb":
                result = s
            case "wkt":
                result = pl.select(from_wkt(s)).to_series()
            case "ewkt":
                result = pl.select(from_ewkt(s)).to_series()
            case "geojson":
                result = pl.select(from_geojson(s)).to_series()
            case "shapely":
                result = pl.select(from_shapely(s)).to_series()
            case "point":
                result = pl.select(point(s)).to_series()
            case "multipoint":
                result = pl.select(multipoint(s)).to_series()
            case "linestring":
                result = pl.select(linestring(s)).to_series()
            case "circularstring":
                result = pl.select(circularstring(s)).to_series()
            case "multilinestring":
                result = pl.select(multilinestring(s)).to_series()
            case "polygon":
                result = pl.select(polygon(s)).to_series()
        return cast("GeoSeries", result)

    def __init__(
        self,
        name: str | ArrayLike | None = None,
        values: ArrayLike | None = None,
        dtype: PolarsDataType | None = None,
        *,
        strict: bool = True,
        nan_to_null: bool = False,
        geometry_format: Literal[
            "wkb",
            "wkt",
            "ewkt",
            "geojson",
            "shapely",
            "point",
            "multipoint",
            "linestring",
            "circularstring",
            "multilinestring",
            "polygon",
        ]
        | None = None,
    ) -> None:
        """Create a new GeoSeries.

        `GeoSeries` is used as an alias for `pl.Series` with type annotations added for the
        [`st`][polars_st.GeoSeries.st] namespace, and an overriden constructor which will parse
        the values into binary EWKB format.

        You can create a GeoSeries from a list of coordinate arrays, WKB, WKT, EWKT or GeoJSON
            strings, or Shapely objects. If `geometry_format` is not set, the geometries will be
            created by infering the correct deserialization operation from its datatype.

        See [`pl.Series`](https://docs.pola.rs/api/python/stable/reference/series/index.html)
        for parameters documentation.

        !!! note

            Because Polars doesn't support subclassing of their types, calling this constructor will
            **NOT** create an instance of `GeoSeries`, but an instance of `pl.Series`.

            As a result, instance checks are not permitted on this class to prevent misuse:
            ```pycon
            >>> s = st.GeoSeries(["POINT(0 0)"])
            >>> type(s)
            <class 'polars.series.series.Series'>
            >>> isinstance(s, st.GeoSeries)
            Traceback (most recent call last):
            ...
            TypeError: instance check on abstract class GeoSeries is not allowed
            ```

        Examples:
            >>> gs = st.GeoSeries([
            ...     "POINT(0 0)",
            ...     "POINT(1 2)",
            ... ])
            >>> gs2 = st.GeoSeries([
            ...     [0, 0],
            ...     [1, 2],
            ... ], geometry_format="point")
            >>> gs.equals(gs2)
            True

            >>> import shapely
            >>> gs = st.GeoSeries([
            ...     shapely.Point(0, 0),
            ...     shapely.Point(1, 2),
            ... ])
            >>> gs2 = st.GeoSeries([
            ...     '{"type": "Point", "coordinates": [0, 0]}',
            ...     '{"type": "Point", "coordinates": [1, 2]}',
            ... ])
            >>> gs.equals(gs2)
            True
        """
        ...


def dispatch(func):  # noqa: ANN001, ANN202 to preserve pylance type hints
    assert is_empty_method(func)  # noqa: S101

    @wraps(func)
    def wrapper(self: GeoSeriesNameSpace, *args: P.args, **kwargs: P.kwargs) -> pl.Series:
        f = getattr(getattr(pl.col(self._series.name), "st"), func.__name__)  # noqa: B009
        return self._series.to_frame().select_seq(f(*args, **kwargs)).to_series()

    return wrapper


@register_series_namespace("st")
class GeoSeriesNameSpace:
    def __init__(self, series: pl.Series) -> None:
        self._series = cast("GeoSeries", series)

    @dispatch
    def geometry_type(self) -> pl.Series:
        """See [`GeoExprNameSpace.geometry_type`][polars_st.GeoExprNameSpace.geometry_type]."""
        ...

    @dispatch
    def dimensions(self) -> pl.Series:
        """See [`GeoExprNameSpace.dimensions`][polars_st.GeoExprNameSpace.dimensions]."""
        ...

    @dispatch
    def coordinate_dimension(self) -> pl.Series:
        """See [`GeoExprNameSpace.coordinate_dimension`][polars_st.GeoExprNameSpace.coordinate_dimension]."""  # noqa: E501
        ...

    @dispatch
    def area(self) -> pl.Series:
        """See [`GeoExprNameSpace.area`][polars_st.GeoExprNameSpace.area]."""
        ...

    @dispatch
    def bounds(self) -> pl.Series:
        """See [`GeoExprNameSpace.bounds`][polars_st.GeoExprNameSpace.bounds]."""
        ...

    @dispatch
    def length(self) -> pl.Series:
        """See [`GeoExprNameSpace.length`][polars_st.GeoExprNameSpace.length]."""
        ...

    @dispatch
    def minimum_clearance(self) -> pl.Series:
        """See [`GeoExprNameSpace.minimum_clearance`][polars_st.GeoExprNameSpace.minimum_clearance]."""  # noqa: E501
        ...

    @dispatch
    def x(self) -> pl.Series:
        """See [`GeoExprNameSpace.x`][polars_st.GeoExprNameSpace.x]."""
        ...

    @dispatch
    def y(self) -> pl.Series:
        """See [`GeoExprNameSpace.y`][polars_st.GeoExprNameSpace.y]."""
        ...

    @dispatch
    def z(self) -> pl.Series:
        """See [`GeoExprNameSpace.z`][polars_st.GeoExprNameSpace.z]."""
        ...

    @dispatch
    def m(self) -> pl.Series:
        """See [`GeoExprNameSpace.m`][polars_st.GeoExprNameSpace.m]."""
        ...

    @dispatch
    def count_coordinates(self) -> pl.Series:
        """See [`GeoExprNameSpace.count_coordinates`][polars_st.GeoExprNameSpace.count_coordinates]."""  # noqa: E501
        ...

    @dispatch
    def coordinates(self, output_dimension: Literal[2, 3] | None = None) -> pl.Series:
        """See [`GeoExprNameSpace.coordinates`][polars_st.GeoExprNameSpace.coordinates]."""
        ...

    @dispatch
    def exterior_ring(self) -> GeoSeries:
        """See [`GeoExprNameSpace.exterior_ring`][polars_st.GeoExprNameSpace.exterior_ring]."""
        ...

    @dispatch
    def interior_rings(self) -> pl.Series:
        """See [`GeoExprNameSpace.interior_rings`][polars_st.GeoExprNameSpace.interior_rings]."""
        ...

    @dispatch
    def count_interior_rings(self) -> pl.Series:
        """See [`GeoExprNameSpace.count_interior_rings`][polars_st.GeoExprNameSpace.count_interior_rings]."""  # noqa: E501
        ...

    @dispatch
    def get_interior_ring(self, index: IntoIntegerExpr) -> GeoSeries:
        """See [`GeoExprNameSpace.get_interior_ring`][polars_st.GeoExprNameSpace.get_interior_ring]."""  # noqa: E501
        ...

    @dispatch
    def count_geometries(self) -> pl.Series:
        """See [`GeoExprNameSpace.count_geometries`][polars_st.GeoExprNameSpace.count_geometries]."""  # noqa: E501
        ...

    @dispatch
    def get_geometry(self, index: IntoIntegerExpr) -> GeoSeries:
        """See [`GeoExprNameSpace.get_geometry`][polars_st.GeoExprNameSpace.get_geometry]."""
        ...

    @dispatch
    def count_points(self) -> pl.Series:
        """See [`GeoExprNameSpace.count_points`][polars_st.GeoExprNameSpace.count_points]."""
        ...

    @dispatch
    def get_point(self, index: IntoIntegerExpr) -> GeoSeries:
        """See [`GeoExprNameSpace.get_point`][polars_st.GeoExprNameSpace.get_point]."""
        ...

    @dispatch
    def parts(self) -> pl.Series:
        """See [`GeoExprNameSpace.parts`][polars_st.GeoExprNameSpace.parts]."""
        ...

    @dispatch
    def precision(self) -> pl.Series:
        """See [`GeoExprNameSpace.precision`][polars_st.GeoExprNameSpace.precision]."""
        ...

    @dispatch
    def set_precision(
        self,
        grid_size: IntoNumericExpr,
        mode: Literal["valid_output", "no_topo", "keep_collapsed"] = "valid_output",
    ) -> GeoSeries:
        """See [`GeoExprNameSpace.set_precision`][polars_st.GeoExprNameSpace.set_precision]."""
        ...

    @dispatch
    def distance(self, other: IntoGeoExprColumn) -> pl.Series:
        """See [`GeoExprNameSpace.distance`][polars_st.GeoExprNameSpace.distance]."""
        ...

    @dispatch
    def hausdorff_distance(
        self,
        other: IntoGeoExprColumn,
        densify: float | None = None,
    ) -> pl.Series:
        """See [`GeoExprNameSpace.hausdorff_distance`][polars_st.GeoExprNameSpace.hausdorff_distance]."""  # noqa: E501
        ...

    @dispatch
    def frechet_distance(
        self,
        other: IntoGeoExprColumn,
        densify: float | None = None,
    ) -> pl.Series:
        """See [`GeoExprNameSpace.frechet_distance`][polars_st.GeoExprNameSpace.frechet_distance]."""  # noqa: E501
        ...

    # Projection operations

    @dispatch
    def srid(self) -> pl.Series:
        """See [`GeoExprNameSpace.srid`][polars_st.GeoExprNameSpace.srid]."""
        ...

    @dispatch
    def set_srid(self, srid: IntoIntegerExpr) -> GeoSeries:
        """See [`GeoExprNameSpace.set_srid`][polars_st.GeoExprNameSpace.set_srid]."""
        ...

    @dispatch
    def to_srid(self, srid: IntoIntegerExpr) -> GeoSeries:
        """See [`GeoExprNameSpace.to_srid`][polars_st.GeoExprNameSpace.to_srid]."""
        ...

    # Serialization

    @dispatch
    def to_wkt(
        self,
        rounding_precision: int | None = 6,
        trim: bool = True,
        output_dimension: Literal[2, 3, 4] = 3,
        old_3d: bool = False,
    ) -> pl.Series:
        """See [`GeoExprNameSpace.to_wkt`][polars_st.GeoExprNameSpace.to_wkt]."""
        ...

    @dispatch
    def to_ewkt(
        self,
        rounding_precision: int | None = 6,
        trim: bool = True,
        output_dimension: Literal[2, 3, 4] = 3,
        old_3d: bool = False,
    ) -> pl.Series:
        """See [`GeoExprNameSpace.to_ewkt`][polars_st.GeoExprNameSpace.to_ewkt]."""
        ...

    @dispatch
    def to_wkb(
        self,
        output_dimension: Literal[2, 3, 4] = 3,
        byte_order: Literal[0, 1] | None = None,
        include_srid: bool = False,
    ) -> pl.Series:
        """See [`GeoExprNameSpace.to_wkb`][polars_st.GeoExprNameSpace.to_wkb]."""
        ...

    @dispatch
    def to_geojson(self, indent: int | None = None) -> pl.Series:
        """See [`GeoExprNameSpace.to_geojson`][polars_st.GeoExprNameSpace.to_geojson]."""
        ...

    @dispatch
    def to_shapely(self) -> pl.Series:
        """See [`GeoExprNameSpace.to_shapely`][polars_st.GeoExprNameSpace.to_shapely]."""
        ...

    @dispatch
    def to_dict(self) -> pl.Series:
        """See [`GeoExprNameSpace.to_dict`][polars_st.GeoExprNameSpace.to_dict]."""
        ...

    @dispatch
    def cast(self, into: IntoExprColumn) -> pl.Series:
        """See [`GeoExprNameSpace.cast`][polars_st.GeoExprNameSpace.cast]."""
        ...

    @dispatch
    def multi(self) -> pl.Series:
        """See [`GeoExprNameSpace.multi`][polars_st.GeoExprNameSpace.multi]."""
        ...

    def to_geopandas(
        self,
        *,
        use_pyarrow_extension_array: bool = False,
        **kwargs: Any,
    ) -> gpd.GeoSeries:
        """Convert this pl.Series to a geopandas GeoSeries."""
        return (
            self._series.to_frame()
            .pipe(st)
            .to_geopandas(
                geometry_name=self._series.name,
                use_pyarrow_extension_array=use_pyarrow_extension_array,
                **kwargs,
            )
            .geometry
        )

    @property
    def __geo_interface__(self) -> dict:
        """Return a GeoJSON GeometryCollection [`dict`][] representation of the DataFrame."""
        return {
            "type": "GeometryCollection",
            "geometries": self.to_dict().to_list(),
        }

    #  Unary predicates

    @dispatch
    def has_z(self) -> pl.Series:
        """See [`GeoExprNameSpace.has_z`][polars_st.GeoExprNameSpace.has_z]."""
        ...

    @dispatch
    def has_m(self) -> pl.Series:
        """See [`GeoExprNameSpace.has_m`][polars_st.GeoExprNameSpace.has_m]."""
        ...

    @dispatch
    def is_ccw(self) -> pl.Series:
        """See [`GeoExprNameSpace.is_ccw`][polars_st.GeoExprNameSpace.is_ccw]."""
        ...

    @dispatch
    def is_closed(self) -> pl.Series:
        """See [`GeoExprNameSpace.is_closed`][polars_st.GeoExprNameSpace.is_closed]."""
        ...

    @dispatch
    def is_empty(self) -> pl.Series:
        """See [`GeoExprNameSpace.is_empty`][polars_st.GeoExprNameSpace.is_empty]."""
        ...

    @dispatch
    def is_ring(self) -> pl.Series:
        """See [`GeoExprNameSpace.is_ring`][polars_st.GeoExprNameSpace.is_ring]."""
        ...

    @dispatch
    def is_simple(self) -> pl.Series:
        """See [`GeoExprNameSpace.is_simple`][polars_st.GeoExprNameSpace.is_simple]."""
        ...

    @dispatch
    def is_valid(self) -> pl.Series:
        """See [`GeoExprNameSpace.is_valid`][polars_st.GeoExprNameSpace.is_valid]."""
        ...

    @dispatch
    def is_valid_reason(self) -> pl.Series:
        """See [`GeoExprNameSpace.is_valid_reason`][polars_st.GeoExprNameSpace.is_valid_reason]."""
        ...

    # Binary predicates

    @dispatch
    def crosses(self, other: IntoGeoExprColumn) -> pl.Series:
        """See [`GeoExprNameSpace.crosses`][polars_st.GeoExprNameSpace.crosses]."""
        ...

    @dispatch
    def contains(self, other: IntoGeoExprColumn) -> pl.Series:
        """See [`GeoExprNameSpace.contains`][polars_st.GeoExprNameSpace.contains]."""
        ...

    @dispatch
    def contains_properly(self, other: IntoGeoExprColumn) -> pl.Series:
        """See [`GeoExprNameSpace.contains_properly`][polars_st.GeoExprNameSpace.contains_properly]."""  # noqa: E501
        ...

    @dispatch
    def covered_by(self, other: IntoGeoExprColumn) -> pl.Series:
        """See [`GeoExprNameSpace.covered_by`][polars_st.GeoExprNameSpace.covered_by]."""
        ...

    @dispatch
    def covers(self, other: IntoGeoExprColumn) -> pl.Series:
        """See [`GeoExprNameSpace.covers`][polars_st.GeoExprNameSpace.covers]."""
        ...

    @dispatch
    def disjoint(self, other: IntoGeoExprColumn) -> pl.Series:
        """See [`GeoExprNameSpace.disjoint`][polars_st.GeoExprNameSpace.disjoint]."""
        ...

    @dispatch
    def dwithin(self, other: IntoGeoExprColumn, distance: IntoNumericExpr) -> pl.Series:
        """See [`GeoExprNameSpace.dwithin`][polars_st.GeoExprNameSpace.dwithin]."""
        ...

    @dispatch
    def intersects(self, other: IntoGeoExprColumn) -> pl.Series:
        """See [`GeoExprNameSpace.intersects`][polars_st.GeoExprNameSpace.intersects]."""
        ...

    @dispatch
    def overlaps(self, other: IntoGeoExprColumn) -> pl.Series:
        """See [`GeoExprNameSpace.overlaps`][polars_st.GeoExprNameSpace.overlaps]."""
        ...

    @dispatch
    def touches(self, other: IntoGeoExprColumn) -> pl.Series:
        """See [`GeoExprNameSpace.touches`][polars_st.GeoExprNameSpace.touches]."""
        ...

    @dispatch
    def within(self, other: IntoGeoExprColumn) -> pl.Series:
        """See [`GeoExprNameSpace.within`][polars_st.GeoExprNameSpace.within]."""
        ...

    @dispatch
    def equals(self, other: IntoGeoExprColumn) -> pl.Series:
        """See [`GeoExprNameSpace.equals`][polars_st.GeoExprNameSpace.equals]."""
        ...

    @dispatch
    def equals_exact(
        self,
        other: IntoGeoExprColumn,
        tolerance: float = 0.0,
    ) -> pl.Series:
        """See [`GeoExprNameSpace.equals_exact`][polars_st.GeoExprNameSpace.equals_exact]."""
        ...

    @dispatch
    def equals_identical(self, other: IntoGeoExprColumn) -> pl.Series:
        """See [`GeoExprNameSpace.equals_identical`][polars_st.GeoExprNameSpace.equals_identical]."""  # noqa: E501
        ...

    @dispatch
    def relate(self, other: IntoGeoExprColumn) -> pl.Series:
        """See [`GeoExprNameSpace.relate`][polars_st.GeoExprNameSpace.relate]."""
        ...

    @dispatch
    def relate_pattern(
        self,
        other: IntoGeoExprColumn,
        pattern: str,
    ) -> pl.Series:
        """See [`GeoExprNameSpace.relate_pattern`][polars_st.GeoExprNameSpace.relate_pattern]."""
        ...

    # Set operations

    @dispatch
    def union(
        self,
        other: IntoGeoExprColumn,
        grid_size: float | None = None,
    ) -> GeoSeries:
        """See [`GeoExprNameSpace.union`][polars_st.GeoExprNameSpace.union]."""
        ...

    @dispatch
    def unary_union(self, grid_size: float | None = None) -> GeoSeries:
        """See [`GeoExprNameSpace.unary_union`][polars_st.GeoExprNameSpace.unary_union]."""
        ...

    @dispatch
    def coverage_union(self) -> GeoSeries:
        """See [`GeoExprNameSpace.coverage_union`][polars_st.GeoExprNameSpace.coverage_union]."""
        ...

    @dispatch
    def intersection(
        self,
        other: IntoGeoExprColumn,
        grid_size: float | None = None,
    ) -> GeoSeries:
        """See [`GeoExprNameSpace.intersection`][polars_st.GeoExprNameSpace.intersection]."""
        ...

    @dispatch
    def difference(
        self,
        other: IntoGeoExprColumn,
        grid_size: float | None = None,
    ) -> GeoSeries:
        """See [`GeoExprNameSpace.difference`][polars_st.GeoExprNameSpace.difference]."""
        ...

    @dispatch
    def symmetric_difference(
        self,
        other: IntoGeoExprColumn,
        grid_size: float | None = None,
    ) -> GeoSeries:
        """See [`GeoExprNameSpace.symmetric_difference`][polars_st.GeoExprNameSpace.symmetric_difference]."""  # noqa: E501
        ...

    # Constructive operations

    @dispatch
    def boundary(self) -> GeoSeries:
        """See [`GeoExprNameSpace.boundary`][polars_st.GeoExprNameSpace.boundary]."""
        ...

    @dispatch
    def buffer(
        self,
        distance: IntoNumericExpr,
        quad_segs: int = 8,
        cap_style: Literal["round", "square", "flat"] = "round",
        join_style: Literal["round", "mitre", "bevel"] = "round",
        mitre_limit: float = 5.0,
        single_sided: bool = False,
    ) -> GeoSeries:
        """See [`GeoExprNameSpace.buffer`][polars_st.GeoExprNameSpace.buffer]."""
        ...

    @dispatch
    def offset_curve(
        self,
        distance: IntoNumericExpr,
        quad_segs: int = 8,
        join_style: Literal["round", "mitre", "bevel"] = "round",
        mitre_limit: float = 5.0,
    ) -> GeoSeries:
        """See [`GeoExprNameSpace.offset_curve`][polars_st.GeoExprNameSpace.offset_curve]."""
        ...

    @dispatch
    def centroid(self) -> GeoSeries:
        """See [`GeoExprNameSpace.centroid`][polars_st.GeoExprNameSpace.centroid]."""
        ...

    @dispatch
    def center(self) -> GeoSeries:
        """See [`GeoExprNameSpace.center`][polars_st.GeoExprNameSpace.center]."""
        ...

    @dispatch
    def clip_by_rect(self, bounds: IntoExprColumn) -> GeoSeries:
        """See [`GeoExprNameSpace.clip_by_rect`][polars_st.GeoExprNameSpace.clip_by_rect]."""
        ...

    @dispatch
    def convex_hull(self) -> GeoSeries:
        """See [`GeoExprNameSpace.convex_hull`][polars_st.GeoExprNameSpace.convex_hull]."""
        ...

    @dispatch
    def concave_hull(self, ratio: float = 0.0, allow_holes: bool = False) -> GeoSeries:
        """See [`GeoExprNameSpace.concave_hull`][polars_st.GeoExprNameSpace.concave_hull]."""
        ...

    @dispatch
    def segmentize(self, max_segment_length: IntoNumericExpr) -> GeoSeries:
        """See [`GeoExprNameSpace.segmentize`][polars_st.GeoExprNameSpace.segmentize]."""
        ...

    @dispatch
    def envelope(self) -> GeoSeries:
        """See [`GeoExprNameSpace.envelope`][polars_st.GeoExprNameSpace.envelope]."""
        ...

    @dispatch
    def extract_unique_points(self) -> GeoSeries:
        """See [`GeoExprNameSpace.extract_unique_points`][polars_st.GeoExprNameSpace.extract_unique_points]."""  # noqa: E501
        ...

    @dispatch
    def build_area(self) -> GeoSeries:
        """See [`GeoExprNameSpace.build_area`][polars_st.GeoExprNameSpace.build_area]."""
        ...

    @dispatch
    def make_valid(self) -> GeoSeries:
        """See [`GeoExprNameSpace.make_valid`][polars_st.GeoExprNameSpace.make_valid]."""
        ...

    @dispatch
    def normalize(self) -> GeoSeries:
        """See [`GeoExprNameSpace.normalize`][polars_st.GeoExprNameSpace.normalize]."""
        ...

    @dispatch
    def node(self) -> GeoSeries:
        """See [`GeoExprNameSpace.node`][polars_st.GeoExprNameSpace.node]."""
        ...

    @dispatch
    def point_on_surface(self) -> GeoSeries:
        """See [`GeoExprNameSpace.point_on_surface`][polars_st.GeoExprNameSpace.point_on_surface]."""  # noqa: E501
        ...

    @dispatch
    def remove_repeated_points(self, tolerance: IntoNumericExpr = 0.0) -> GeoSeries:
        """See [`GeoExprNameSpace.remove_repeated_points`][polars_st.GeoExprNameSpace.remove_repeated_points]."""  # noqa: E501
        ...

    @dispatch
    def reverse(self) -> GeoSeries:
        """See [`GeoExprNameSpace.reverse`][polars_st.GeoExprNameSpace.reverse]."""
        ...

    @dispatch
    def simplify(
        self,
        tolerance: IntoNumericExpr,
        preserve_topology: bool = True,
    ) -> GeoSeries:
        """See [`GeoExprNameSpace.simplify`][polars_st.GeoExprNameSpace.simplify]."""
        ...

    @dispatch
    def force_2d(self) -> GeoSeries:
        """See [`GeoExprNameSpace.force_2d`][polars_st.GeoExprNameSpace.force_2d]."""
        ...

    @dispatch
    def force_3d(self, z: IntoNumericExpr = 0.0) -> GeoSeries:
        """See [`GeoExprNameSpace.force_3d`][polars_st.GeoExprNameSpace.force_3d]."""
        ...

    @dispatch
    def flip_coordinates(self) -> GeoSeries:
        """See [`GeoExprNameSpace.flip_coordinates`][polars_st.GeoExprNameSpace.flip_coordinates]."""  # noqa: E501
        ...

    @dispatch
    def minimum_rotated_rectangle(self) -> GeoSeries:
        """See [`GeoExprNameSpace.minimum_rotated_rectangle`][polars_st.GeoExprNameSpace.minimum_rotated_rectangle]."""  # noqa: E501
        ...

    @dispatch
    def snap(
        self,
        other: IntoGeoExprColumn,
        tolerance: IntoNumericExpr,
    ) -> GeoSeries:
        """See [`GeoExprNameSpace.snap`][polars_st.GeoExprNameSpace.snap]."""
        ...

    @dispatch
    def shortest_line(self, other: IntoGeoExprColumn) -> GeoSeries:
        """See [`GeoExprNameSpace.shortest_line`][polars_st.GeoExprNameSpace.shortest_line]."""
        ...

    # Affine transforms

    @dispatch
    def affine_transform(self, matrix: IntoExprColumn | Sequence[float]) -> GeoSeries:
        """See [`GeoExprNameSpace.affine_transform`][polars_st.GeoExprNameSpace.affine_transform]."""  # noqa: E501
        ...

    @dispatch
    def translate(
        self,
        x: IntoNumericExpr = 0.0,
        y: IntoNumericExpr = 0.0,
        z: IntoNumericExpr = 0.0,
    ) -> GeoSeries:
        """See [`GeoExprNameSpace.translate`][polars_st.GeoExprNameSpace.translate]."""
        ...

    @dispatch
    def rotate(
        self,
        angle: IntoNumericExpr,
        origin: Literal["center", "centroid"] | Sequence[float] = "center",
    ) -> GeoSeries:
        """See [`GeoExprNameSpace.rotate`][polars_st.GeoExprNameSpace.rotate]."""
        ...

    @dispatch
    def scale(
        self,
        x: IntoNumericExpr = 1.0,
        y: IntoNumericExpr = 1.0,
        z: IntoNumericExpr = 1.0,
        origin: Literal["center", "centroid"] | Sequence[float] = "center",
    ) -> GeoSeries:
        """See [`GeoExprNameSpace.scale`][polars_st.GeoExprNameSpace.scale]."""
        ...

    @dispatch
    def skew(
        self,
        x: IntoNumericExpr = 0.0,
        y: IntoNumericExpr = 0.0,
        z: IntoNumericExpr = 0.0,
        origin: Literal["center", "centroid"] | Sequence[float] = "center",
    ) -> GeoSeries:
        """See [`GeoExprNameSpace.skew`][polars_st.GeoExprNameSpace.skew]."""
        ...

    # LineString operations

    @dispatch
    def interpolate(
        self,
        distance: IntoNumericExpr,
        normalized: bool = False,
    ) -> GeoSeries:
        """See [`GeoExprNameSpace.interpolate`][polars_st.GeoExprNameSpace.interpolate]."""
        ...

    @dispatch
    def project(
        self,
        other: IntoGeoExprColumn,
        normalized: bool = False,
    ) -> pl.Series:
        """See [`GeoExprNameSpace.project`][polars_st.GeoExprNameSpace.project]."""
        ...

    @dispatch
    def substring(
        self,
        start: IntoNumericExpr,
        end: IntoNumericExpr,
    ) -> GeoSeries:
        """See [`GeoExprNameSpace.substring`][polars_st.GeoExprNameSpace.substring]."""
        ...

    @dispatch
    def line_merge(self, directed: bool = False) -> GeoSeries:
        """See [`GeoExprNameSpace.line_merge`][polars_st.GeoExprNameSpace.line_merge]."""
        ...

    @dispatch
    def shared_paths(
        self,
        other: IntoGeoExprColumn,
    ) -> GeoSeries:
        """See [`GeoExprNameSpace.shared_paths`][polars_st.GeoExprNameSpace.shared_paths]."""
        ...

    # Aggregations

    @dispatch
    def total_bounds(self) -> pl.Series:
        """See [`GeoExprNameSpace.total_bounds`][polars_st.GeoExprNameSpace.total_bounds]."""
        ...

    @dispatch
    def collect(self, into: GeometryType | None = None) -> GeoSeries:
        """See [`GeoExprNameSpace.collect`][polars_st.GeoExprNameSpace.collect]."""
        ...

    @dispatch
    def union_all(self, grid_size: float | None = None) -> GeoSeries:
        """See [`GeoExprNameSpace.union_all`][polars_st.GeoExprNameSpace.union_all]."""
        ...

    @dispatch
    def coverage_union_all(self) -> GeoSeries:
        """See [`GeoExprNameSpace.coverage_union_all`][polars_st.GeoExprNameSpace.coverage_union_all]."""  # noqa: E501
        ...

    @dispatch
    def intersection_all(self, grid_size: float | None = None) -> GeoSeries:
        """See [`GeoExprNameSpace.intersection_all`][polars_st.GeoExprNameSpace.intersection_all]."""  # noqa: E501
        ...

    @dispatch
    def difference_all(self, grid_size: float | None = None) -> GeoSeries:
        """See [`GeoExprNameSpace.difference_all`][polars_st.GeoExprNameSpace.difference_all]."""
        ...

    @dispatch
    def symmetric_difference_all(self, grid_size: float | None = None) -> GeoSeries:
        """See [`GeoExprNameSpace.symmetric_difference_all`][polars_st.GeoExprNameSpace.symmetric_difference_all]."""  # noqa: E501
        ...

    @dispatch
    def polygonize(self) -> GeoSeries:
        """See [`GeoExprNameSpace.polygonize`][polars_st.GeoExprNameSpace.polygonize]."""
        ...

    @dispatch
    def voronoi_polygons(
        self,
        tolerance: float = 0.0,
        extend_to: bytes | None = None,
        only_edges: bool = False,
    ) -> GeoSeries:
        """See [`GeoExprNameSpace.voronoi_polygons`][polars_st.GeoExprNameSpace.voronoi_polygons]."""  # noqa: E501
        ...

    @dispatch
    def delaunay_triangles(
        self,
        tolerance: float = 0.0,
        only_edges: bool = False,
    ) -> GeoSeries:
        """See [`GeoExprNameSpace.delaunay_triangles`][polars_st.GeoExprNameSpace.delaunay_triangles]."""  # noqa: E501
        ...

    def plot(self, **kwargs: Unpack[MarkConfigKwds]) -> alt.Chart:
        """Draw map plot.

        Polars does not implement plotting logic itself but instead defers to
        [`Altair`](https://altair-viz.github.io/).

        `df.st.plot(**kwargs)` is shorthand for
        `alt.Chart(df).mark_geoshape(**kwargs).interactive()`. Please read Altair
        [GeoShape](https://altair-viz.github.io/user_guide/marks/geoshape.html) documentation
        for available options.
        """
        return self._series.to_frame().pipe(st).plot(**kwargs)

    def explore(
        self,
        *,
        scatterplot_kwargs: ScatterplotLayerKwargs | None = None,
        path_kwargs: PathLayerKwargs | None = None,
        polygon_kwargs: PolygonLayerKwargs | None = None,
        map_kwargs: MapKwargs | None = None,
    ) -> Map:
        return (
            self._series.to_frame()
            .pipe(st)
            .explore(
                self._series.name,
                scatterplot_kwargs=scatterplot_kwargs,
                path_kwargs=path_kwargs,
                polygon_kwargs=polygon_kwargs,
                map_kwargs=map_kwargs,
            )
        )
