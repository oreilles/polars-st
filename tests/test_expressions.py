# ruff: noqa: E501

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Concatenate, ParamSpec

import numpy as np
import polars as pl
import pytest

import polars_st as st
from polars_st.geoexpr import GeoExprNameSpace as Geo
from polars_st.geometry import GeometryType, PolarsGeometryType

empty_frame = pl.Series("geometry", [], pl.Binary()).to_frame()
none_frame = pl.Series("geometry", [None], pl.Binary()).to_frame()

point_empty = st.GeoDataFrame(["POINT EMPTY"])
point_2d = st.GeoDataFrame(["POINT (1 2)"])
point_3d = st.GeoDataFrame(["POINT (1 2 3)"])
line_empty = st.GeoDataFrame(["LINESTRING EMPTY"])
line_2d = st.GeoDataFrame(["LINESTRING (0 0, 1 1)"])
line_3d = st.GeoDataFrame(["LINESTRING Z (0 0 0, 1 1 1, 2 2 2)"])
poly_empty = st.GeoDataFrame(["POLYGON EMPTY"])
poly_2d = st.GeoDataFrame(["POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))"])
poly_3d = st.GeoDataFrame(["POLYGON Z ((0 0 1, 1 0 0, 1 1 1, 0 1 0, 0 0 1))"])
multipoint_empty = st.GeoDataFrame(["MULTIPOINT EMPTY"])
multipoint_2d = st.GeoDataFrame(["MULTIPOINT ((0 0), (1 1))"])
multipoint_3d = st.GeoDataFrame(["MULTIPOINT Z ((0 0 0), (1 1 1))"])
multiline_empty = st.GeoDataFrame(["MULTILINESTRING EMPTY"])
multiline_2d = st.GeoDataFrame(["MULTILINESTRING ((0 0, 1 1), (2 2, 3 3))"])
multiline_3d = st.GeoDataFrame(["MULTILINESTRING Z ((0 0 0, 1 1 1), (2 2 2, 3 3 3))"])
multipoly_empty = st.GeoDataFrame(["MULTIPOLYGON EMPTY"])
multipoly_2d = st.GeoDataFrame(["MULTIPOLYGON (((0 0, 1 0, 0 1, 0 0)), ((2 2, 3 2, 2 3, 2 2)))"])
multipoly_3d = st.GeoDataFrame([
    "MULTIPOLYGON Z (((0 0 0, 1 0 0, 0 1 1, 0 0 0)), ((2 2 2, 3 2 3, 2 3 2, 2 2 2)))",
])
collection_empty = st.GeoDataFrame(["GEOMETRYCOLLECTION EMPTY"])
collection_2d = st.GeoDataFrame(["GEOMETRYCOLLECTION (POINT (0 0), LINESTRING (0 0, 1 1))"])
collection_3d = st.GeoDataFrame([
    "GEOMETRYCOLLECTION (POINT (0 0), LINESTRING (0 0, 1 1), POLYGON ((0 0, 1 0, 1 1, 0 0)))",
])
collection_mixed = st.GeoDataFrame([
    "GEOMETRYCOLLECTION (POINT Z (0 0 0), LINESTRING (0 0, 1 1), POLYGON ((0 0, 1 0, 1 1, 0 0)))",
])

base_types = [
    point_empty,
    point_2d,
    point_3d,
    line_empty,
    line_2d,
    line_3d,
    poly_empty,
    poly_2d,
    poly_3d,
    multipoint_empty,
    multipoint_2d,
    multipoint_3d,
    multiline_empty,
    multiline_2d,
    multiline_3d,
    multipoly_empty,
    multipoly_2d,
    multipoly_3d,
    collection_empty,
    collection_2d,
    collection_3d,
    collection_mixed,
]

dummy_point = point_2d.item()
dummy_line = line_2d.item()

P = ParamSpec("P")
FunctionCallable = Callable[Concatenate[Geo, P], pl.Expr]


@dataclass()
class Function:
    call: FunctionCallable
    dtype: pl.DataType
    args: dict[str, Any] = field(default_factory=dict)

    def __call__(self):
        return self.call(st.geom().st, **self.args)


functions = [
    Function(Geo.geometry_type, PolarsGeometryType),
    Function(Geo.dimensions, pl.Int32()),
    Function(Geo.coordinate_dimension, pl.UInt32()),
    Function(Geo.srid, pl.Int32()),
    Function(Geo.set_srid, pl.Binary(), {"srid": 3857}),
    Function(Geo.to_srid, pl.Binary(), {"srid": 3857}),
    Function(Geo.x, pl.Float64()),
    Function(Geo.y, pl.Float64()),
    Function(Geo.z, pl.Float64()),
    Function(Geo.m, pl.Float64()),
    Function(Geo.exterior_ring, pl.Binary()),
    Function(Geo.count_points, pl.UInt32()),
    Function(Geo.count_interior_rings, pl.UInt32()),
    Function(Geo.count_geometries, pl.UInt32()),
    Function(Geo.get_point, pl.Binary(), {"index": 0}),
    Function(Geo.get_interior_ring, pl.Binary(), {"index": 0}),
    Function(Geo.get_geometry, pl.Binary(), {"index": 0}),
    Function(Geo.parts, pl.List(pl.Binary())),
    Function(Geo.interior_rings, pl.List(pl.Binary())),
    Function(Geo.precision, pl.Float64()),
    Function(Geo.set_precision, pl.Binary(), {"grid_size": 1.0, "mode": "valid_output"}),
    Function(Geo.set_precision, pl.Binary(), {"grid_size": 1.0, "mode": "no_topo"}),
    Function(Geo.set_precision, pl.Binary(), {"grid_size": 1.0, "mode": "keep_collapsed"}),
    Function(Geo.to_wkt, pl.String()),
    Function(Geo.to_ewkt, pl.String()),
    Function(Geo.to_wkb, pl.Binary()),
    Function(Geo.to_geojson, pl.String()),
    Function(Geo.to_dict, pl.Object()),
    Function(Geo.to_shapely, pl.Object()),
    Function(Geo.area, pl.Float64()),
    Function(Geo.bounds, pl.Array(pl.Float64, 4)),
    Function(Geo.length, pl.Float64()),
    Function(Geo.distance, pl.Float64(), {"other": dummy_point}),
    Function(Geo.hausdorff_distance, pl.Float64(), {"other": dummy_point, "densify": None}),
    Function(Geo.hausdorff_distance, pl.Float64(), {"other": dummy_point, "densify": 0.5}),
    Function(Geo.frechet_distance, pl.Float64(), {"other": dummy_point, "densify": None}),
    Function(Geo.frechet_distance, pl.Float64(), {"other": dummy_point, "densify": 0.5}),
    Function(Geo.minimum_clearance, pl.Float64()),
    Function(Geo.has_z, pl.Boolean()),
    Function(Geo.has_m, pl.Boolean()),
    Function(Geo.is_ccw, pl.Boolean()),
    Function(Geo.is_closed, pl.Boolean()),
    Function(Geo.is_empty, pl.Boolean()),
    Function(Geo.is_ring, pl.Boolean()),
    Function(Geo.is_simple, pl.Boolean()),
    Function(Geo.is_valid, pl.Boolean()),
    Function(Geo.is_valid_reason, pl.String()),
    Function(Geo.crosses, pl.Boolean(), {"other": dummy_point}),
    Function(Geo.contains, pl.Boolean(), {"other": dummy_point}),
    Function(Geo.contains_properly, pl.Boolean(), {"other": dummy_point}),
    Function(Geo.covered_by, pl.Boolean(), {"other": dummy_point}),
    Function(Geo.covers, pl.Boolean(), {"other": dummy_point}),
    Function(Geo.disjoint, pl.Boolean(), {"other": dummy_point}),
    Function(Geo.dwithin, pl.Boolean(), {"other": dummy_point, "distance": 1.0}),
    Function(Geo.intersects, pl.Boolean(), {"other": dummy_point}),
    Function(Geo.overlaps, pl.Boolean(), {"other": dummy_point}),
    Function(Geo.touches, pl.Boolean(), {"other": dummy_point}),
    Function(Geo.within, pl.Boolean(), {"other": dummy_point}),
    Function(Geo.equals, pl.Boolean(), {"other": dummy_point}),
    Function(Geo.equals_exact, pl.Boolean(), {"other": dummy_point}),
    Function(Geo.equals_identical, pl.Boolean(), {"other": dummy_point}),
    Function(Geo.relate, pl.String(), {"other": dummy_point}),
    Function(Geo.relate_pattern, pl.Boolean(), {"other": dummy_point, "pattern": "*********"}),
    Function(Geo.difference, pl.Binary(), {"other": dummy_point, "grid_size": None}),
    Function(Geo.difference, pl.Binary(), {"other": dummy_point, "grid_size": 0.5}),
    Function(Geo.intersection, pl.Binary(), {"other": dummy_point, "grid_size": None}),
    Function(Geo.intersection, pl.Binary(), {"other": dummy_point, "grid_size": 0.5}),
    Function(Geo.symmetric_difference, pl.Binary(), {"other": dummy_point, "grid_size": None}),
    Function(Geo.symmetric_difference, pl.Binary(), {"other": dummy_point, "grid_size": 0.5}),
    Function(Geo.union, pl.Binary(), {"other": dummy_point, "grid_size": None}),
    Function(Geo.union, pl.Binary(), {"other": dummy_point, "grid_size": 0.5}),
    Function(Geo.unary_union, pl.Binary(), {"grid_size": None}),
    Function(Geo.unary_union, pl.Binary(), {"grid_size": 0.5}),
    Function(Geo.cast, pl.Binary(), {"into": "GeometryCollection"}),
    Function(Geo.multi, pl.Binary()),
    Function(Geo.boundary, pl.Binary()),
    Function(Geo.coverage_union, pl.Binary()),
    Function(Geo.buffer, pl.Binary(), {"distance": 1.0}),
    Function(Geo.offset_curve, pl.Binary(), {"distance": 1.0}),
    Function(Geo.centroid, pl.Binary()),
    Function(Geo.center, pl.Binary()),
    Function(Geo.clip_by_rect, pl.Binary(), {"bounds": [0.0, 0.0, 1.0, 1.0]}),
    Function(Geo.concave_hull, pl.Binary()),
    Function(Geo.convex_hull, pl.Binary()),
    Function(Geo.segmentize, pl.Binary(), {"max_segment_length": 1.0}),
    Function(Geo.envelope, pl.Binary()),
    Function(Geo.extract_unique_points, pl.Binary()),
    Function(Geo.build_area, pl.Binary()),
    Function(Geo.make_valid, pl.Binary()),
    Function(Geo.normalize, pl.Binary()),
    Function(Geo.node, pl.Binary()),
    Function(Geo.point_on_surface, pl.Binary()),
    Function(Geo.remove_repeated_points, pl.Binary()),
    Function(Geo.reverse, pl.Binary()),
    Function(Geo.snap, pl.Binary(), {"other": dummy_point, "tolerance": 1.0}),
    Function(Geo.simplify, pl.Binary(), {"tolerance": 1.0, "preserve_topology": False}),
    Function(Geo.simplify, pl.Binary(), {"tolerance": 1.0, "preserve_topology": True}),
    Function(Geo.flip_coordinates, pl.Binary()),
    Function(Geo.minimum_rotated_rectangle, pl.Binary()),
    Function(Geo.translate, pl.Binary()),
    Function(Geo.rotate, pl.Binary(), {"angle": 90}),
    Function(Geo.scale, pl.Binary()),
    Function(Geo.skew, pl.Binary()),
    Function(Geo.interpolate, pl.Binary(), {"distance": 1.0, "normalized": False}),
    Function(Geo.interpolate, pl.Binary(), {"distance": 1.0, "normalized": True}),
    Function(Geo.project, pl.Float64(), {"other": dummy_point, "normalized": False}),
    Function(Geo.project, pl.Float64(), {"other": dummy_point, "normalized": True}),
    Function(Geo.substring, pl.Binary(), {"start": 0.0, "end": 0.0}),
    Function(Geo.line_merge, pl.Binary(), {"directed": True}),
    Function(Geo.line_merge, pl.Binary(), {"directed": False}),
    Function(Geo.shared_paths, pl.Binary(), {"other": dummy_line}),
    Function(Geo.shortest_line, pl.Binary(), {"other": dummy_point}),
    Function(Geo.count_coordinates, pl.UInt32()),
    Function(Geo.coordinates, pl.List(pl.List(pl.Float64))),
]


@dataclass()
class Aggregate:
    call: FunctionCallable
    dtype: pl.DataType
    identity: Any


aggregates = [
    Aggregate(Geo.voronoi_polygons, pl.Binary(), collection_empty.item()),
    Aggregate(Geo.delaunay_triangles, pl.Binary(), collection_empty.item()),
    Aggregate(Geo.polygonize, pl.Binary(), collection_empty.item()),
    Aggregate(Geo.total_bounds, pl.Array(pl.Float64(), 4), np.full(4, np.nan)),
    Aggregate(Geo.intersection_all, pl.Binary(), collection_empty.item()),
    Aggregate(Geo.symmetric_difference_all, pl.Binary(), collection_empty.item()),
    Aggregate(Geo.union_all, pl.Binary(), collection_empty.item()),
    Aggregate(Geo.coverage_union_all, pl.Binary(), collection_empty.item()),
    Aggregate(Geo.collect, pl.Binary(), collection_empty.item()),
]


@pytest.mark.parametrize("frame", [empty_frame])
@pytest.mark.parametrize("func", functions)
def test_functions_empty_frame(frame: pl.DataFrame, func: Function):
    """Functions should work on empty frames."""
    result = frame.select(func())
    assert result.schema == pl.Schema([("geometry", func.dtype)])
    assert len(result) == 0


@pytest.mark.parametrize("frame", [none_frame])
@pytest.mark.parametrize("func", functions)
def test_functions_none_frame(frame: pl.DataFrame, func: Function):
    """Functions should work on full-null frames."""
    result = frame.select(func())
    assert result.schema == pl.Schema([("geometry", func.dtype)])
    assert len(result) == 1
    assert result.item() is None


@pytest.mark.parametrize("frame", [empty_frame])
@pytest.mark.parametrize("func", functions)
def test_functions_empty_frame_agg(frame: pl.DataFrame, func: Function):
    """Functions should work on empty frames in aggregation context."""
    # Should file a bug report in polars for that (cannot concatenate empty list of arrays)
    if func.call == Geo.bounds:
        return
    result = frame.group_by(0).agg(func()).drop("literal")
    assert result.schema == pl.Schema([("geometry", pl.List(func.dtype))])
    assert len(result) == 0


@pytest.mark.parametrize("frame", [none_frame])
@pytest.mark.parametrize("func", functions)
def test_functions_none_frame_agg(frame: pl.DataFrame, func: Function):
    """Functions should work on full-null frames in aggregation context."""
    # Skip since List(Object) is not supported
    if func.call in {Geo.to_dict, Geo.to_shapely}:
        return
    result = frame.group_by(0).agg(func()).drop("literal")
    assert result.schema == pl.Schema([("geometry", pl.List(func.dtype))])
    assert len(result) == 1
    assert result.get_column("geometry").list.len().item() == 1
    assert result.get_column("geometry").list.get(0).item() is None


@pytest.mark.parametrize("frame", [empty_frame.group_by(0).agg(st.geom())])
@pytest.mark.parametrize("func", functions)
def test_functions_empty_list_frame(frame: pl.DataFrame, func: Function):
    """Functions should work on empty lists."""
    # Skip since List(Object) is not supported
    if func.call in {Geo.to_dict, Geo.to_shapely}:
        return
    result = frame.select(st.geom().list.eval(func.call(st.element().st, **func.args)))
    assert result.schema == pl.Schema([("geometry", pl.List(func.dtype))])
    assert len(result) == 0


@pytest.mark.parametrize("frame", [none_frame.group_by(0).agg(st.geom())])
@pytest.mark.parametrize("func", functions)
def test_functions_none_list_frame(frame: pl.DataFrame, func: Function):
    """Functions should work on full-null lists."""
    # Skip since List(Object) is not supported
    if func.call in {Geo.to_dict, Geo.to_shapely}:
        return
    result = frame.select(st.geom().list.eval(func.call(st.element().st, **func.args)))
    assert result.schema == pl.Schema([("geometry", pl.List(func.dtype))])
    assert result.item().item() is None


@pytest.mark.parametrize("frame", [none_frame, empty_frame])
@pytest.mark.parametrize("func", aggregates)
def test_aggregates(frame: pl.DataFrame, func: Aggregate):
    """Aggregations should work on empty and full-null frames."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="invalid value encountered in coverage_union")
        result = frame.select(func.call(st.geom().st))
    assert result.schema == pl.Schema([("geometry", func.dtype)])
    assert len(result) == 1
    item = result.item()
    if isinstance(item, pl.Series):
        # item() returns a series if dtype is List or Array (for st.total_bounds)
        assert np.array_equal(item.to_numpy(), func.identity, equal_nan=True)
    else:
        assert item == func.identity


@pytest.mark.parametrize("frame", base_types)
@pytest.mark.parametrize("func", functions)
def test_functions_all_types_frame(frame: pl.DataFrame, func: Function):  # noqa: C901, PLR0912
    """Functions should work on every geometry type."""
    geom_type: GeometryType = frame.select(st.geometry_type()).item()
    geom_empty: bool = frame.select(st.is_empty()).item()
    error = None

    if (
        func.call
        in {
            Geo.difference,
            Geo.symmetric_difference,
            Geo.union,
            Geo.coverage_union,
        }
        and func.args.get("grid_size", 0) is not None
        and geom_type == "GeometryCollection"
        and not geom_empty
    ):
        error = "IllegalArgumentException: Overlay input is mixed-dimension"

    if func.call == Geo.coverage_union and frame is collection_mixed:
        error = "IllegalArgumentException: Overlay input is mixed-dimension"

    if func.call == Geo.shared_paths and geom_type not in {"LineString", "MultiLineString"}:
        error = "IllegalArgumentException: Geometry is not lineal"

    if func.call == Geo.get_interior_ring and geom_type not in {"Polygon", "CurvePolygon"}:
        error = "IllegalArgumentException: Argument is not a Surface"

    if func.call == Geo.get_point and geom_type not in {"LineString"}:
        error = "IllegalArgumentException: Argument is not a SimpleCurve"

    if (
        func.call == Geo.substring
        and geom_type not in {"LineString", "MultiLineString"}
        and not (geom_type == "GeometryCollection" and geom_empty)
    ):
        error = "IllegalArgumentException: Input geometry must be linear"

    if (
        func.call == Geo.project
        and geom_type != "LineString"
        and not (geom_type == "MultiLineString" and not geom_empty)
    ):
        error = "IllegalArgumentException: LinearIterator only supports lineal geometry components"
        if geom_type != "Point" and geom_empty:
            geos_func = "GEOSProjectNormalized_r" if func.args["normalized"] else "GEOSProject_r"
            error = f"{geos_func} failed"

    if (
        func.call == Geo.interpolate
        and geom_type not in {"LineString", "MultiLineString"}
        and not (geom_type in {"MultiPoint", "MultiPolygon", "GeometryCollection"} and geom_empty)
        and not (func.args["normalized"] and geom_empty)
    ):
        error = "IllegalArgumentException: LinearIterator only supports lineal geometry components"
        if func.args["normalized"] and geom_type in {"Point", "MultiPoint"} and not geom_empty:
            error = "IllegalArgumentException: LinearLocation::getCoordinate only works with LineString geometries"

    if func.call == Geo.coverage_union and geom_type not in {
        "MultiPoint",
        "MultiLineString",
        "MultiPolygon",
        "GeometryCollection",
        "CompoundCurve",
        "MultiCurve",
        "MultiSurface",
    }:
        error = "Geometry must be a collection"

    if func.call == Geo.to_srid:
        frame = frame.select(st.geom().st.set_srid(4326))

    if error is not None:
        with pytest.raises(pl.exceptions.ComputeError, match=error):
            frame.select(func())
        return

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="invalid value encountered in voronoi_polygons")
        result = frame.select(func())

    assert result.schema == pl.Schema([("geometry", func.dtype)])
