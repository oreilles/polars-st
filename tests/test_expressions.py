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
from polars_st.geometry import GeometryType


def gdf(values: list[str | None]):
    s = pl.Series(values, dtype=pl.String())
    return pl.select(geometry=st.from_wkt(s))


empty_frame = pl.Series("geometry", [], pl.Binary()).to_frame()
none_frame = pl.Series("geometry", [None], pl.Binary()).to_frame()

point_empty = gdf(["POINT EMPTY"])
point_2d = gdf(["POINT (1 2)"])
point_3d = gdf(["POINT (1 2 3)"])
line_empty = gdf(["LINESTRING EMPTY"])
line_2d = gdf(["LINESTRING (0 0, 1 1)"])
line_3d = gdf(["LINESTRING Z (0 0 0, 1 1 1, 2 2 2)"])
poly_empty = gdf(["POLYGON EMPTY"])
poly_2d = gdf(["POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))"])
poly_3d = gdf(["POLYGON Z ((0 0 1, 1 0 0, 1 1 1, 0 1 0, 0 0 1))"])
multipoint_empty = gdf(["MULTIPOINT EMPTY"])
multipoint_2d = gdf(["MULTIPOINT ((0 0), (1 1))"])
multipoint_3d = gdf(["MULTIPOINT Z ((0 0 0), (1 1 1))"])
multiline_empty = gdf(["MULTILINESTRING EMPTY"])
multiline_2d = gdf(["MULTILINESTRING ((0 0, 1 1), (2 2, 3 3))"])
multiline_3d = gdf(["MULTILINESTRING Z ((0 0 0, 1 1 1), (2 2 2, 3 3 3))"])
multipoly_empty = gdf(["MULTIPOLYGON EMPTY"])
multipoly_2d = gdf(["MULTIPOLYGON (((0 0, 1 0, 0 1, 0 0)), ((2 2, 3 2, 2 3, 2 2)))"])
multipoly_3d = gdf([
    "MULTIPOLYGON Z (((0 0 0, 1 0 0, 0 1 1, 0 0 0)), ((2 2 2, 3 2 3, 2 3 2, 2 2 2)))",
])
collection_empty = gdf(["GEOMETRYCOLLECTION EMPTY"])
collection_2d = gdf(["GEOMETRYCOLLECTION (POINT (0 0), LINESTRING (0 0, 1 1))"])
collection_3d = gdf([
    "GEOMETRYCOLLECTION (POINT (0 0), LINESTRING (0 0, 1 1), POLYGON ((0 0, 1 0, 1 1, 0 0)))",
])
collection_mixed = gdf([
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


functions = [
    Function(Geo.geometry_type, pl.UInt32()),
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
    Function(Geo.boundary, pl.Binary()),
    Function(Geo.coverage_union, pl.Binary()),
    Function(Geo.buffer, pl.Binary(), {"distance": 1.0}),
    Function(Geo.offset_curve, pl.Binary(), {"distance": 1.0}),
    Function(Geo.centroid, pl.Binary()),
    Function(Geo.center, pl.Binary()),
    Function(Geo.clip_by_rect, pl.Binary(), {"xmin": 0.0, "ymin": 0.0, "xmax": 1.0, "ymax": 1.0}),
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
    Function(Geo.interpolate, pl.Binary(), {"distance": 1.0, "normalized": False}),
    Function(Geo.interpolate, pl.Binary(), {"distance": 1.0, "normalized": True}),
    Function(Geo.project, pl.Float64(), {"other": dummy_point, "normalized": True}),
    Function(Geo.project, pl.Float64(), {"other": dummy_point, "normalized": False}),
    Function(Geo.line_merge, pl.Binary(), {"directed": True}),
    Function(Geo.line_merge, pl.Binary(), {"directed": False}),
    Function(Geo.shared_paths, pl.Binary(), {"other": dummy_line}),
    Function(Geo.shortest_line, pl.Binary(), {"other": dummy_point}),
    Function(Geo.count_coordinates, pl.UInt32()),
    Function(Geo.coordinates, pl.List(pl.Array(pl.Float64, 2))),
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
    Aggregate(Geo.multipoint, pl.Binary(), multipoint_empty.item()),
    Aggregate(Geo.multilinestring, pl.Binary(), multiline_empty.item()),
    Aggregate(Geo.multipolygon, pl.Binary(), multipoly_empty.item()),
    Aggregate(Geo.geometrycollection, pl.Binary(), collection_empty.item()),
    Aggregate(Geo.collect, pl.Binary(), collection_empty.item()),
]


@pytest.mark.parametrize("frame", [empty_frame])
@pytest.mark.parametrize("func", functions)
def test_functions_empty_frame(frame: pl.DataFrame, func: Function):
    """Functions should work on empty frames."""
    result = frame.select(func.call(st.geom().st, **func.args))
    assert result.schema == pl.Schema([("geometry", func.dtype)])
    assert len(result) == 0


@pytest.mark.parametrize("frame", [none_frame])
@pytest.mark.parametrize("func", functions)
def test_functions_none_frame(frame: pl.DataFrame, func: Function):
    """Functions should work on full-null frames."""
    result = frame.select(func.call(st.geom().st, **func.args))
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
    result = frame.group_by(0).agg(func.call(st.geom().st, **func.args)).drop("literal")
    assert result.schema == pl.Schema([("geometry", pl.List(func.dtype))])
    assert len(result) == 0


@pytest.mark.parametrize("frame", [none_frame])
@pytest.mark.parametrize("func", functions)
def test_functions_none_frame_agg(frame: pl.DataFrame, func: Function):
    """Functions should work on full-null frames in aggregation context."""
    # Skip since List(Object) is not supported
    if func.call in {Geo.to_dict, Geo.to_shapely}:
        return
    result = frame.group_by(0).agg(func.call(st.geom().st, **func.args)).drop("literal")
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
def test_functions_all_types_frame(frame: pl.DataFrame, func: Function):
    """Functions should work on every geometry type."""
    geom_type: GeometryType = frame.select(st.geometry_type()).item()
    geom_empty: bool = frame.select(st.is_empty()).item()

    if func.call in {Geo.coverage_union} and frame is collection_mixed:
        match = "IllegalArgumentException: Overlay input is mixed-dimensio"
        with pytest.raises(pl.exceptions.ComputeError, match=match):
            frame.select(func.call(st.geom().st, **func.args))
        return

    if (
        func.call
        in {
            Geo.difference,
            Geo.symmetric_difference,
            Geo.union,
            Geo.coverage_union,
        }
        and func.args.get("grid_size", 0) is not None
        and geom_type == GeometryType.GeometryCollection
        and not geom_empty
    ):
        match = "IllegalArgumentException: Overlay input is mixed-dimensio"
        with pytest.raises(pl.exceptions.ComputeError, match=match):
            frame.select(func.call(st.geom().st, **func.args))
        return

    if func.call in {Geo.shared_paths} and geom_type not in {
        GeometryType.LineString,
        GeometryType.MultiLineString,
    }:
        match = "IllegalArgumentException: Geometry is not linea"
        with pytest.raises(pl.exceptions.ComputeError, match=match):
            frame.select(func.call(st.geom().st, **func.args))
        return

    if func.call in {Geo.get_interior_ring} and geom_type not in {
        GeometryType.Polygon,
        GeometryType.CurvePolygon,
    }:
        match = "generic error: Geometry must be a Polygon or CurvePolygon"
        with pytest.raises(pl.exceptions.ComputeError, match=match):
            frame.select(func.call(st.geom().st, **func.args))
        return

    if func.call in {
        Geo.offset_curve,
        Geo.interpolate,
        Geo.project,
        Geo.get_point,
    } and geom_type not in {GeometryType.LineString}:
        match = "generic error: Geometry must be a LineString"
        with pytest.raises(pl.exceptions.ComputeError, match=match):
            frame.select(func.call(st.geom().st, **func.args))
        return

    if func.call in {Geo.coverage_union} and geom_type not in {
        GeometryType.MultiPoint,
        GeometryType.MultiLineString,
        GeometryType.MultiPolygon,
        GeometryType.GeometryCollection,
        GeometryType.CompoundCurve,
        GeometryType.MultiCurve,
        GeometryType.MultiSurface,
    }:
        match = "generic error: Geometry must be a collection"
        with pytest.raises(pl.exceptions.ComputeError, match=match):
            frame.select(func.call(st.geom().st, **func.args))
        return

    if func.call in {Geo.to_srid}:
        frame = frame.select(st.geom().st.set_srid(4326))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="invalid value encountered in voronoi_polygons")
        result = frame.select(func.call(st.geom().st, **func.args))

    assert result.schema == pl.Schema([("geometry", func.dtype)])
