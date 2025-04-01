from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from polars_st.selectors import geom

if TYPE_CHECKING:
    from collections.abc import Sequence

    import polars as pl

    from polars_st.typing import (
        IntoDecimalExpr,
        IntoExprColumn,
        IntoIntegerExpr,
    )

    from .geoexpr import GeoExpr


__all__ = [
    "affine_transform",
    "area",
    "boundary",
    "bounds",
    "buffer",
    "build_area",
    "center",
    "centroid",
    "clip_by_rect",
    "collect",
    "concave_hull",
    "convex_hull",
    "coordinate_dimension",
    "coordinates",
    "count_coordinates",
    "count_geometries",
    "count_interior_rings",
    "count_points",
    "coverage_union",
    "coverage_union_all",
    "delaunay_triangles",
    "difference_all",
    "dimensions",
    "envelope",
    "exterior_ring",
    "extract_unique_points",
    "flip_coordinates",
    "force_2d",
    "force_3d",
    "geometry_type",
    "geometrycollection",
    "get_geometry",
    "get_interior_ring",
    "get_point",
    "has_m",
    "has_z",
    "interior_rings",
    "interpolate",
    "intersection_all",
    "is_ccw",
    "is_closed",
    "is_empty",
    "is_ring",
    "is_simple",
    "is_valid",
    "is_valid_reason",
    "length",
    "line_merge",
    "m",
    "make_valid",
    "minimum_clearance",
    "minimum_rotated_rectangle",
    "multilinestring",
    "multipoint",
    "multipolygon",
    "node",
    "normalize",
    "offset_curve",
    "parts",
    "point_on_surface",
    "polygonize",
    "precision",
    "remove_repeated_points",
    "reverse",
    "rotate",
    "scale",
    "segmentize",
    "set_precision",
    "set_srid",
    "simplify",
    "srid",
    "symmetric_difference_all",
    "to_dict",
    "to_ewkt",
    "to_geojson",
    "to_shapely",
    "to_srid",
    "to_wkb",
    "to_wkt",
    "total_bounds",
    "translate",
    "unary_union",
    "union_all",
    "voronoi_polygons",
    "x",
    "y",
    "z",
]


def geometry_type(*columns: str) -> pl.Expr:
    """This function is syntactic sugar for <code>st.geom(columns).st.[geometry_type()][polars_st.GeoExprNameSpace.geometry_type]</code>."""  # noqa: E501
    return geom(*columns).st.geometry_type()


def dimensions(*columns: str) -> pl.Expr:
    """This function is syntactic sugar for <code>st.geom(columns).st.[dimensions()][polars_st.GeoExprNameSpace.dimensions]</code>."""  # noqa: E501
    return geom(*columns).st.dimensions()


def coordinate_dimension(*columns: str) -> pl.Expr:
    """This function is syntactic sugar for <code>st.geom(columns).st.[coordinate_dimension()][polars_st.GeoExprNameSpace.coordinate_dimension]</code>."""  # noqa: E501
    return geom(*columns).st.coordinate_dimension()


def area(*columns: str) -> pl.Expr:
    """This function is syntactic sugar for <code>st.geom(columns).st.[area()][polars_st.GeoExprNameSpace.area]</code>."""  # noqa: E501
    return geom(*columns).st.area()


def bounds(*columns: str) -> pl.Expr:
    """This function is syntactic sugar for <code>st.geom(columns).st.[bounds()][polars_st.GeoExprNameSpace.bounds]</code>."""  # noqa: E501
    return geom(*columns).st.bounds()


def length(*columns: str) -> pl.Expr:
    """This function is syntactic sugar for <code>st.geom(columns).st.[length()][polars_st.GeoExprNameSpace.length]</code>."""  # noqa: E501
    return geom(*columns).st.length()


def minimum_clearance(*columns: str) -> pl.Expr:
    """This function is syntactic sugar for <code>st.geom(columns).st.[minimum_clearance()][polars_st.GeoExprNameSpace.minimum_clearance]</code>."""  # noqa: E501
    return geom(*columns).st.minimum_clearance()


def x(*columns: str) -> pl.Expr:
    """This function is syntactic sugar for <code>st.geom(columns).st.[x()][polars_st.GeoExprNameSpace.x]</code>."""  # noqa: E501
    return geom(*columns).st.x()


def y(*columns: str) -> pl.Expr:
    """This function is syntactic sugar for <code>st.geom(columns).st.[y()][polars_st.GeoExprNameSpace.y]</code>."""  # noqa: E501
    return geom(*columns).st.y()


def z(*columns: str) -> pl.Expr:
    """This function is syntactic sugar for <code>st.geom(columns).st.[z()][polars_st.GeoExprNameSpace.z]</code>."""  # noqa: E501
    return geom(*columns).st.z()


def m(*columns: str) -> pl.Expr:
    """This function is syntactic sugar for <code>st.geom(columns).st.[m()][polars_st.GeoExprNameSpace.m]</code>."""  # noqa: E501
    return geom(*columns).st.m()


def count_coordinates(*columns: str) -> pl.Expr:
    """This function is syntactic sugar for <code>st.geom(columns).st.[count_coordinates()][polars_st.GeoExprNameSpace.count_coordinates]</code>."""  # noqa: E501
    return geom(*columns).st.count_coordinates()


def coordinates(*columns: str, output_dimension: Literal[2, 3] = 2) -> pl.Expr:
    """This function is syntactic sugar for <code>st.geom(columns).st.[coordinates(...)][polars_st.GeoExprNameSpace.coordinates]</code>."""  # noqa: E501
    return geom(*columns).st.coordinates(output_dimension)


def exterior_ring(*columns: str) -> GeoExpr:
    """This function is syntactic sugar for <code>st.geom(columns).st.[exterior_ring()][polars_st.GeoExprNameSpace.exterior_ring]</code>."""  # noqa: E501
    return geom(*columns).st.exterior_ring()


def interior_rings(*columns: str) -> pl.Expr:
    """This function is syntactic sugar for <code>st.geom(columns).st.[interior_rings()][polars_st.GeoExprNameSpace.interior_rings]</code>."""  # noqa: E501
    return geom(*columns).st.interior_rings()


def count_interior_rings(*columns: str) -> pl.Expr:
    """This function is syntactic sugar for <code>st.geom(columns).st.[count_interior_rings()][polars_st.GeoExprNameSpace.count_interior_rings]</code>."""  # noqa: E501
    return geom(*columns).st.count_interior_rings()


def get_interior_ring(*columns: str, index: IntoIntegerExpr) -> GeoExpr:
    """This function is syntactic sugar for <code>st.geom(columns).st.[get_interior_ring(...)][polars_st.GeoExprNameSpace.get_interior_ring]</code>."""  # noqa: E501
    return geom(*columns).st.get_interior_ring(index)


def count_geometries(*columns: str) -> pl.Expr:
    """This function is syntactic sugar for <code>st.geom(columns).st.[count_geometries()][polars_st.GeoExprNameSpace.count_geometries]</code>."""  # noqa: E501
    return geom(*columns).st.count_geometries()


def get_geometry(*columns: str, index: IntoIntegerExpr) -> GeoExpr:
    """This function is syntactic sugar for <code>st.geom(columns).st.[get_geometry(...)][polars_st.GeoExprNameSpace.get_geometry]</code>."""  # noqa: E501
    return geom(*columns).st.get_geometry(index)


def count_points(*columns: str) -> pl.Expr:
    """This function is syntactic sugar for <code>st.geom(columns).st.[count_points()][polars_st.GeoExprNameSpace.count_points]</code>."""  # noqa: E501
    return geom(*columns).st.count_points()


def get_point(*columns: str, index: IntoIntegerExpr) -> GeoExpr:
    """This function is syntactic sugar for <code>st.geom(columns).st.[get_point(...)][polars_st.GeoExprNameSpace.get_point]</code>."""  # noqa: E501
    return geom(*columns).st.get_point(index)


def parts(*columns: str) -> pl.Expr:
    """This function is syntactic sugar for <code>st.geom(columns).st.[parts()][polars_st.GeoExprNameSpace.parts]</code>."""  # noqa: E501
    return geom(*columns).st.parts()


def precision(*columns: str) -> pl.Expr:
    """This function is syntactic sugar for <code>st.geom(columns).st.[precision()][polars_st.GeoExprNameSpace.precision]</code>."""  # noqa: E501
    return geom(*columns).st.precision()


def set_precision(
    *columns: str,
    grid_size: IntoDecimalExpr,
    mode: Literal["valid_output", "no_topo", "keep_collapsed"] = "valid_output",
) -> GeoExpr:
    """This function is syntactic sugar for <code>st.geom(columns).st.[set_precision(...)][polars_st.GeoExprNameSpace.set_precision]</code>."""  # noqa: E501
    return geom(*columns).st.set_precision(grid_size, mode)


def srid(*columns: str) -> pl.Expr:
    """This function is syntactic sugar for <code>st.geom(columns).st.[srid()][polars_st.GeoExprNameSpace.srid]</code>."""  # noqa: E501
    return geom(*columns).st.srid()


def set_srid(*columns: str, srid: IntoIntegerExpr) -> GeoExpr:
    """This function is syntactic sugar for <code>st.geom(columns).st.[set_srid(...)][polars_st.GeoExprNameSpace.set_srid]</code>."""  # noqa: E501
    return geom(*columns).st.set_srid(srid)


def to_srid(*columns: str, srid: int) -> GeoExpr:
    """This function is syntactic sugar for <code>st.geom(columns).st.[to_srid(...)][polars_st.GeoExprNameSpace.to_srid]</code>."""  # noqa: E501
    return geom(*columns).st.to_srid(srid)


def to_wkt(
    *columns: str,
    rounding_precision: int | None = 6,
    trim: bool = True,
    output_dimension: Literal[2, 3, 4] = 3,
    old_3d: bool = False,
) -> pl.Expr:
    """This function is syntactic sugar for <code>st.geom(columns).st.[to_wkt(...)][polars_st.GeoExprNameSpace.to_wkt]</code>."""  # noqa: E501
    return geom(*columns).st.to_wkt(rounding_precision, trim, output_dimension, old_3d)


def to_ewkt(
    *columns: str,
    rounding_precision: int | None = 6,
    trim: bool = True,
    output_dimension: Literal[2, 3, 4] = 3,
    old_3d: bool = False,
) -> pl.Expr:
    """This function is syntactic sugar for <code>st.geom(columns).st.[to_ewkt(...)][polars_st.GeoExprNameSpace.to_ewkt]</code>."""  # noqa: E501
    return geom(*columns).st.to_ewkt(rounding_precision, trim, output_dimension, old_3d)


def to_wkb(
    *columns: str,
    output_dimension: Literal[2, 3, 4] = 3,
    byte_order: Literal[0, 1] | None = None,
    include_srid: bool = False,
) -> pl.Expr:
    """This function is syntactic sugar for <code>st.geom(columns).st.[to_wkb(...)][polars_st.GeoExprNameSpace.to_wkb]</code>."""  # noqa: E501
    return geom(*columns).st.to_wkb(output_dimension, byte_order, include_srid)


def to_geojson(*columns: str, indent: int | None = None) -> pl.Expr:
    """This function is syntactic sugar for <code>st.geom(columns).st.[to_geojson(...)][polars_st.GeoExprNameSpace.to_geojson]</code>."""  # noqa: E501
    return geom(*columns).st.to_geojson(indent)


def to_shapely(*columns: str) -> pl.Expr:
    """This function is syntactic sugar for <code>st.geom(columns).st.[to_shapely()][polars_st.GeoExprNameSpace.to_shapely]</code>."""  # noqa: E501
    return geom(*columns).st.to_shapely()


def to_dict(*columns: str) -> pl.Expr:
    """This function is syntactic sugar for <code>st.geom(columns).st.[to_dict()][polars_st.GeoExprNameSpace.to_dict]</code>."""  # noqa: E501
    return geom(*columns).st.to_dict()


def has_z(*columns: str) -> pl.Expr:
    """This function is syntactic sugar for <code>st.geom(columns).st.[has_z()][polars_st.GeoExprNameSpace.has_z]</code>."""  # noqa: E501
    return geom(*columns).st.has_z()


def has_m(*columns: str) -> pl.Expr:
    """This function is syntactic sugar for <code>st.geom(columns).st.[has_m()][polars_st.GeoExprNameSpace.has_m]</code>."""  # noqa: E501
    return geom(*columns).st.has_m()


def is_ccw(*columns: str) -> pl.Expr:
    """This function is syntactic sugar for <code>st.geom(columns).st.[is_ccw()][polars_st.GeoExprNameSpace.is_ccw]</code>."""  # noqa: E501
    return geom(*columns).st.is_ccw()


def is_closed(*columns: str) -> pl.Expr:
    """This function is syntactic sugar for <code>st.geom(columns).st.[is_closed()][polars_st.GeoExprNameSpace.is_closed]</code>."""  # noqa: E501
    return geom(*columns).st.is_closed()


def is_empty(*columns: str) -> pl.Expr:
    """This function is syntactic sugar for <code>st.geom(columns).st.[is_empty()][polars_st.GeoExprNameSpace.is_empty]</code>."""  # noqa: E501
    return geom(*columns).st.is_empty()


def is_ring(*columns: str) -> pl.Expr:
    """This function is syntactic sugar for <code>st.geom(columns).st.[is_ring()][polars_st.GeoExprNameSpace.is_ring]</code>."""  # noqa: E501
    return geom(*columns).st.is_ring()


def is_simple(*columns: str) -> pl.Expr:
    """This function is syntactic sugar for <code>st.geom(columns).st.[is_simple()][polars_st.GeoExprNameSpace.is_simple]</code>."""  # noqa: E501
    return geom(*columns).st.is_simple()


def is_valid(*columns: str) -> pl.Expr:
    """This function is syntactic sugar for <code>st.geom(columns).st.[is_valid()][polars_st.GeoExprNameSpace.is_valid]</code>."""  # noqa: E501
    return geom(*columns).st.is_valid()


def is_valid_reason(*columns: str) -> pl.Expr:
    """This function is syntactic sugar for <code>st.geom(columns).st.[is_valid_reason()][polars_st.GeoExprNameSpace.is_valid_reason]</code>."""  # noqa: E501
    return geom(*columns).st.is_valid_reason()


def unary_union(*columns: str, grid_size: float | None = None) -> GeoExpr:
    """This function is syntactic sugar for <code>st.geom(columns).st.[unary_union(...)][polars_st.GeoExprNameSpace.unary_union]</code>."""  # noqa: E501
    return geom(*columns).st.unary_union(grid_size)


def coverage_union(*columns: str) -> GeoExpr:
    """This function is syntactic sugar for <code>st.geom(columns).st.[coverage_union()][polars_st.GeoExprNameSpace.coverage_union]</code>."""  # noqa: E501
    return geom(*columns).st.coverage_union()


def boundary(*columns: str) -> GeoExpr:
    """This function is syntactic sugar for <code>st.geom(columns).st.[boundary()][polars_st.GeoExprNameSpace.boundary]</code>."""  # noqa: E501
    return geom(*columns).st.boundary()


def buffer(
    *columns: str,
    distance: IntoDecimalExpr,
    quad_segs: int = 8,
    cap_style: Literal["round", "square", "flat"] = "round",
    join_style: Literal["round", "mitre", "bevel"] = "round",
    mitre_limit: float = 5.0,
    single_sided: bool = False,
) -> GeoExpr:
    """This function is syntactic sugar for <code>st.geom(columns).st.[buffer(...)][polars_st.GeoExprNameSpace.buffer]</code>."""  # noqa: E501
    return geom(*columns).st.buffer(
        distance,
        quad_segs,
        cap_style,
        join_style,
        mitre_limit,
        single_sided,
    )


def offset_curve(
    *columns: str,
    distance: IntoDecimalExpr,
    quad_segs: int = 8,
    join_style: Literal["round", "mitre", "bevel"] = "round",
    mitre_limit: float = 5.0,
) -> GeoExpr:
    """This function is syntactic sugar for <code>st.geom(columns).st.[offset_curve(...)][polars_st.GeoExprNameSpace.offset_curve]</code>."""  # noqa: E501
    return geom(*columns).st.offset_curve(distance, quad_segs, join_style, mitre_limit)


def centroid(*columns: str) -> GeoExpr:
    """This function is syntactic sugar for <code>st.geom(columns).st.[centroid()][polars_st.GeoExprNameSpace.centroid]</code>."""  # noqa: E501
    return geom(*columns).st.centroid()


def center(*columns: str) -> GeoExpr:
    """This function is syntactic sugar for <code>st.geom(columns).st.[center()][polars_st.GeoExprNameSpace.center]</code>."""  # noqa: E501
    return geom(*columns).st.center()


def clip_by_rect(
    *columns: str,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
) -> GeoExpr:
    """This function is syntactic sugar for <code>st.geom(columns).st.[clip_by_rect()][polars_st.GeoExprNameSpace.clip_by_rect]</code>."""  # noqa: E501
    return geom(*columns).st.clip_by_rect(xmin, ymin, xmax, ymax)


def convex_hull(*columns: str) -> GeoExpr:
    """This function is syntactic sugar for <code>st.geom(columns).st.[convex_hull()][polars_st.GeoExprNameSpace.convex_hull]</code>."""  # noqa: E501
    return geom(*columns).st.convex_hull()


def concave_hull(*columns: str, ratio: float = 0.0, allow_holes: bool = False) -> GeoExpr:
    """This function is syntactic sugar for <code>st.geom(columns).st.[concave_hull(...)][polars_st.GeoExprNameSpace.concave_hull]</code>."""  # noqa: E501
    return geom(*columns).st.concave_hull(ratio, allow_holes)


def segmentize(*columns: str, max_segment_length: IntoDecimalExpr) -> GeoExpr:
    """This function is syntactic sugar for <code>st.geom(columns).st.[segmentize(max_segment_length)][polars_st.GeoExprNameSpace.segmentize]</code>."""  # noqa: E501
    return geom(*columns).st.segmentize(max_segment_length)


def envelope(*columns: str) -> GeoExpr:
    """This function is syntactic sugar for <code>st.geom(columns).st.[envelope()][polars_st.GeoExprNameSpace.envelope]</code>."""  # noqa: E501
    return geom(*columns).st.envelope()


def extract_unique_points(*columns: str) -> GeoExpr:
    """This function is syntactic sugar for <code>st.geom(columns).st.[extract_unique_points()][polars_st.GeoExprNameSpace.extract_unique_points]</code>."""  # noqa: E501
    return geom(*columns).st.extract_unique_points()


def build_area(*columns: str) -> GeoExpr:
    """This function is syntactic sugar for <code>st.geom(columns).st.[build_area()][polars_st.GeoExprNameSpace.build_area]</code>."""  # noqa: E501
    return geom(*columns).st.build_area()


def make_valid(*columns: str) -> GeoExpr:
    """This function is syntactic sugar for <code>st.geom(columns).st.[make_valid()][polars_st.GeoExprNameSpace.make_valid]</code>."""  # noqa: E501
    return geom(*columns).st.make_valid()


def normalize(*columns: str) -> GeoExpr:
    """This function is syntactic sugar for <code>st.geom(columns).st.[normalize()][polars_st.GeoExprNameSpace.normalize]</code>."""  # noqa: E501
    return geom(*columns).st.normalize()


def node(*columns: str) -> GeoExpr:
    """This function is syntactic sugar for <code>st.geom(columns).st.[node()][polars_st.GeoExprNameSpace.node]</code>."""  # noqa: E501
    return geom(*columns).st.node()


def point_on_surface(*columns: str) -> GeoExpr:
    """This function is syntactic sugar for <code>st.geom(columns).st.[point_on_surface()][polars_st.GeoExprNameSpace.point_on_surface]</code>."""  # noqa: E501
    return geom(*columns).st.point_on_surface()


def remove_repeated_points(*columns: str, tolerance: IntoDecimalExpr = 0.0) -> GeoExpr:
    """This function is syntactic sugar for <code>st.geom(columns).st.[remove_repeated_points(...)][polars_st.GeoExprNameSpace.remove_repeated_points]</code>."""  # noqa: E501
    return geom(*columns).st.remove_repeated_points(tolerance)


def reverse(*columns: str) -> GeoExpr:
    """This function is syntactic sugar for <code>st.geom(columns).st.[reverse()][polars_st.GeoExprNameSpace.reverse]</code>."""  # noqa: E501
    return geom(*columns).st.reverse()


def simplify(
    *columns: str,
    tolerance: IntoDecimalExpr,
    preserve_topology: bool = True,
) -> GeoExpr:
    """This function is syntactic sugar for <code>st.geom(columns).st.[simplify(...)][polars_st.GeoExprNameSpace.simplify]</code>."""  # noqa: E501
    return geom(*columns).st.simplify(tolerance, preserve_topology)


def force_2d(*columns: str) -> GeoExpr:
    """This function is syntactic sugar for <code>st.geom(columns).st.[force_2d()][polars_st.GeoExprNameSpace.force_2d]</code>."""  # noqa: E501
    return geom(*columns).st.force_2d()


def force_3d(*columns: str, z: IntoDecimalExpr = 0.0) -> GeoExpr:
    """This function is syntactic sugar for <code>st.geom(columns).st.[force_3d(...)][polars_st.GeoExprNameSpace.force_3d]</code>."""  # noqa: E501
    return geom(*columns).st.force_3d(z)


def flip_coordinates(*columns: str) -> GeoExpr:
    """This function is syntactic sugar for <code>st.geom(columns).st.[flip_coordinates()][polars_st.GeoExprNameSpace.flip_coordinates]</code>."""  # noqa: E501
    return geom(*columns).st.flip_coordinates()


def minimum_rotated_rectangle(*columns: str) -> GeoExpr:
    """This function is syntactic sugar for <code>st.geom(columns).st.[minimum_rotated_rectangle()][polars_st.GeoExprNameSpace.minimum_rotated_rectangle]</code>."""  # noqa: E501
    return geom(*columns).st.minimum_rotated_rectangle()


def affine_transform(*columns: str, matrix: IntoExprColumn | Sequence[float]) -> GeoExpr:
    """This function is syntactic sugar for <code>st.geom(columns).st.[affine_transform(...)][polars_st.GeoExprNameSpace.affine_transform]</code>."""  # noqa: E501
    return geom(*columns).st.affine_transform(matrix)


def translate(
    *columns: str,
    x: IntoDecimalExpr = 0.0,
    y: IntoDecimalExpr = 0.0,
    z: IntoDecimalExpr = 0.0,
) -> GeoExpr:
    """This function is syntactic sugar for <code>st.geom(columns).st.[translate(...)][polars_st.GeoExprNameSpace.translate]</code>."""  # noqa: E501
    return geom(*columns).st.translate(x, y, z)


def rotate(
    *columns: str,
    angle: IntoDecimalExpr,
    origin: Literal["center", "centroid"] | Sequence[float] | pl.Expr | pl.Series = "center",
) -> GeoExpr:
    """This function is syntactic sugar for <code>st.geom(columns).st.[rotate(...)][polars_st.GeoExprNameSpace.rotate]</code>."""  # noqa: E501
    return geom(*columns).st.rotate(angle, origin)


def scale(
    *columns: str,
    x: IntoDecimalExpr = 1.0,
    y: IntoDecimalExpr = 1.0,
    z: IntoDecimalExpr = 1.0,
    origin: Literal["center", "centroid"] | Sequence[float] | pl.Expr | pl.Series = "center",
) -> GeoExpr:
    """This function is syntactic sugar for <code>st.geom(columns).st.[scale(...)][polars_st.GeoExprNameSpace.scale]</code>."""  # noqa: E501
    return geom(*columns).st.scale(x, y, z, origin)


def interpolate(
    *columns: str,
    distance: IntoDecimalExpr,
    normalized: bool = False,
) -> GeoExpr:
    """This function is syntactic sugar for <code>st.geom(columns).st.[interpolate()][polars_st.GeoExprNameSpace.interpolate]</code>."""  # noqa: E501
    return geom(*columns).st.interpolate(distance, normalized)


def line_merge(*columns: str, directed: bool = False) -> GeoExpr:
    """This function is syntactic sugar for <code>st.geom(columns).st.[line_merge()][polars_st.GeoExprNameSpace.line_merge]</code>."""  # noqa: E501
    return geom(*columns).st.line_merge(directed)


def total_bounds(*columns: str) -> pl.Expr:
    """This function is syntactic sugar for <code>st.geom(columns).st.[total_bounds()][polars_st.GeoExprNameSpace.total_bounds]</code>."""  # noqa: E501
    return geom(*columns).st.total_bounds()


def multipoint(*columns: str) -> GeoExpr:
    """This function is syntactic sugar for <code>st.geom(columns).st.[multipoint()][polars_st.GeoExprNameSpace.multipoint]</code>."""  # noqa: E501
    return geom(*columns).st.multipoint()


def multilinestring(*columns: str) -> GeoExpr:
    """This function is syntactic sugar for <code>st.geom(columns).st.[multilinestring()][polars_st.GeoExprNameSpace.multilinestring]</code>."""  # noqa: E501
    return geom(*columns).st.multilinestring()


def multipolygon(*columns: str) -> GeoExpr:
    """This function is syntactic sugar for <code>st.geom(columns).st.[multipolygon()][polars_st.GeoExprNameSpace.multipolygon]</code>."""  # noqa: E501
    return geom(*columns).st.multipolygon()


def geometrycollection(*columns: str) -> GeoExpr:
    """This function is syntactic sugar for <code>st.geom(columns).st.[geometrycollection()][polars_st.GeoExprNameSpace.geometrycollection]</code>."""  # noqa: E501
    return geom(*columns).st.geometrycollection()


def collect(*columns: str) -> GeoExpr:
    """This function is syntactic sugar for <code>st.geom(columns).st.[geometrycollection()][polars_st.GeoExprNameSpace.geometrycollection]</code>."""  # noqa: E501
    return geom(*columns).st.collect()


def union_all(*columns: str, grid_size: float | None = None) -> GeoExpr:
    """This function is syntactic sugar for <code>st.geom(columns).st.[union_all(...)][polars_st.GeoExprNameSpace.union_all]</code>."""  # noqa: E501
    return geom(*columns).st.union_all(grid_size)


def coverage_union_all(*columns: str) -> GeoExpr:
    """This function is syntactic sugar for <code>st.geom(columns).st.[coverage_union_all()][polars_st.GeoExprNameSpace.coverage_union_all]</code>."""  # noqa: E501
    return geom(*columns).st.coverage_union_all()


def intersection_all(*columns: str, grid_size: float | None = None) -> GeoExpr:
    """This function is syntactic sugar for <code>st.geom(columns).st.[intersection_all()][polars_st.GeoExprNameSpace.intersection_all]</code>."""  # noqa: E501
    return geom(*columns).st.intersection_all(grid_size)


def difference_all(*columns: str, grid_size: float | None = None) -> GeoExpr:
    """This function is syntactic sugar for <code>st.geom(columns).st.[difference_all(...)][polars_st.GeoExprNameSpace.difference_all]</code>."""  # noqa: E501
    return geom(*columns).st.difference_all(grid_size)


def symmetric_difference_all(*columns: str, grid_size: float | None = None) -> GeoExpr:
    """This function is syntactic sugar for <code>st.geom(columns).st.[symmetric_difference_all(...)][polars_st.GeoExprNameSpace.symmetric_difference_all]</code>."""  # noqa: E501
    return geom(*columns).st.symmetric_difference_all(grid_size)


def polygonize(*columns: str) -> GeoExpr:
    """This function is syntactic sugar for <code>st.geom(columns).st.[polygonize()][polars_st.GeoExprNameSpace.polygonize]</code>."""  # noqa: E501
    return geom(*columns).st.polygonize()


def voronoi_polygons(
    *columns: str,
    tolerance: float = 0.0,
    extend_to: bytes | None = None,
    only_edges: bool = False,
) -> GeoExpr:
    """This function is syntactic sugar for <code>st.geom(columns).st.[voronoi_polygons(...)][polars_st.GeoExprNameSpace.voronoi_polygons]</code>."""  # noqa: E501
    return geom(*columns).st.voronoi_polygons(tolerance, extend_to, only_edges)


def delaunay_triangles(
    *columns: str,
    tolerance: float = 0.0,
    only_edges: bool = False,
) -> GeoExpr:
    """This function is syntactic sugar for <code>st.geom(columns).st.[delaunay_triangles(...)][polars_st.GeoExprNameSpace.delaunay_triangles]</code>."""  # noqa: E501
    return geom(*columns).st.delaunay_triangles(tolerance, only_edges)
