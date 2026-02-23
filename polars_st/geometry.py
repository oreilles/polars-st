from __future__ import annotations

from typing import Literal, TypeAlias, get_args

import polars as pl

__all__ = [
    "CoordinateType",
    "DimensionType",
    "GeometryType",
    "PolarsCoordinateType",
    "PolarsDimensionType",
    "PolarsGeometryType",
]

GeometryType: TypeAlias = Literal[
    "Unknown",
    "Point",
    "LineString",
    "Polygon",
    "MultiPoint",
    "MultiLineString",
    "MultiPolygon",
    "GeometryCollection",
    "CircularString",
    "CompoundCurve",
    "CurvePolygon",
    "MultiCurve",
    "MultiSurface",
    "Curve",
    "Surface",
    "PolyhedralSurface",
    "Tin",
    "Triangle",
]

DimensionType: TypeAlias = Literal[
    "Point",
    "Curve",
    "Surface",
]

CoordinateType: TypeAlias = Literal[
    "XY",
    "XYZ",
    "XYZM",
    "XYM",
]

PolarsGeometryType = pl.Enum(get_args(GeometryType))
PolarsDimensionType = pl.Enum(get_args(DimensionType))
PolarsCoordinateType = pl.Enum(get_args(CoordinateType))
