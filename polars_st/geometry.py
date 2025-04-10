from __future__ import annotations

from typing import Literal, TypeAlias, get_args

import polars as pl

__all__ = ["GeometryType", "PolarsGeometryType"]

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

PolarsGeometryType = pl.Enum(get_args(GeometryType))
