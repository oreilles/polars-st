from __future__ import annotations

from enum import StrEnum

import polars as pl

__all__ = ["GeometryType", "PolarsGeometryType"]


class GeometryType(StrEnum):
    Unknown = "Unknown"
    Point = "Point"
    LineString = "LineString"
    Polygon = "Polygon"
    MultiPoint = "MultiPoint"
    MultiLineString = "MultiLineString"
    MultiPolygon = "MultiPolygon"
    GeometryCollection = "GeometryCollection"
    CircularString = "CircularString"
    CompoundCurve = "CompoundCurve"
    CurvePolygon = "CurvePolygon"
    MultiCurve = "MultiCurve"
    MultiSurface = "MultiSurface"
    Curve = "Curve"
    Surface = "Surface"
    PolyhedralSurface = "PolyhedralSurface"
    Tin = "Tin"
    Triangle = "Triangle"


PolarsGeometryType = pl.Enum(GeometryType)
