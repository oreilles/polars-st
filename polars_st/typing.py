from __future__ import annotations

from typing import Literal, TypeAlias

import polars as pl

IntoExprColumn: TypeAlias = pl.Expr | pl.Series | str
IntoGeoExprColumn: TypeAlias = IntoExprColumn
IntoIntegerExpr: TypeAlias = IntoExprColumn | int
IntoNumericExpr: TypeAlias = IntoExprColumn | int | float

GeometryFormat: TypeAlias = Literal[
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
    "rectangle",
]

SjoinPredicate: TypeAlias = Literal[
    "intersects_bbox",
    "intersects",
    "within",
    "dwithin",
    "contains",
    "overlaps",
    "crosses",
    "touches",
    "covers",
    "covered_by",
    "contains_properly",
    "relate_pattern",
]

CapStyle: TypeAlias = Literal["round", "square", "flat"]
JoinStyle: TypeAlias = Literal["round", "mitre", "bevel"]
TransformOrigin: TypeAlias = Literal["center", "centroid"]
OutputDimension: TypeAlias = Literal[2, 3, 4]
ByteOrder: TypeAlias = Literal[0, 1]
MakeValidMethod: TypeAlias = Literal["linework", "structure"]
PrecisionMode: TypeAlias = Literal["valid_output", "no_topo", "keep_collapsed"]
