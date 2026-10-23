from typing import TypedDict

import pytest

import polars_st as st
from polars_st.typing import SjoinPredicate

# Each case checks a single left/right geometry pair against a single predicate, verifying
# both that matching pairs are found, and that the predicate is evaluated in the correct
# direction (`left <predicate> right`, not the other way around).


cases = [
    (
        "intersects_bbox",
        {},
        "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))",
        "POLYGON ((0.5 0.5, 2 0.5, 2 2, 0.5 2, 0.5 0.5))",
        True,
    ),
    (
        "intersects_bbox",
        {},
        "POINT (0 0)",
        "POINT (10 10)",
        False,
    ),
    (
        "intersects",
        {},
        "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))",
        "POLYGON ((1 1, 3 1, 3 3, 1 3, 1 1))",
        True,
    ),
    (
        "intersects",
        {},
        "POINT (0 0)",
        "POINT (10 10)",
        False,
    ),
    (
        "within",
        {},
        "POINT (1 1)",
        "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))",
        True,
    ),
    (
        "within",
        {},
        "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))",
        "POINT (1 1)",
        False,
    ),
    (
        "contains",
        {},
        "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))",
        "POINT (1 1)",
        True,
    ),
    (
        "contains",
        {},
        "POINT (1 1)",
        "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))",
        False,
    ),
    (
        "overlaps",
        {},
        "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))",
        "POLYGON ((1 1, 3 1, 3 3, 1 3, 1 1))",
        True,
    ),
    (
        "overlaps",
        {},
        "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))",
        "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))",
        False,
    ),
    (
        "crosses",
        {},
        "LINESTRING (0 0, 2 2)",
        "LINESTRING (0 2, 2 0)",
        True,
    ),
    (
        "crosses",
        {},
        "LINESTRING (0 0, 1 1)",
        "LINESTRING (5 5, 6 6)",
        False,
    ),
    (
        "touches",
        {},
        "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))",
        "POLYGON ((1 0, 2 0, 2 1, 1 1, 1 0))",
        True,
    ),
    (
        "touches",
        {},
        "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))",
        "POLYGON ((1 1, 3 1, 3 3, 1 3, 1 1))",
        False,
    ),
    (
        "covers",
        {},
        "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))",
        "LINESTRING (0 0, 2 0)",
        True,
    ),
    (
        "covers",
        {},
        "LINESTRING (0 0, 2 0)",
        "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))",
        False,
    ),
    (
        "covered_by",
        {},
        "LINESTRING (0 0, 2 0)",
        "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))",
        True,
    ),
    (
        "covered_by",
        {},
        "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))",
        "LINESTRING (0 0, 2 0)",
        False,
    ),
    (
        "contains_properly",
        {},
        "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))",
        "POINT (2 2)",
        True,
    ),
    (
        "contains_properly",
        {},
        "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))",
        "POINT (0 0)",
        False,
    ),
    (
        "dwithin",
        {"distance": 3.0},
        "POINT (0 0)",
        "POINT (2 0)",
        True,
    ),
    (
        "dwithin",
        {"distance": 1.0},
        "POINT (0 0)",
        "POINT (2 0)",
        False,
    ),
]


class SjoinKwargs(TypedDict, total=False):
    distance: float
    pattern: str


@pytest.mark.parametrize(("predicate", "kwargs", "left_wkt", "right_wkt", "expected"), cases)
def test_sjoin_predicate(
    predicate: SjoinPredicate,
    kwargs: SjoinKwargs,
    left_wkt: str,
    right_wkt: str,
    expected: bool,
) -> None:
    left = st.GeoDataFrame({"geometry": [left_wkt]})
    right = st.GeoDataFrame({"geometry": [right_wkt]})

    result = left.st.sjoin(right, predicate=predicate, how="inner", **kwargs)

    assert len(result) == (1 if expected else 0)


# Small geometries against detailed polygons: the polygons are the larger geometries and, with the
# small ones repeated, each of them is reused many times, so `sjoin` prepares the right side when
# they are on the right (transposing the predicate) and the left side when they are on the left.
# Both must give the pairs the element-wise predicate gives on every pair.
small = [
    "POINT (1 1)",  # inside the first polygon
    "POINT (0 2)",  # on its boundary
    "POINT (5 5)",  # inside its hole
    "POINT (12 1)",  # inside the second part of the multipolygon
    "POINT (20 20)",  # in nothing
    "LINESTRING (9 5, 11 5)",  # crosses the first polygon's boundary
    "LINESTRING (13 0, 13 2)",  # lies on the multipolygon's boundary
    "POLYGON ((9 9, 11 9, 11 11, 9 11, 9 9))",  # overlaps the first polygon
]
polygons = [
    "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0), (4 4, 6 4, 6 6, 4 6, 4 4))",
    "MULTIPOLYGON (((11 0, 13 0, 13 2, 11 2, 11 0)), ((20 0, 22 0, 22 2, 20 2, 20 0)))",
]
predicates: list[tuple[SjoinPredicate, SjoinKwargs]] = [
    ("intersects", {}),
    ("within", {}),
    ("contains", {}),
    ("covers", {}),
    ("covered_by", {}),
    ("contains_properly", {}),
    ("overlaps", {}),
    ("crosses", {}),
    ("touches", {}),
    ("dwithin", {"distance": 1.5}),
    ("relate_pattern", {"pattern": "T*F**F***"}),
]


@pytest.mark.parametrize(("predicate", "kwargs"), predicates)
@pytest.mark.parametrize("small_on_left", [True, False])
def test_sjoin_matches_elementwise_predicate(
    predicate: SjoinPredicate, kwargs: SjoinKwargs, small_on_left: bool
) -> None:
    repeated = small * 1000
    smalls = st.GeoDataFrame({"p": range(len(repeated)), "geometry": repeated})
    polys = st.GeoDataFrame(
        st.GeoDataFrame({"z": range(len(polygons)), "geometry": polygons}).with_columns(
            st.geom().st.segmentize(0.1)
        )
    )
    left, right = (smalls, polys) if small_on_left else (polys, smalls)

    joined = left.st.sjoin(right, predicate=predicate, how="inner", **kwargs)
    expected = left.join(right, how="cross").filter(
        getattr(st.geom("geometry").st, predicate)(st.geom("geometry_right"), **kwargs)
    )

    assert sorted(joined.select("p", "z").rows()) == sorted(expected.select("p", "z").rows())
