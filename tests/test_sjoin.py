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
