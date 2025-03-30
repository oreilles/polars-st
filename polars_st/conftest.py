import pprint

import polars as pl
import pytest

import polars_st as st


@pytest.fixture(autouse=True)
def setup(doctest_namespace: dict) -> None:
    doctest_namespace["pl"] = pl
    doctest_namespace["st"] = st
    doctest_namespace["pprint"] = pprint
