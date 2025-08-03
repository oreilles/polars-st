import inspect
from typing import TYPE_CHECKING

import polars_st as st

if TYPE_CHECKING:
    from collections.abc import Callable


def signature_matches(left: inspect.Signature, right: inspect.Signature) -> bool:
    """Returns True if both signatures share the same parameters names and default values."""
    assert left.return_annotation == right.return_annotation.replace("Series", "Expr")

    assert set(left.parameters) == set(right.parameters)

    for a, b in zip(
        left.parameters.values(),
        right.parameters.values(),
        strict=True,
    ):
        assert a.name == b.name
        assert a.default == b.default
        assert str(a.annotation) == str(b.annotation)
    return True


def patch_sugar_signature(sig: inspect.Signature) -> inspect.Signature:
    """Returns a sugar function signature with patched parameters for comparison with GeoExpr."""
    params = [
        p.replace(name="self", annotation=inspect._empty) if i == 0 else p  # noqa: SLF001
        for i, p in enumerate(sig.parameters.values())
    ]
    return sig.replace(parameters=params)


def test_series_signatures_matches_expr():
    """All GeoExpr functions should have a matching GeoSeries function."""
    expr_funcs: dict[str, Callable] = {
        name: func
        for name, func in st.geoexpr.GeoExprNameSpace.__dict__.items()
        if callable(func) and not name.startswith("_")
    }
    series_funcs: dict[str, Callable] = {
        name: func
        for name, func in st.geoseries.GeoSeriesNameSpace.__dict__.items()
        if callable(func)
        and not name.startswith("_")
        and name not in {"plot", "explore", "to_geopandas"}
    }
    assert set(expr_funcs.keys()) == set(series_funcs.keys())
    for expr_func, series_func in zip(expr_funcs.values(), series_funcs.values(), strict=True):
        assert expr_func.__name__ == series_func.__name__
        assert signature_matches(
            inspect.signature(expr_func),
            inspect.signature(series_func),
        ), expr_func.__name__


def test_sugar_signatures_matches_expr():
    """All GeoExpr functions should have a matching GeoSeries function."""
    expr_funcs: dict[str, Callable] = {
        name: func
        for name, func in st.geoexpr.GeoExprNameSpace.__dict__.items()
        if callable(func) and not name.startswith("_")
    }
    sugar_funcs: dict[str, Callable] = {
        name: func
        for name, func in st.sugar.__dict__.items()
        if callable(func) and name in st.sugar.__all__
    }
    expr_funcs = {name: func for name, func in expr_funcs.items() if name in sugar_funcs}

    assert set(sugar_funcs.keys()).issubset(expr_funcs.keys())
    for expr_func, sugar_func in zip(expr_funcs.values(), sugar_funcs.values(), strict=True):
        assert expr_func.__name__ == sugar_func.__name__
        assert signature_matches(
            inspect.signature(expr_func),
            patch_sugar_signature(inspect.signature(sugar_func)),
        )
