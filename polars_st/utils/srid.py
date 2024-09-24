from __future__ import annotations

from logging import warning

from polars_st._lib import get_crs_auth_code


def get_crs_srid_or_warn(crs: str) -> int | None:
    try:
        _auth, code = get_crs_auth_code(crs)
        if code.isdigit():
            return int(code, base=10)
        warning.warn(
            f"Found an authority for {crs} but couldn't"
            f'convert code "{code}" to an integer srid. ',
            "The geometries SRID will be set to 0.",
        )
    except ValueError:
        warning.warn(
            f'Couldn\'t find a matching crs for "{crs}". The geometries SRID will be set to 0.'
        )
    return None
