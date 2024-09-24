from __future__ import annotations

from logging import warning
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyproj import CRS


def get_crs_srid_or_warn(crs: CRS) -> int | None:
    if authority := crs.to_authority():
        _auth, code = authority
        if code.isdigit():
            return int(code, base=10)
        warning.warn(
            f"Found an authority for {crs} but couldn't"
            f'convert code "{code}" to an integer srid. ',
            "The geometries SRID will be set to 0.",
        )
    else:
        warning.warn(
            f'Couldn\'t find an authority for crs "{crs}". The geometries SRID will be set to 0.'
        )
    return None
