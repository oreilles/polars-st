from __future__ import annotations

import logging

from polars_st._lib import get_crs_auth_code

logger = logging.getLogger(__name__)


def get_crs_srid_or_warn(crs: str) -> int | None:
    try:
        _auth, code = get_crs_auth_code(crs)
        if code.isdigit():
            return int(code, base=10)
        msg = (
            f'Found an authority for {crs} but couldn\'t convert code "{code}" to an integer srid.'
        )
        logger.warning(msg)
    except ValueError:
        msg = f'Couldn\'t find a matching crs for "{crs}". The geometries SRID will be set to 0.'
        logger.warning(msg)
    return None
