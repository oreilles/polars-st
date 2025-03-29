from __future__ import annotations

import logging

from polars_st._lib import get_crs_authority

logger = logging.getLogger(__name__)


def get_crs_srid_or_warn(crs: str) -> int | None:
    authority = get_crs_authority(crs)
    if authority is None:
        msg = f"Couldn't infer authority from {crs}.\n The geometries SRID will be set to 0."
        logger.warning(msg)
        return None

    name, code = authority
    if not code.isdigit():
        msg = f'Couldn\'t convert authority "{name}:{code}" to an integer srid.'
        logger.warning(msg)
        return None

    return int(code, base=10)
