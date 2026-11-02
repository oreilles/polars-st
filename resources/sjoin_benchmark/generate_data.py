"""Synthetic inputs for the sjoin benchmark: zonings of a made-up country, and smaller polygons.

The shapes stand in for a national zoning and a mobile-network Voronoi tessellation, with the sizes
that matter for sjoin: 615 zones of about 930 km² (~365 / ~2.4k / ~11k vertices each once
densified) and 1,922 Voronoi cells of about 300 km², in a projected CRS in metres, inside a
bounding box of 1,200 by 1,390 km of which the country covers about a third. Everything is seeded:
the same files on every run.

Usage: python generate_data.py OUT_DIR

Writes zones.parquet, zones_d50.parquet, zones_d10.parquet (columns zone_id, geometry) and
voronoi.parquet (columns ID, geometry), geometries as WKB.
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl
import shapely

SEED = 88
WIDTH, HEIGHT = 1_200_000.0, 1_390_000.0
COUNTRY_KM2 = 570_000.0
ZONES = 615
VORONOI = 1922
# Densification of the zones' edges, in metres: about 365, 2.4k and 11k vertices per zone.
DENSIFY = {"zones": 340.0, "zones_d50": 52.0, "zones_d10": 11.2}
VORONOI_DENSIFY = 1700.0


def country(rng: np.random.Generator) -> shapely.Polygon:
    """An elongated outline of `COUNTRY_KM2` running corner to corner across the bounding box.

    A wiggly diagonal spine is widened until the outline, fitted to the box, has the wanted area: a
    long country fills a third of its box, as a real one does, and most of a grid over it is sea.
    """
    t = np.linspace(0, 1, 200)
    wiggle = sum(
        rng.uniform(0.02, 0.05) * np.sin(k * np.pi * t + rng.uniform(0, 2 * np.pi))
        for k in (2, 3, 5)
    )
    spine = shapely.LineString(np.column_stack([t + wiggle, 1 - t + wiggle[::-1]]))

    def fitted(width: float) -> shapely.Polygon:
        shape = spine.buffer(width, quad_segs=16)
        xmin, ymin, xmax, ymax = shape.bounds
        shape = shapely.affinity.translate(shape, -xmin, -ymin)
        return shapely.affinity.scale(
            shape, WIDTH / (xmax - xmin), HEIGHT / (ymax - ymin), origin=(0, 0)
        )

    low, high = 0.01, 0.5
    for _ in range(40):  # the area grows with the width: bisect for it
        middle = (low + high) / 2
        low, high = (middle, high) if fitted(middle).area < COUNTRY_KM2 * 1e6 else (low, middle)
    return fitted((low + high) / 2)


def tessellation(rng: np.random.Generator, outline: shapely.Polygon, count: int) -> np.ndarray:
    """The Voronoi cells of `count` random points of `outline`, clipped to it, in their order."""
    xmin, ymin, xmax, ymax = outline.bounds
    seeds: list[np.ndarray] = []
    while sum(len(s) for s in seeds) < count:
        candidates = rng.uniform((xmin, ymin), (xmax, ymax), (count, 2))
        seeds.append(candidates[shapely.contains_xy(outline, candidates[:, 0], candidates[:, 1])])
    points = shapely.points(np.concatenate(seeds)[:count])
    cells = shapely.get_parts(
        shapely.voronoi_polygons(shapely.multipoints(points), extend_to=outline)
    )
    # Cells come in no particular order: each is matched to the seed it holds.
    tree = shapely.STRtree(cells)
    seed_index, cell_index = tree.query(points, predicate="within")
    ordered = cells[cell_index[np.argsort(seed_index)]]
    return shapely.intersection(ordered, outline)


def write(path: Path, id_column: str, ids: list[str], geoms: np.ndarray) -> None:
    pl.DataFrame({
        id_column: ids,
        "geometry": pl.Series(shapely.to_wkb(geoms), dtype=pl.Binary),
    }).write_parquet(path)
    vertices = shapely.get_num_coordinates(geoms)
    area = shapely.area(geoms).mean() / 1e6
    print(f"{path.name}: {len(geoms)} polygons, {vertices.mean():.0f} vertices, {area:.0f} km²")


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    outline = country(rng)
    zones = tessellation(rng, outline, ZONES)
    for name, spacing in DENSIFY.items():
        write(
            out / f"{name}.parquet",
            "zone_id",
            [f"z{i:04d}" for i in range(ZONES)],
            shapely.segmentize(zones, spacing),
        )
    cells = shapely.segmentize(tessellation(rng, outline, VORONOI), VORONOI_DENSIFY)
    write(out / "voronoi.parquet", "ID", [f"v{i:04d}" for i in range(VORONOI)], cells)


if __name__ == "__main__":
    main()
