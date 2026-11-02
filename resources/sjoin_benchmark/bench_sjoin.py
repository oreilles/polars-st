"""Time `sjoin` of the installed polars-st on the data of generate_data.py.

Usage: python bench_sjoin.py DATA_DIR OUT.json

Each case prints the best of 3 runs and a signature of the sorted pairs it returns, so that builds
can be compared on time and checked to return the same pairs. Run it once per build, with the same
data, on an otherwise idle machine; compare the results with compare.py.
"""

import json
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import polars as pl
import shapely

import polars_st as st

STEP = 2000.0  # metres between the points of the grid: ~416k points over the bounding box


def grid(zones: pl.DataFrame, step: float) -> tuple[np.ndarray, np.ndarray]:
    xmin, ymin, xmax, ymax = shapely.total_bounds(shapely.from_wkb(zones["geometry"].to_numpy()))
    xs, ys = np.meshgrid(
        np.arange(xmin, xmax, step) + step / 2, np.arange(ymin, ymax, step) + step / 2
    )
    return xs.ravel(), ys.ravel()


def frame(geoms: np.ndarray) -> pl.DataFrame:
    return pl.DataFrame({
        "p": np.arange(len(geoms)),
        "geometry": pl.Series(shapely.to_wkb(geoms), dtype=pl.Binary),
    })


def run(
    left: pl.DataFrame, right: pl.DataFrame, predicate: str, **kwargs: object
) -> tuple[float, int, int]:
    best, out = float("inf"), None
    for _ in range(3):
        t = time.perf_counter()
        out = left.st.sjoin(right, predicate=predicate, how="inner", **kwargs)
        best = min(best, time.perf_counter() - t)
    return best, out.height, int(out.select("p", "z").sort(pl.all()).hash_rows().sum())


Case = tuple[str, pl.DataFrame, pl.DataFrame, str, dict[str, object]]


def cases(data: Path) -> Iterator[Case]:
    voronoi = pl.read_parquet(data / "voronoi.parquet").select(pl.col("ID").alias("p"), "geometry")
    for name in ("zones", "zones_d50", "zones_d10"):
        zones = pl.read_parquet(data / f"{name}.parquet").select(
            pl.col("zone_id").alias("z"), "geometry"
        )
        xs, ys = grid(zones, STEP)
        points = frame(shapely.points(xs, ys))
        squares = frame(shapely.box(xs - 500, ys - 500, xs + 500, ys + 500))  # 1 km cells
        segments = frame(
            shapely.linestrings(
                np.stack([np.stack([xs - 700, ys], 1), np.stack([xs + 700, ys], 1)], 1)
            )
        )
        n = len(points)
        yield f"{n} points intersects {name}", points, zones, "intersects", {}
        yield f"{n} points within {name}", points, zones, "within", {}
        yield f"{n} points contains_properly {name}", points, zones, "contains_properly", {}
        yield (
            f"{n} points relate T*F**F*** {name}",
            points,
            zones,
            "relate_pattern",
            {"pattern": "T*F**F***"},
        )
        yield f"{n} points dwithin 500 m {name}", points, zones, "dwithin", {"distance": 500.0}
        yield f"{n} 1 km squares overlaps {name}", squares, zones, "overlaps", {}
        yield f"{n} 1 km squares covered_by {name}", squares, zones, "covered_by", {}
        yield f"{n} 1.4 km segments crosses {name}", segments, zones, "crosses", {}
        yield f"{n} 1.4 km segments touches {name}", segments, zones, "touches", {}
        if name == "zones":
            # Four times the points, and the polygons on the left (each repeated once per match).
            xs4, ys4 = grid(zones, STEP / 2)
            yield (
                f"{len(xs4)} points intersects {name}",
                frame(shapely.points(xs4, ys4)),
                zones,
                "intersects",
                {},
            )
            yield (
                f"{name} contains {n} points",
                zones.rename({"z": "p"}),
                points.rename({"p": "z"}),
                "contains",
                {},
            )
        yield f"{len(voronoi)} voronois intersects {name}", voronoi, zones, "intersects", {}
    yield (
        f"{len(voronoi)} voronois touches voronois",
        voronoi,
        voronoi.rename({"p": "z"}),
        "touches",
        {},
    )


def main() -> None:
    data, out = Path(sys.argv[1]), Path(sys.argv[2])
    results = []
    for label, left, right, predicate, kwargs in cases(data):
        seconds, rows, sig = run(left, right, predicate, **kwargs)
        print(f"{label:<50} {seconds:8.3f} s  rows={rows:<8} sig={sig}", flush=True)
        results.append({"case": label, "seconds": seconds, "rows": rows, "sig": sig})
    out.write_text(
        json.dumps(
            {"polars_st": st.__version__, "polars": pl.__version__, "results": results}, indent=1
        )
    )


if __name__ == "__main__":
    main()
