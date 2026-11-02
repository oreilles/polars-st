# sjoin benchmark

The inputs and the code behind the `sjoin` timings discussed in #88 and #89.

The data is synthetic and seeded, so every run produces the same files. It stands in for a national
zoning (615 zones) and a mobile-network Voronoi tessellation (1,922 cells), with the properties that
drive `sjoin`'s cost: many small geometries on one side (a regular grid of ~416k points, 1 km squares
or 1.4 km segments) against a few large polygons on the other. The polygons are densified to three
levels of detail, about 365, 2.4k and 11k vertices each.

| File | Polygons | Vertices per polygon (mean) | Area per polygon (mean) |
|---|---|---|---|
| `zones.parquet` | 615 | ~365 | ~930 km² |
| `zones_d50.parquet` | 615 | ~2.4k | ~930 km² |
| `zones_d10.parquet` | 615 | ~11k | ~930 km² |
| `voronoi.parquet` | 1,922 | ~45 | ~300 km² |

Coordinates are in metres in a projected CRS. The country fills about a third of a 1,200 × 1,390 km
bounding box, so most grid points fall outside every zone, as they would for a real country.

## Run

```sh
python resources/sjoin_benchmark/generate_data.py /tmp/sjoin_data   # a few seconds
python resources/sjoin_benchmark/bench_sjoin.py /tmp/sjoin_data main.json
```

Each case is the best of 3 runs. It also records a signature of the sorted
pairs, so builds can be checked to return the same result.

`compare.py` prints the runs side by side as a Markdown table, with a column saying whether every
build returned the same pairs:

```sh
python resources/sjoin_benchmark/compare.py main=main.json "#89"=pr89.json
```

On `main`, the dense zonings take minutes in all, and the `relate_pattern` cases need the two-line
`param` fix in `geolazyframe.py` that #89 includes.

## Cases

For each zoning, on the left, against the zones on the right:

- the ~416k points with `intersects`, `within`, `contains_properly`, `relate_pattern` (`T*F**F***`)
  and `dwithin` (500 m);
- 1 km squares around the same points with `overlaps` and `covered_by`;
- 1.4 km horizontal segments through them with `crosses` and `touches`;
- the 1,922 Voronoi cells with `intersects`.

On the least dense zoning only:

- four times the points (~1.66M) with `intersects`;
- the zones on the left `contains` the points.

Finally, the Voronoi cells `touches` themselves.
