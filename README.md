[![Banner](https://raw.githubusercontent.com/Oreilles/polars-st/main/assets/banner.svg)](#)

# Polars ST

Polars ST provides spatial operations on [Polars](https://github.com/pola-rs/polars) DataFrames, Series, and Expressions. Just like [Shapely](https://github.com/shapely/shapely/) and [GeoPandas](https://github.com/geopandas/geopandas), it make use of the [GEOS](https://github.com/libgeos/geos) library, meaning that its API is mostly identical to theirs.

* Documentation: https://oreilles.github.io/polars-st/

```pycon
>>> import polars_st as st
>>> gdf = st.GeoDataFrame({
...     "category": ["A", "A", "B"],
...     "geometry": [
...         "POLYGON((0 0, 0 4, 4 2, 0 0))",
...         "POLYGON((4 0, 4 4, 0 2, 4 0))",
...         "POLYGON((0 0, 2 2, 2 0, 0 0))",
...     ]
... })
>>> gdf = gdf.group_by("category").agg(st.intersection_all()).with_columns(area=st.area())
>>> gdf.st.to_wkt()
shape: (2, 3)
┌──────────┬─────────────────────────────────────┬──────┐
│ category ┆ geometry                            ┆ area │
│ ---      ┆ ---                                 ┆ ---  │
│ str      ┆ str                                 ┆ f64  │
╞══════════╪═════════════════════════════════════╪══════╡
│ B        ┆ POLYGON ((0 0, 2 2, 2 0, 0 0))      ┆ 2.0  │
│ A        ┆ POLYGON ((4 2, 2 1, 0 2, 2 3, 4 2)) ┆ 4.0  │
└──────────┴─────────────────────────────────────┴──────┘
```

## Installation

Polars ST is published on PyPI so you can install it with your preferred package manager.

```sh
pip install polars-st
```

## How it works

Geometries are stored as EWKB in regular Polars Binary columns. EWKB is a extension to the WKB standard popularized by PostGIS, that can also store information about the CRS of each geometry as an integer code called SRID.

For every spatial operations, the WKB binary blob will be parsed into a Geometry object so the operation can be done. If the operation result is a geometry, it will then be serialized back to EWKB. Because of that round-trip, some operations might turn out to be slower than GeoPandas. In most cases however, the performance penalty of that round-trip will be marginal compared to the spatial operation, and you will fully benefit from the parallelization capabilities of Polars.


## About GeoPolars

[GeoPolars](https://github.com/geopolars/geopolars) is an incredibly promising tool for manipulating geographic data in Polars based on [GeoArrow](https://github.com/geoarrow/geoarrow) that likely will outperform this library's performance by a long shot. It however seems to be quite a long way from being ready and feature-complete, mostly due to Polars lack of support for [Arrow Extension Types](https://github.com/pola-rs/polars/issues/9112) and [subclassing of core datatypes](https://github.com/pola-rs/polars/issues/2846#issuecomment-1711799869).

Storing geometry as EWKB and delegating core functionality to GEOS allows `polars-st` to be ready now, and provide additional features such as XYZM coordinates, curved geometry types and per-geometry CRS information.

I really hope Geopolars get there soon, and that maybe some of the API design explorations made here will have helped make it even more pleasant to use.

## About Polars

This project is not affiliated with Polars. The design language was made very close to that of Polars because I found it amazingly appealing and liked the challenge of adding geographic meaning to it, and to also highlight the fact that this project is an exclusive extension to Polars.


## About GEOS

In order to save myself (and yourself) from the burden of GEOS version-specific features and bugs, `polars-st` statically link to GEOS `main`. This means that all documented features are guaranteed to be available. This also means that this project is LGPL licensed, the same way GEOS is.

## License

Copyright (C) 2024  Aurèle Ferotin

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301
USA
