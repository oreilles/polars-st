![banner](https://raw.githubusercontent.com/Oreilles/polars-st/main/assets/banner.svg)

# Getting Started

## Installing polars-st

```sh
pip install polars-st
```

## Basics

`polars-st` provides geometry operations under the namespace `st` on Polars [`Expr`][polars_st.GeoExpr], [`Series`][polars_st.GeoSeries], [`DataFrame`][polars_st.GeoDataFrame] and  [`LazyFrame`][polars_st.GeoLazyFrame]. Functions used to read files or parse geometries are available at the module root. Here's a basic example:

``` pycon
>>> import polars_st as st
>>> gdf = st.GeoDataFrame({
...     "geometry": [
...         "POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))",
...         "POLYGON ((0 0, 0 1, 1 1, 0 0))",
...     ]
... })
>>> area = gdf.select(st.geom("geometry").st.area())
```

Behind the scenes, the GeoDataFrame constructor will parse each geometry from the user input into its EWKB internal representation. WKT is used as an example here but [many different type of input are supported](api-reference/creation.md). If the spatial data you have is already in a Series, you can also call the parser functions directly;

``` pycon
>>> s = pl.Series(["POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))", "POLYGON ((0 0, 0 1, 1 1, 0 0))"])
>>> gdf = pl.select(geometry=st.from_wkt("geometry"))
```

In polars-st, the `GeoDataFrame` constructor is just a utility returning a regular polars `DataFrame`. 
As a result, the `gdf` object will be exactly identical in the two examples above. 

In order to support autocompletions and type checking for the `st` namespace, `polars-st` provides a utility function [`st.geom`][polars_st.geom], with the same signature as [`pl.col`](https://docs.pola.rs/api/python/stable/reference/expressions/col.html), but which returns [`GeoExpr`][polars_st.GeoExpr] instead of [`Expr`](https://docs.pola.rs/api/python/stable/reference/expressions/index.html). [`GeoExpr`][polars_st.GeoExpr] is (kinda) just a type alias to polars `Expr` with type annotations added for the `st` namespace. It is therefore recommended that you use [`st.geom`][polars_st.geom] instead of `pl.col` to query geometry columns - even though both will yield identical results.

In addition to type checking, [`st.geom`][polars_st.geom] also has a trick up its sleeve: assuming the geometry column matches the default ("geometry"), you can even omit typing the column name entirely:

```pycon
>>> area = gdf.select(st.geom().st.area())
```

Even better, operations that involves a single geometry can be called in a simpler form:

``` pycon
>>> area = gdf.select(st.area())
```
