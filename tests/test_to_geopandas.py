import geopandas as gpd
import pytest

import polars_st as st


def test_to_geopandas_valid_conversion():
    geometry = ["POINT (0 0)", "POINT (1 2)", None]
    name = ["A", "B", "C"]
    gdf = st.GeoDataFrame({
        "geometry": st.GeoSeries(geometry).st.set_srid(4326),
        "name": name,
    })

    geopandas_gdf = gdf.st.to_geopandas()

    assert isinstance(geopandas_gdf, gpd.GeoDataFrame)
    assert list(geopandas_gdf.columns) == ["geometry", "name"]
    assert list(geopandas_gdf.geometry.to_wkt()) == geometry
    assert list(geopandas_gdf["name"]) == name
    assert geopandas_gdf.crs == "EPSG:4326"


def test_to_geopandas_no_srid():
    gdf = st.GeoDataFrame(["POINT (0 0)", "POINT (1 2)", None])
    geopandas_gdf = gdf.st.to_geopandas()
    assert geopandas_gdf.crs is None


def test_to_geopandas_named():
    gdf = st.GeoDataFrame({"geom": ["POINT (0 0)", "POINT (1 2)", None]}, geometry_name="geom")
    geopandas_gdf = gdf.st.to_geopandas(geometry_name="geom")
    assert geopandas_gdf.geometry.name == "geom"


def test_to_geopandas_empty_dataframe():
    gdf = st.GeoDataFrame({"geometry": []})
    geopandas_gdf = gdf.st.to_geopandas()

    assert isinstance(geopandas_gdf, gpd.GeoDataFrame)
    assert geopandas_gdf.empty


def test_to_geopandas_mixed_srids():
    gdf = st.GeoDataFrame({
        "geometry": ["POINT(0 0)", "POINT(1 2)"],
        "srid": [4326, 3857],
    }).with_columns(st.set_srid(srid="srid"))
    msg = "DataFrame with mixed SRIDs aren't supported for this operation"
    with pytest.raises(ValueError, match=msg):
        gdf.pipe(st.st).to_geopandas()


def test_series_to_geopandas():
    gs = st.GeoSeries(["POINT (0 0)", "POINT (1 2)", None])
    geopandas_gs = gs.st.to_geopandas()
    assert isinstance(geopandas_gs, gpd.GeoSeries)


def test_series_to_geopandas_named():
    gs = st.GeoSeries("geom", ["POINT (0 0)", "POINT (1 2)", None])
    geopandas_gs = gs.st.to_geopandas()
    assert isinstance(geopandas_gs, gpd.GeoSeries)
    assert geopandas_gs.geometry.name == "geom"
