import geopandas as gpd
import polars as pl
import pytest

import polars_st as st


@pytest.mark.parametrize("srid, expected_crs", [
    (None, None),
    (4326, "EPSG:4326"),
])
def test_to_geopandas_valid_conversion(srid, expected_crs):
    wkts = ["POINT (0 0)", "POINT (1 2)"]
    names = ["Foo", "Bar"]
    gdf = st.GeoDataFrame({"geometry": wkts, "name": names})

    if srid is not None:
        gdf = gdf.with_columns(
            pl.col("geometry").st.set_srid(srid)
        )

    geopandas_gdf = gdf.st.to_geopandas()

    assert isinstance(geopandas_gdf, gpd.GeoDataFrame)
    assert list(geopandas_gdf.geometry.to_wkt()) == wkts
    assert list(geopandas_gdf["name"]) == names
    assert geopandas_gdf.crs == expected_crs


def test_to_geopandas_empty_dataframe():
    gdf = st.GeoDataFrame({"geometry": []})
    geopandas_gdf = gdf.st.to_geopandas()

    assert isinstance(geopandas_gdf, gpd.GeoDataFrame)
    assert geopandas_gdf.empty


def test_to_geopandas_mixed_srids():
    gdf = st.GeoDataFrame({
        "geometry": ["POINT(0 0)", "POINT(1 2)"],
        "srid": [4326, 3857],
    }).with_columns(pl.col("geometry").st.set_srid("srid"))
    with pytest.raises(
            ValueError, match="DataFrame with mixed SRIDs aren't supported in GeoPandas"
    ):
        gdf.st.to_geopandas()


def test_to_geopandas_no_geometry_column():
    with pytest.raises(ValueError, match="geometry column geometry not found"):
        st.GeoDataFrame({"name": ["Foo", "Bar"]})
