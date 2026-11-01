from pathlib import Path

import polars_st as st


def test_write_file_path(tmp_path: Path):
    gdf = st.GeoDataFrame({
        "geometry": st.GeoSeries(["POINT (0 0)", "POINT (1 2)"]).st.set_srid(4326),
        "name": ["A", "B"],
    })
    path = tmp_path / "test.gpkg"

    gdf.st.write_file(path)

    result = st.read_file(path).select(st.geom("geom").st.to_wkt(), "name")
    assert result.rows() == [("POINT (0 0)", "A"), ("POINT (1 2)", "B")]
