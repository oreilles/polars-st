import polars_st as st


def test_to_dict_point():
    gs = st.GeoSeries(["POINT (1 2)"])
    assert gs.st.to_dict().item() == {"type": "Point", "coordinates": [1.0, 2.0]}


def test_to_dict_point_empty():
    gs = st.GeoSeries(["POINT EMPTY"])
    assert gs.st.to_dict().item() == {"type": "Point", "coordinates": []}


def test_to_dict_linestring():
    gs = st.GeoSeries(["LINESTRING (0 0, 1 1)"])
    assert gs.st.to_dict().item() == {"type": "LineString", "coordinates": [[0.0, 0.0], [1.0, 1.0]]}


def test_to_dict_circularstring():
    gs = st.GeoSeries(["CIRCULARSTRING (0 0, 1 1, 2 0)"])
    assert gs.st.to_dict().item() == {
        "type": "CircularString",
        "coordinates": [[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]],
    }


def test_to_dict_polygon():
    gs = st.GeoSeries(["POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))"])
    assert gs.st.to_dict().item() == {
        "type": "Polygon",
        "coordinates": [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]],
    }


def test_to_dict_polygon_with_hole():
    gs = st.GeoSeries([
        "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0), (1 1, 2 1, 2 2, 1 2, 1 1))",
    ])
    assert gs.st.to_dict().item() == {
        "type": "Polygon",
        "coordinates": [
            [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0], [0.0, 0.0]],
            [[1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [1.0, 2.0], [1.0, 1.0]],
        ],
    }


def test_to_dict_polygon_empty():
    gs = st.GeoSeries(["POLYGON EMPTY"])
    assert gs.st.to_dict().item() == {"type": "Polygon", "coordinates": []}


def test_to_dict_multipoint():
    gs = st.GeoSeries(["MULTIPOINT ((0 0), (1 1))"])
    assert gs.st.to_dict().item() == {"type": "MultiPoint", "coordinates": [[0.0, 0.0], [1.0, 1.0]]}


def test_to_dict_multilinestring():
    gs = st.GeoSeries(["MULTILINESTRING ((0 0, 1 1), (2 2, 3 3))"])
    assert gs.st.to_dict().item() == {
        "type": "MultiLineString",
        "coordinates": [[[0.0, 0.0], [1.0, 1.0]], [[2.0, 2.0], [3.0, 3.0]]],
    }


def test_to_dict_multipolygon():
    gs = st.GeoSeries([
        "MULTIPOLYGON (((0 0, 1 0, 0 1, 0 0)), ((2 2, 3 2, 2 3, 2 2)))",
    ])
    assert gs.st.to_dict().item() == {
        "type": "MultiPolygon",
        "coordinates": [
            [[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]],
            [[[2.0, 2.0], [3.0, 2.0], [2.0, 3.0], [2.0, 2.0]]],
        ],
    }


def test_to_dict_geometrycollection():
    gs = st.GeoSeries([
        "GEOMETRYCOLLECTION (POINT (0 0), LINESTRING (0 0, 1 1))",
    ])
    assert gs.st.to_dict().item() == {
        "type": "GeometryCollection",
        "geometries": [
            {"type": "Point", "coordinates": [0.0, 0.0]},
            {"type": "LineString", "coordinates": [[0.0, 0.0], [1.0, 1.0]]},
        ],
    }


def test_to_dict_compoundcurve():
    gs = st.GeoSeries([
        "COMPOUNDCURVE (CIRCULARSTRING (0 0, 1 1, 2 0), (2 0, 3 0))",
    ])
    assert gs.st.to_dict().item() == {
        "type": "CompoundCurve",
        "geometries": [
            {"type": "CircularString", "coordinates": [[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]]},
            {"type": "LineString", "coordinates": [[2.0, 0.0], [3.0, 0.0]]},
        ],
    }


def test_to_dict_compoundcurve_empty():
    gs = st.GeoSeries(["COMPOUNDCURVE EMPTY"])
    assert gs.st.to_dict().item() == {"type": "CompoundCurve", "geometries": []}


def test_to_dict_curvepolygon():
    gs = st.GeoSeries([
        "CURVEPOLYGON (CIRCULARSTRING (0 0, 1 1, 2 0, 1 -1, 0 0))",
    ])
    assert gs.st.to_dict().item() == {
        "type": "CurvePolygon",
        "geometries": [
            {
                "type": "CircularString",
                "coordinates": [
                    [0.0, 0.0],
                    [1.0, 1.0],
                    [2.0, 0.0],
                    [1.0, -1.0],
                    [0.0, 0.0],
                ],
            },
        ],
    }


def test_to_dict_curvepolygon_empty():
    gs = st.GeoSeries(["CURVEPOLYGON EMPTY"])
    assert gs.st.to_dict().item() == {"type": "CurvePolygon", "geometries": []}


def test_to_dict_multicurve():
    gs = st.GeoSeries(["MULTICURVE (CIRCULARSTRING (0 0, 1 1, 2 0), (2 0, 3 0))"])
    assert gs.st.to_dict().item() == {
        "type": "MultiCurve",
        "geometries": [
            {"type": "CircularString", "coordinates": [[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]]},
            {"type": "LineString", "coordinates": [[2.0, 0.0], [3.0, 0.0]]},
        ],
    }


def test_to_dict_multicurve_empty():
    gs = st.GeoSeries(["MULTICURVE EMPTY"])
    assert gs.st.to_dict().item() == {"type": "MultiCurve", "geometries": []}


def test_to_dict_multisurface():
    gs = st.GeoSeries([
        "MULTISURFACE (CURVEPOLYGON (CIRCULARSTRING (0 0, 1 1, 2 0, 1 -1, 0 0)), ((10 10, 11 10, 11 11, 10 11, 10 10)))",  # noqa: E501
    ])
    assert gs.st.to_dict().item() == {
        "type": "MultiSurface",
        "geometries": [
            {
                "type": "CurvePolygon",
                "geometries": [
                    {
                        "type": "CircularString",
                        "coordinates": [
                            [0.0, 0.0],
                            [1.0, 1.0],
                            [2.0, 0.0],
                            [1.0, -1.0],
                            [0.0, 0.0],
                        ],
                    },
                ],
            },
            {
                "type": "Polygon",
                "coordinates": [
                    [
                        [10.0, 10.0],
                        [11.0, 10.0],
                        [11.0, 11.0],
                        [10.0, 11.0],
                        [10.0, 10.0],
                    ],
                ],
            },
        ],
    }


def test_to_dict_null():
    gs = st.GeoSeries([None])
    assert gs.st.to_dict().item() is None
