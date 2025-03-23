import polars_st as st


def test_config_context():
    with st.Config(geometry_column="hello"):
        assert st.Config.get_geometry_column() == "hello"
    assert st.Config.get_geometry_column() == "geometry"


def test_config_context_object():
    with st.Config() as config:
        config.set_geometry_column("hello")
        assert st.Config.get_geometry_column() == "hello"
    assert st.Config.get_geometry_column() == "geometry"


def test_config_decorator():
    @st.Config(geometry_column="hello")
    def foo():
        return st.Config.get_geometry_column()

    assert foo() == "hello"
    assert st.Config.get_geometry_column() == "geometry"


def test_config_global():
    st.Config.set_geometry_column("hello")
    assert st.Config.get_geometry_column() == "hello"
    st.Config.set_geometry_column()
    assert st.Config.get_geometry_column() == "geometry"
