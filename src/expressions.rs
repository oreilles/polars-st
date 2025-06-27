use crate::{
    args,
    functions::{self, GeometryUtils},
    utils::try_reduce,
    wkb::WKBGeometryType,
};
use geos::{Geom, Geometry};
use polars::{datatypes::DataType as D, prelude::array::ArrayNameSpace};
use polars::{error::to_compute_err, prelude::*};
use polars_arrow::array::{Array, FixedSizeListArray, Float64Array, Utf8ViewArray};
use pyo3::prelude::*;
use pyo3_polars::{derive::polars_expr, error::PyPolarsErr, PySeries};

fn first_field_name(fields: &[Field]) -> PolarsResult<&PlSmallStr> {
    fields
        .first()
        .map(Field::name)
        .ok_or_else(|| to_compute_err("Invalid number of arguments."))
}

fn output_type_bounds(input_fields: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        first_field_name(input_fields)?.clone(),
        D::Array(D::Float64.into(), 4),
    ))
}

fn output_type_coordinates(input_fields: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        first_field_name(input_fields)?.clone(),
        D::List(D::List(D::Float64.into()).into()),
    ))
}

fn output_type_geometry_list(input_fields: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        first_field_name(input_fields)?.clone(),
        D::List(D::Binary.into()),
    ))
}

fn geometry_enum() -> DataType {
    static GEOMETRY_TYPES: [Option<&str>; 18] = [
        Some("Unknown"),
        Some("Point"),
        Some("LineString"),
        Some("Polygon"),
        Some("MultiPoint"),
        Some("MultiLineString"),
        Some("MultiPolygon"),
        Some("GeometryCollection"),
        Some("CircularString"),
        Some("CompoundCurve"),
        Some("CurvePolygon"),
        Some("MultiCurve"),
        Some("MultiSurface"),
        Some("Curve"),
        Some("Surface"),
        Some("PolyhedralSurface"),
        Some("Tin"),
        Some("Triangle"),
    ];
    let cats = Utf8ViewArray::from_slice(GEOMETRY_TYPES);
    let rev_mapping = RevMapping::build_local(cats);
    D::Enum(Some(rev_mapping.into()), CategoricalOrdering::Physical)
}

fn output_type_sjoin(input_fields: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        first_field_name(input_fields)?.clone(),
        D::Struct(vec![
            Field::new("left_index".into(), D::UInt32),
            Field::new("right_index".into(), D::UInt32),
        ]),
    ))
}

fn validate_inputs_length<const M: usize>(inputs: &[Series]) -> PolarsResult<&[Series; M]> {
    inputs
        .try_into()
        .map_err(|_| polars_err!(InvalidOperation: format!("invalid number of arguments: expected {}, got {}", M, inputs.len())))
}

fn validate_wkb(s: &Series) -> PolarsResult<&BinaryChunked> {
    s.binary()
        .map_err(|_| polars_err!(InvalidOperation: "invalid series dtype: expected `binary`, got `{}` for geoseries with name `{}`", s.dtype(), s.name()))
}

#[polars_expr(output_type=Binary)]
fn from_wkb(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    functions::from_wkb(inputs[0].binary()?)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn from_wkt(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    functions::from_wkt(inputs[0].str()?)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn from_ewkt(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    functions::from_ewkt(inputs[0].str()?)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn from_geojson(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    functions::from_geojson(inputs[0].str()?)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn rectangle(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let rect = inputs[0].strict_cast(&D::Array(D::Float64.into(), 4))?;
    functions::rectangle(rect.array().unwrap())
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn from_coords(inputs: &[Series], kwargs: args::CollectKwargs) -> PolarsResult<Series> {
    fn validate_point_coords(dtype: &DataType) -> PolarsResult<()> {
        match &dtype {
            &D::List(inner) if inner.is_primitive_numeric() => Ok(()),
            &D::Array(inner, 2..=4) if inner.is_primitive_numeric() => Ok(()),
            &D::Array(inner, t) if inner.is_primitive_numeric() => {
                Err(polars_err!(InvalidOperation: "invalid coordinates dimension: {t}"))
            }
            t => Err(polars_err!(InvalidOperation: "invalid coordinates dtype: {t}")),
        }
    }
    fn validate_line_coords(dtype: &DataType) -> PolarsResult<()> {
        match &dtype {
            &D::List(inner) | &D::Array(inner, _) => validate_point_coords(inner),
            t => Err(polars_err!(InvalidOperation: "invalid coordinates dtype: {t}")),
        }
    }
    fn validate_polygon_coords(dtype: &DataType) -> PolarsResult<()> {
        match &dtype {
            &D::List(inner) | &D::Array(inner, _) => validate_line_coords(inner),
            t => Err(polars_err!(InvalidOperation: "invalid coordinates dtype: {t}")),
        }
    }

    fn point(coords: &Series) -> PolarsResult<BinaryChunked> {
        let coords = coords.cast(&D::List(D::Float64.into()))?;
        let coords = coords.list().unwrap();
        functions::point(coords).map_err(to_compute_err)
    }
    fn multipoint(coords: &Series) -> PolarsResult<BinaryChunked> {
        let coords = coords.cast(&D::List(D::List(D::Float64.into()).into()))?;
        let coords = coords.list().unwrap();
        functions::multipoint(coords).map_err(to_compute_err)
    }
    fn linestring(coords: &Series) -> PolarsResult<BinaryChunked> {
        let coords = coords.cast(&D::List(D::List(D::Float64.into()).into()))?;
        let coords = coords.list().unwrap();
        functions::linestring(coords).map_err(to_compute_err)
    }
    fn circularstring(coords: &Series) -> PolarsResult<BinaryChunked> {
        let coords = coords.cast(&D::List(D::List(D::Float64.into()).into()))?;
        let coords = coords.list().unwrap();
        functions::circularstring(coords).map_err(to_compute_err)
    }
    fn multilinestring(coords: &Series) -> PolarsResult<BinaryChunked> {
        let coords = coords.cast(&D::List(D::List(D::List(D::Float64.into()).into()).into()))?;
        let coords = coords.list().unwrap();
        functions::multilinestring(coords).map_err(to_compute_err)
    }
    fn polygon(coords: &Series) -> PolarsResult<BinaryChunked> {
        let coords = coords.cast(&D::List(D::List(D::List(D::Float64.into()).into()).into()))?;
        let coords = coords.list().unwrap();
        functions::polygon(coords).map_err(to_compute_err)
    }

    use WKBGeometryType::*;
    let inputs = validate_inputs_length::<1>(inputs)?;
    let coords = &inputs[0];
    match (kwargs.into, &coords.dtype()) {
        (Some(Point), t) => validate_point_coords(t).and_then(|()| point(coords)),
        (Some(MultiPoint), t) => validate_line_coords(t).and_then(|()| multipoint(coords)),
        (Some(LineString), t) => validate_line_coords(t).and_then(|()| linestring(coords)),
        (Some(CircularString), t) => validate_line_coords(t).and_then(|()| circularstring(coords)),
        (Some(MultiLineString), t) => {
            validate_polygon_coords(t).and_then(|()| multilinestring(coords))
        }
        (Some(Polygon), t) => validate_polygon_coords(t).and_then(|()| polygon(coords)),
        (None, t) if validate_point_coords(t).is_ok() => point(coords),
        (None, t) if validate_line_coords(t).is_ok() => linestring(coords),
        (None, t) if validate_polygon_coords(t).is_ok() => polygon(coords),
        (None, t) => Err(polars_err!(InvalidOperation: "invalid coordinates dtype: {t}")),
        (Some(g), _) => Err(polars_err!(InvalidOperation: "unsupported geometry type: {g:?}")),
    }
    .map(IntoSeries::into_series)
}

#[polars_expr(output_type=UInt32)]
fn geometry_type(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    functions::get_type_id(wkb)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Int32)]
fn dimensions(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    functions::get_num_dimensions(wkb)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=UInt32)]
fn coordinate_dimension(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    functions::get_coordinate_dimension(wkb)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type_func=output_type_coordinates)]
fn coordinates(inputs: &[Series], kwargs: args::GetCoordinatesKwargs) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    functions::get_coordinates(wkb, kwargs.output_dimension)
        .map_err(to_compute_err)?
        .into_series()
        .with_name(wkb.name().clone())
        .strict_cast(&D::List(D::List(D::Float64.into()).into()))
}

#[polars_expr(output_type=Int32)]
fn srid(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    functions::get_srid(wkb)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn set_srid(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    let srid = inputs[1].strict_cast(&D::Int32)?;
    let srid = srid.i32().unwrap();
    functions::set_srid(wkb, srid)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Float64)]
fn x(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    functions::get_x(wkb)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Float64)]
fn y(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    functions::get_y(wkb)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Float64)]
fn z(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    functions::get_z(wkb)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Float64)]
fn m(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    functions::get_m(wkb)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn exterior_ring(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    functions::get_exterior_ring(wkb)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type_func=output_type_geometry_list)]
fn interior_rings(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    functions::get_interior_rings(wkb)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=UInt32)]
fn count_points(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    functions::get_num_points(wkb)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=UInt32)]
fn count_interior_rings(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    functions::get_num_interior_rings(wkb)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=UInt32)]
fn count_geometries(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    functions::get_num_geometries(wkb)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=UInt32)]
fn count_coordinates(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    functions::get_num_coordinates(wkb)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn get_point(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    let index = inputs[1].strict_cast(&D::UInt32)?;
    let index = index.u32().unwrap();
    functions::get_point_n(wkb, index)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn get_interior_ring(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    let index = inputs[1].strict_cast(&D::UInt32)?;
    let index = index.u32().unwrap();
    functions::get_interior_ring_n(wkb, index)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn get_geometry(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    let index = inputs[1].strict_cast(&D::UInt32)?;
    let index = index.u32()?;
    functions::get_geometry_n(wkb, index)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type_func=output_type_geometry_list)]
fn parts(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    functions::get_parts(wkb)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Float64)]
fn precision(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    functions::get_precision(wkb)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn set_precision(inputs: &[Series], kwargs: args::SetPrecisionKwargs) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    let precision = inputs[1].strict_cast(&D::Float64)?;
    let precision = precision.f64().unwrap();
    functions::set_precision(wkb, precision, &kwargs)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=String)]
fn to_wkt(inputs: &[Series], kwargs: args::ToWktKwargs) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    functions::to_wkt(wkb, &kwargs)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=String)]
fn to_ewkt(inputs: &[Series], kwargs: args::ToWktKwargs) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    functions::to_ewkt(wkb, &kwargs)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn to_wkb(inputs: &[Series], kwargs: args::ToWkbKwargs) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    functions::to_wkb(wkb, &kwargs)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=String)]
fn to_geojson(inputs: &[Series], kwargs: args::ToGeoJsonKwargs) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    functions::to_geojson(wkb, &kwargs)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[pyfunction]
pub fn to_python_dict(
    py: Python,
    pyseries: PySeries,
) -> Result<Vec<Option<PyObject>>, PyPolarsErr> {
    let wkb = validate_wkb(&pyseries.0)?;
    functions::to_python_dict(wkb, py)
        .map_err(to_compute_err)
        .map_err(Into::into)
}

#[polars_expr(output_type=Binary)]
fn cast(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    let into = &inputs[1].strict_cast(&geometry_enum())?;
    let into = into.categorical()?;
    functions::cast(wkb, into)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn multi(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    functions::multi(wkb)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Float64)]
fn area(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    functions::area(wkb)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type_func=output_type_bounds)]
fn bounds(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    functions::bounds(wkb)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type_func=output_type_bounds)]
fn total_bounds(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    let bounds = functions::bounds(wkb).map_err(to_compute_err)?;
    let arrow_dt = bounds.dtype().to_arrow(CompatLevel::newest());
    let i = |i| Int64Chunked::new("".into(), [i]);
    let total: Box<dyn Array> = Box::new(Float64Array::from_slice([
        bounds.array_get(&i(0), false)?.min()?.unwrap_or(f64::NAN),
        bounds.array_get(&i(1), false)?.min()?.unwrap_or(f64::NAN),
        bounds.array_get(&i(2), false)?.max()?.unwrap_or(f64::NAN),
        bounds.array_get(&i(3), false)?.max()?.unwrap_or(f64::NAN),
    ]));
    let total = FixedSizeListArray::new(arrow_dt, 1, total, None);
    Ok(ArrayChunked::from_chunk_iter(wkb.name().clone(), [total]).into_series())
}

#[polars_expr(output_type=Float64)]
fn length(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    functions::length(wkb)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Float64)]
fn distance(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let left = validate_wkb(&inputs[0])?;
    let right = validate_wkb(&inputs[1])?;
    functions::distance(left, right)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Float64)]
fn hausdorff_distance(
    inputs: &[Series],
    kwargs: args::DistanceDensifyKwargs,
) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let left = validate_wkb(&inputs[0])?;
    let right = validate_wkb(&inputs[1])?;
    match kwargs.densify {
        Some(densify) => functions::hausdorff_distance_densify(left, right, densify),
        None => functions::hausdorff_distance(left, right),
    }
    .map_err(to_compute_err)
    .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Float64)]
fn frechet_distance(
    inputs: &[Series],
    kwargs: args::DistanceDensifyKwargs,
) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let left = validate_wkb(&inputs[0])?;
    let right = validate_wkb(&inputs[1])?;
    match kwargs.densify {
        Some(densify) => functions::frechet_distance_densify(left, right, densify),
        None => functions::frechet_distance(left, right),
    }
    .map_err(to_compute_err)
    .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Float64)]
fn minimum_clearance(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    functions::minimum_clearance(wkb)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

// Predicates

#[polars_expr(output_type=Boolean)]
fn has_z(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    functions::has_z(wkb)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Boolean)]
fn has_m(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    functions::has_m(wkb)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Boolean)]
fn is_ccw(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    functions::is_ccw(wkb)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Boolean)]
fn is_closed(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    functions::is_closed(wkb)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Boolean)]
fn is_empty(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    functions::is_empty(wkb)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Boolean)]
fn is_ring(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    functions::is_ring(wkb)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Boolean)]
fn is_simple(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    functions::is_simple(wkb)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Boolean)]
fn is_valid(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    functions::is_valid(wkb)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=String)]
fn is_valid_reason(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    functions::is_valid_reason(wkb)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Boolean)]
fn crosses(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let left = validate_wkb(&inputs[0])?;
    let right = validate_wkb(&inputs[1])?;
    functions::crosses(left, right)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Boolean)]
fn contains(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let left = validate_wkb(&inputs[0])?;
    let right = validate_wkb(&inputs[1])?;
    functions::contains(left, right)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Boolean)]
fn contains_properly(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let left = validate_wkb(&inputs[0])?;
    let right = validate_wkb(&inputs[1])?;
    functions::contains_properly(left, right)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Boolean)]
fn covered_by(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let left = validate_wkb(&inputs[0])?;
    let right = validate_wkb(&inputs[1])?;
    functions::covered_by(left, right)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Boolean)]
fn covers(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let left = validate_wkb(&inputs[0])?;
    let right = validate_wkb(&inputs[1])?;
    functions::covers(left, right)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Boolean)]
fn disjoint(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let left = validate_wkb(&inputs[0])?;
    let right = validate_wkb(&inputs[1])?;
    functions::disjoint(left, right)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Boolean)]
fn dwithin(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<3>(inputs)?;
    let left = validate_wkb(&inputs[0])?;
    let right = validate_wkb(&inputs[1])?;
    let distance = inputs[2].strict_cast(&D::Float64)?;
    let distance = distance.f64().unwrap();
    functions::dwithin(left, right, distance)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Boolean)]
fn intersects(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let left = validate_wkb(&inputs[0])?;
    let right = validate_wkb(&inputs[1])?;
    functions::intersects(left, right)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Boolean)]
fn overlaps(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let left = validate_wkb(&inputs[0])?;
    let right = validate_wkb(&inputs[1])?;
    functions::overlaps(left, right)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Boolean)]
fn touches(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let left = validate_wkb(&inputs[0])?;
    let right = validate_wkb(&inputs[1])?;
    functions::touches(left, right)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Boolean)]
fn within(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let left = validate_wkb(&inputs[0])?;
    let right = validate_wkb(&inputs[1])?;
    functions::within(left, right)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Boolean)]
fn equals(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let left = validate_wkb(&inputs[0])?;
    let right = validate_wkb(&inputs[1])?;
    functions::equals(left, right)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Boolean)]
fn equals_identical(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let left = validate_wkb(&inputs[0])?;
    let right = validate_wkb(&inputs[1])?;
    functions::equals_identical(left, right)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Boolean)]
fn equals_exact(inputs: &[Series], kwargs: args::EqualsExactKwargs) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let left = validate_wkb(&inputs[0])?;
    let right = validate_wkb(&inputs[1])?;
    functions::equals_exact(left, right, kwargs.tolerance)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=String)]
fn relate(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let left = validate_wkb(&inputs[0])?;
    let right = validate_wkb(&inputs[1])?;
    functions::relate(left, right)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Boolean)]
fn relate_pattern(inputs: &[Series], kwargs: args::RelatePatternKwargs) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let left = validate_wkb(&inputs[0])?;
    let right = validate_wkb(&inputs[1])?;
    functions::relate_pattern(left, right, &kwargs.pattern)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn intersects_xy(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    let s = inputs[1].struct_()?;
    let x = s.field_by_name("x")?.strict_cast(&D::Float64)?;
    let y = s.field_by_name("y")?.strict_cast(&D::Float64)?;
    let x = x.f64().unwrap();
    let y = y.f64().unwrap();
    functions::intersects_xy(wkb, x, y)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn contains_xy(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    let s = inputs[1].struct_()?;
    let x = s.field_by_name("x")?.strict_cast(&D::Float64)?;
    let y = s.field_by_name("y")?.strict_cast(&D::Float64)?;
    let x = x.f64().unwrap();
    let y = y.f64().unwrap();
    functions::contains_xy(wkb, x, y)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn difference(inputs: &[Series], kwargs: args::SetOperationKwargs) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let left = validate_wkb(&inputs[0])?;
    let right = validate_wkb(&inputs[1])?;
    match kwargs.grid_size {
        Some(grid_size) => functions::difference_prec(left, right, grid_size),
        None => functions::difference(left, right),
    }
    .map_err(to_compute_err)
    .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn difference_all(inputs: &[Series], kwargs: args::SetOperationKwargs) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    let it = wkb.into_iter().flatten().map(Geometry::new_from_wkb);
    match kwargs.grid_size {
        Some(g) => try_reduce(it.flatten(), |a, b| a.difference_prec(&b, g)),
        None => try_reduce(it.flatten(), |a, b| a.difference(&b)),
    }
    .map(|geom| geom.unwrap_or_else(|| Geometry::new_from_wkt("GEOMETRYCOLLECTION EMPTY").unwrap()))
    .and_then(|geom| geom.to_ewkb())
    .map(|res| Series::new(wkb.name().clone(), [res]))
    .map_err(to_compute_err)
}

#[polars_expr(output_type=Binary)]
fn intersection(inputs: &[Series], kwargs: args::SetOperationKwargs) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let left = validate_wkb(&inputs[0])?;
    let right = validate_wkb(&inputs[1])?;
    match kwargs.grid_size {
        Some(grid_size) => functions::intersection_prec(left, right, grid_size),
        None => functions::intersection(left, right),
    }
    .map_err(to_compute_err)
    .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn intersection_all(inputs: &[Series], kwargs: args::SetOperationKwargs) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    let it = wkb.into_iter().flatten().map(Geometry::new_from_wkb);
    match kwargs.grid_size {
        Some(g) => try_reduce(it.flatten(), |a, b| a.intersection_prec(&b, g)),
        None => try_reduce(it.flatten(), |a, b| a.intersection(&b)),
    }
    .map(|geom| geom.unwrap_or_else(|| Geometry::new_from_wkt("GEOMETRYCOLLECTION EMPTY").unwrap()))
    .and_then(|geom| geom.to_ewkb())
    .map_err(to_compute_err)
    .map(|res| Series::new(wkb.name().clone(), [res]))
}

#[polars_expr(output_type=Binary)]
fn symmetric_difference(
    inputs: &[Series],
    kwargs: args::SetOperationKwargs,
) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let left = validate_wkb(&inputs[0])?;
    let right = validate_wkb(&inputs[1])?;
    match kwargs.grid_size {
        Some(grid_size) => functions::sym_difference_prec(left, right, grid_size),
        None => functions::sym_difference(left, right),
    }
    .map_err(to_compute_err)
    .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn symmetric_difference_all(
    inputs: &[Series],
    kwargs: args::SetOperationKwargs,
) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    let it = wkb.into_iter().flatten().map(Geometry::new_from_wkb);
    match kwargs.grid_size {
        Some(g) => try_reduce(it.flatten(), |a, b| a.sym_difference_prec(&b, g)),
        None => try_reduce(it.flatten(), |a, b| a.sym_difference(&b)),
    }
    .map(|geom| geom.unwrap_or_else(|| Geometry::new_from_wkt("GEOMETRYCOLLECTION EMPTY").unwrap()))
    .and_then(|geom| geom.to_ewkb())
    .map_err(to_compute_err)
    .map(|res| Series::new(wkb.name().clone(), [res]))
}

#[polars_expr(output_type=Binary)]
fn unary_union(inputs: &[Series], kwargs: args::SetOperationKwargs) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let geom = validate_wkb(&inputs[0])?;
    match kwargs.grid_size {
        Some(grid_size) => functions::unary_union_prec(geom, grid_size),
        None => functions::unary_union(geom),
    }
    .map_err(to_compute_err)
    .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn disjoint_subset_union(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    functions::disjoint_subset_union(wkb)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn union(inputs: &[Series], kwargs: args::SetOperationKwargs) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let left = validate_wkb(&inputs[0])?;
    let right = validate_wkb(&inputs[1])?;
    match kwargs.grid_size {
        Some(grid_size) => functions::union_prec(left, right, grid_size),
        None => functions::union(left, right),
    }
    .map_err(to_compute_err)
    .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn union_all(inputs: &[Series], kwargs: args::SetOperationKwargs) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let geom = validate_wkb(&inputs[0])?;
    let it = geom.into_iter().flatten().map(Geometry::new_from_wkb);
    match kwargs.grid_size {
        Some(g) => try_reduce(it.flatten(), |a, b| a.union_prec(&b, g)),
        None => try_reduce(it.flatten(), |a, b| a.union(&b)),
    }
    .map(|geom| geom.unwrap_or_else(|| Geometry::new_from_wkt("GEOMETRYCOLLECTION EMPTY").unwrap()))
    .and_then(|geom| geom.to_ewkb())
    .map_err(to_compute_err)
    .map(|wkb| Series::new(geom.name().clone(), [wkb]))
}

#[polars_expr(output_type=Binary)]
fn coverage_union(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    functions::coverage_union(wkb)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn coverage_union_all(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    functions::coverage_union_all(wkb)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn polygonize(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    functions::polygonize(wkb)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn collect(inputs: &[Series], kwargs: args::CollectKwargs) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    functions::collect(wkb, kwargs.into)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn boundary(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    functions::boundary(wkb)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn buffer(inputs: &[Series], kwargs: args::BufferKwargs) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    let distance = inputs[1].strict_cast(&D::Float64)?;
    let distance = distance.f64().unwrap();
    functions::buffer(wkb, distance, &kwargs)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn offset_curve(inputs: &[Series], kwargs: args::OffsetCurveKwargs) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    let distance = inputs[1].strict_cast(&D::Float64)?;
    let distance = distance.f64().unwrap();
    functions::offset_curve(wkb, distance, &kwargs)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn convex_hull(inputs: &[Series]) -> PolarsResult<Series> {
    let wkb = validate_wkb(&inputs[0])?;
    functions::convex_hull(wkb)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn concave_hull(inputs: &[Series], kwargs: args::ConcaveHullKwargs) -> PolarsResult<Series> {
    let wkb = validate_wkb(&inputs[0])?;
    functions::concave_hull(wkb, &kwargs)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn clip_by_rect(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    let rect = inputs[1].strict_cast(&D::Array(D::Float64.into(), 4))?;
    let rect = rect.array()?;
    functions::clip_by_rect(wkb, rect)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn centroid(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    functions::get_centroid(wkb)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn center(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    functions::get_center(wkb)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn delaunay_triangles(
    inputs: &[Series],
    kwargs: args::DelaunayTrianlesKwargs,
) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    functions::delaunay_triangulation(wkb, &kwargs)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn segmentize(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    let tolerance = inputs[1].strict_cast(&D::Float64)?;
    let tolerance = tolerance.f64().unwrap();
    functions::densify(wkb, tolerance)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn envelope(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    functions::envelope(wkb)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn extract_unique_points(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    functions::extract_unique_points(wkb)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn build_area(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    functions::build_area(wkb)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
pub fn make_valid(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    functions::make_valid(wkb)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
pub fn normalize(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    functions::normalize(wkb)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
pub fn node(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    functions::node(wkb)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
pub fn point_on_surface(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    functions::point_on_surface(wkb)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
pub fn remove_repeated_points(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    let tolerance = inputs[1].strict_cast(&D::Float64)?;
    let tolerance = tolerance.f64().unwrap();
    functions::remove_repeated_points(wkb, tolerance)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
pub fn reverse(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    functions::reverse(wkb)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
pub fn simplify(inputs: &[Series], kwargs: args::SimplifyKwargs) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    let tolerance = inputs[1].strict_cast(&D::Float64)?;
    let tolerance = tolerance.f64().unwrap();
    match kwargs.preserve_topology {
        true => functions::topology_preserve_simplify(wkb, tolerance),
        false => functions::simplify(wkb, tolerance),
    }
    .map_err(to_compute_err)
    .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
pub fn force_2d(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    functions::force_2d(wkb)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
pub fn force_3d(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    let z = inputs[1].strict_cast(&D::Float64)?;
    let z = z.f64().unwrap();
    functions::force_3d(wkb, z)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
pub fn snap(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<3>(inputs)?;
    let left = validate_wkb(&inputs[0])?;
    let right = validate_wkb(&inputs[1])?;
    let tolerance = inputs[2].strict_cast(&D::Float64)?;
    let tolerance = tolerance.f64().unwrap();
    functions::snap(left, right, tolerance)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
pub fn voronoi_polygons(inputs: &[Series], kwargs: args::VoronoiKwargs) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    functions::voronoi_polygons(wkb, &kwargs)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
pub fn minimum_rotated_rectangle(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    functions::minimum_rotated_rectangle(wkb)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
pub fn translate(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    let factors = inputs[1].strict_cast(&D::Array(D::Float64.into(), 3))?;
    let factors = factors.array()?;
    functions::translate(wkb, factors)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
pub fn rotate(inputs: &[Series], kwargs: args::TransformKwargs) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    let angle = inputs[1].strict_cast(&D::Float64)?;
    let angle = angle.f64().unwrap();
    match kwargs.origin {
        args::TransformOrigin::XY(o) => functions::rotate_around_point(wkb, angle, &o),
        args::TransformOrigin::XYZ(o) => functions::rotate_around_point(wkb, angle, &(o.0, o.1)),
        args::TransformOrigin::Center => functions::rotate_around_center(wkb, angle),
        args::TransformOrigin::Centroid => functions::rotate_around_centroid(wkb, angle),
    }
    .map_err(to_compute_err)
    .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
pub fn scale(inputs: &[Series], kwargs: args::TransformKwargs) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    let factors = inputs[1].strict_cast(&D::Array(D::Float64.into(), 3))?;
    let factors = factors.array()?;
    match kwargs.origin {
        args::TransformOrigin::XY(o) => functions::scale_from_point(wkb, factors, &(o.0, o.1, 0.0)),
        args::TransformOrigin::XYZ(origin) => functions::scale_from_point(wkb, factors, &origin),
        args::TransformOrigin::Center => functions::scale_from_center(wkb, factors),
        args::TransformOrigin::Centroid => functions::scale_from_centroid(wkb, factors),
    }
    .map_err(to_compute_err)
    .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
pub fn skew(inputs: &[Series], kwargs: args::TransformKwargs) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    let factors = inputs[1].strict_cast(&D::Array(D::Float64.into(), 3))?;
    let factors = factors.array()?;
    match kwargs.origin {
        args::TransformOrigin::XY(o) => functions::skew_from_point(wkb, factors, &(o.0, o.1, 0.0)),
        args::TransformOrigin::XYZ(origin) => functions::skew_from_point(wkb, factors, &origin),
        args::TransformOrigin::Center => functions::skew_from_center(wkb, factors),
        args::TransformOrigin::Centroid => functions::skew_from_centroid(wkb, factors),
    }
    .map_err(to_compute_err)
    .map(IntoSeries::into_series)
}
#[polars_expr(output_type=Binary)]
pub fn affine_transform(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    let matrix = &inputs[1];
    match matrix.dtype() {
        D::Array(.., 6) => {
            let matrix = matrix.strict_cast(&D::Array(D::Float64.into(), 6))?;
            functions::affine_transform_2d(wkb, matrix.array()?).map_err(to_compute_err)
        }
        D::Array(.., 12) => {
            let matrix = matrix.strict_cast(&D::Array(D::Float64.into(), 12))?;
            functions::affine_transform_3d(wkb, matrix.array()?).map_err(to_compute_err)
        }
        _ => Err(to_compute_err(
            "matrix parameter should be of type array with shape (6 | 12)",
        )),
    }
    .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
pub fn interpolate(inputs: &[Series], kwargs: args::InterpolateKwargs) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    let distance = inputs[1].strict_cast(&D::Float64)?;
    let distance = distance.f64().unwrap();
    match kwargs.normalized {
        true => functions::interpolate_normalized(wkb, distance),
        false => functions::interpolate(wkb, distance),
    }
    .map_err(to_compute_err)
    .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Float64)]
pub fn project(inputs: &[Series], kwargs: args::InterpolateKwargs) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let left = validate_wkb(&inputs[0])?;
    let right = validate_wkb(&inputs[1])?;
    match kwargs.normalized {
        true => functions::project_normalized(left, right),
        false => functions::project(left, right),
    }
    .map_err(to_compute_err)
    .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
pub fn substring(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<3>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    let start = inputs[1].strict_cast(&D::Float64)?;
    let end = inputs[2].strict_cast(&D::Float64)?;
    let start = start.f64().unwrap();
    let end = end.f64().unwrap();
    functions::substring(wkb, start, end)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
pub fn line_merge(inputs: &[Series], kwargs: args::LineMergeKwargs) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    match kwargs.directed {
        true => functions::line_merge_directed(wkb),
        false => functions::line_merge(wkb),
    }
    .map_err(to_compute_err)
    .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
pub fn shared_paths(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let left = validate_wkb(&inputs[0])?;
    let right = validate_wkb(&inputs[1])?;
    functions::shared_paths(left, right)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
pub fn shortest_line(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let left = validate_wkb(&inputs[0])?;
    let right = validate_wkb(&inputs[1])?;
    functions::shortest_line(left, right)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type_func=output_type_sjoin)]
pub fn sjoin(inputs: &[Series], kwargs: args::SpatialJoinKwargs) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let left = validate_wkb(&inputs[0])?;
    let right = validate_wkb(&inputs[1])?;
    functions::sjoin(left, right, kwargs.predicate)
        .map_err(to_compute_err)
        .map(|(left_index, right_index)| {
            StructChunked::from_columns(
                left.name().clone(),
                left_index.len(),
                &[left_index.into_column(), right_index.into_column()],
            )
            .map(IntoSeries::into_series)
        })?
}

#[polars_expr(output_type=Binary)]
pub fn flip_coordinates(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    functions::flip_coordinates(wkb)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
pub fn to_srid(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    let srid = inputs[1].strict_cast(&D::Int64)?;
    let srid = srid.i64()?;

    functions::to_srid(wkb, srid)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}
