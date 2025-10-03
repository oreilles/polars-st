use crate::{
    args,
    functions::{self, GeometryUtils},
    utils::try_reduce,
};
use geos::{Geom, Geometry};
use polars::{datatypes::DataType as D, prelude::array::ArrayNameSpace};
use polars::{error::to_compute_err, prelude::*};
use polars_arrow::array::{Array, FixedSizeListArray, Float64Array};
use polars_python::{error::PyPolarsErr, PySeries};
use pyo3::prelude::*;
use pyo3_polars::derive::polars_expr;

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

fn geometry_enum() -> &'static DataType {
    use std::sync::OnceLock;
    static GEOMETRY_ENUM: OnceLock<DataType> = OnceLock::new();

    GEOMETRY_ENUM.get_or_init(|| {
        let cats = FrozenCategories::new([
            "Unknown",
            "Point",
            "LineString",
            "Polygon",
            "MultiPoint",
            "MultiLineString",
            "MultiPolygon",
            "GeometryCollection",
            "CircularString",
            "CompoundCurve",
            "CurvePolygon",
            "MultiCurve",
            "MultiSurface",
            "Curve",
            "Surface",
            "PolyhedralSurface",
            "Tin",
            "Triangle",
        ])
        .unwrap();
        D::from_frozen_categories(cats)
    })
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
        .map_err(|_| polars_err!(InvalidOperation: "invalid dtype for geoseries `{}`: expected `binary`, got `{}`", s.dtype(), s.name()))
}

macro_rules! extract {
    ($out:ident, $s:expr, $dt:expr, $f:ident) => {
        let type_err = |_| polars_err!(InvalidOperation:"invalid dtype for series `{}`: `{}`", $s.dtype(), $s.name());
        let tmp = $s.strict_cast(&$dt).map_err(type_err)?;
        let $out = tmp.$f().unwrap();
    };
}

macro_rules! wrap {
    ($func:ident($first_arg:expr $(, $rest:expr)*)) => {
        functions::$func($first_arg $(, $rest)*)
            .map_err(to_compute_err)
            .map(IntoSeries::into_series)
    };
}

#[polars_expr(output_type=Binary)]
fn from_wkb(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    wrap!(from_wkb(wkb))
}

#[polars_expr(output_type=Binary)]
fn from_wkt(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    wrap!(from_wkt(inputs[0].str()?))
}

#[polars_expr(output_type=Binary)]
fn from_ewkt(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    wrap!(from_ewkt(inputs[0].str()?))
}

#[polars_expr(output_type=Binary)]
fn from_geojson(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    wrap!(from_geojson(inputs[0].str()?))
}

#[polars_expr(output_type=Binary)]
fn rectangle(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    extract!(rect, inputs[0], D::Array(D::Float64.into(), 4), array);
    extract!(srid, inputs[1], D::Int32, i32);
    wrap!(rectangle(rect, srid))
}

macro_rules! create_geometry {
    ($name:ident, $cast_type:expr) => {
        #[polars_expr(output_type = Binary)]
        fn $name(inputs: &[Series]) -> PolarsResult<Series> {
            let inputs = validate_inputs_length::<2>(inputs)?;
            let coords = &inputs[0];
            let coords = coords
                .cast(&$cast_type)
                .map_err(|_| polars_err!(InvalidOperation: "invalid coordinates dtype for {}: {}", stringify!($name), coords.dtype()))?;
            let coords = coords.list().unwrap();
            extract!(srid, inputs[1], D::Int32, i32);
            wrap!($name(coords, srid))
        }
    };
}

create_geometry!(point, D::Float64.implode());
create_geometry!(linestring, D::Float64.implode().implode());
create_geometry!(multipoint, D::Float64.implode().implode());
create_geometry!(circularstring, D::Float64.implode().implode());
create_geometry!(multilinestring, D::Float64.implode().implode().implode());
create_geometry!(polygon, D::Float64.implode().implode().implode());

#[polars_expr(output_type=UInt32)]
fn geometry_type(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    wrap!(get_type_id(wkb))
}

#[polars_expr(output_type=Int32)]
fn dimensions(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    wrap!(get_num_dimensions(wkb))
}

#[polars_expr(output_type=UInt32)]
fn coordinate_dimension(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    wrap!(get_coordinate_dimension(wkb))
}

#[polars_expr(output_type_func=output_type_coordinates)]
fn coordinates(inputs: &[Series], kwargs: args::GetCoordinatesKwargs) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    wrap!(get_coordinates(wkb, kwargs.output_dimension))?
        .with_name(wkb.name().clone())
        .strict_cast(&D::List(D::List(D::Float64.into()).into()))
}

#[polars_expr(output_type=Int32)]
fn srid(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    wrap!(get_srid(wkb))
}

#[polars_expr(output_type=Binary)]
fn set_srid(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    extract!(srid, inputs[1], D::Int32, i32);
    wrap!(set_srid(wkb, srid))
}

#[polars_expr(output_type=Float64)]
fn x(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    wrap!(get_x(wkb))
}

#[polars_expr(output_type=Float64)]
fn y(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    wrap!(get_y(wkb))
}

#[polars_expr(output_type=Float64)]
fn z(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    wrap!(get_z(wkb))
}

#[polars_expr(output_type=Float64)]
fn m(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    wrap!(get_m(wkb))
}

#[polars_expr(output_type=Binary)]
fn exterior_ring(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    wrap!(get_exterior_ring(wkb))
}

#[polars_expr(output_type_func=output_type_geometry_list)]
fn interior_rings(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    wrap!(get_interior_rings(wkb))
}

#[polars_expr(output_type=UInt32)]
fn count_points(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    wrap!(get_num_points(wkb))
}

#[polars_expr(output_type=UInt32)]
fn count_interior_rings(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    wrap!(get_num_interior_rings(wkb))
}

#[polars_expr(output_type=UInt32)]
fn count_geometries(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    wrap!(get_num_geometries(wkb))
}

#[polars_expr(output_type=UInt32)]
fn count_coordinates(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    wrap!(get_num_coordinates(wkb))
}

#[polars_expr(output_type=Binary)]
fn get_point(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    extract!(index, inputs[1], D::UInt32, u32);
    wrap!(get_point_n(wkb, index))
}

#[polars_expr(output_type=Binary)]
fn get_interior_ring(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    extract!(index, inputs[1], D::UInt32, u32);
    wrap!(get_interior_ring_n(wkb, index))
}

#[polars_expr(output_type=Binary)]
fn get_geometry(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    extract!(index, inputs[1], D::UInt32, u32);
    wrap!(get_geometry_n(wkb, index))
}

#[polars_expr(output_type_func=output_type_geometry_list)]
fn parts(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    wrap!(get_parts(wkb))
}

#[polars_expr(output_type=Float64)]
fn precision(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    wrap!(get_precision(wkb))
}

#[polars_expr(output_type=Binary)]
fn set_precision(inputs: &[Series], kwargs: args::SetPrecisionKwargs) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    extract!(precision, inputs[1], D::Float64, f64);
    wrap!(set_precision(wkb, precision, &kwargs))
}

#[polars_expr(output_type=String)]
fn to_wkt(inputs: &[Series], kwargs: args::ToWktKwargs) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    wrap!(to_wkt(wkb, &kwargs))
}

#[polars_expr(output_type=String)]
fn to_ewkt(inputs: &[Series], kwargs: args::ToWktKwargs) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    wrap!(to_ewkt(wkb, &kwargs))
}

#[polars_expr(output_type=Binary)]
fn to_wkb(inputs: &[Series], kwargs: args::ToWkbKwargs) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    wrap!(to_wkb(wkb, &kwargs))
}

#[polars_expr(output_type=String)]
fn to_geojson(inputs: &[Series], kwargs: args::ToGeoJsonKwargs) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    wrap!(to_geojson(wkb, &kwargs))
}

#[pyfunction]
pub fn to_python_dict(
    py: Python,
    capsule: &Bound<'_, PyAny>,
) -> Result<Vec<Option<PyObject>>, PyPolarsErr> {
    let pyseries = PySeries::from_arrow_c_stream(&py.get_type::<pyo3::types::PyNone>(), capsule)?;
    let series = pyseries.series.read();
    let wkb = validate_wkb(&series)?;
    functions::to_python_dict(wkb, py)
        .map_err(to_compute_err)
        .map_err(Into::into)
}

#[polars_expr(output_type=Binary)]
fn cast(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    extract!(into, inputs[1], geometry_enum(), cat8);
    wrap!(cast(wkb, into))
}

#[polars_expr(output_type=Binary)]
fn multi(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    wrap!(multi(wkb))
}

#[polars_expr(output_type=Float64)]
fn area(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    wrap!(area(wkb))
}

#[polars_expr(output_type_func=output_type_bounds)]
fn bounds(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    wrap!(bounds(wkb))
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
    wrap!(length(wkb))
}

#[polars_expr(output_type=Float64)]
fn distance(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let left = validate_wkb(&inputs[0])?;
    let right = validate_wkb(&inputs[1])?;
    wrap!(distance(left, right))
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
        Some(densify) => wrap!(hausdorff_distance_densify(left, right, densify)),
        None => wrap!(hausdorff_distance(left, right)),
    }
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
        Some(densify) => wrap!(frechet_distance_densify(left, right, densify)),
        None => wrap!(frechet_distance(left, right)),
    }
}

#[polars_expr(output_type=Float64)]
fn minimum_clearance(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    wrap!(minimum_clearance(wkb))
}

// Predicates

#[polars_expr(output_type=Boolean)]
fn has_z(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    wrap!(has_z(wkb))
}

#[polars_expr(output_type=Boolean)]
fn has_m(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    wrap!(has_m(wkb))
}

#[polars_expr(output_type=Boolean)]
fn is_ccw(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    wrap!(is_ccw(wkb))
}

#[polars_expr(output_type=Boolean)]
fn is_closed(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    wrap!(is_closed(wkb))
}

#[polars_expr(output_type=Boolean)]
fn is_empty(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    wrap!(is_empty(wkb))
}

#[polars_expr(output_type=Boolean)]
fn is_ring(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    wrap!(is_ring(wkb))
}

#[polars_expr(output_type=Boolean)]
fn is_simple(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    wrap!(is_simple(wkb))
}

#[polars_expr(output_type=Boolean)]
fn is_valid(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    wrap!(is_valid(wkb))
}

#[polars_expr(output_type=String)]
fn is_valid_reason(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    wrap!(is_valid_reason(wkb))
}

#[polars_expr(output_type=Boolean)]
fn crosses(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let left = validate_wkb(&inputs[0])?;
    let right = validate_wkb(&inputs[1])?;
    wrap!(crosses(left, right))
}

#[polars_expr(output_type=Boolean)]
fn contains(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let left = validate_wkb(&inputs[0])?;
    let right = validate_wkb(&inputs[1])?;
    wrap!(contains(left, right))
}

#[polars_expr(output_type=Boolean)]
fn contains_properly(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let left = validate_wkb(&inputs[0])?;
    let right = validate_wkb(&inputs[1])?;
    wrap!(contains_properly(left, right))
}

#[polars_expr(output_type=Boolean)]
fn covered_by(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let left = validate_wkb(&inputs[0])?;
    let right = validate_wkb(&inputs[1])?;
    wrap!(covered_by(left, right))
}

#[polars_expr(output_type=Boolean)]
fn covers(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let left = validate_wkb(&inputs[0])?;
    let right = validate_wkb(&inputs[1])?;
    wrap!(covers(left, right))
}

#[polars_expr(output_type=Boolean)]
fn disjoint(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let left = validate_wkb(&inputs[0])?;
    let right = validate_wkb(&inputs[1])?;
    wrap!(disjoint(left, right))
}

#[polars_expr(output_type=Boolean)]
fn dwithin(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<3>(inputs)?;
    let left = validate_wkb(&inputs[0])?;
    let right = validate_wkb(&inputs[1])?;
    extract!(distance, inputs[2], D::Float64, f64);
    wrap!(dwithin(left, right, distance))
}

#[polars_expr(output_type=Boolean)]
fn intersects(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let left = validate_wkb(&inputs[0])?;
    let right = validate_wkb(&inputs[1])?;
    wrap!(intersects(left, right))
}

#[polars_expr(output_type=Boolean)]
fn overlaps(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let left = validate_wkb(&inputs[0])?;
    let right = validate_wkb(&inputs[1])?;
    wrap!(overlaps(left, right))
}

#[polars_expr(output_type=Boolean)]
fn touches(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let left = validate_wkb(&inputs[0])?;
    let right = validate_wkb(&inputs[1])?;
    wrap!(touches(left, right))
}

#[polars_expr(output_type=Boolean)]
fn within(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let left = validate_wkb(&inputs[0])?;
    let right = validate_wkb(&inputs[1])?;
    wrap!(within(left, right))
}

#[polars_expr(output_type=Boolean)]
fn equals(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let left = validate_wkb(&inputs[0])?;
    let right = validate_wkb(&inputs[1])?;
    wrap!(equals(left, right))
}

#[polars_expr(output_type=Boolean)]
fn equals_identical(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let left = validate_wkb(&inputs[0])?;
    let right = validate_wkb(&inputs[1])?;
    wrap!(equals_identical(left, right))
}

#[polars_expr(output_type=Boolean)]
fn equals_exact(inputs: &[Series], kwargs: args::EqualsExactKwargs) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let left = validate_wkb(&inputs[0])?;
    let right = validate_wkb(&inputs[1])?;
    wrap!(equals_exact(left, right, kwargs.tolerance))
}

#[polars_expr(output_type=String)]
fn relate(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let left = validate_wkb(&inputs[0])?;
    let right = validate_wkb(&inputs[1])?;
    wrap!(relate(left, right))
}

#[polars_expr(output_type=Boolean)]
fn relate_pattern(inputs: &[Series], kwargs: args::RelatePatternKwargs) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let left = validate_wkb(&inputs[0])?;
    let right = validate_wkb(&inputs[1])?;
    wrap!(relate_pattern(left, right, &kwargs.pattern))
}

#[polars_expr(output_type=Binary)]
fn difference(inputs: &[Series], kwargs: args::SetOperationKwargs) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let left = validate_wkb(&inputs[0])?;
    let right = validate_wkb(&inputs[1])?;
    match kwargs.grid_size {
        Some(grid_size) => wrap!(difference_prec(left, right, grid_size)),
        None => wrap!(difference(left, right)),
    }
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
        Some(grid_size) => wrap!(intersection_prec(left, right, grid_size)),
        None => wrap!(intersection(left, right)),
    }
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
        Some(grid_size) => wrap!(sym_difference_prec(left, right, grid_size)),
        None => wrap!(sym_difference(left, right)),
    }
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
        Some(grid_size) => wrap!(unary_union_prec(geom, grid_size)),
        None => wrap!(unary_union(geom)),
    }
}

#[polars_expr(output_type=Binary)]
fn disjoint_subset_union(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    wrap!(disjoint_subset_union(wkb))
}

#[polars_expr(output_type=Binary)]
fn union(inputs: &[Series], kwargs: args::SetOperationKwargs) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let left = validate_wkb(&inputs[0])?;
    let right = validate_wkb(&inputs[1])?;
    match kwargs.grid_size {
        Some(grid_size) => wrap!(union_prec(left, right, grid_size)),
        None => wrap!(union(left, right)),
    }
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
    wrap!(coverage_union(wkb))
}

#[polars_expr(output_type=Binary)]
fn coverage_union_all(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    wrap!(coverage_union_all(wkb))
}

#[polars_expr(output_type=Binary)]
fn polygonize(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    wrap!(polygonize(wkb))
}

#[polars_expr(output_type=Binary)]
fn collect(inputs: &[Series], kwargs: args::CollectKwargs) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    wrap!(collect(wkb, kwargs.into))
}

#[polars_expr(output_type=Binary)]
fn boundary(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    wrap!(boundary(wkb))
}

#[polars_expr(output_type=Binary)]
fn buffer(inputs: &[Series], kwargs: args::BufferKwargs) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    extract!(distance, inputs[1], D::Float64, f64);
    wrap!(buffer(wkb, distance, &kwargs))
}

#[polars_expr(output_type=Binary)]
fn offset_curve(inputs: &[Series], kwargs: args::OffsetCurveKwargs) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    extract!(distance, inputs[1], D::Float64, f64);
    wrap!(offset_curve(wkb, distance, &kwargs))
}

#[polars_expr(output_type=Binary)]
fn convex_hull(inputs: &[Series]) -> PolarsResult<Series> {
    let wkb = validate_wkb(&inputs[0])?;
    wrap!(convex_hull(wkb))
}

#[polars_expr(output_type=Binary)]
fn concave_hull(inputs: &[Series], kwargs: args::ConcaveHullKwargs) -> PolarsResult<Series> {
    let wkb = validate_wkb(&inputs[0])?;
    wrap!(concave_hull(wkb, &kwargs))
}

#[polars_expr(output_type=Binary)]
fn clip_by_rect(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    extract!(rect, inputs[1], D::Array(D::Float64.into(), 4), array);
    wrap!(clip_by_rect(wkb, rect))
}

#[polars_expr(output_type=Binary)]
fn centroid(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    wrap!(get_centroid(wkb))
}

#[polars_expr(output_type=Binary)]
fn center(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    wrap!(get_center(wkb))
}

#[polars_expr(output_type=Binary)]
fn delaunay_triangles(
    inputs: &[Series],
    kwargs: args::DelaunayTrianlesKwargs,
) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    wrap!(delaunay_triangulation(wkb, &kwargs))
}

#[polars_expr(output_type=Binary)]
fn segmentize(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    extract!(tolerance, inputs[1], D::Float64, f64);
    wrap!(densify(wkb, tolerance))
}

#[polars_expr(output_type=Binary)]
fn envelope(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    wrap!(envelope(wkb))
}

#[polars_expr(output_type=Binary)]
fn extract_unique_points(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    wrap!(extract_unique_points(wkb))
}

#[polars_expr(output_type=Binary)]
fn build_area(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    wrap!(build_area(wkb))
}

#[polars_expr(output_type=Binary)]
pub fn make_valid(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    wrap!(make_valid(wkb))
}

#[polars_expr(output_type=Binary)]
pub fn normalize(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    wrap!(normalize(wkb))
}

#[polars_expr(output_type=Binary)]
pub fn node(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    wrap!(node(wkb))
}

#[polars_expr(output_type=Binary)]
pub fn point_on_surface(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    wrap!(point_on_surface(wkb))
}

#[polars_expr(output_type=Binary)]
pub fn remove_repeated_points(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    extract!(tolerance, inputs[1], D::Float64, f64);
    wrap!(remove_repeated_points(wkb, tolerance))
}

#[polars_expr(output_type=Binary)]
pub fn reverse(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    wrap!(reverse(wkb))
}

#[polars_expr(output_type=Binary)]
pub fn simplify(inputs: &[Series], kwargs: args::SimplifyKwargs) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    extract!(tolerance, inputs[1], D::Float64, f64);
    match kwargs.preserve_topology {
        true => wrap!(topology_preserve_simplify(wkb, tolerance)),
        false => wrap!(simplify(wkb, tolerance)),
    }
}

#[polars_expr(output_type=Binary)]
pub fn force_2d(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    wrap!(force_2d(wkb))
}

#[polars_expr(output_type=Binary)]
pub fn force_3d(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    extract!(z, inputs[1], D::Float64, f64);
    wrap!(force_3d(wkb, z))
}

#[polars_expr(output_type=Binary)]
pub fn snap(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<3>(inputs)?;
    let left = validate_wkb(&inputs[0])?;
    let right = validate_wkb(&inputs[1])?;
    extract!(tolerance, inputs[2], D::Float64, f64);
    wrap!(snap(left, right, tolerance))
}

#[polars_expr(output_type=Binary)]
pub fn voronoi_polygons(inputs: &[Series], kwargs: args::VoronoiKwargs) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    wrap!(voronoi_polygons(wkb, &kwargs))
}

#[polars_expr(output_type=Binary)]
pub fn minimum_rotated_rectangle(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    wrap!(minimum_rotated_rectangle(wkb))
}

#[polars_expr(output_type=Binary)]
pub fn translate(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    extract!(factors, inputs[1], D::Array(D::Float64.into(), 3), array);
    wrap!(translate(wkb, factors))
}

#[polars_expr(output_type=Binary)]
pub fn rotate(inputs: &[Series], kwargs: args::TransformKwargs) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    extract!(angle, inputs[1], D::Float64, f64);
    match kwargs.origin {
        args::TransformOrigin::XY(o) => wrap!(rotate_around_point(wkb, angle, &o)),
        args::TransformOrigin::XYZ(o) => wrap!(rotate_around_point(wkb, angle, &(o.0, o.1))),
        args::TransformOrigin::Center => wrap!(rotate_around_center(wkb, angle)),
        args::TransformOrigin::Centroid => wrap!(rotate_around_centroid(wkb, angle)),
    }
}

#[polars_expr(output_type=Binary)]
pub fn scale(inputs: &[Series], kwargs: args::TransformKwargs) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    extract!(factors, inputs[1], D::Array(D::Float64.into(), 3), array);
    match kwargs.origin {
        args::TransformOrigin::XY(o) => wrap!(scale_from_point(wkb, factors, &(o.0, o.1, 0.0))),
        args::TransformOrigin::XYZ(origin) => wrap!(scale_from_point(wkb, factors, &origin)),
        args::TransformOrigin::Center => wrap!(scale_from_center(wkb, factors)),
        args::TransformOrigin::Centroid => wrap!(scale_from_centroid(wkb, factors)),
    }
}

#[polars_expr(output_type=Binary)]
pub fn skew(inputs: &[Series], kwargs: args::TransformKwargs) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    extract!(factors, inputs[1], D::Array(D::Float64.into(), 3), array);
    match kwargs.origin {
        args::TransformOrigin::XY(o) => wrap!(skew_from_point(wkb, factors, &(o.0, o.1, 0.0))),
        args::TransformOrigin::XYZ(origin) => wrap!(skew_from_point(wkb, factors, &origin)),
        args::TransformOrigin::Center => wrap!(skew_from_center(wkb, factors)),
        args::TransformOrigin::Centroid => wrap!(skew_from_centroid(wkb, factors)),
    }
}

#[polars_expr(output_type=Binary)]
pub fn affine_transform(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    let matrix = &inputs[1];
    match matrix.dtype() {
        D::Array(.., 6) => {
            extract!(matrix, matrix, D::Array(D::Float64.into(), 6), array);
            wrap!(affine_transform_2d(wkb, matrix))
        }
        D::Array(.., 12) => {
            extract!(matrix, matrix, D::Array(D::Float64.into(), 12), array);
            wrap!(affine_transform_3d(wkb, matrix))
        }
        _ => Err(to_compute_err(
            "matrix parameter should be of type array with shape (6 | 12)",
        )),
    }
}

#[polars_expr(output_type=Binary)]
pub fn interpolate(inputs: &[Series], kwargs: args::InterpolateKwargs) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    extract!(distance, inputs[1], D::Float64, f64);
    match kwargs.normalized {
        true => wrap!(interpolate_normalized(wkb, distance)),
        false => wrap!(interpolate(wkb, distance)),
    }
}

#[polars_expr(output_type=Float64)]
pub fn project(inputs: &[Series], kwargs: args::InterpolateKwargs) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let left = validate_wkb(&inputs[0])?;
    let right = validate_wkb(&inputs[1])?;
    match kwargs.normalized {
        true => wrap!(project_normalized(left, right)),
        false => wrap!(project(left, right)),
    }
}

#[polars_expr(output_type=Binary)]
pub fn substring(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<3>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    extract!(start, inputs[1], D::Float64, f64);
    extract!(end, inputs[2], D::Float64, f64);
    wrap!(substring(wkb, start, end))
}

#[polars_expr(output_type=Binary)]
pub fn line_merge(inputs: &[Series], kwargs: args::LineMergeKwargs) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    match kwargs.directed {
        true => wrap!(line_merge_directed(wkb)),
        false => wrap!(line_merge(wkb)),
    }
}

#[polars_expr(output_type=Binary)]
pub fn shared_paths(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let left = validate_wkb(&inputs[0])?;
    let right = validate_wkb(&inputs[1])?;
    wrap!(shared_paths(left, right))
}

#[polars_expr(output_type=Binary)]
pub fn shortest_line(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let left = validate_wkb(&inputs[0])?;
    let right = validate_wkb(&inputs[1])?;
    wrap!(shortest_line(left, right))
}

#[polars_expr(output_type_func=output_type_sjoin)]
pub fn sjoin(inputs: &[Series], kwargs: args::SjoinKwargs) -> PolarsResult<Series> {
    use args::SjoinPredicate::Dwithin;
    let inputs = validate_inputs_length::<2>(inputs)?;
    let left = validate_wkb(&inputs[0])?;
    let right = validate_wkb(&inputs[1])?;
    match kwargs.predicate {
        Dwithin(distance) => functions::sjoin_dwithin(left, right, distance),
        predicate => functions::sjoin(left, right, predicate),
    }
    .map(|(left, right)| {
        let left = Series::from_vec("left_index".into(), left);
        let right = Series::from_vec("right_index".into(), right);
        StructChunked::from_series("".into(), left.len(), [left, right].iter())
    })
    .map_err(to_compute_err)?
    .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
pub fn flip_coordinates(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    wrap!(flip_coordinates(wkb))
}

#[polars_expr(output_type=Binary)]
pub fn to_srid(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let wkb = validate_wkb(&inputs[0])?;
    extract!(srid, inputs[1], D::Int64, i64);
    wrap!(to_srid(wkb, srid))
}
