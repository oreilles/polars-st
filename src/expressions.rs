use crate::{
    args,
    functions::{self, ToEwkb},
};
use geos::{Geom, Geometry};
use polars::{error::to_compute_err, prelude::*};
use polars_plan::{
    plans::{Literal, NULL},
    prelude::{ApplyOptions, FunctionFlags, FunctionOptions},
};
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
        DataType::Array(DataType::Float64.into(), 4),
    ))
}

fn output_type_geometry_list(input_fields: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        first_field_name(input_fields)?.clone(),
        DataType::List(DataType::Binary.into()),
    ))
}

fn output_type_sjoin(input_fields: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        first_field_name(input_fields)?.clone(),
        DataType::Struct(vec![
            Field::new("left_index".into(), DataType::Float64),
            Field::new("right_index".into(), DataType::Float64),
        ]),
    ))
}

fn validate_inputs_length<const M: usize>(inputs: &[Series]) -> PolarsResult<&[Series; M]> {
    polars_ensure!(
        inputs.len() == M,
        InvalidOperation: "Invalid number of arguments."
    );
    let inputs: &[Series; M] = inputs.try_into().unwrap();
    Ok(inputs)
}

#[polars_expr(output_type=Binary)]
fn from_wkt(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;

    functions::from_wkt(inputs[0].str()?)
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
fn from_xy(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let fields = inputs[0].struct_()?.fields_as_series();
    let x = fields[0].strict_cast(&DataType::Float64)?;
    let y = fields[1].strict_cast(&DataType::Float64)?;
    let z = fields
        .get(2)
        .map(|s| s.strict_cast(&DataType::Float64))
        .transpose()?;
    let x = x.f64()?;
    let y = y.f64()?;
    let z = z.as_ref().map(|s| s.f64()).transpose()?;
    functions::from_xy(x, y, z)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=UInt32)]
fn geometry_type(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    functions::get_type_id(inputs[0].binary()?)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Int32)]
fn dimensions(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    functions::get_num_dimensions(inputs[0].binary()?)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=UInt32)]
fn coordinate_dimension(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    functions::get_coordinate_dimension(inputs[0].binary()?)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Float64)]
fn coordinates(inputs: &[Series], kwargs: args::GetCoordinatesKwargs) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = inputs[0].binary()?;
    functions::get_coordinates(wkb, kwargs.output_dimension)
        .map_err(to_compute_err)?
        .into_series()
        .with_name(wkb.name().clone())
        .cast(&DataType::List(
            DataType::Array(DataType::Float64.into(), 2).into(),
        ))
}

#[polars_expr(output_type=Int32)]
fn srid(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    functions::get_srid(inputs[0].binary()?)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn set_srid(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let wkb = inputs[0].binary()?;
    let srid = inputs[1].cast(&DataType::Int32)?;
    let srid = srid.i32()?;
    functions::set_srid(wkb, srid)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Float64)]
fn x(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    functions::get_x(inputs[0].binary()?)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Float64)]
fn y(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    functions::get_y(inputs[0].binary()?)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Float64)]
fn z(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    functions::get_z(inputs[0].binary()?)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Float64)]
fn m(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    functions::get_m(inputs[0].binary()?)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn exterior_ring(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    functions::get_exterior_ring(inputs[0].binary()?)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type_func=output_type_geometry_list)]
fn interior_rings(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = inputs[0].binary()?;
    functions::get_interior_rings(wkb)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)?
        .with_name(wkb.name().clone())
        .cast(&DataType::List(DataType::Binary.into()))
}

#[polars_expr(output_type=UInt32)]
fn count_points(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    functions::get_num_points(inputs[0].binary()?)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=UInt32)]
fn count_interior_rings(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    functions::get_num_interior_rings(inputs[0].binary()?)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=UInt32)]
fn count_geometries(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    functions::get_num_geometries(inputs[0].binary()?)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=UInt32)]
fn count_coordinates(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    functions::get_num_coordinates(inputs[0].binary()?)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn get_point(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let wkb = inputs[0].binary()?;
    let index = inputs[1].strict_cast(&DataType::UInt32)?;
    let index = index.u32()?;
    functions::get_point_n(wkb, index)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn get_interior_ring(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let wkb = inputs[0].binary()?;
    let index = inputs[1].strict_cast(&DataType::UInt32)?;
    let index = index.u32()?;
    functions::get_interior_ring_n(wkb, index)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn get_geometry(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let wkb = inputs[0].binary()?;
    let index = inputs[1].strict_cast(&DataType::UInt32)?;
    let index = index.u32()?;
    functions::get_geometry_n(wkb, index)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type_func=output_type_geometry_list)]
fn parts(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = inputs[0].binary()?;
    functions::get_parts(wkb)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)?
        .with_name(wkb.name().clone())
        .cast(&DataType::List(DataType::Binary.into()))
}

#[polars_expr(output_type=Float64)]
fn precision(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    functions::get_precision(inputs[0].binary()?)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn set_precision(inputs: &[Series], kwargs: args::SetPrecisionKwargs) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let wkb = inputs[0].binary()?;
    let precision = inputs[1].strict_cast(&DataType::Float64)?;
    let precision = precision.f64()?;
    functions::set_precision(wkb, precision, &kwargs)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=String)]
fn to_wkt(inputs: &[Series], kwargs: args::ToWktKwargs) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    functions::to_wkt(inputs[0].binary()?, &kwargs)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=String)]
fn to_ewkt(inputs: &[Series], kwargs: args::ToWktKwargs) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    functions::to_ewkt(inputs[0].binary()?, &kwargs)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn to_wkb(inputs: &[Series], kwargs: args::ToWkbKwargs) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    functions::to_wkb(inputs[0].binary()?, &kwargs)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=String)]
fn to_geojson(inputs: &[Series], kwargs: args::ToGeoJsonKwargs) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    functions::to_geojson(inputs[0].binary()?, &kwargs)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Float64)]
fn area(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    functions::area(inputs[0].binary()?)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type_func=output_type_bounds)]
fn bounds(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = inputs[0].binary()?;
    functions::bounds(wkb)
        .map_err(to_compute_err)?
        .into_series()
        .with_name(wkb.name().clone())
        .cast(&DataType::Array(DataType::Float64.into(), 4))
}

#[polars_expr(output_type_func=output_type_bounds)]
fn par_bounds(inputs: &[Series]) -> PolarsResult<Series> {
    functions::bounds(inputs[0].binary()?)
        .map_err(to_compute_err)?
        .into_series()
        .cast(&DataType::Array(DataType::Float64.into(), 4))
}

#[polars_expr(output_type_func=output_type_bounds)]
fn total_bounds(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let bounds = functions::bounds(inputs[0].binary()?)
        .map_err(to_compute_err)?
        .cast(&DataType::List(DataType::Float64.into()))?;
    let bounds = bounds.list()?;
    let mut builder = ListPrimitiveChunkedBuilder::<Float64Type>::new(
        bounds.name().clone(),
        1,
        4,
        DataType::Float64,
    );
    builder.append_slice(&[
        bounds.lst_get(0, false)?.min()?.unwrap_or(f64::NAN),
        bounds.lst_get(1, false)?.min()?.unwrap_or(f64::NAN),
        bounds.lst_get(2, false)?.max()?.unwrap_or(f64::NAN),
        bounds.lst_get(3, false)?.max()?.unwrap_or(f64::NAN),
    ]);
    builder
        .finish()
        .into_series()
        .cast(&DataType::Array(DataType::Float64.into(), 4))
}

#[polars_expr(output_type=Float64)]
fn length(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    functions::length(inputs[0].binary()?)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn distance(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let left = inputs[0].binary()?;
    let right = inputs[1].binary()?;
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
    let left = inputs[0].binary()?;
    let right = inputs[1].binary()?;
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
    let left = inputs[0].binary()?;
    let right = inputs[1].binary()?;
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
    functions::minimum_clearance(inputs[0].binary()?)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

// Predicates

#[polars_expr(output_type=Boolean)]
fn has_z(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    functions::has_z(inputs[0].binary()?)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Boolean)]
fn has_m(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    functions::has_m(inputs[0].binary()?)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Boolean)]
fn is_ccw(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    functions::is_ccw(inputs[0].binary()?)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Boolean)]
fn is_closed(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    functions::is_closed(inputs[0].binary()?)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Boolean)]
fn is_empty(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    functions::is_empty(inputs[0].binary()?)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Boolean)]
fn is_ring(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    functions::is_ring(inputs[0].binary()?)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Boolean)]
fn is_simple(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    functions::is_simple(inputs[0].binary()?)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Boolean)]
fn is_valid(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    functions::is_valid(inputs[0].binary()?)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=String)]
fn is_valid_reason(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    functions::is_valid_reason(inputs[0].binary()?)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Boolean)]
fn crosses(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let left = inputs[0].binary()?;
    let right = inputs[1].binary()?;
    functions::crosses(left, right)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Boolean)]
fn contains(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let left = inputs[0].binary()?;
    let right = inputs[1].binary()?;
    functions::contains(left, right)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Boolean)]
fn contains_properly(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let left = inputs[0].binary()?;
    let right = inputs[1].binary()?;
    functions::contains_properly(left, right)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Boolean)]
fn covered_by(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let left = inputs[0].binary()?;
    let right = inputs[1].binary()?;
    functions::covered_by(left, right)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Boolean)]
fn covers(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let left = inputs[0].binary()?;
    let right = inputs[1].binary()?;
    functions::covers(left, right)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Boolean)]
fn disjoint(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let left = inputs[0].binary()?;
    let right = inputs[1].binary()?;
    functions::disjoint(left, right)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Boolean)]
fn dwithin(inputs: &[Series], kwargs: args::DWithinKwargs) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let left = inputs[0].binary()?;
    let right = inputs[1].binary()?;
    functions::dwithin(left, right, kwargs.distance)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Boolean)]
fn intersects(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let left = inputs[0].binary()?;
    let right = inputs[1].binary()?;
    functions::intersects(left, right)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Boolean)]
fn overlaps(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let left = inputs[0].binary()?;
    let right = inputs[1].binary()?;
    functions::overlaps(left, right)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Boolean)]
fn touches(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let left = inputs[0].binary()?;
    let right = inputs[1].binary()?;
    functions::touches(left, right)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Boolean)]
fn within(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let left = inputs[0].binary()?;
    let right = inputs[1].binary()?;
    functions::within(left, right)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Boolean)]
fn equals(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let left = inputs[0].binary()?;
    let right = inputs[1].binary()?;
    functions::equals(left, right)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Boolean)]
fn equals_identical(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let left = inputs[0].binary()?;
    let right = inputs[1].binary()?;
    functions::equals_identical(left, right)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Boolean)]
fn equals_exact(inputs: &[Series], kwargs: args::EqualsExactKwargs) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let left = inputs[0].binary()?;
    let right = inputs[1].binary()?;
    functions::equals_exact(left, right, kwargs.tolerance)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=String)]
fn relate(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let left = inputs[0].binary()?;
    let right = inputs[1].binary()?;
    functions::relate(left, right)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Boolean)]
fn relate_pattern(inputs: &[Series], kwargs: args::RelatePatternKwargs) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let left = inputs[0].binary()?;
    let right = inputs[1].binary()?;
    functions::relate_pattern(left, right, &kwargs.pattern)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn intersects_xy(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let wkb = inputs[0].binary()?;
    let s = inputs[1].struct_()?;
    let x = s.field_by_name("x")?.strict_cast(&DataType::Float64)?;
    let y = s.field_by_name("y")?.strict_cast(&DataType::Float64)?;
    let x = x.f64()?;
    let y = y.f64()?;
    functions::intersects_xy(wkb, x, y)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn contains_xy(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let wkb = inputs[0].binary()?;
    let s = inputs[1].struct_()?;
    let x = s.field_by_name("x")?.strict_cast(&DataType::Float64)?;
    let y = s.field_by_name("y")?.strict_cast(&DataType::Float64)?;
    let x = x.f64()?;
    let y = y.f64()?;
    functions::contains_xy(wkb, x, y)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn difference(inputs: &[Series], kwargs: args::SetOperationKwargs) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let left = inputs[0].binary()?;
    let right = inputs[1].binary()?;
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
    let wkb = inputs[0].binary()?;
    let it = wkb.into_iter().flatten().map(Geometry::new_from_wkb);
    match kwargs.grid_size {
        Some(g) => it
            .flatten()
            .try_reduce(|left, right| left.difference_prec(&right, g)),
        None => it
            .flatten()
            .try_reduce(|left, right| left.difference(&right)),
    }
    .map(|geom| geom.unwrap_or_else(|| Geometry::new_from_wkt("GEOMETRYCOLLECTION EMPTY").unwrap()))
    .and_then(|geom| geom.to_ewkb())
    .map(|res| Series::new(wkb.name().clone(), [res]))
    .map_err(to_compute_err)
}

#[polars_expr(output_type=Binary)]
fn intersection(inputs: &[Series], kwargs: args::SetOperationKwargs) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let left = inputs[0].binary()?;
    let right = inputs[1].binary()?;
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
    let wkb = inputs[0].binary()?;
    let it = wkb.into_iter().flatten().map(Geometry::new_from_wkb);
    match kwargs.grid_size {
        Some(g) => it.flatten().try_reduce(|a, b| a.intersection_prec(&b, g)),
        None => it.flatten().try_reduce(|a, b| a.intersection(&b)),
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
    let left = inputs[0].binary()?;
    let right = inputs[1].binary()?;
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
    let wkb = inputs[0].binary()?;
    let it = wkb.into_iter().flatten().map(Geometry::new_from_wkb);
    match kwargs.grid_size {
        Some(g) => it.flatten().try_reduce(|a, b| a.sym_difference_prec(&b, g)),
        None => it.flatten().try_reduce(|a, b| a.sym_difference(&b)),
    }
    .map(|geom| geom.unwrap_or_else(|| Geometry::new_from_wkt("GEOMETRYCOLLECTION EMPTY").unwrap()))
    .and_then(|geom| geom.to_ewkb())
    .map_err(to_compute_err)
    .map(|res| Series::new(wkb.name().clone(), [res]))
}

#[polars_expr(output_type=Binary)]
fn unary_union(inputs: &[Series], kwargs: args::SetOperationKwargs) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let geom = inputs[0].binary()?;
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
    functions::disjoint_subset_union(inputs[0].binary()?)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn union(inputs: &[Series], kwargs: args::SetOperationKwargs) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let left = inputs[0].binary()?;
    let right = inputs[1].binary()?;
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
    let geom = inputs[0].binary()?;
    let it = geom.into_iter().flatten().map(Geometry::new_from_wkb);
    match kwargs.grid_size {
        Some(g) => it
            .flatten()
            .try_reduce(|left, right| left.union_prec(&right, g)),
        None => it.flatten().try_reduce(|left, right| left.union(&right)),
    }
    .map(|geom| geom.unwrap_or_else(|| Geometry::new_from_wkt("GEOMETRYCOLLECTION EMPTY").unwrap()))
    .and_then(|geom| geom.to_ewkb())
    .map_err(to_compute_err)
    .map(|wkb| Series::new(geom.name().clone(), [wkb]))
}

#[polars_expr(output_type=Binary)]
fn coverage_union(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    functions::coverage_union(inputs[0].binary()?)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn coverage_union_all(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    functions::coverage_union_all(inputs[0].binary()?)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn polygonize(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    functions::polygonize(inputs[0].binary()?)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn multipoint(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    functions::multipoint(inputs[0].binary()?)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn multilinestring(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    functions::multilinestring(inputs[0].binary()?)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn multipolygon(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    functions::multipolygon(inputs[0].binary()?)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn geometrycollection(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    functions::geometrycollection(inputs[0].binary()?)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn collect(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    functions::collect(inputs[0].binary()?)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn boundary(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    functions::boundary(inputs[0].binary()?)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn buffer(inputs: &[Series], kwargs: args::BufferKwargs) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let wkb = inputs[0].binary()?;
    let distance = inputs[1].strict_cast(&DataType::Float64)?;
    let distance = distance.f64()?;
    functions::buffer(wkb, distance, &kwargs)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn offset_curve(inputs: &[Series], kwargs: args::OffsetCurveKwargs) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let wkb = inputs[0].binary()?;
    let distance = inputs[1].strict_cast(&DataType::Float64)?;
    let distance = distance.f64()?;
    functions::offset_curve(wkb, distance, &kwargs)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn convex_hull(inputs: &[Series]) -> PolarsResult<Series> {
    let wkb = inputs[0].binary()?;
    functions::convex_hull(wkb)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn concave_hull(inputs: &[Series], kwargs: args::ConcaveHullKwargs) -> PolarsResult<Series> {
    let wkb = inputs[0].binary()?;
    functions::concave_hull(wkb, &kwargs)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn clip_by_rect(inputs: &[Series], kwargs: args::ClipByRectKwargs) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    functions::clip_by_rect(inputs[0].binary()?, &kwargs)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn centroid(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    functions::get_centroid(inputs[0].binary()?)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn center(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    functions::get_center(inputs[0].binary()?)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn delaunay_triangles(
    inputs: &[Series],
    kwargs: args::DelaunayTrianlesKwargs,
) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    functions::delaunay_triangulation(inputs[0].binary()?, &kwargs)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn segmentize(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let wkb = inputs[0].binary()?;
    let tolerance = inputs[1].strict_cast(&DataType::Float64)?;
    let tolerance = tolerance.f64()?;
    functions::densify(wkb, tolerance)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn envelope(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    functions::envelope(inputs[0].binary()?)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn extract_unique_points(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    functions::extract_unique_points(inputs[0].binary()?)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn build_area(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    functions::build_area(inputs[0].binary()?)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
pub fn make_valid(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    functions::make_valid(inputs[0].binary()?)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
pub fn normalize(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    functions::normalize(inputs[0].binary()?)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
pub fn node(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    functions::node(inputs[0].binary()?)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
pub fn point_on_surface(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    functions::point_on_surface(inputs[0].binary()?)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
pub fn remove_repeated_points(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let wkb = inputs[0].binary()?;
    let tolerance = inputs[1].strict_cast(&DataType::Float64)?;
    let tolerance = tolerance.f64()?;
    functions::remove_repeated_points(wkb, tolerance)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
pub fn reverse(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = inputs[0].binary()?;
    functions::reverse(wkb)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
pub fn simplify(inputs: &[Series], kwargs: args::SimplifyKwargs) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let wkb = inputs[0].binary()?;
    let tolerance = inputs[1].strict_cast(&DataType::Float64)?;
    let tolerance = tolerance.f64()?;
    match kwargs.preserve_topology {
        true => functions::topology_preserve_simplify(wkb, tolerance),
        false => functions::simplify(wkb, tolerance),
    }
    .map_err(to_compute_err)
    .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
pub fn snap(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<3>(inputs)?;
    let left = inputs[0].binary()?;
    let right = inputs[1].binary()?;
    let tolerance = inputs[2].cast(&DataType::Float64)?;
    let tolerance = tolerance.f64()?;
    functions::snap(left, right, tolerance)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
pub fn voronoi_polygons(inputs: &[Series], kwargs: args::VoronoiKwargs) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    functions::voronoi_polygons(inputs[0].binary()?, &kwargs)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
pub fn minimum_rotated_rectangle(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    functions::minimum_rotated_rectangle(inputs[0].binary()?)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
pub fn affine_transform(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let wkb = inputs[0].binary()?;
    let matrix = &inputs[1];
    let matrix_size = match matrix.dtype() {
        DataType::Array(.., 6) => Ok(6),
        DataType::Array(.., 12) => Ok(12),
        _ => Err(to_compute_err(
            "matrix parameter should be an numeric array with shape (6 | 12)",
        )),
    }?;
    let matrix = matrix.cast(&DataType::Array(DataType::Float64.into(), matrix_size))?;
    match matrix_size {
        6 => functions::affine_transform_2d(wkb, matrix.array()?),
        12 => functions::affine_transform_3d(wkb, matrix.array()?),
        _ => unreachable!(),
    }
    .map_err(to_compute_err)
    .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
pub fn interpolate(inputs: &[Series], kwargs: args::InterpolateKwargs) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let wkb = inputs[0].binary()?;
    let distance = inputs[1].strict_cast(&DataType::Float64)?;
    let distance = distance.f64()?;
    match kwargs.normalized {
        true => functions::interpolate_normalized(wkb, distance),
        false => functions::interpolate(wkb, distance),
    }
    .map_err(to_compute_err)
    .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
pub fn line_merge(inputs: &[Series], kwargs: args::LineMergeKwargs) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = inputs[0].binary()?;
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
    let left = inputs[0].binary()?;
    let right = inputs[1].binary()?;
    functions::shared_paths(left, right)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
pub fn shortest_line(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let left = inputs[0].binary()?;
    let right = inputs[1].binary()?;
    functions::shortest_line(left, right)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type_func=output_type_sjoin)]
pub fn sjoin(inputs: &[Series], kwargs: args::SpatialJoinKwargs) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let left = inputs[0].binary()?;
    let right = inputs[1].binary()?;
    functions::sjoin(left, right, kwargs.predicate)
        .map_err(to_compute_err)
        .map(|(left_index, right_index)| {
            StructChunked::from_series(
                left.name().clone(),
                &[left_index.into_series(), right_index.into_series()],
            )
            .map(IntoSeries::into_series)
        })?
}

#[polars_expr(output_type=Binary)]
pub fn flip_coordinates(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = inputs[0].binary()?;
    functions::flip_coordinates(wkb)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[pyfunction]
pub fn apply_coordinates(
    py: Python,
    pyseries: PySeries,
    pyfunc: PyObject,
) -> Result<PySeries, PyPolarsErr> {
    fn apply_coordinates<F>(inputs: &[Series], func: F) -> PolarsResult<Series>
    where
        F: Fn(Series, Series, Option<Series>) -> PolarsResult<(Series, Series, Option<Series>)>,
    {
        let wkb = inputs[0].binary()?;
        functions::apply_coordinates(wkb, func)
            .map_err(to_compute_err)
            .map(IntoSeries::into_series)
    }

    let res = apply_coordinates(&[pyseries.0], |x, y, z| {
        let (tx, ty, tz): (PySeries, PySeries, Option<PySeries>) = pyfunc
            .call1(py, (PySeries(x), PySeries(y), z.map(PySeries)))
            .map_err(to_compute_err)?
            .extract(py)
            .map_err(to_compute_err)?;
        Ok((
            tx.0.cast(&DataType::Float64)?,
            ty.0.cast(&DataType::Float64)?,
            tz.map(|tz| tz.0.cast(&DataType::Float64)).transpose()?,
        ))
    })?;

    Ok(PySeries(res))
}

#[polars_expr(output_type=Binary)]
pub fn to_srid(inputs: &[Series], kwargs: args::ToSridKwargs) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = inputs[0].binary()?;

    if wkb.len() == wkb.null_count() {
        return Ok(Series::full_null(
            wkb.name().clone(),
            wkb.len(),
            wkb.dtype(),
        ));
    }

    let srids = functions::get_srid(wkb).map_err(to_compute_err)?;
    let unique_srids = srids.unique()?.drop_nulls();

    if unique_srids.len() == 1 {
        return functions::to_srid(wkb, unique_srids.get(0).unwrap(), kwargs.srid)
            .map_err(to_compute_err)
            .map(IntoSeries::into_series);
    }

    let chained_then = unique_srids.into_iter().flatten().fold(
        // Must repeat this twice to get a ChainedThen expression
        when(false).then(NULL.lit()).when(false).then(NULL.lit()),
        |chain, srid| {
            let function = move |s: &mut [Series]| {
                functions::to_srid(s[0].binary()?, srid, kwargs.srid)
                    .map_err(to_compute_err)
                    .map(IntoSeries::into_series)
                    .map(Some)
            };
            chain
                .when(col("srid").eq(srid))
                .then(Expr::AnonymousFunction {
                    input: vec![col("wkb")],
                    function: SpecialEq::new(Arc::new(function)),
                    output_type: GetOutput::from_type(DataType::Binary),
                    options: FunctionOptions {
                        collect_groups: ApplyOptions::ElementWise,
                        fmt_str: "transform_xy",
                        flags: FunctionFlags::default(),
                        ..Default::default()
                    },
                })
        },
    );
    let res = df! {"wkb" => &inputs[0], "srid" => srids }?
        .lazy()
        .select([chained_then.otherwise(NULL.lit())])
        .collect()?;

    Ok(res.get_columns()[0].clone())
}
