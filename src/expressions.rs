use crate::{
    geo::{self, ToEwkb},
    kwargs,
};
use geos::{Geom, Geometry};
use polars::{error::to_compute_err, prelude::*};
use polars_plan::{
    plans::{Literal, NULL},
    prelude::{ApplyOptions, FunctionFlags, FunctionOptions},
};
use pyo3::prelude::*;
use pyo3_polars::{derive::polars_expr, PySeries};

fn first_field_name(fields: &[Field]) -> PolarsResult<&SmartString> {
    fields
        .first()
        .map(Field::name)
        .ok_or_else(|| to_compute_err("Invalid number of arguments."))
}

fn output_type_bounds(input_fields: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        first_field_name(input_fields)?,
        DataType::Array(DataType::Float64.into(), 4),
    ))
}

fn output_type_geometry_list(input_fields: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        first_field_name(input_fields)?,
        DataType::List(DataType::Binary.into()),
    ))
}

fn output_type_sjoin(input_fields: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        first_field_name(input_fields)?,
        DataType::Struct(vec![
            Field::new("left_index", DataType::Float64),
            Field::new("right_index", DataType::Float64),
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

    geo::from_wkt(inputs[0].str()?)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn from_geojson(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    geo::from_geojson(inputs[0].str()?)
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
    geo::from_xy(x, y, z)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=UInt32)]
fn geometry_type(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    geo::get_type_id(inputs[0].binary()?)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Int32)]
fn dimensions(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    geo::get_num_dimensions(inputs[0].binary()?)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=UInt32)]
fn coordinate_dimension(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    geo::get_coordinate_dimension(inputs[0].binary()?)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Float64)]
fn coordinates(inputs: &[Series], kwargs: kwargs::GetCoordinatesKwargs) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = inputs[0].binary()?;
    geo::get_coordinates(wkb, kwargs.output_dimension)
        .map_err(to_compute_err)?
        .into_series()
        .with_name(wkb.name())
        .cast(&DataType::List(
            DataType::Array(DataType::Float64.into(), 2).into(),
        ))
}

#[polars_expr(output_type=Int32)]
fn srid(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    geo::get_srid(inputs[0].binary()?)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn set_srid(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let wkb = inputs[0].binary()?;
    let srid = inputs[1].cast(&DataType::Int32)?;
    let srid = srid.i32()?;
    geo::set_srid(wkb, srid)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Float64)]
fn x(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    geo::get_x(inputs[0].binary()?)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Float64)]
fn y(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    geo::get_y(inputs[0].binary()?)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Float64)]
fn z(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    geo::get_z(inputs[0].binary()?)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Float64)]
fn m(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    geo::get_m(inputs[0].binary()?)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn exterior_ring(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    geo::get_exterior_ring(inputs[0].binary()?)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type_func=output_type_geometry_list)]
fn interior_rings(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = inputs[0].binary()?;
    geo::get_interior_rings(wkb)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)?
        .with_name(wkb.name())
        .cast(&DataType::List(DataType::Binary.into()))
}

#[polars_expr(output_type=UInt32)]
fn count_points(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    geo::get_num_points(inputs[0].binary()?)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=UInt32)]
fn count_interior_rings(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    geo::get_num_interior_rings(inputs[0].binary()?)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=UInt32)]
fn count_geometries(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    geo::get_num_geometries(inputs[0].binary()?)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=UInt32)]
fn count_coordinates(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    geo::get_num_coordinates(inputs[0].binary()?)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn get_point(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let wkb = inputs[0].binary()?;
    let index = inputs[1].strict_cast(&DataType::UInt32)?;
    let index = index.u32()?;
    geo::get_point_n(wkb, index)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn get_interior_ring(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let wkb = inputs[0].binary()?;
    let index = inputs[1].strict_cast(&DataType::UInt32)?;
    let index = index.u32()?;
    geo::get_interior_ring_n(wkb, index)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn get_geometry(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let wkb = inputs[0].binary()?;
    let index = inputs[1].strict_cast(&DataType::UInt32)?;
    let index = index.u32()?;
    geo::get_geometry_n(wkb, index)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type_func=output_type_geometry_list)]
fn parts(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = inputs[0].binary()?;
    geo::get_parts(wkb)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)?
        .with_name(wkb.name())
        .cast(&DataType::List(DataType::Binary.into()))
}

#[polars_expr(output_type=Float64)]
fn precision(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    geo::get_precision(inputs[0].binary()?)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn set_precision(inputs: &[Series], kwargs: kwargs::SetPrecisionKwargs) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let wkb = inputs[0].binary()?;
    let precision = inputs[1].strict_cast(&DataType::Float64)?;
    let precision = precision.f64()?;
    geo::set_precision(wkb, precision, &kwargs)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=String)]
fn to_wkt(inputs: &[Series], kwargs: kwargs::ToWktKwargs) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    geo::to_wkt(inputs[0].binary()?, &kwargs)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=String)]
fn to_ewkt(inputs: &[Series], kwargs: kwargs::ToWktKwargs) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    geo::to_ewkt(inputs[0].binary()?, &kwargs)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn to_wkb(inputs: &[Series], kwargs: kwargs::ToWkbKwargs) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    geo::to_wkb(inputs[0].binary()?, &kwargs)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=String)]
fn to_geojson(inputs: &[Series], kwargs: kwargs::ToGeoJsonKwargs) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    geo::to_geojson(inputs[0].binary()?, &kwargs)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Float64)]
fn area(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    geo::area(inputs[0].binary()?)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type_func=output_type_bounds)]
fn bounds(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = inputs[0].binary()?;
    geo::bounds(wkb)
        .map_err(to_compute_err)?
        .into_series()
        .with_name(wkb.name())
        .cast(&DataType::Array(DataType::Float64.into(), 4))
}

#[polars_expr(output_type_func=output_type_bounds)]
fn par_bounds(inputs: &[Series]) -> PolarsResult<Series> {
    geo::bounds(inputs[0].binary()?)
        .map_err(to_compute_err)?
        .into_series()
        .cast(&DataType::Array(DataType::Float64.into(), 4))
}

#[polars_expr(output_type_func=output_type_bounds)]
fn total_bounds(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let bounds = geo::bounds(inputs[0].binary()?)
        .map_err(to_compute_err)?
        .cast(&DataType::List(DataType::Float64.into()))?;
    let bounds = bounds.list()?;
    let mut builder =
        ListPrimitiveChunkedBuilder::<Float64Type>::new(bounds.name(), 1, 4, DataType::Float64);
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
    geo::length(inputs[0].binary()?)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn distance(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let left = inputs[0].binary()?;
    let right = inputs[1].binary()?;
    geo::distance(left, right)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Float64)]
fn hausdorff_distance(
    inputs: &[Series],
    kwargs: kwargs::DistanceDensifyKwargs,
) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let left = inputs[0].binary()?;
    let right = inputs[1].binary()?;
    match kwargs.densify {
        Some(densify) => geo::hausdorff_distance_densify(left, right, densify),
        None => geo::hausdorff_distance(left, right),
    }
    .map_err(to_compute_err)
    .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Float64)]
fn frechet_distance(
    inputs: &[Series],
    kwargs: kwargs::DistanceDensifyKwargs,
) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let left = inputs[0].binary()?;
    let right = inputs[1].binary()?;
    match kwargs.densify {
        Some(densify) => geo::frechet_distance_densify(left, right, densify),
        None => geo::frechet_distance(left, right),
    }
    .map_err(to_compute_err)
    .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Float64)]
fn minimum_clearance(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    geo::minimum_clearance(inputs[0].binary()?)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

// Predicates

#[polars_expr(output_type=Boolean)]
fn has_z(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    geo::has_z(inputs[0].binary()?)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Boolean)]
fn has_m(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    geo::has_m(inputs[0].binary()?)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Boolean)]
fn is_ccw(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    geo::is_ccw(inputs[0].binary()?)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Boolean)]
fn is_closed(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    geo::is_closed(inputs[0].binary()?)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Boolean)]
fn is_empty(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    geo::is_empty(inputs[0].binary()?)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Boolean)]
fn is_ring(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    geo::is_ring(inputs[0].binary()?)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Boolean)]
fn is_simple(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    geo::is_simple(inputs[0].binary()?)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Boolean)]
fn is_valid(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    geo::is_valid(inputs[0].binary()?)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=String)]
fn is_valid_reason(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    geo::is_valid_reason(inputs[0].binary()?)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Boolean)]
fn crosses(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let left = inputs[0].binary()?;
    let right = inputs[1].binary()?;
    geo::crosses(left, right)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Boolean)]
fn contains(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let left = inputs[0].binary()?;
    let right = inputs[1].binary()?;
    geo::contains(left, right)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Boolean)]
fn contains_properly(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let left = inputs[0].binary()?;
    let right = inputs[1].binary()?;
    geo::contains_properly(left, right)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Boolean)]
fn covered_by(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let left = inputs[0].binary()?;
    let right = inputs[1].binary()?;
    geo::covered_by(left, right)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Boolean)]
fn covers(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let left = inputs[0].binary()?;
    let right = inputs[1].binary()?;
    geo::covers(left, right)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Boolean)]
fn disjoint(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let left = inputs[0].binary()?;
    let right = inputs[1].binary()?;
    geo::disjoint(left, right)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Boolean)]
fn dwithin(inputs: &[Series], kwargs: kwargs::DWithinKwargs) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let left = inputs[0].binary()?;
    let right = inputs[1].binary()?;
    geo::dwithin(left, right, kwargs.distance)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Boolean)]
fn intersects(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let left = inputs[0].binary()?;
    let right = inputs[1].binary()?;
    geo::intersects(left, right)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Boolean)]
fn overlaps(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let left = inputs[0].binary()?;
    let right = inputs[1].binary()?;
    geo::overlaps(left, right)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Boolean)]
fn touches(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let left = inputs[0].binary()?;
    let right = inputs[1].binary()?;
    geo::touches(left, right)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Boolean)]
fn within(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let left = inputs[0].binary()?;
    let right = inputs[1].binary()?;
    geo::within(left, right)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Boolean)]
fn equals(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let left = inputs[0].binary()?;
    let right = inputs[1].binary()?;
    geo::equals(left, right)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Boolean)]
fn equals_identical(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let left = inputs[0].binary()?;
    let right = inputs[1].binary()?;
    geo::equals_identical(left, right)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Boolean)]
fn equals_exact(inputs: &[Series], kwargs: kwargs::EqualsExactKwargs) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let left = inputs[0].binary()?;
    let right = inputs[1].binary()?;
    geo::equals_exact(left, right, kwargs.tolerance)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=String)]
fn relate(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let left = inputs[0].binary()?;
    let right = inputs[1].binary()?;
    geo::relate(left, right)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Boolean)]
fn relate_pattern(inputs: &[Series], kwargs: kwargs::RelatePatternKwargs) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let left = inputs[0].binary()?;
    let right = inputs[1].binary()?;
    geo::relate_pattern(left, right, &kwargs.pattern)
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
    geo::intersects_xy(wkb, x, y)
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
    geo::contains_xy(wkb, x, y)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn difference(inputs: &[Series], kwargs: kwargs::SetOperationKwargs) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let left = inputs[0].binary()?;
    let right = inputs[1].binary()?;
    match kwargs.grid_size {
        Some(grid_size) => geo::difference_prec(left, right, grid_size),
        None => geo::difference(left, right),
    }
    .map_err(to_compute_err)
    .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn difference_all(inputs: &[Series], kwargs: kwargs::SetOperationKwargs) -> PolarsResult<Series> {
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
    .map(|res| Series::new(wkb.name(), [res]))
    .map_err(to_compute_err)
}

#[polars_expr(output_type=Binary)]
fn intersection(inputs: &[Series], kwargs: kwargs::SetOperationKwargs) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let left = inputs[0].binary()?;
    let right = inputs[1].binary()?;
    match kwargs.grid_size {
        Some(grid_size) => geo::intersection_prec(left, right, grid_size),
        None => geo::intersection(left, right),
    }
    .map_err(to_compute_err)
    .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn intersection_all(inputs: &[Series], kwargs: kwargs::SetOperationKwargs) -> PolarsResult<Series> {
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
    .map(|res| Series::new(wkb.name(), [res]))
}

#[polars_expr(output_type=Binary)]
fn symmetric_difference(
    inputs: &[Series],
    kwargs: kwargs::SetOperationKwargs,
) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let left = inputs[0].binary()?;
    let right = inputs[1].binary()?;
    match kwargs.grid_size {
        Some(grid_size) => geo::sym_difference_prec(left, right, grid_size),
        None => geo::sym_difference(left, right),
    }
    .map_err(to_compute_err)
    .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn symmetric_difference_all(
    inputs: &[Series],
    kwargs: kwargs::SetOperationKwargs,
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
    .map(|res| Series::new(wkb.name(), [res]))
}

#[polars_expr(output_type=Binary)]
fn unary_union(inputs: &[Series], kwargs: kwargs::SetOperationKwargs) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let geom = inputs[0].binary()?;
    match kwargs.grid_size {
        Some(grid_size) => geo::unary_union_prec(geom, grid_size),
        None => geo::unary_union(geom),
    }
    .map_err(to_compute_err)
    .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn disjoint_subset_union(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    geo::disjoint_subset_union(inputs[0].binary()?)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn union(inputs: &[Series], kwargs: kwargs::SetOperationKwargs) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let left = inputs[0].binary()?;
    let right = inputs[1].binary()?;
    match kwargs.grid_size {
        Some(grid_size) => geo::union_prec(left, right, grid_size),
        None => geo::union(left, right),
    }
    .map_err(to_compute_err)
    .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn union_all(inputs: &[Series], kwargs: kwargs::SetOperationKwargs) -> PolarsResult<Series> {
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
    .map(|wkb| Series::new(geom.name(), [wkb]))
}

#[polars_expr(output_type=Binary)]
fn coverage_union(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    geo::coverage_union(inputs[0].binary()?)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn coverage_union_all(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    geo::coverage_union_all(inputs[0].binary()?)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn polygonize(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    geo::polygonize(inputs[0].binary()?)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn multipoint(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    geo::multipoint(inputs[0].binary()?)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn multilinestring(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    geo::multilinestring(inputs[0].binary()?)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn multipolygon(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    geo::multipolygon(inputs[0].binary()?)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn geometrycollection(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    geo::geometrycollection(inputs[0].binary()?)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn collect(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    geo::collect(inputs[0].binary()?)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn boundary(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    geo::boundary(inputs[0].binary()?)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn buffer(inputs: &[Series], kwargs: kwargs::BufferKwargs) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let wkb = inputs[0].binary()?;
    let distance = inputs[1].strict_cast(&DataType::Float64)?;
    let distance = distance.f64()?;
    geo::buffer(wkb, distance, &kwargs)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn offset_curve(inputs: &[Series], kwargs: kwargs::OffsetCurveKwargs) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let wkb = inputs[0].binary()?;
    let distance = inputs[1].strict_cast(&DataType::Float64)?;
    let distance = distance.f64()?;
    geo::offset_curve(wkb, distance, &kwargs)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn convex_hull(inputs: &[Series]) -> PolarsResult<Series> {
    let wkb = inputs[0].binary()?;
    geo::convex_hull(wkb)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn concave_hull(inputs: &[Series], kwargs: kwargs::ConcaveHullKwargs) -> PolarsResult<Series> {
    let wkb = inputs[0].binary()?;
    geo::concave_hull(wkb, &kwargs)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn clip_by_rect(inputs: &[Series], kwargs: kwargs::ClipByRectKwargs) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    geo::clip_by_rect(inputs[0].binary()?, &kwargs)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn centroid(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    geo::get_centroid(inputs[0].binary()?)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn center(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    geo::get_center(inputs[0].binary()?)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn delaunay_triangles(
    inputs: &[Series],
    kwargs: kwargs::DelaunayTrianlesKwargs,
) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    geo::delaunay_triangulation(inputs[0].binary()?, &kwargs)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn segmentize(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let wkb = inputs[0].binary()?;
    let tolerance = inputs[1].strict_cast(&DataType::Float64)?;
    let tolerance = tolerance.f64()?;
    geo::densify(wkb, tolerance)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn envelope(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    geo::envelope(inputs[0].binary()?)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn extract_unique_points(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    geo::extract_unique_points(inputs[0].binary()?)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
fn build_area(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    geo::build_area(inputs[0].binary()?)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
pub fn make_valid(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    geo::make_valid(inputs[0].binary()?)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
pub fn normalize(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    geo::normalize(inputs[0].binary()?)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
pub fn node(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    geo::node(inputs[0].binary()?)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
pub fn point_on_surface(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    geo::point_on_surface(inputs[0].binary()?)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
pub fn remove_repeated_points(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let wkb = inputs[0].binary()?;
    let tolerance = inputs[1].strict_cast(&DataType::Float64)?;
    let tolerance = tolerance.f64()?;
    geo::remove_repeated_points(wkb, tolerance)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
pub fn reverse(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = inputs[0].binary()?;
    geo::reverse(wkb)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
pub fn simplify(inputs: &[Series], kwargs: kwargs::SimplifyKwargs) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let wkb = inputs[0].binary()?;
    let tolerance = inputs[1].strict_cast(&DataType::Float64)?;
    let tolerance = tolerance.f64()?;
    match kwargs.preserve_topology {
        true => geo::topology_preserve_simplify(wkb, tolerance),
        false => geo::simplify(wkb, tolerance),
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
    geo::snap(left, right, tolerance)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
pub fn voronoi_polygons(inputs: &[Series], kwargs: kwargs::VoronoiKwargs) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    geo::voronoi_polygons(inputs[0].binary()?, &kwargs)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
pub fn minimum_rotated_rectangle(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    geo::minimum_rotated_rectangle(inputs[0].binary()?)
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
        6 => geo::affine_transform_2d(wkb, matrix.array()?),
        12 => geo::affine_transform_3d(wkb, matrix.array()?),
        _ => unreachable!(),
    }
    .map_err(to_compute_err)
    .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
pub fn interpolate(inputs: &[Series], kwargs: kwargs::InterpolateKwargs) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let wkb = inputs[0].binary()?;
    let distance = inputs[1].strict_cast(&DataType::Float64)?;
    let distance = distance.f64()?;
    match kwargs.normalized {
        true => geo::interpolate_normalized(wkb, distance),
        false => geo::interpolate(wkb, distance),
    }
    .map_err(to_compute_err)
    .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
pub fn line_merge(inputs: &[Series], kwargs: kwargs::LineMergeKwargs) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = inputs[0].binary()?;
    match kwargs.directed {
        true => geo::line_merge_directed(wkb),
        false => geo::line_merge(wkb),
    }
    .map_err(to_compute_err)
    .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
pub fn shared_paths(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let left = inputs[0].binary()?;
    let right = inputs[1].binary()?;
    geo::shared_paths(left, right)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type=Binary)]
pub fn shortest_line(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let left = inputs[0].binary()?;
    let right = inputs[1].binary()?;
    geo::shortest_line(left, right)
        .map_err(to_compute_err)
        .map(IntoSeries::into_series)
}

#[polars_expr(output_type_func=output_type_sjoin)]
pub fn sjoin(inputs: &[Series], kwargs: kwargs::SpatialJoinKwargs) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<2>(inputs)?;
    let left = inputs[0].binary()?;
    let right = inputs[1].binary()?;
    geo::sjoin(left, right, kwargs.predicate)
        .map_err(to_compute_err)
        .map(|(left_index, right_index)| {
            StructChunked::from_series(
                left.name(),
                &[left_index.into_series(), right_index.into_series()],
            )
            .map(IntoSeries::into_series)
        })?
}

#[pyfunction]
pub fn apply_coordinates(py: Python, pyseries: PySeries, pyfunc: PyObject) -> PyResult<PySeries> {
    fn apply_coordinates<F>(inputs: &[Series], func: F) -> PolarsResult<Series>
    where
        F: Fn(Series, Series, Option<Series>) -> PolarsResult<(Series, Series, Option<Series>)>,
    {
        let wkb = inputs[0].binary()?;
        geo::apply_coordinates(wkb, func)
            .map_err(to_compute_err)
            .map(IntoSeries::into_series)
    }

    let res = apply_coordinates(&[pyseries.0], |x, y, z| {
        let (tx, ty, tz): (PySeries, PySeries, Option<PySeries>) = pyfunc
            .call1(py, (PySeries(x), PySeries(y), z.map(PySeries)))
            .unwrap()
            .extract(py)
            .map_err(to_compute_err)?;
        Ok((
            tx.0.cast(&DataType::Float64)?,
            ty.0.cast(&DataType::Float64)?,
            tz.map(|tz| tz.0.cast(&DataType::Float64)).transpose()?,
        ))
    })
    .expect("failed to apply transform");

    Ok(PySeries(res))
}

#[polars_expr(output_type=Binary)]
pub fn to_srid(inputs: &[Series], kwargs: kwargs::ToSridKwargs) -> PolarsResult<Series> {
    let inputs = validate_inputs_length::<1>(inputs)?;
    let wkb = inputs[0].binary()?;

    if wkb.len() == wkb.null_count() {
        return Ok(Series::full_null(wkb.name(), wkb.len(), wkb.dtype()));
    }

    let srids = geo::get_srid(wkb).map_err(to_compute_err)?;
    let unique_srids = srids.unique()?.drop_nulls();

    if unique_srids.len() == 1 {
        return geo::to_srid(wkb, unique_srids.get(0).unwrap(), kwargs.srid)
            .map_err(to_compute_err)
            .map(IntoSeries::into_series);
    }

    let chained_then = unique_srids.into_iter().flatten().fold(
        // Must repeat this twice to get a ChainedThen expression
        when(false).then(NULL.lit()).when(false).then(NULL.lit()),
        |chain, srid| {
            let function = move |s: &mut [Series]| {
                geo::to_srid(s[0].binary()?, srid, kwargs.srid)
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

    Ok(res.get_columns()[0].to_owned())
}
