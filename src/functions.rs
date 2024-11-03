use crate::{
    args::{
        BufferKwargs, ClipByRectKwargs, ConcaveHullKwargs, DelaunayTrianlesKwargs,
        OffsetCurveKwargs, SetPrecisionKwargs, SpatialJoinPredicate, ToGeoJsonKwargs, ToWkbKwargs,
        ToWktKwargs, VoronoiKwargs,
    },
    arity::{
        broadcast_try_binary_elementwise_values, broadcast_try_ternary_elementwise_values,
        try_binary_elementwise_values, try_ternary_elementwise_values,
    },
    wkb::{read_ewkb_header, WKBGeometryType},
};
use geos::{
    BufferParams, CoordSeq, GResult, GeoJSONWriter, Geom, Geometry, GeometryTypes::*,
    PreparedGeometry, STRtree, SpatialIndex, WKBWriter, WKTWriter,
};
use polars::prelude::arity::{broadcast_try_binary_elementwise, try_unary_elementwise};
use polars::prelude::*;
use proj::Proj;
use pyo3_polars::export::polars_core::utils::arrow::array::Float64Array;

fn ewkb_writer() -> GResult<WKBWriter> {
    let mut writer = WKBWriter::new()?;
    writer.set_include_SRID(true);
    Ok(writer)
}

pub trait ToEwkb {
    fn to_ewkb(&self) -> GResult<Vec<u8>>;
}

impl<T> ToEwkb for T
where
    T: Geom,
{
    fn to_ewkb(&self) -> GResult<Vec<u8>> {
        let mut writer = ewkb_writer()?;
        Ok(writer.write_wkb(self)?.into())
    }
}

pub fn from_wkt(wkt: &StringChunked) -> GResult<BinaryChunked> {
    wkt.try_apply_nonnull_values_generic(|wkt| Geometry::new_from_wkt(wkt)?.to_ewkb())
}

pub fn from_geojson(json: &StringChunked) -> GResult<BinaryChunked> {
    json.try_apply_nonnull_values_generic(|json| Geometry::new_from_geojson(json)?.to_ewkb())
}

pub fn from_xy(
    x: &Float64Chunked,
    y: &Float64Chunked,
    z: Option<&Float64Chunked>,
) -> GResult<BinaryChunked> {
    match z {
        Some(z) => try_ternary_elementwise_values(x, y, z, |x, y, z| {
            let seq = CoordSeq::new_from_vec(&[&[x, y, z]])?;
            Geometry::create_point(seq)?.to_ewkb()
        }),
        None => try_binary_elementwise_values(x, y, |x, y| {
            let seq = CoordSeq::new_from_vec(&[&[x, y]])?;
            Geometry::create_point(seq)?.to_ewkb()
        }),
    }
}

pub fn get_type_id(wkb: &BinaryChunked) -> GResult<UInt32Chunked> {
    wkb.try_apply_nonnull_values_generic(|mut wkb| {
        read_ewkb_header(&mut wkb)
            .map_err(|_| geos::Error::InvalidGeometry("Invalid WKB header".into()))
            .map(|header| WKBGeometryType::try_from(header.base_type))?
            .map_err(|e| geos::Error::InvalidGeometry(format!("Invalid geometry type: {e}")))
            .map(u32::from)
    })
}

pub fn get_num_dimensions(wkb: &BinaryChunked) -> GResult<Int32Chunked> {
    wkb.try_apply_nonnull_values_generic(|wkb| Geometry::new_from_wkb(wkb)?.get_num_dimensions())
}

pub fn get_coordinate_dimension(wkb: &BinaryChunked) -> GResult<UInt32Chunked> {
    wkb.try_apply_nonnull_values_generic(|mut wkb| {
        read_ewkb_header(&mut wkb)
            .map_err(|_| geos::Error::InvalidGeometry("Invalid header".into()))
            .map(|header| 2 + u32::from(header.has_z) + u32::from(header.has_m))
    })
}

pub fn get_srid(wkb: &BinaryChunked) -> GResult<Int32Chunked> {
    wkb.try_apply_nonnull_values_generic(|mut wkb| {
        read_ewkb_header(&mut wkb)
            .map_err(|_| geos::Error::InvalidGeometry("Invalid header".into()))
            .map(|header| header.srid)
    })
}

pub fn set_srid(wkb: &BinaryChunked, srid: &Int32Chunked) -> GResult<BinaryChunked> {
    broadcast_try_binary_elementwise_values(wkb, srid, |wkb, srid| {
        let mut geom = Geometry::new_from_wkb(wkb)?;
        geom.set_srid(srid);
        geom.to_ewkb()
    })
}

pub fn get_x(wkb: &BinaryChunked) -> GResult<Float64Chunked> {
    wkb.try_apply_nonnull_values_generic(|wkb| {
        let geom = Geometry::new_from_wkb(wkb)?;
        if geom.geometry_type()? == Point && !geom.is_empty()? {
            geom.get_x()
        } else {
            Ok(f64::NAN)
        }
    })
}

pub fn get_y(wkb: &BinaryChunked) -> GResult<Float64Chunked> {
    wkb.try_apply_nonnull_values_generic(|wkb| {
        let geom = Geometry::new_from_wkb(wkb)?;
        if geom.geometry_type()? == Point && !geom.is_empty()? {
            geom.get_y()
        } else {
            Ok(f64::NAN)
        }
    })
}

pub fn get_z(wkb: &BinaryChunked) -> GResult<Float64Chunked> {
    wkb.try_apply_nonnull_values_generic(|wkb| {
        let geom = Geometry::new_from_wkb(wkb)?;
        if geom.geometry_type()? == Point && !geom.is_empty()? {
            geom.get_z()
        } else {
            Ok(f64::NAN)
        }
    })
}

pub fn get_m(wkb: &BinaryChunked) -> GResult<Float64Chunked> {
    wkb.try_apply_nonnull_values_generic(|wkb| {
        let geom = Geometry::new_from_wkb(wkb)?;
        if geom.geometry_type()? == Point && !geom.is_empty()? {
            geom.get_m()
        } else {
            Ok(f64::NAN)
        }
    })
}

pub fn get_exterior_ring(wkb: &BinaryChunked) -> GResult<BinaryChunked> {
    try_unary_elementwise(wkb, |wkb| {
        if let Some(wkb) = wkb {
            let geom = Geometry::new_from_wkb(wkb)?;
            if geom.geometry_type()? == Polygon {
                return Ok(Some(geom.get_exterior_ring()?.to_ewkb()?));
            }
        }
        Ok(None)
    })
}

pub fn get_interior_rings(wkb: &BinaryChunked) -> GResult<ListChunked> {
    fn get_geometry_rings(wkb: &[u8]) -> GResult<Series> {
        let geom = Geometry::new_from_wkb(wkb)?;
        if geom.geometry_type()? != Polygon {
            return Ok(Series::new_empty("".into(), &DataType::Binary));
        }
        let num_rings = geom.get_num_interior_rings()?;
        let mut rings = BinaryChunkedBuilder::new("".into(), num_rings + 1);
        for n in 0..num_rings {
            let ring = geom.get_interior_ring_n(n as u32)?;
            rings.append_value(ring.to_ewkb()?);
        }
        Ok(rings.finish().into_series())
    }
    wkb.into_iter()
        .map(|wkb| wkb.map(get_geometry_rings).transpose())
        .collect()
}

pub fn get_num_points(wkb: &BinaryChunked) -> GResult<UInt32Chunked> {
    wkb.try_apply_nonnull_values_generic(|wkb| {
        let geom = Geometry::new_from_wkb(wkb)?;
        match geom.geometry_type()? {
            LineString | LinearRing => Ok(geom.get_num_points()? as u32),
            _ => Ok(0),
        }
    })
}

pub fn get_num_interior_rings(wkb: &BinaryChunked) -> GResult<UInt32Chunked> {
    wkb.try_apply_nonnull_values_generic(|wkb| {
        let geom = Geometry::new_from_wkb(wkb)?;
        match geom.geometry_type()? {
            Polygon => Ok(geom.get_num_interior_rings()? as u32),
            _ => Ok(0),
        }
    })
}

pub fn get_num_geometries(wkb: &BinaryChunked) -> GResult<UInt32Chunked> {
    wkb.try_apply_nonnull_values_generic(|wkb| {
        Geometry::new_from_wkb(wkb)?
            .get_num_geometries()
            .map(|n| n as u32)
    })
}

pub fn get_num_coordinates(wkb: &BinaryChunked) -> GResult<UInt32Chunked> {
    wkb.try_apply_nonnull_values_generic(|wkb| {
        Geometry::new_from_wkb(wkb)?
            .get_num_coordinates()
            .map(|n| n as u32)
    })
}

pub fn get_coordinates(wkb_array: &BinaryChunked, dimension: usize) -> GResult<ListChunked> {
    fn get_coordinates(wkb: &[u8], dimension: usize) -> GResult<Series> {
        let geom = Geometry::new_from_wkb(wkb)?;
        if geom.is_empty()? {
            let dt = &DataType::Array(DataType::Float64.into(), dimension);
            return Ok(Series::new_empty("".into(), dt));
        }
        match geom.geometry_type()? {
            Point | LineString | CircularString => {
                let seq = geom.get_coord_seq()?;
                Series::new("".into(), seq.as_buffer(Some(dimension))?)
                    .reshape_array(&[seq.size()? as i64, dimension as _])
                    .map_err(|_| geos::Error::GenericError("Invalid coordinate sequence".into()))
            }
            _ => {
                let n_coords = wkb.len() / 8 * dimension;
                let mut builder = ListPrimitiveChunkedBuilder::<Float64Type>::new(
                    "".into(),
                    n_coords,
                    n_coords * dimension,
                    DataType::Float64,
                );
                match dimension {
                    2 => geom.transform_xy(|x, y| {
                        builder.append_slice(&[*x, *y]);
                        1
                    })?,
                    3 => geom.transform_xyz(|x, y, z| {
                        builder.append_slice(&[*x, *y, *z]);
                        1
                    })?,
                    _ => unreachable!(),
                };
                let dt = &DataType::Array(DataType::Float64.into(), dimension);
                builder
                    .finish()
                    .cast(dt)
                    .map_err(|_| geos::Error::GenericError("Invalid coordinate sequence".into()))
            }
        }
    }
    wkb_array
        .iter()
        .map(|wkb| wkb.map(|wkb| get_coordinates(wkb, dimension)).transpose())
        .collect()
}

pub fn flip_coordinates(wkb: &BinaryChunked) -> GResult<BinaryChunked> {
    wkb.try_apply_nonnull_values_generic(|wkb| {
        Geometry::new_from_wkb(wkb)?
            .transform_xy(|x, y| {
                (*x, *y) = (*y, *x);
                1
            })?
            .to_ewkb()
    })
}

pub fn apply_coordinates<F>(wkb: &BinaryChunked, transform: F) -> GResult<BinaryChunked>
where
    F: Fn(Series, Series, Option<Series>) -> PolarsResult<(Series, Series, Option<Series>)>,
{
    wkb.try_apply_nonnull_values_generic(|wkb| {
        let geom = Geometry::new_from_wkb(wkb)?;
        let coords_len = geom.get_num_coordinates()?;
        let transformed = if 3 > geom.get_coordinate_dimension()?.into() {
            let mut coords_x = PrimitiveChunkedBuilder::<Float64Type>::new("x".into(), coords_len);
            let mut coords_y = PrimitiveChunkedBuilder::<Float64Type>::new("y".into(), coords_len);
            geom.transform_xy(|x, y| {
                coords_x.append_value(*x);
                coords_y.append_value(*y);
                1
            })?;
            transform(
                coords_x.finish().into_series(),
                coords_y.finish().into_series(),
                None,
            )
        } else {
            let mut coords_x = PrimitiveChunkedBuilder::<Float64Type>::new("x".into(), coords_len);
            let mut coords_y = PrimitiveChunkedBuilder::<Float64Type>::new("y".into(), coords_len);
            let mut coords_z = PrimitiveChunkedBuilder::<Float64Type>::new("z".into(), coords_len);
            geom.transform_xyz(|x, y, z| {
                coords_x.append_value(*x);
                coords_y.append_value(*y);
                coords_z.append_value(*z);
                1
            })?;
            transform(
                coords_x.finish().into_series(),
                coords_y.finish().into_series(),
                Some(coords_z.finish().into_series()),
            )
        }
        .map_err(|e| geos::Error::GenericError(format!("{e}")))?;
        let mut transformed_x = transformed.0.f64().unwrap().into_iter();
        let mut transformed_y = transformed.1.f64().unwrap().into_iter();
        if let Some(transformed_z) = transformed.2 {
            let mut transformed_z = transformed_z.f64().unwrap().into_iter();
            geom.transform_xyz(|x, y, z| {
                *x = transformed_x.next().unwrap().unwrap_or(f64::NAN);
                *y = transformed_y.next().unwrap().unwrap_or(f64::NAN);
                *z = transformed_z.next().unwrap().unwrap_or(f64::NAN);
                1
            })?
        } else {
            geom.transform_xy(|x, y| {
                *x = transformed_x.next().unwrap().unwrap_or(f64::NAN);
                *y = transformed_y.next().unwrap().unwrap_or(f64::NAN);
                1
            })?
        }
        .to_ewkb()
    })
}

pub fn get_point_n(wkb: &BinaryChunked, index: &UInt32Chunked) -> GResult<BinaryChunked> {
    broadcast_try_binary_elementwise(wkb, index, |wkb, index| {
        if let (Some(wkb), Some(index)) = (wkb, index) {
            let index = index as usize;
            let geom = Geometry::new_from_wkb(wkb)?;
            let num_points = geom.get_num_points()?;
            if index < num_points {
                return Some(geom.get_point_n(index)?.to_ewkb()).transpose();
            }
        }
        Ok(None)
    })
}

pub fn get_interior_ring_n(wkb: &BinaryChunked, index: &UInt32Chunked) -> GResult<BinaryChunked> {
    broadcast_try_binary_elementwise(wkb, index, |wkb, index| {
        if let (Some(wkb), Some(index)) = (wkb, index) {
            let geom = Geometry::new_from_wkb(wkb)?;
            let num_rings = geom.get_num_interior_rings()? as u32;
            if index < num_rings {
                return Some(geom.get_interior_ring_n(index)?.to_ewkb()).transpose();
            }
        }
        Ok(None)
    })
}

pub fn get_geometry_n(wkb: &BinaryChunked, index: &UInt32Chunked) -> GResult<BinaryChunked> {
    broadcast_try_binary_elementwise(wkb, index, |wkb, index| {
        if let (Some(wkb), Some(index)) = (wkb, index) {
            let index = index as usize;
            let geom = Geometry::new_from_wkb(wkb)?;
            let num_geom = geom.get_num_geometries()?;
            if index < num_geom {
                return Some(geom.get_geometry_n(index)?.to_ewkb()).transpose();
            }
        }
        Ok(None)
    })
}

pub fn get_parts(wkb: &BinaryChunked) -> GResult<ListChunked> {
    fn get_geometry_parts(wkb: &[u8]) -> GResult<Series> {
        let geom = Geometry::new_from_wkb(wkb)?;
        let num_geom = geom.get_num_geometries()?;
        let mut parts = BinaryChunkedBuilder::new("".into(), num_geom);
        for n in 0..num_geom {
            let part = geom.get_geometry_n(n)?;
            parts.append_value(part.to_ewkb()?);
        }
        Ok(parts.finish().into_series())
    }
    wkb.into_iter()
        .map(|wkb| wkb.map(get_geometry_parts).transpose())
        .collect()
}

pub fn get_precision(wkb: &BinaryChunked) -> GResult<Float64Chunked> {
    wkb.try_apply_nonnull_values_generic(|wkb| Geometry::new_from_wkb(wkb)?.get_precision())
}

pub fn set_precision(
    wkb: &BinaryChunked,
    grid_size: &Float64Chunked,
    params: &SetPrecisionKwargs,
) -> GResult<BinaryChunked> {
    broadcast_try_binary_elementwise_values(wkb, grid_size, |wkb, grid_size| {
        Geometry::new_from_wkb(wkb)?
            .set_precision(grid_size, params.mode.into())?
            .to_ewkb()
    })
}

pub fn to_wkt(wkb: &BinaryChunked, params: &ToWktKwargs) -> GResult<StringChunked> {
    let mut writer = WKTWriter::new()?;
    if let Some(rounding_precision) = params.rounding_precision {
        writer.set_rounding_precision(rounding_precision);
    }
    writer.set_old_3D(params.old_3d);
    writer.set_trim(params.trim);
    writer.set_output_dimension(params.output_dimension.try_into()?);
    wkb.try_apply_nonnull_values_generic(|wkb| {
        let geom = Geometry::new_from_wkb(wkb)?;
        writer.write(&geom)
    })
}

pub fn to_ewkt(wkb: &BinaryChunked, params: &ToWktKwargs) -> GResult<StringChunked> {
    let mut writer = WKTWriter::new()?;
    if let Some(rounding_precision) = params.rounding_precision {
        writer.set_rounding_precision(rounding_precision);
    }
    writer.set_old_3D(params.old_3d);
    writer.set_trim(params.trim);
    writer.set_output_dimension(params.output_dimension.try_into()?);
    wkb.try_apply_nonnull_values_generic(|wkb| {
        let geom = Geometry::new_from_wkb(wkb)?;
        match geom.get_srid()? {
            0 => writer.write(&geom),
            srid => writer.write(&geom).map(|s| format!("SRID={srid};{s}")),
        }
    })
}

pub fn to_wkb(wkb: &BinaryChunked, params: &ToWkbKwargs) -> GResult<BinaryChunked> {
    let mut writer = WKBWriter::new()?;
    if let Some(byte_order) = params.byte_order {
        writer.set_wkb_byte_order(byte_order.try_into()?);
    }
    writer.set_include_SRID(params.include_srid);
    writer.set_output_dimension(params.output_dimension.try_into()?);
    wkb.try_apply_nonnull_values_generic(|wkb| {
        let geom = Geometry::new_from_wkb(wkb)?;
        let res: Vec<u8> = writer.write_wkb(&geom)?.into();
        Ok(res)
    })
}

pub fn to_geojson(wkb: &BinaryChunked, params: &ToGeoJsonKwargs) -> GResult<StringChunked> {
    let mut writer = GeoJSONWriter::new()?;
    wkb.try_apply_nonnull_values_generic(|wkb| {
        let geom = Geometry::new_from_wkb(wkb)?;
        writer.write(&geom, params.indent.unwrap_or(-1))
    })
}

pub fn area(wkb: &BinaryChunked) -> GResult<Float64Chunked> {
    wkb.try_apply_nonnull_values_generic(|wkb| Geometry::new_from_wkb(wkb)?.area())
}

pub fn bounds(wkb: &BinaryChunked) -> GResult<ListChunked> {
    fn get_bounds(wkb: &[u8]) -> GResult<Series> {
        let geom = Geometry::new_from_wkb(wkb)?;
        let res = if geom.is_empty()? {
            Series::new("".into(), [f64::NAN, f64::NAN, f64::NAN, f64::NAN])
        } else {
            let x_min = geom.get_x_min()?;
            let y_min = geom.get_y_min()?;
            let x_max = geom.get_x_max()?;
            let y_max = geom.get_y_max()?;
            Series::new("".into(), [x_min, y_min, x_max, y_max])
        };
        Ok(res)
    }
    wkb.iter()
        .map(|wkb| wkb.map(get_bounds).transpose())
        .collect()
}

pub fn length(wkb: &BinaryChunked) -> GResult<Float64Chunked> {
    wkb.try_apply_nonnull_values_generic(|wkb| Geometry::new_from_wkb(wkb)?.length())
}

pub fn distance(a: &BinaryChunked, b: &BinaryChunked) -> GResult<Float64Chunked> {
    broadcast_try_binary_elementwise_values(a, b, |a, b| {
        let a = Geometry::new_from_wkb(a)?;
        let b = Geometry::new_from_wkb(b)?;
        if a.is_empty()? || b.is_empty()? {
            Ok(f64::NAN) // Match `hausdorff_distance` and `frechet_distance` behavior
        } else {
            a.distance(&b)
        }
    })
}

pub fn hausdorff_distance(a: &BinaryChunked, b: &BinaryChunked) -> GResult<Float64Chunked> {
    broadcast_try_binary_elementwise_values(a, b, |a, b| {
        let a = Geometry::new_from_wkb(a)?;
        let b = Geometry::new_from_wkb(b)?;
        a.hausdorff_distance(&b)
    })
}

pub fn hausdorff_distance_densify(
    a: &BinaryChunked,
    b: &BinaryChunked,
    densify: f64,
) -> GResult<Float64Chunked> {
    broadcast_try_binary_elementwise_values(a, b, |a, b| {
        let a = Geometry::new_from_wkb(a)?;
        let b = Geometry::new_from_wkb(b)?;
        a.hausdorff_distance_densify(&b, densify)
    })
}

pub fn frechet_distance(a: &BinaryChunked, b: &BinaryChunked) -> GResult<Float64Chunked> {
    broadcast_try_binary_elementwise_values(a, b, |a, b| {
        let a = Geometry::new_from_wkb(a)?;
        let b = Geometry::new_from_wkb(b)?;
        // TODO: bug report to GEOS
        if a.is_empty()? || b.is_empty()? {
            Ok(f64::NAN)
        } else {
            a.frechet_distance(&b)
        }
    })
}

pub fn frechet_distance_densify(
    a: &BinaryChunked,
    b: &BinaryChunked,
    densify: f64,
) -> GResult<Float64Chunked> {
    broadcast_try_binary_elementwise_values(a, b, |a, b| {
        let a = Geometry::new_from_wkb(a)?;
        let b = Geometry::new_from_wkb(b)?;
        // TODO: bug report to GEOS
        if a.is_empty()? || b.is_empty()? {
            Ok(f64::NAN)
        } else {
            a.frechet_distance_densify(&b, densify)
        }
    })
}

pub fn minimum_clearance(wkb: &BinaryChunked) -> GResult<Float64Chunked> {
    wkb.try_apply_nonnull_values_generic(|wkb| Geometry::new_from_wkb(wkb)?.minimum_clearance())
}

pub fn has_z(wkb: &BinaryChunked) -> GResult<BooleanChunked> {
    wkb.try_apply_nonnull_values_generic(|wkb| Geometry::new_from_wkb(wkb)?.has_z())
}

pub fn has_m(wkb: &BinaryChunked) -> GResult<BooleanChunked> {
    wkb.try_apply_nonnull_values_generic(|wkb| Geometry::new_from_wkb(wkb)?.has_m())
}

pub fn is_ccw(wkb: &BinaryChunked) -> GResult<BooleanChunked> {
    wkb.try_apply_nonnull_values_generic(|wkb| {
        let geom = Geometry::new_from_wkb(wkb)?;
        match geom.geometry_type()? {
            Point | LineString | LinearRing => geom.get_coord_seq()?.is_ccw(),
            _ => Ok(false),
        }
    })
}

pub fn is_closed(wkb: &BinaryChunked) -> GResult<BooleanChunked> {
    wkb.try_apply_nonnull_values_generic(|wkb| {
        let geom = Geometry::new_from_wkb(wkb)?;
        match geom.geometry_type()? {
            LineString | LinearRing | MultiLineString => geom.is_closed(),
            _ => Ok(false),
        }
    })
}

pub fn is_empty(wkb: &BinaryChunked) -> GResult<BooleanChunked> {
    wkb.try_apply_nonnull_values_generic(|wkb| Geometry::new_from_wkb(wkb)?.is_empty())
}

pub fn is_ring(wkb: &BinaryChunked) -> GResult<BooleanChunked> {
    wkb.try_apply_nonnull_values_generic(|wkb| Geometry::new_from_wkb(wkb)?.is_ring())
}

pub fn is_simple(wkb: &BinaryChunked) -> GResult<BooleanChunked> {
    wkb.try_apply_nonnull_values_generic(|wkb| Geometry::new_from_wkb(wkb)?.is_simple())
}

pub fn is_valid(wkb: &BinaryChunked) -> GResult<BooleanChunked> {
    wkb.try_apply_nonnull_values_generic(|wkb| Ok(Geometry::new_from_wkb(wkb)?.is_valid()))
}

pub fn is_valid_reason(wkb: &BinaryChunked) -> GResult<StringChunked> {
    wkb.try_apply_nonnull_values_generic(|wkb| Geometry::new_from_wkb(wkb)?.is_valid_reason())
}

pub fn crosses(a: &BinaryChunked, b: &BinaryChunked) -> GResult<BooleanChunked> {
    broadcast_try_binary_elementwise_values(a, b, |a, b| {
        let a = Geometry::new_from_wkb(a)?;
        let b = Geometry::new_from_wkb(b)?;
        Geometry::crosses(&a, &b)
    })
}

pub fn contains(a: &BinaryChunked, b: &BinaryChunked) -> GResult<BooleanChunked> {
    broadcast_try_binary_elementwise_values(a, b, |a, b| {
        let a = Geometry::new_from_wkb(a)?;
        let b = Geometry::new_from_wkb(b)?;
        Geometry::contains(&a, &b)
    })
}

pub fn contains_properly(a: &BinaryChunked, b: &BinaryChunked) -> GResult<BooleanChunked> {
    broadcast_try_binary_elementwise_values(a, b, |a, b| {
        let a = Geometry::new_from_wkb(a)?;
        let b = Geometry::new_from_wkb(b)?;
        a.to_prepared_geom()?.contains_properly(&b)
    })
}

pub fn covered_by(a: &BinaryChunked, b: &BinaryChunked) -> GResult<BooleanChunked> {
    broadcast_try_binary_elementwise_values(a, b, |a, b| {
        let a = Geometry::new_from_wkb(a)?;
        let b = Geometry::new_from_wkb(b)?;
        Geometry::covered_by(&a, &b)
    })
}

pub fn covers(a: &BinaryChunked, b: &BinaryChunked) -> GResult<BooleanChunked> {
    broadcast_try_binary_elementwise_values(a, b, |a, b| {
        let a = Geometry::new_from_wkb(a)?;
        let b = Geometry::new_from_wkb(b)?;
        Geometry::covers(&a, &b)
    })
}

pub fn disjoint(a: &BinaryChunked, b: &BinaryChunked) -> GResult<BooleanChunked> {
    broadcast_try_binary_elementwise_values(a, b, |a, b| {
        let a = Geometry::new_from_wkb(a)?;
        let b = Geometry::new_from_wkb(b)?;
        Geometry::disjoint(&a, &b)
    })
}

pub fn dwithin(a: &BinaryChunked, b: &BinaryChunked, distance: f64) -> GResult<BooleanChunked> {
    broadcast_try_binary_elementwise_values(a, b, |a, b| {
        let a = Geometry::new_from_wkb(a)?;
        let b = Geometry::new_from_wkb(b)?;
        Geometry::distance(&a, &b).map(|d| d < distance)
    })
}

pub fn intersects(a: &BinaryChunked, b: &BinaryChunked) -> GResult<BooleanChunked> {
    broadcast_try_binary_elementwise_values(a, b, |a, b| {
        let a = Geometry::new_from_wkb(a)?;
        let b = Geometry::new_from_wkb(b)?;
        Geometry::intersects(&a, &b)
    })
}

pub fn overlaps(a: &BinaryChunked, b: &BinaryChunked) -> GResult<BooleanChunked> {
    broadcast_try_binary_elementwise_values(a, b, |a, b| {
        let a = Geometry::new_from_wkb(a)?;
        let b = Geometry::new_from_wkb(b)?;
        Geometry::overlaps(&a, &b)
    })
}

pub fn touches(a: &BinaryChunked, b: &BinaryChunked) -> GResult<BooleanChunked> {
    broadcast_try_binary_elementwise_values(a, b, |a, b| {
        let a = Geometry::new_from_wkb(a)?;
        let b = Geometry::new_from_wkb(b)?;
        Geometry::touches(&a, &b)
    })
}

pub fn within(a: &BinaryChunked, b: &BinaryChunked) -> GResult<BooleanChunked> {
    broadcast_try_binary_elementwise_values(a, b, |a, b| {
        let a = Geometry::new_from_wkb(a)?;
        let b = Geometry::new_from_wkb(b)?;
        Geometry::within(&a, &b)
    })
}

pub fn equals(a: &BinaryChunked, b: &BinaryChunked) -> GResult<BooleanChunked> {
    broadcast_try_binary_elementwise_values(a, b, |a, b| {
        let a = Geometry::new_from_wkb(a)?;
        let b = Geometry::new_from_wkb(b)?;
        Geometry::equals(&a, &b)
    })
}

pub fn equals_identical(a: &BinaryChunked, b: &BinaryChunked) -> GResult<BooleanChunked> {
    broadcast_try_binary_elementwise_values(a, b, |a, b| {
        let a = Geometry::new_from_wkb(a)?;
        let b = Geometry::new_from_wkb(b)?;
        Geometry::equals_identical(&a, &b)
    })
}

pub fn equals_exact(
    a: &BinaryChunked,
    b: &BinaryChunked,
    tolerance: f64,
) -> GResult<BooleanChunked> {
    broadcast_try_binary_elementwise_values(a, b, |a, b| {
        let a = Geometry::new_from_wkb(a)?;
        let b = Geometry::new_from_wkb(b)?;
        Geometry::equals_exact(&a, &b, tolerance)
    })
}

pub fn relate(a: &BinaryChunked, b: &BinaryChunked) -> GResult<StringChunked> {
    broadcast_try_binary_elementwise_values(a, b, |a, b| {
        let a = Geometry::new_from_wkb(a)?;
        let b = Geometry::new_from_wkb(b)?;
        Geometry::relate(&a, &b)
    })
}

pub fn relate_pattern(
    a: &BinaryChunked,
    b: &BinaryChunked,
    pattern: &str,
) -> GResult<BooleanChunked> {
    broadcast_try_binary_elementwise_values(a, b, |a, b| {
        let a = Geometry::new_from_wkb(a)?;
        let b = Geometry::new_from_wkb(b)?;
        Geometry::relate_pattern(&a, &b, pattern)
    })
}

pub fn intersects_xy(
    wkb: &BinaryChunked,
    x: &Float64Chunked,
    y: &Float64Chunked,
) -> GResult<BooleanChunked> {
    try_ternary_elementwise_values(wkb, x, y, |wkb, x, y| {
        Geometry::new_from_wkb(wkb)?
            .to_prepared_geom()?
            .intersects_xy(x, y)
    })
}

pub fn contains_xy(
    wkb: &BinaryChunked,
    x: &Float64Chunked,
    y: &Float64Chunked,
) -> GResult<BooleanChunked> {
    try_ternary_elementwise_values(wkb, x, y, |wkb, x, y| {
        Geometry::new_from_wkb(wkb)?
            .to_prepared_geom()?
            .contains_xy(x, y)
    })
}

pub fn difference(a: &BinaryChunked, b: &BinaryChunked) -> GResult<BinaryChunked> {
    broadcast_try_binary_elementwise_values(a, b, |a, b| {
        let a = Geometry::new_from_wkb(a)?;
        let b = Geometry::new_from_wkb(b)?;
        Geometry::difference(&a, &b)?.to_ewkb()
    })
}

pub fn difference_prec(
    a: &BinaryChunked,
    b: &BinaryChunked,
    grid_size: f64,
) -> GResult<BinaryChunked> {
    broadcast_try_binary_elementwise_values(a, b, |a, b| {
        let a = Geometry::new_from_wkb(a)?;
        let b = Geometry::new_from_wkb(b)?;
        Geometry::difference_prec(&a, &b, grid_size)?.to_ewkb()
    })
}

pub fn intersection(a: &BinaryChunked, b: &BinaryChunked) -> GResult<BinaryChunked> {
    broadcast_try_binary_elementwise_values(a, b, |a, b| {
        let a = Geometry::new_from_wkb(a)?;
        let b = Geometry::new_from_wkb(b)?;
        Geometry::intersection(&a, &b)?.to_ewkb()
    })
}

pub fn intersection_prec(
    a: &BinaryChunked,
    b: &BinaryChunked,
    grid_size: f64,
) -> GResult<BinaryChunked> {
    broadcast_try_binary_elementwise_values(a, b, |a, b| {
        let a = Geometry::new_from_wkb(a)?;
        let b = Geometry::new_from_wkb(b)?;
        Geometry::intersection_prec(&a, &b, grid_size)?.to_ewkb()
    })
}

pub fn sym_difference(a: &BinaryChunked, b: &BinaryChunked) -> GResult<BinaryChunked> {
    broadcast_try_binary_elementwise_values(a, b, |a, b| {
        let a = Geometry::new_from_wkb(a)?;
        let b = Geometry::new_from_wkb(b)?;
        Geometry::sym_difference(&a, &b)?.to_ewkb()
    })
}

pub fn sym_difference_prec(
    a: &BinaryChunked,
    b: &BinaryChunked,
    grid_size: f64,
) -> GResult<BinaryChunked> {
    broadcast_try_binary_elementwise_values(a, b, |a, b| {
        let a = Geometry::new_from_wkb(a)?;
        let b = Geometry::new_from_wkb(b)?;
        Geometry::sym_difference_prec(&a, &b, grid_size)?.to_ewkb()
    })
}

pub fn unary_union(wkb: &BinaryChunked) -> GResult<BinaryChunked> {
    wkb.try_apply_nonnull_values_generic(|wkb| {
        Geometry::new_from_wkb(wkb)?.unary_union()?.to_ewkb()
    })
}

pub fn unary_union_prec(wkb: &BinaryChunked, grid_size: f64) -> GResult<BinaryChunked> {
    wkb.try_apply_nonnull_values_generic(|wkb| {
        Geometry::new_from_wkb(wkb)?
            .unary_union_prec(grid_size)?
            .to_ewkb()
    })
}

pub fn disjoint_subset_union(wkb: &BinaryChunked) -> GResult<BinaryChunked> {
    wkb.try_apply_nonnull_values_generic(|wkb| {
        Geometry::new_from_wkb(wkb)?
            .disjoint_subset_union()?
            .to_ewkb()
    })
}

pub fn union(a: &BinaryChunked, b: &BinaryChunked) -> GResult<BinaryChunked> {
    broadcast_try_binary_elementwise_values(a, b, |a, b| {
        let a = Geometry::new_from_wkb(a)?;
        let b = Geometry::new_from_wkb(b)?;
        Geometry::union(&a, &b)?.to_ewkb()
    })
}

pub fn union_prec(a: &BinaryChunked, b: &BinaryChunked, grid_size: f64) -> GResult<BinaryChunked> {
    broadcast_try_binary_elementwise_values(a, b, |a, b| {
        let a = Geometry::new_from_wkb(a)?;
        let b = Geometry::new_from_wkb(b)?;
        Geometry::union_prec(&a, &b, grid_size)?.to_ewkb()
    })
}

pub fn coverage_union(wkb: &BinaryChunked) -> GResult<BinaryChunked> {
    wkb.try_apply_nonnull_values_generic(|wkb| {
        let geom = Geometry::new_from_wkb(wkb)?;
        if geom.geometry_type()?.is_collection() {
            geom.coverage_union()?.to_ewkb()
        } else {
            let msg = "Geometry must be a collection";
            Err(geos::Error::GenericError(msg.into()))
        }
    })
}

fn collect_geometry_vec(wkb: &BinaryChunked) -> GResult<Vec<Geometry>> {
    wkb.into_iter()
        .flatten()
        .map(Geometry::new_from_wkb)
        .collect()
}

pub fn coverage_union_all(wkb: &BinaryChunked) -> GResult<BinaryChunked> {
    collect_geometry_vec(wkb)
        .and_then(Geometry::create_geometry_collection)
        .and_then(|geom| geom.coverage_union())
        .and_then(|geom| geom.to_ewkb())
        .map(|res| BinaryChunked::from_slice(wkb.name().clone(), &[res]))
}

pub fn polygonize(wkb: &BinaryChunked) -> GResult<BinaryChunked> {
    collect_geometry_vec(wkb)
        .and_then(|vec| Geometry::polygonize(&vec))
        .and_then(|geom| geom.to_ewkb())
        .map(|res| BinaryChunked::from_slice(wkb.name().clone(), &[res]))
}

fn aggregate_with<F>(wkb: &BinaryChunked, func: F) -> GResult<BinaryChunked>
where
    F: FnOnce(Vec<Geometry>) -> GResult<Geometry>,
{
    collect_geometry_vec(wkb)
        .and_then(func)
        .and_then(|geom| geom.to_ewkb())
        .map(|res| BinaryChunked::from_slice(wkb.name().clone(), &[res]))
}

pub fn multipoint(wkb: &BinaryChunked) -> GResult<BinaryChunked> {
    aggregate_with(wkb, Geometry::create_multipoint)
}

pub fn multilinestring(wkb: &BinaryChunked) -> GResult<BinaryChunked> {
    aggregate_with(wkb, Geometry::create_multiline_string)
}

pub fn multipolygon(wkb: &BinaryChunked) -> GResult<BinaryChunked> {
    aggregate_with(wkb, Geometry::create_multipolygon)
}

pub fn geometrycollection(wkb: &BinaryChunked) -> GResult<BinaryChunked> {
    aggregate_with(wkb, Geometry::create_geometry_collection)
}

pub fn collect(wkb: &BinaryChunked) -> GResult<BinaryChunked> {
    let geometry_types = get_type_id(wkb)?
        .unique()
        .map_err(|_| geos::Error::GenericError("Couldn't get geometry types".into()))?;
    match geometry_types.len() {
        1 => match geometry_types.get(0) {
            Some(t) if t == WKBGeometryType::Point as u32 => multipoint(wkb),
            Some(t) if t == WKBGeometryType::LineString as u32 => multilinestring(wkb),
            Some(t) if t == WKBGeometryType::Polygon as u32 => multipolygon(wkb),
            _ => geometrycollection(wkb),
        },
        _ => geometrycollection(wkb),
    }
}

pub fn boundary(wkb: &BinaryChunked) -> GResult<BinaryChunked> {
    wkb.try_apply_nonnull_values_generic(|wkb| {
        let geom = Geometry::new_from_wkb(wkb)?;
        match geom.geometry_type()? {
            GeometryCollection => Geometry::create_empty_collection(GeometryCollection),
            _ => geom.boundary(),
        }?
        .to_ewkb()
    })
}

pub fn buffer(
    wkb: &BinaryChunked,
    distance: &Float64Chunked,
    params: &BufferKwargs,
) -> GResult<BinaryChunked> {
    let buffer_params: BufferParams = params.try_into()?;
    broadcast_try_binary_elementwise_values(wkb, distance, |wkb, distance| {
        Geometry::new_from_wkb(wkb)?
            .buffer_with_params(distance, &buffer_params)?
            .to_ewkb()
    })
}

pub fn offset_curve(
    wkb: &BinaryChunked,
    distance: &Float64Chunked,
    params: &OffsetCurveKwargs,
) -> GResult<BinaryChunked> {
    broadcast_try_binary_elementwise_values(wkb, distance, |wkb, distance| {
        Geometry::new_from_wkb(wkb)?
            .offset_curve(
                distance,
                params.quad_segs,
                params.join_style.into(),
                params.mitre_limit,
            )?
            .to_ewkb()
    })
}

pub fn get_centroid(wkb: &BinaryChunked) -> GResult<BinaryChunked> {
    wkb.try_apply_nonnull_values_generic(|wkb| {
        Geometry::new_from_wkb(wkb)?.get_centroid()?.to_ewkb()
    })
}

pub fn get_center(wkb: &BinaryChunked) -> GResult<BinaryChunked> {
    wkb.try_apply_nonnull_values_generic(|wkb| {
        let geom = Geometry::new_from_wkb(wkb)?;
        if geom.is_empty()? {
            return Geometry::create_empty_point()?.to_ewkb();
        };
        let x_min = geom.get_x_min()?;
        let x_max = geom.get_x_max()?;
        let y_min = geom.get_y_min()?;
        let y_max = geom.get_y_max()?;
        let coords = CoordSeq::new_from_vec(&[&[(x_min + x_max) / 2.0, (y_min + y_max) / 2.0]])?;
        Geometry::create_point(coords)?.to_ewkb()
    })
}

pub fn clip_by_rect(wkb: &BinaryChunked, params: &ClipByRectKwargs) -> GResult<BinaryChunked> {
    wkb.try_apply_nonnull_values_generic(|wkb| {
        Geometry::new_from_wkb(wkb)?
            .clip_by_rect(params.xmin, params.ymin, params.xmax, params.ymax)?
            .to_ewkb()
    })
}

pub fn convex_hull(wkb: &BinaryChunked) -> GResult<BinaryChunked> {
    wkb.try_apply_nonnull_values_generic(|wkb| {
        Geometry::new_from_wkb(wkb)?.convex_hull()?.to_ewkb()
    })
}

pub fn concave_hull(wkb: &BinaryChunked, params: &ConcaveHullKwargs) -> GResult<BinaryChunked> {
    wkb.try_apply_nonnull_values_generic(|wkb| {
        Geometry::new_from_wkb(wkb)?
            .concave_hull(params.ratio, params.allow_holes)?
            .to_ewkb()
    })
}

pub fn delaunay_triangulation(
    wkb: &BinaryChunked,
    params: &DelaunayTrianlesKwargs,
) -> GResult<BinaryChunked> {
    collect_geometry_vec(wkb)
        .and_then(Geometry::create_geometry_collection)
        .and_then(|geom| geom.delaunay_triangulation(params.tolerance, params.only_edges))
        .and_then(|geom| geom.to_ewkb())
        .map(|res| BinaryChunked::from_slice(wkb.name().clone(), &[res]))
}

pub fn densify(wkb: &BinaryChunked, tolerance: &Float64Chunked) -> GResult<BinaryChunked> {
    broadcast_try_binary_elementwise_values(wkb, tolerance, |wkb, tolerance| {
        Geometry::new_from_wkb(wkb)?.densify(tolerance)?.to_ewkb()
    })
}

pub fn envelope(wkb: &BinaryChunked) -> GResult<BinaryChunked> {
    wkb.try_apply_nonnull_values_generic(|wkb| Geometry::new_from_wkb(wkb)?.envelope()?.to_ewkb())
}

pub fn extract_unique_points(wkb: &BinaryChunked) -> GResult<BinaryChunked> {
    wkb.try_apply_nonnull_values_generic(|wkb| {
        Geometry::new_from_wkb(wkb)?
            .extract_unique_points()?
            .to_ewkb()
    })
}

pub fn build_area(wkb: &BinaryChunked) -> GResult<BinaryChunked> {
    wkb.try_apply_nonnull_values_generic(|wkb| Geometry::new_from_wkb(wkb)?.build_area()?.to_ewkb())
}

pub fn make_valid(wkb: &BinaryChunked) -> GResult<BinaryChunked> {
    wkb.try_apply_nonnull_values_generic(|wkb| Geometry::new_from_wkb(wkb)?.make_valid()?.to_ewkb())
}

pub fn normalize(wkb: &BinaryChunked) -> GResult<BinaryChunked> {
    wkb.try_apply_nonnull_values_generic(|wkb| {
        let mut geom = Geometry::new_from_wkb(wkb)?;
        geom.normalize()?;
        geom.to_ewkb()
    })
}

pub fn node(wkb: &BinaryChunked) -> GResult<BinaryChunked> {
    wkb.try_apply_nonnull_values_generic(|wkb| Geometry::new_from_wkb(wkb)?.node()?.to_ewkb())
}

pub fn point_on_surface(wkb: &BinaryChunked) -> GResult<BinaryChunked> {
    wkb.try_apply_nonnull_values_generic(|wkb| {
        Geometry::new_from_wkb(wkb)?.point_on_surface()?.to_ewkb()
    })
}

pub fn remove_repeated_points(
    wkb: &BinaryChunked,
    tolerance: &Float64Chunked,
) -> GResult<BinaryChunked> {
    broadcast_try_binary_elementwise_values(wkb, tolerance, |wkb, tolerance| {
        Geometry::new_from_wkb(wkb)?
            .remove_repeated_points(tolerance)?
            .to_ewkb()
    })
}

pub fn reverse(wkb: &BinaryChunked) -> GResult<BinaryChunked> {
    wkb.try_apply_nonnull_values_generic(|wkb| Geometry::new_from_wkb(wkb)?.reverse()?.to_ewkb())
}

pub fn simplify(wkb: &BinaryChunked, tolerance: &Float64Chunked) -> GResult<BinaryChunked> {
    broadcast_try_binary_elementwise_values(wkb, tolerance, |wkb, tolerance| {
        Geometry::new_from_wkb(wkb)?.simplify(tolerance)?.to_ewkb()
    })
}

pub fn topology_preserve_simplify(
    wkb: &BinaryChunked,
    tolerance: &Float64Chunked,
) -> GResult<BinaryChunked> {
    broadcast_try_binary_elementwise_values(wkb, tolerance, |wkb, tolerance| {
        Geometry::new_from_wkb(wkb)?
            .topology_preserve_simplify(tolerance)?
            .to_ewkb()
    })
}

pub fn minimum_rotated_rectangle(wkb: &BinaryChunked) -> GResult<BinaryChunked> {
    wkb.try_apply_nonnull_values_generic(|wkb| {
        Geometry::new_from_wkb(wkb)?
            .minimum_rotated_rectangle()?
            .to_ewkb()
    })
}

#[allow(clippy::too_many_arguments)]
fn apply_affine_transform(
    geom: &Geometry,
    m11: f64,
    m12: f64,
    m13: f64,
    m21: f64,
    m22: f64,
    m23: f64,
    m31: f64,
    m32: f64,
    m33: f64,
    tx: f64,
    ty: f64,
    tz: f64,
) -> GResult<Geometry> {
    let dims: i32 = geom.get_coordinate_dimension()?.into();
    if dims < 3 {
        geom.transform_xy(|x, y| {
            *x = *x * m11 + *y * m12 + tx;
            *y = *x * m21 + *y * m22 + ty;
            1
        })
    } else {
        geom.transform_xyz(|x, y, z| {
            *x = *x * m11 + *y * m12 + m13 * *z + tx;
            *y = *x * m21 + *y * m22 + m23 * *z + ty;
            *z = *x * m31 + *y * m32 + m33 * *z + tz;
            1
        })
    }
}

pub fn affine_transform_2d(wkb: &BinaryChunked, matrix: &ArrayChunked) -> GResult<BinaryChunked> {
    broadcast_try_binary_elementwise_values(wkb, matrix, |wkb, matrix| {
        let matrix = unsafe { matrix.as_any().downcast_ref_unchecked::<Float64Array>() };
        apply_affine_transform(
            &Geometry::new_from_wkb(wkb)?,
            unsafe { matrix.get_unchecked(0) }.unwrap_or(f64::NAN),
            unsafe { matrix.get_unchecked(1) }.unwrap_or(f64::NAN),
            0.0,
            unsafe { matrix.get_unchecked(2) }.unwrap_or(f64::NAN),
            unsafe { matrix.get_unchecked(3) }.unwrap_or(f64::NAN),
            0.0,
            0.0,
            0.0,
            1.0,
            unsafe { matrix.get_unchecked(4) }.unwrap_or(f64::NAN),
            unsafe { matrix.get_unchecked(5) }.unwrap_or(f64::NAN),
            0.0,
        )?
        .to_ewkb()
    })
}

pub fn affine_transform_3d(wkb: &BinaryChunked, matrix: &ArrayChunked) -> GResult<BinaryChunked> {
    broadcast_try_binary_elementwise_values(wkb, matrix, |wkb, matrix| {
        let matrix = unsafe { matrix.as_any().downcast_ref_unchecked::<Float64Array>() };
        apply_affine_transform(
            &Geometry::new_from_wkb(wkb)?,
            unsafe { matrix.get_unchecked(0) }.unwrap_or(f64::NAN),
            unsafe { matrix.get_unchecked(1) }.unwrap_or(f64::NAN),
            unsafe { matrix.get_unchecked(2) }.unwrap_or(f64::NAN),
            unsafe { matrix.get_unchecked(3) }.unwrap_or(f64::NAN),
            unsafe { matrix.get_unchecked(4) }.unwrap_or(f64::NAN),
            unsafe { matrix.get_unchecked(5) }.unwrap_or(f64::NAN),
            unsafe { matrix.get_unchecked(6) }.unwrap_or(f64::NAN),
            unsafe { matrix.get_unchecked(7) }.unwrap_or(f64::NAN),
            unsafe { matrix.get_unchecked(8) }.unwrap_or(f64::NAN),
            unsafe { matrix.get_unchecked(9) }.unwrap_or(f64::NAN),
            unsafe { matrix.get_unchecked(10) }.unwrap_or(f64::NAN),
            unsafe { matrix.get_unchecked(11) }.unwrap_or(f64::NAN),
        )?
        .to_ewkb()
    })
}

pub fn interpolate(wkb: &BinaryChunked, distance: &Float64Chunked) -> GResult<BinaryChunked> {
    broadcast_try_binary_elementwise_values(wkb, distance, |wkb, distance| {
        Geometry::new_from_wkb(wkb)?
            .interpolate(distance)?
            .to_ewkb()
    })
}

pub fn interpolate_normalized(
    wkb: &BinaryChunked,
    distance: &Float64Chunked,
) -> GResult<BinaryChunked> {
    broadcast_try_binary_elementwise_values(wkb, distance, |wkb, distance| {
        Geometry::new_from_wkb(wkb)?
            .interpolate_normalized(distance)?
            .to_ewkb()
    })
}

pub fn line_merge(wkb: &BinaryChunked) -> GResult<BinaryChunked> {
    wkb.try_apply_nonnull_values_generic(|wkb| Geometry::new_from_wkb(wkb)?.line_merge()?.to_ewkb())
}

pub fn line_merge_directed(wkb: &BinaryChunked) -> GResult<BinaryChunked> {
    wkb.try_apply_nonnull_values_generic(|wkb| {
        Geometry::new_from_wkb(wkb)?
            .line_merge_directed()?
            .to_ewkb()
    })
}

pub fn shared_paths(a: &BinaryChunked, b: &BinaryChunked) -> GResult<BinaryChunked> {
    broadcast_try_binary_elementwise_values(a, b, |a, b| {
        let a = Geometry::new_from_wkb(a)?;
        let b = Geometry::new_from_wkb(b)?;
        a.shared_paths(&b)?.to_ewkb()
    })
}

pub fn shortest_line(a: &BinaryChunked, b: &BinaryChunked) -> GResult<BinaryChunked> {
    broadcast_try_binary_elementwise_values(a, b, |a, b| {
        let a = Geometry::new_from_wkb(a)?;
        let b = Geometry::new_from_wkb(b)?;
        if a.is_empty()? || b.is_empty()? {
            Geometry::create_empty_line_string()?
        } else {
            let seq = a.nearest_points(&b)?;
            Geometry::create_line_string(seq)?
        }
        .to_ewkb()
    })
}

pub fn snap(
    a: &BinaryChunked,
    b: &BinaryChunked,
    tolerance: &Float64Chunked,
) -> GResult<BinaryChunked> {
    broadcast_try_ternary_elementwise_values(a, b, tolerance, |a, b, tolerance| {
        let a = Geometry::new_from_wkb(a)?;
        let b = Geometry::new_from_wkb(b)?;
        Geometry::snap(&a, &b, tolerance)?.to_ewkb()
    })
}

pub fn voronoi_polygons(wkb: &BinaryChunked, params: &VoronoiKwargs) -> GResult<BinaryChunked> {
    let extend_to = params
        .extend_to
        .as_ref()
        .map(|wkb| Geometry::new_from_wkb(wkb))
        .transpose()?;
    collect_geometry_vec(wkb)
        .and_then(Geometry::create_geometry_collection)
        .and_then(|geom| geom.voronoi(extend_to.as_ref(), params.tolerance, params.only_edges))
        .and_then(|geom| geom.to_ewkb())
        .map(|res| BinaryChunked::from_slice(wkb.name().clone(), &[res]))
}

fn strtree(geoms: &[Option<Geometry>]) -> GResult<STRtree<usize>> {
    let length = geoms.len();
    geoms.iter().enumerate().try_fold(
        STRtree::<usize>::with_capacity(length)?,
        |mut tree, (index, geom)| {
            if let Some(geom) = geom {
                tree.insert(geom, index);
            }
            Ok(tree)
        },
    )
}

pub fn sjoin(
    left: &BinaryChunked,
    right: &BinaryChunked,
    predicate: SpatialJoinPredicate,
) -> GResult<(UInt32Chunked, UInt32Chunked)> {
    let predicate = match predicate {
        SpatialJoinPredicate::IntersectsBbox => |_: &_, _: &_| Ok(true),
        SpatialJoinPredicate::Intersects => PreparedGeometry::intersects,
        SpatialJoinPredicate::Within => PreparedGeometry::within,
        SpatialJoinPredicate::Contains => PreparedGeometry::contains,
        SpatialJoinPredicate::Overlaps => PreparedGeometry::overlaps,
        SpatialJoinPredicate::Crosses => PreparedGeometry::crosses,
        SpatialJoinPredicate::Touches => PreparedGeometry::touches,
        SpatialJoinPredicate::Covers => PreparedGeometry::covers,
        SpatialJoinPredicate::CoveredBy => PreparedGeometry::covered_by,
        SpatialJoinPredicate::ContainsProperly => PreparedGeometry::contains_properly,
    };
    let left_geoms = left
        .into_iter()
        .map(|v| v.map(Geometry::new_from_wkb).transpose())
        .collect::<GResult<Vec<_>>>()?;
    let spatial_index = strtree(&left_geoms)?;
    let left_geoms = left_geoms
        .iter()
        .map(|v| v.as_ref().map(Geom::to_prepared_geom).transpose())
        .collect::<GResult<Vec<_>>>()?;
    let mut left_index_builder = PrimitiveChunkedBuilder::<UInt32Type>::new(
        "left_index".into(),
        core::cmp::max(left.len(), right.len()),
    );
    let mut right_index_builder = PrimitiveChunkedBuilder::<UInt32Type>::new(
        "right_index".into(),
        core::cmp::max(left.len(), right.len()),
    );

    for (right_index, wkb) in right.into_iter().enumerate() {
        if wkb.is_none() {
            continue;
        };
        let right_geom = Geometry::new_from_wkb(wkb.unwrap())?;
        spatial_index.query(&right_geom, |left_index| {
            let left_geom = left_geoms[*left_index]
                .as_ref()
                .expect("Shouldn't be able to match None");
            if matches!(predicate(left_geom, &right_geom), Ok(true)) {
                left_index_builder.append_value(*left_index as u32);
                right_index_builder.append_value(right_index as u32);
            }
        });
    }
    Ok((left_index_builder.finish(), right_index_builder.finish()))
}

fn apply_proj_transformation(geometry: Geometry, transformation: &Proj) -> GResult<Geometry> {
    if geometry.is_empty()? {
        return Ok(geometry);
    }
    if 3 > geometry.get_coordinate_dimension()?.into() {
        geometry.transform_xy(|x, y| match transformation.convert((*x, *y)) {
            Ok(projected) => {
                *x = projected.0;
                *y = projected.1;
                1
            }
            Err(_) => 0,
        })
    } else {
        geometry.transform_xyz(|x, y, z| match transformation.convert((*x, *y, *z)) {
            Ok(projected) => {
                *x = projected.0;
                *y = projected.1;
                *z = projected.2;
                1
            }
            Err(_) => 0,
        })
    }
}

pub fn to_srid(wkb: &BinaryChunked, from_srid: i32, to_srid: i32) -> GResult<BinaryChunked> {
    let from_crs = format!("EPSG:{from_srid}");
    let to_crs = format!("EPSG:{to_srid}");
    let transformation = Proj::try_from((from_crs.as_str(), to_crs.as_str())).map_err(|_| {
        let err = format!("Couldn't create transformation from {from_crs} to {to_crs}");
        geos::Error::GenericError(err)
    })?;

    wkb.try_apply_nonnull_values_generic(|wkb| {
        Geometry::new_from_wkb(wkb)
            .and_then(|geom| apply_proj_transformation(geom, &transformation))
            .map(|mut geom| {
                geom.set_srid(to_srid);
                geom
            })?
            .to_ewkb()
    })
}
