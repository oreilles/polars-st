use geos::GeometryTypes;
use num_enum::IntoPrimitive;
use num_enum::TryFromPrimitive;
use scroll::{Endian, IOread};
use serde::Deserialize;
use serde::Serialize;
use std::io::{Error, Read};

pub struct WkbInfo {
    pub base_type: u32,
    pub has_z: bool,
    pub has_m: bool,
    pub srid: i32,
}

pub fn read_ewkb_header<R: Read>(raw: &mut R) -> Result<WkbInfo, Error> {
    let byte_order = raw.ioread::<u8>()?;
    let is_little_endian = byte_order != 0;
    let endian = Endian::from(is_little_endian);
    let type_id = raw.ioread_with::<u32>(endian)?;
    let srid = if type_id & 0x2000_0000 == 0x2000_0000 {
        raw.ioread_with::<i32>(endian)?
    } else {
        0
    };

    let info = WkbInfo {
        base_type: type_id & 0xFF,
        has_z: type_id & 0x8000_0000 == 0x8000_0000,
        has_m: type_id & 0x4000_0000 == 0x4000_0000,
        srid,
    };
    Ok(info)
}

#[derive(Clone, Copy, Debug, IntoPrimitive, TryFromPrimitive, Serialize, Deserialize)]
#[repr(u32)]
pub enum WKBGeometryType {
    Unknown = 0,
    Point = 1,
    LineString = 2,
    Polygon = 3,
    MultiPoint = 4,
    MultiLineString = 5,
    MultiPolygon = 6,
    GeometryCollection = 7,
    CircularString = 8,
    CompoundCurve = 9,
    CurvePolygon = 10,
    MultiCurve = 11,
    MultiSurface = 12,
    Curve = 13,
    Surface = 14,
    PolyhedralSurface = 15,
    Tin = 16,
    Triangle = 17,
}

impl TryInto<GeometryTypes> for WKBGeometryType {
    type Error = geos::Error;

    fn try_into(self) -> Result<GeometryTypes, Self::Error> {
        match self {
            Self::Point => Ok(GeometryTypes::Point),
            Self::LineString => Ok(GeometryTypes::LineString),
            Self::Polygon => Ok(GeometryTypes::Polygon),
            Self::MultiPoint => Ok(GeometryTypes::MultiPoint),
            Self::MultiLineString => Ok(GeometryTypes::MultiLineString),
            Self::MultiPolygon => Ok(GeometryTypes::MultiPolygon),
            Self::GeometryCollection => Ok(GeometryTypes::GeometryCollection),
            Self::CircularString => Ok(GeometryTypes::CircularString),
            Self::CompoundCurve => Ok(GeometryTypes::CompoundCurve),
            Self::CurvePolygon => Ok(GeometryTypes::CurvePolygon),
            Self::MultiCurve => Ok(GeometryTypes::MultiCurve),
            Self::MultiSurface => Ok(GeometryTypes::MultiSurface),
            t => Err(geos::Error::GenericError(format!(
                "unsupported geometry type: {t:?}"
            ))),
        }
    }
}
