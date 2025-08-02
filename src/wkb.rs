use geos::GeometryTypes;
use num_enum::{IntoPrimitive, TryFromPrimitive};
use scroll::{Endian, IOread};
use serde::{Deserialize, Serialize};
use std::io;

pub struct WKBHeader {
    pub geometry_type: WKBGeometryType,
    pub has_z: bool,
    pub has_m: bool,
    pub srid: i32,
}

impl TryFrom<&[u8]> for WKBHeader {
    type Error = geos::Error;

    fn try_from(mut wkb: &[u8]) -> Result<Self, Self::Error> {
        fn get_type_id_and_srid(wkb: &mut &[u8]) -> Result<(u32, i32), io::Error> {
            let byte_order = wkb.ioread::<u8>()?;
            let is_little_endian = byte_order != 0;
            let endian = Endian::from(is_little_endian);
            let type_id = wkb.ioread_with::<u32>(endian)?;
            let srid = if type_id & 0x2000_0000 == 0x2000_0000 {
                wkb.ioread_with::<i32>(endian)?
            } else {
                0
            };
            Ok((type_id, srid))
        }

        let (type_id, srid) = get_type_id_and_srid(&mut wkb)
            .map_err(|_| geos::Error::GenericError("Invalid WKB Header".into()))?;

        let has_z = type_id & 0x8000_0000 != 0;
        let has_m = type_id & 0x4000_0000 != 0;

        let type_id: u8 = (type_id & 0xFF)
            .try_into()
            .map_err(|_| geos::Error::GenericError("Invalid WKB Header".into()))?;

        let geometry_type = WKBGeometryType::try_from(type_id).map_err(|_| {
            geos::Error::GenericError(format!("Invalid geometry type id: {type_id}"))
        })?;

        Ok(Self {
            geometry_type,
            has_z,
            has_m,
            srid,
        })
    }
}

#[derive(Clone, Copy, Debug, IntoPrimitive, TryFromPrimitive, Serialize, Deserialize)]
#[repr(u8)]
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
