use num_enum::IntoPrimitive;
use num_enum::TryFromPrimitive;
use scroll::{Endian, IOread};
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

#[derive(Debug, IntoPrimitive, TryFromPrimitive)]
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
