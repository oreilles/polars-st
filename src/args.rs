use serde::{Deserialize, Serialize};

use crate::wkb::WKBGeometryType;

#[derive(Deserialize)]
pub struct ToWktKwargs {
    pub rounding_precision: Option<u32>,
    pub trim: bool,
    pub output_dimension: i32,
    pub old_3d: bool,
}

#[derive(Deserialize)]
pub struct ToWkbKwargs {
    pub output_dimension: i32,
    pub byte_order: Option<i32>,
    pub include_srid: bool,
}

#[derive(Deserialize)]
pub struct ToGeoJsonKwargs {
    pub indent: Option<i32>,
}

#[derive(Deserialize, Clone, Copy)]
#[serde(rename_all = "snake_case")]
pub enum PrecisionMode {
    ValidOutput,
    NoTopo,
    KeepCollapsed,
}

impl From<PrecisionMode> for geos::Precision {
    #[inline]
    fn from(val: PrecisionMode) -> Self {
        match val {
            PrecisionMode::ValidOutput => Self::ValidOutput,
            PrecisionMode::NoTopo => Self::NoTopo,
            PrecisionMode::KeepCollapsed => Self::KeepCollapsed,
        }
    }
}

#[derive(Deserialize)]
pub struct SetPrecisionKwargs {
    pub mode: PrecisionMode,
}

#[derive(Deserialize)]
pub struct SimplifyKwargs {
    pub preserve_topology: bool,
}

#[derive(Deserialize)]
pub struct DistanceDensifyKwargs {
    pub densify: Option<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CapStyle {
    Round,
    Flat,
    Square,
}

impl From<CapStyle> for geos::CapStyle {
    #[inline]
    fn from(val: CapStyle) -> Self {
        match val {
            CapStyle::Round => Self::Round,
            CapStyle::Flat => Self::Flat,
            CapStyle::Square => Self::Square,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JoinStyle {
    Round,
    Mitre,
    Bevel,
}

impl From<JoinStyle> for geos::JoinStyle {
    #[inline]
    fn from(val: JoinStyle) -> Self {
        match val {
            JoinStyle::Round => Self::Round,
            JoinStyle::Mitre => Self::Mitre,
            JoinStyle::Bevel => Self::Bevel,
        }
    }
}

#[derive(Deserialize)]
pub struct BufferKwargs {
    quad_segs: i32,
    cap_style: CapStyle,
    join_style: JoinStyle,
    mitre_limit: f64,
    single_sided: bool,
}

impl TryInto<geos::BufferParams> for &BufferKwargs {
    type Error = geos::Error;

    #[inline]
    fn try_into(self) -> Result<geos::BufferParams, Self::Error> {
        geos::BufferParams::builder()
            .quadrant_segments(self.quad_segs)
            .end_cap_style(self.cap_style.into())
            .join_style(self.join_style.into())
            .mitre_limit(self.mitre_limit)
            .single_sided(self.single_sided)
            .build()
    }
}

#[derive(Deserialize)]
pub struct OffsetCurveKwargs {
    pub quad_segs: i32,
    pub join_style: JoinStyle,
    pub mitre_limit: f64,
}

#[derive(Deserialize)]
pub struct ConcaveHullKwargs {
    pub ratio: f64,
    pub allow_holes: bool,
}

#[derive(Deserialize)]
pub struct InterpolateKwargs {
    pub normalized: bool,
}

#[derive(Deserialize)]
pub struct SetOperationKwargs {
    pub grid_size: Option<f64>,
}

#[derive(Deserialize)]
pub struct EqualsExactKwargs {
    pub tolerance: f64,
}

#[derive(Deserialize)]
pub struct DelaunayTrianlesKwargs {
    pub only_edges: bool,
    pub tolerance: f64,
}

#[derive(Deserialize)]
pub struct VoronoiKwargs {
    pub tolerance: f64,
    pub extend_to: Option<Vec<u8>>,
    pub only_edges: bool,
}

#[derive(Deserialize)]
pub struct LineMergeKwargs {
    pub directed: bool,
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
#[serde(tag = "type", content = "param")]
pub enum SjoinPredicate {
    IntersectsBbox,
    Intersects,
    Within,
    Contains,
    Overlaps,
    Crosses,
    Touches,
    Covers,
    CoveredBy,
    ContainsProperly,
    Dwithin(f64),
}

#[derive(Deserialize)]
pub struct SjoinKwargs {
    pub predicate: SjoinPredicate,
}

#[derive(Deserialize)]
pub struct GetCoordinatesKwargs {
    pub output_dimension: Option<usize>,
}

#[derive(Deserialize)]
pub struct RelatePatternKwargs {
    pub pattern: String,
}

#[derive(Deserialize, Clone, Copy)]
#[serde(rename_all = "snake_case")]
pub enum TransformOrigin {
    Center,
    Centroid,
    #[serde(untagged)]
    XY((f64, f64)),
    #[serde(untagged)]
    XYZ((f64, f64, f64)),
}

impl TryInto<(f64, f64, f64)> for TransformOrigin {
    type Error = u8;
    fn try_into(self) -> Result<(f64, f64, f64), Self::Error> {
        match self {
            Self::XY((x, y)) => Ok((x, y, 0.0)),
            Self::XYZ((x, y, z)) => Ok((x, y, z)),
            _ => Err(1),
        }
    }
}

#[derive(Deserialize)]
pub struct TransformKwargs {
    pub origin: TransformOrigin,
}

#[derive(Deserialize)]
pub struct CollectKwargs {
    pub into: Option<WKBGeometryType>,
}
