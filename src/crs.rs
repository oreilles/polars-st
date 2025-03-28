use proj4wkt::builder::{Builder, Node};
use pyo3::prelude::*;

fn wkt_to_authority(i: &str) -> Option<(&str, &str)> {
    match Builder::new().parse(i) {
        Ok(Node::PROJCRS(p)) => p.projection.authority.map(|a| (a.name, a.code)),
        _ => None,
    }
}

#[pyfunction]
pub fn get_crs_authority(definition: &str) -> Option<(&str, &str)> {
    if let Some(("EPSG", code)) = definition.split_once(':') {
        Some(("EPSG", code))
    } else {
        wkt_to_authority(definition)
    }
}

#[pyfunction]
pub fn get_crs_from_code(srid: i64) -> Option<String> {
    srid.try_into()
        .ok()
        .and_then(crs_definitions::from_code)
        .map(|def| def.wkt.to_owned())
}
