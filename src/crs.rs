use proj4wkt::{Builder, Node};
use pyo3::prelude::*;

fn wkt_to_authority(i: &str) -> Option<(&str, &str)> {
    match Builder::new().parse(i) {
        Ok(Node::PROJCRS(p)) => p.projection.authority.map(|a| (a.name, a.code)),
        _ => None,
    }
}

#[pyfunction]
pub fn get_crs_auth_code(definition: &str) -> PyResult<Option<(&str, &str)>> {
    if let Some(("EPSG", code)) = definition.split_once(':') {
        Ok(Some(("EPSG", code)))
    } else {
        Ok(wkt_to_authority(definition))
    }
}
