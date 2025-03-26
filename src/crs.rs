use proj4wkt::{parser::Processor, Builder};
use pyo3::prelude::*;

fn wkt_to_authority(i: &str) -> Option<(&str, &str)> {
    type Node<'a> = <proj4wkt::Builder as Processor<'a>>::Output;
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
