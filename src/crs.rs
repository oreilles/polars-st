use proj::Proj;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyfunction]
pub fn get_crs_auth_code(definition: &str) -> PyResult<(String, String)> {
    let crs = Proj::new(definition).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((crs.id_auth_name()?.to_owned(), crs.id_code()?.to_owned()))
}
