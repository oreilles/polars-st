#![feature(downcast_unchecked)]
#![feature(iterator_try_reduce)]
#![deny(clippy::pedantic)]
#![allow(clippy::get_first)]
#![allow(clippy::match_bool)]
#![allow(clippy::unused_unit)]
#![allow(clippy::enum_glob_use)]
#![allow(clippy::single_call_fn)]
#![allow(clippy::needless_pass_by_value)]
#![allow(clippy::module_name_repetitions)]
// TODO: Actually fix those
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::cast_possible_truncation)]

use pyo3::prelude::*;

mod args;
mod arity;
mod crs;
mod expressions;
mod functions;
mod wkb;

#[pymodule]
fn _lib(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(crs::get_crs_auth_code, m)?)?;
    m.add_function(wrap_pyfunction!(expressions::apply_coordinates, m)?)?;
    Ok(())
}
