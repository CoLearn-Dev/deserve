pub mod sede;
pub mod server;

use pyo3::prelude::*;

/// A Python module implemented in Rust.
#[pymodule]
fn _deserve_network_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<server::PyServer>()?;
    m.add_class::<server::PyClient>()?;
    Ok(())
}
