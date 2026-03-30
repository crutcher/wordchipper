use pyo3::{
    PyResult,
    pyclass,
    pymethods,
};

use crate::wc;

/// A policy for filtering special tokens.
#[pyclass(from_py_object)]
#[derive(Clone, Debug, PartialEq)]
pub struct SpecialFilter {
    inner: wc::SpecialFilter,
}

impl SpecialFilter {
    pub fn inner(&self) -> &wc::SpecialFilter {
        &self.inner
    }
}

#[pymethods]
impl SpecialFilter {
    #[staticmethod]
    fn all() -> Self {
        SpecialFilter {
            inner: wc::SpecialFilter::All,
        }
    }

    #[staticmethod]
    fn none() -> Self {
        SpecialFilter {
            inner: wc::SpecialFilter::None,
        }
    }

    #[staticmethod]
    fn new(tokens: Vec<String>) -> PyResult<Self> {
        Ok(SpecialFilter {
            inner: wc::SpecialFilter::Include(tokens.into_iter().collect()),
        })
    }

    /// Does the given token match the special filter?
    fn __contains__(
        &self,
        token: &str,
    ) -> bool {
        self.inner.contains(token)
    }
}
