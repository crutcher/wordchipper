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
    fn include_all() -> Self {
        SpecialFilter {
            inner: wc::SpecialFilter::IncludeAll,
        }
    }

    #[staticmethod]
    fn include_none() -> Self {
        SpecialFilter {
            inner: wc::SpecialFilter::IncludeNone,
        }
    }

    #[staticmethod]
    fn include(tokens: Vec<String>) -> PyResult<Self> {
        Ok(SpecialFilter {
            inner: wc::SpecialFilter::Include(tokens.into_iter().collect()),
        })
    }

    fn is_all(&self) -> bool {
        matches!(self.inner, wc::SpecialFilter::IncludeAll)
    }

    /// Does the given token match the special filter?
    fn __contains__(
        &self,
        token: &str,
    ) -> bool {
        self.inner.contains(token)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_filter() {
        let filter = SpecialFilter::include_all();
        assert_eq!(filter.inner(), &wc::SpecialFilter::IncludeAll);
        assert!(filter.__contains__("foo"));
    }

    #[test]
    fn test_none_filter() {
        let filter = SpecialFilter::include_none();
        assert_eq!(filter.inner(), &wc::SpecialFilter::IncludeNone);
        assert!(!filter.__contains__("foo"));
    }

    #[test]
    fn test_include_filter() {
        let filter = SpecialFilter::include(vec!["foo".to_string(), "bar".to_string()]).unwrap();
        assert_eq!(
            filter.inner(),
            &wc::SpecialFilter::Include(
                vec!["foo".to_string(), "bar".to_string()]
                    .into_iter()
                    .collect()
            )
        );
        assert!(filter.__contains__("foo"));
        assert!(filter.__contains__("bar"));
        assert!(!filter.__contains__("baz"));
    }
}
