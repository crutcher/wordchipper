# wordchipper style guide

This document describes the style guide for the wordchipper project.
API rules provided in this document should be uniformly applied across
all crates in the workspace, including unpublished ones.

### Field Accessors

```rust
pub struct Foo {
    pub bar: i32,
}

impl Foo {
    /// Get ${bar short descrip / name}.
    pub fn bar(&self) -> i32 {
        self.bar
    }

    /// Set ${bar short descrip / name}.
    ///
    /// This is an in-place setter.
    pub fn set_bar(&mut self, bar: i32) {
        self.bar = bar;
    }

    /// Set ${bar short descrip / name}.
    ///
    /// This is a builder setter.
    pub fn with_bar(mut self, bar: i32) -> Self {
        self.set_bar(bar);
        self
    }
}


