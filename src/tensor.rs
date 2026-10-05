use crate::{Error, Result};
use serde::{Deserialize, Serialize};

/// Contiguous row-major f32 tile. Axis names carry meaning; axis order is arbitrary.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub axes: Vec<String>,
    pub values: Vec<f32>,
}

pub(crate) const MAX_ELEMENTS: usize = 16 * 1024 * 1024;

pub(crate) fn validate_shape(shape: &[usize], axes: &[String]) -> Result<usize> {
    if shape.is_empty() || shape.len() > 8 || shape.len() != axes.len() {
        return Err(Error::Invalid("expected 1..=8 named axes".into()));
    }
    let mut size = 1usize;
    for (i, (&dim, axis)) in shape.iter().zip(axes).enumerate() {
        if dim == 0 || axis.is_empty() || axes[..i].contains(axis) {
            return Err(Error::Invalid(
                "axes must be unique and dimensions nonzero".into(),
            ));
        }
        size = size
            .checked_mul(dim)
            .filter(|&n| n <= MAX_ELEMENTS)
            .ok_or_else(|| Error::Invalid("tile exceeds 16M elements".into()))?;
    }
    Ok(size)
}

impl Tensor {
    pub fn validate(&self) -> Result<()> {
        if validate_shape(&self.shape, &self.axes)? != self.values.len() {
            return Err(Error::Invalid("shape does not match value count".into()));
        }
        if self.values.iter().any(|x| !x.is_finite()) {
            return Err(Error::Invalid("NaN and infinity are unsupported".into()));
        }
        Ok(())
    }

    pub(crate) fn axis(&self, name: &str) -> Result<usize> {
        self.axes
            .iter()
            .position(|x| x == name)
            .ok_or_else(|| Error::Invalid(format!("missing axis {name}")))
    }
}

/// Enumerate groups along one axis, retaining all other coordinates.
/// Kept deterministic so the decoder needs no per-element position catalog.
pub(crate) fn groups(shape: &[usize], axis: usize, group_size: usize) -> Vec<Vec<usize>> {
    let inner: usize = shape[axis + 1..].iter().product();
    let outer: usize = shape[..axis].iter().product();
    let width = shape[axis];
    let mut result = Vec::new();
    for o in 0..outer {
        for i in 0..inner {
            for start in (0..width).step_by(group_size) {
                result.push(
                    (start..(start + group_size).min(width))
                        .map(|j| (o * width + j) * inner + i)
                        .collect(),
                );
            }
        }
    }
    result
}
