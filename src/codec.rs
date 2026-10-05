use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use yesno_core::{roaring_format::serialize_u64, OrdSet};

use crate::tensor::{groups, validate_shape};
use crate::{Error, Result, Tensor};

/// Numeric error bounds are local reconstruction bounds, not model-quality guarantees.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Policy {
    pub group_axis: String,
    pub group_size: usize,
    pub candidate_bits: Vec<u8>,
    pub max_abs_error: f64,
    /// Try excluding this fraction from each end of a group's range.
    /// Excluded values are preserved exactly, including their positions.
    pub outlier_fraction: f64,
    /// Groups touching this many final positions of token_axis remain exact.
    pub residual_tokens: usize,
    pub token_axis: String,
}

impl Policy {
    /// KIVI-inspired grouping: each channel is quantized across tokens.
    pub fn keys(max_abs_error: f64) -> Self {
        Self {
            group_axis: "token".into(),
            group_size: 32,
            candidate_bits: vec![2, 4, 8],
            max_abs_error,
            outlier_fraction: 0.02,
            residual_tokens: 16,
            token_axis: "token".into(),
        }
    }

    /// KIVI-inspired grouping: each token is quantized across channels.
    pub fn values(max_abs_error: f64) -> Self {
        Self {
            group_axis: "channel".into(),
            ..Self::keys(max_abs_error)
        }
    }

    fn validate(&self, shape: &[usize], axes: &[String]) -> Result<usize> {
        validate_shape(shape, axes)?;
        let axis = axes
            .iter()
            .position(|x| x == &self.group_axis)
            .ok_or_else(|| Error::Invalid("group axis not found".into()))?;
        if self.group_size == 0
            || self.group_size > 65536
            || self.candidate_bits.is_empty()
            || self.candidate_bits.iter().any(|b| ![2, 4, 8].contains(b))
            || !self.max_abs_error.is_finite()
            || self.max_abs_error < 0.0
            || !self.outlier_fraction.is_finite()
            || !(0.0..=0.1).contains(&self.outlier_fraction)
        {
            return Err(Error::Invalid("invalid quantization policy".into()));
        }
        if self.residual_tokens > 0 && !axes.contains(&self.token_axis) {
            return Err(Error::Invalid(
                "residual policy requires a token axis".into(),
            ));
        }
        let count: usize =
            shape.iter().product::<usize>() / shape[axis] * shape[axis].div_ceil(self.group_size);
        if count > 262144 {
            return Err(Error::Invalid("tile exceeds 262144 groups".into()));
        }
        Ok(axis)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Group {
    pub offset: u64,
    pub bits: u8,
    pub minimum: f64,
    pub step: f64,
    pub exceptions: usize,
}

impl Group {
    fn span(&self, n: usize) -> u64 {
        (n * self.bits as usize
            + if self.exceptions > 0 {
                n + self.exceptions * 32
            } else {
                0
            }) as u64
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Descriptor {
    pub version: u32,
    pub shape: Vec<usize>,
    pub axes: Vec<String>,
    pub policy: Policy,
    pub groups: Vec<Group>,
    pub measured_max_abs_error: f64,
    pub measured_rmse: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Report {
    pub groups_by_bits: BTreeMap<u8, usize>,
    pub exception_values: usize,
    pub raw_values: usize,
    pub max_abs_error: f64,
    pub rmse: f64,
    pub raw_f32_bytes: usize,
    /// Portable Roaring64 serialization of the complete payload.
    pub payload_bytes: usize,
    /// Cardinality and logical span, including exact exceptions and their masks.
    pub payload_set_bits: u64,
    pub payload_bit_len: u64,
    pub packed_payload_bytes: u64,
    /// Diagnostic JSON size. Persistence uses compact binary group descriptors.
    pub descriptor_json_bytes: usize,
}

pub struct EncodedTile {
    pub(crate) descriptor: Descriptor,
    pub(crate) payload: OrdSet,
}

impl EncodedTile {
    pub fn report(&self) -> Result<Report> {
        let d = &self.descriptor;
        let axis = d.policy.validate(&d.shape, &d.axes)?;
        let positions = groups(&d.shape, axis, d.policy.group_size);
        let mut groups_by_bits = BTreeMap::new();
        let mut raw_values = 0;
        let mut exception_values = 0;
        let mut payload_bit_len = 0;
        for (g, indices) in d.groups.iter().zip(positions) {
            *groups_by_bits.entry(g.bits).or_default() += 1;
            exception_values += g.exceptions;
            payload_bit_len += g.span(indices.len());
            if g.bits == 32 {
                raw_values += indices.len();
            }
        }
        Ok(Report {
            groups_by_bits,
            raw_values,
            exception_values,
            max_abs_error: d.measured_max_abs_error,
            rmse: d.measured_rmse,
            raw_f32_bytes: d.shape.iter().product::<usize>() * 4,
            payload_bytes: serialize_u64(&self.payload).len(),
            payload_set_bits: self.payload.len(),
            payload_bit_len,
            packed_payload_bytes: payload_bit_len.div_ceil(8),
            descriptor_json_bytes: serde_json::to_vec(d)?.len(),
        })
    }
}

struct Candidate {
    group: Group,
    payload: OrdSet,
    decoded: Vec<f32>,
}

fn put_word(ordinals: &mut Vec<u64>, base: u64, value: u32, width: u8, stride: usize) {
    for bit in 0..width {
        if value & (1u32 << bit) != 0 {
            ordinals.push(base + u64::from(bit) * stride as u64);
        }
    }
}

fn word(payload: &OrdSet, base: u64, width: u8, stride: usize) -> u32 {
    let mut result = 0;
    for bit in 0..width {
        if payload.contains(base + u64::from(bit) * stride as u64) {
            result |= 1u32 << bit;
        }
    }
    result
}

fn candidate(values: &[f32], bits: u8, trim: usize) -> Candidate {
    let n = values.len();
    let (minimum, step, maximum) = if bits == 32 {
        (0.0, 0.0, 0.0)
    } else {
        let mut sorted = values.to_vec();
        sorted.sort_by(f32::total_cmp);
        let minimum = sorted[trim] as f64;
        let maximum = sorted[n - 1 - trim] as f64;
        (
            minimum,
            (maximum - minimum) / ((1u32 << bits) - 1) as f64,
            maximum,
        )
    };
    let mut ordinals = Vec::new();
    let mut exceptions = Vec::new();
    let mut decoded = Vec::with_capacity(n);
    for (i, &value) in values.iter().enumerate() {
        let x = value as f64;
        if bits != 32 && (x < minimum || x > maximum) {
            exceptions.push((i, value.to_bits()));
            decoded.push(value);
            continue;
        }
        let code = if bits == 32 {
            value.to_bits()
        } else if step == 0.0 {
            0
        } else {
            ((x - minimum) / step).round() as u32
        };
        put_word(&mut ordinals, i as u64, code, bits, n);
        decoded.push(if bits == 32 {
            value
        } else {
            (minimum + step * code as f64) as f32
        });
    }
    if !exceptions.is_empty() {
        let mask = n as u64 * bits as u64;
        for (rank, &(i, value)) in exceptions.iter().enumerate() {
            ordinals.push(mask + i as u64);
            put_word(
                &mut ordinals,
                mask + n as u64 + rank as u64 * 32,
                value,
                32,
                1,
            );
        }
    }
    let mut payload = OrdSet::from_iter_unsorted(ordinals);
    payload.optimize();
    Candidate {
        group: Group {
            offset: 0,
            bits,
            minimum,
            step,
            exceptions: exceptions.len(),
        },
        payload,
        decoded,
    }
}

fn acceptable(original: &[f32], decoded: &[f32], error: f64) -> bool {
    original.iter().zip(decoded).all(|(&a, &b)| {
        b.is_finite()
            && (a as f64 - b as f64).abs() <= error
            && (error != 0.0 || a.to_bits() == b.to_bits())
    })
}

fn cost(candidate: &Candidate) -> Result<usize> {
    Ok(serialize_u64(&candidate.payload).len() + 29)
}

/// Choose each group's smallest measured standalone encoding satisfying the bound.
/// This is a local heuristic: concatenated container boundaries and database page
/// overhead are not an additive function of standalone group sizes.
pub fn encode(tensor: &Tensor, policy: &Policy) -> Result<EncodedTile> {
    tensor.validate()?;
    let axis = policy.validate(&tensor.shape, &tensor.axes)?;
    let positions = groups(&tensor.shape, axis, policy.group_size);
    let token = if policy.residual_tokens > 0 {
        Some(tensor.axis(&policy.token_axis)?)
    } else {
        None
    };
    let mut all_bits = Vec::new();
    let mut descriptors = Vec::new();
    let mut offset = 0;
    let mut max_error = 0.0f64;
    let mut squared_error = 0.0;
    for indices in positions {
        let values: Vec<_> = indices.iter().map(|&i| tensor.values[i]).collect();
        let raw = token.is_some_and(|axis| {
            let stride: usize = tensor.shape[axis + 1..].iter().product();
            let first_raw = tensor.shape[axis].saturating_sub(policy.residual_tokens);
            indices
                .iter()
                .any(|i| (i / stride) % tensor.shape[axis] >= first_raw)
        });
        let mut best = candidate(&values, 32, 0);
        let mut best_cost = cost(&best)?;
        if !raw {
            let trim = (values.len() as f64 * policy.outlier_fraction).floor() as usize;
            for &bits in &policy.candidate_bits {
                for t in [0, trim] {
                    let c = candidate(&values, bits, t);
                    if acceptable(&values, &c.decoded, policy.max_abs_error) {
                        let bytes = cost(&c)?;
                        if bytes < best_cost || (bytes == best_cost && bits < best.group.bits) {
                            best = c;
                            best_cost = bytes;
                        }
                    }
                    if trim == 0 {
                        break;
                    }
                }
            }
        }
        for (&a, &b) in values.iter().zip(&best.decoded) {
            let error = (a as f64 - b as f64).abs();
            max_error = max_error.max(error);
            squared_error += error * error;
        }
        best.group.offset = offset;
        all_bits.extend(best.payload.iter().map(|x| x + offset));
        offset += best.group.span(values.len());
        descriptors.push(best.group);
    }
    let mut payload = OrdSet::from_iter_unsorted(all_bits);
    payload.optimize();
    Ok(EncodedTile {
        descriptor: Descriptor {
            version: 1,
            shape: tensor.shape.clone(),
            axes: tensor.axes.clone(),
            policy: policy.clone(),
            groups: descriptors,
            measured_max_abs_error: max_error,
            measured_rmse: (squared_error / tensor.values.len() as f64).sqrt(),
        },
        payload,
    })
}

pub fn decode(tile: &EncodedTile) -> Result<Tensor> {
    let d = &tile.descriptor;
    if d.version != 1 {
        return Err(Error::Corrupt("unsupported codec version".into()));
    }
    let axis = d.policy.validate(&d.shape, &d.axes)?;
    let positions = groups(&d.shape, axis, d.policy.group_size);
    if positions.len() != d.groups.len() {
        return Err(Error::Corrupt("group count mismatch".into()));
    }
    let mut values = vec![0.0; d.shape.iter().product()];
    let mut end = 0;
    for (g, indices) in d.groups.iter().zip(positions) {
        let n = indices.len();
        if g.offset != end
            || ![2, 4, 8, 32].contains(&g.bits)
            || !g.minimum.is_finite()
            || !g.step.is_finite()
            || g.step < 0.0
            || g.exceptions > n
            || (g.bits == 32 && g.exceptions != 0)
        {
            return Err(Error::Corrupt("invalid group descriptor".into()));
        }
        let mask = g.offset + n as u64 * g.bits as u64;
        let mut rank = 0;
        for (i, &destination) in indices.iter().enumerate() {
            let code = word(&tile.payload, g.offset + i as u64, g.bits, n);
            let value = if g.exceptions > 0 && tile.payload.contains(mask + i as u64) {
                if rank >= g.exceptions || code != 0 {
                    return Err(Error::Corrupt("invalid exception mask or code".into()));
                }
                let bits = word(&tile.payload, mask + n as u64 + rank as u64 * 32, 32, 1);
                rank += 1;
                f32::from_bits(bits)
            } else if g.bits == 32 {
                f32::from_bits(code)
            } else {
                (g.minimum + g.step * code as f64) as f32
            };
            if !value.is_finite() {
                return Err(Error::Corrupt("decoded nonfinite value".into()));
            }
            values[destination] = value;
        }
        if rank != g.exceptions {
            return Err(Error::Corrupt("exception count mismatch".into()));
        }
        end += g.span(n);
    }
    if tile.payload.max().is_some_and(|x| x >= end) {
        return Err(Error::Corrupt("payload exceeds descriptor".into()));
    }
    Ok(Tensor {
        shape: d.shape.clone(),
        axes: d.axes.clone(),
        values,
    })
}
