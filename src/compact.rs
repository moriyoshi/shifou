//! Packed low-bit KV tiles with two shared f32 parameters per quantized group.
//! Groups use a named axis, and recent token values are retained exactly.
use serde::{Deserialize, Serialize};

use crate::tensor::{groups, validate_shape};
use crate::{Error, PackedBuffer, PackedSnapshot, Result, Tensor};

const FORMAT: &str = "shifou/compact-kv-v1";
const MAX_META: usize = 16 * 1024;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompactPolicy {
    pub bits: u8,
    pub group_axis: String,
    pub token_axis: String,
    pub residual_tokens: usize,
}

impl CompactPolicy {
    pub fn keys(bits: u8) -> Self {
        Self {
            bits,
            group_axis: "token".into(),
            token_axis: "token".into(),
            residual_tokens: 32,
        }
    }

    pub fn values(bits: u8) -> Self {
        Self {
            group_axis: "channel".into(),
            ..Self::keys(bits)
        }
    }

    fn validate(&self, shape: &[usize], axes: &[String]) -> Result<(usize, usize)> {
        validate_shape(shape, axes)?;
        if ![1, 2, 4].contains(&self.bits) {
            return Err(Error::Invalid("compact bits must be 1, 2, or 4".into()));
        }
        let group = axes
            .iter()
            .position(|name| name == &self.group_axis)
            .ok_or_else(|| Error::Invalid("compact group axis missing".into()))?;
        let token = axes
            .iter()
            .position(|name| name == &self.token_axis)
            .ok_or_else(|| Error::Invalid("compact token axis missing".into()))?;
        Ok((group, token))
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Metadata {
    version: u32,
    shape: Vec<usize>,
    axes: Vec<String>,
    policy: CompactPolicy,
}

fn is_tail(index: usize, shape: &[usize], token_axis: usize, residual_tokens: usize) -> bool {
    let stride: usize = shape[token_axis + 1..].iter().product();
    let token = (index / stride) % shape[token_axis];
    token >= shape[token_axis].saturating_sub(residual_tokens)
}

fn two_centroids(values: &[f32]) -> (f32, f32) {
    let mut low = values.iter().copied().fold(f32::INFINITY, f32::min) as f64;
    let mut high = values.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    for _ in 0..8 {
        let mid = (low + high) * 0.5;
        let (mut sum_low, mut sum_high) = (0.0, 0.0);
        let (mut n_low, mut n_high) = (0usize, 0usize);
        for &value in values {
            if (value as f64) <= mid {
                sum_low += value as f64;
                n_low += 1;
            } else {
                sum_high += value as f64;
                n_high += 1;
            }
        }
        if n_low == 0 || n_high == 0 {
            break;
        }
        low = sum_low / n_low as f64;
        high = sum_high / n_high as f64;
    }
    (low as f32, high as f32)
}

fn put_code(bytes: &mut Vec<u8>, rank: usize, bits: u8, code: u8) {
    let bit = rank * bits as usize;
    let byte = bit / 8;
    if byte == bytes.len() {
        bytes.push(0);
    }
    bytes[byte] |= code << (bit % 8);
}

fn get_code(bytes: &[u8], rank: usize, bits: u8) -> u8 {
    let bit = rank * bits as usize;
    (bytes[bit / 8] >> (bit % 8)) & ((1 << bits) - 1)
}

fn buffer(name: &str, dtype: &str, bytes: Vec<u8>, width: usize) -> PackedBuffer {
    PackedBuffer {
        name: name.into(),
        dtype: dtype.into(),
        shape: vec![bytes.len() / width],
        bytes,
    }
}

/// Encode a finite f32 tensor as dense 1/2/4-bit codes plus shared parameters.
/// For one bit the parameters are two learned centroids; for two/four bits they
/// are the affine minimum and step. The exact tail uses bf16 words when
/// all source values permit this, and otherwise retains original f32 words.
pub fn encode_compact(tensor: &Tensor, policy: &CompactPolicy) -> Result<PackedSnapshot> {
    tensor.validate()?;
    let (group_axis, token_axis) = policy.validate(&tensor.shape, &tensor.axes)?;
    let metadata = serde_json::to_vec(&Metadata {
        version: 1,
        shape: tensor.shape.clone(),
        axes: tensor.axes.clone(),
        policy: policy.clone(),
    })?;
    if metadata.len() > MAX_META {
        return Err(Error::Invalid("compact metadata too large".into()));
    }
    let mut codes = Vec::new();
    let mut params = Vec::new();
    let mut tail_values = Vec::new();
    let mut rank = 0;
    for indices in groups(&tensor.shape, group_axis, tensor.shape[group_axis]) {
        let quantized: Vec<f32> = indices
            .iter()
            .filter(|&&i| !is_tail(i, &tensor.shape, token_axis, policy.residual_tokens))
            .map(|&i| tensor.values[i])
            .collect();
        if !quantized.is_empty() {
            let (a, b) = if policy.bits == 1 {
                two_centroids(&quantized)
            } else {
                let minimum = quantized.iter().copied().fold(f32::INFINITY, f32::min);
                let maximum = quantized.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let step = (maximum as f64 - minimum as f64) / ((1u32 << policy.bits) - 1) as f64;
                (minimum, step as f32)
            };
            if !a.is_finite() || !b.is_finite() {
                return Err(Error::Invalid("compact parameters are not finite".into()));
            }
            params.extend_from_slice(&a.to_le_bytes());
            params.extend_from_slice(&b.to_le_bytes());
            for &value in &quantized {
                let code = if policy.bits == 1 {
                    u8::from((value as f64) > (a as f64 + b as f64) * 0.5)
                } else if b == 0.0 {
                    0
                } else {
                    (((value as f64 - a as f64) / b as f64).round() as i64)
                        .clamp(0, ((1u32 << policy.bits) - 1) as i64) as u8
                };
                put_code(&mut codes, rank, policy.bits, code);
                rank += 1;
            }
        }
        for &i in &indices {
            if is_tail(i, &tensor.shape, token_axis, policy.residual_tokens) {
                tail_values.push(tensor.values[i]);
            }
        }
    }
    let mut buffers = vec![buffer("metadata", "u8", metadata, 1)];
    if !codes.is_empty() {
        buffers.push(buffer("codes", "u8", codes, 1));
    }
    if !params.is_empty() {
        buffers.push(buffer("parameters", "f32-le", params, 4));
    }
    if !tail_values.is_empty() {
        if tail_values.iter().all(|v| v.to_bits() & 0xffff == 0) {
            let mut tail = Vec::with_capacity(tail_values.len() * 2);
            for value in tail_values {
                tail.extend_from_slice(&((value.to_bits() >> 16) as u16).to_le_bytes());
            }
            buffers.push(buffer("tail", "bf16-le", tail, 2));
        } else {
            let mut tail = Vec::with_capacity(tail_values.len() * 4);
            for value in tail_values {
                tail.extend_from_slice(&value.to_le_bytes());
            }
            buffers.push(buffer("tail", "f32-le", tail, 4));
        }
    }
    let snapshot = PackedSnapshot {
        format: FORMAT.into(),
        buffers,
    };
    snapshot.validate()?;
    Ok(snapshot)
}

fn bytes<'a>(snapshot: &'a PackedSnapshot, name: &str, dtype: &str) -> Result<&'a [u8]> {
    match snapshot.buffers.iter().find(|b| b.name == name) {
        Some(b) if b.dtype == dtype => Ok(&b.bytes),
        Some(_) => Err(Error::Corrupt("compact buffer dtype mismatch".into())),
        None => Ok(&[]),
    }
}

/// Decode a self-describing compact tile, rejecting length and padding errors.
pub fn decode_compact(snapshot: &PackedSnapshot) -> Result<Tensor> {
    snapshot.validate()?;
    if snapshot.format != FORMAT
        || snapshot
            .buffers
            .iter()
            .any(|b| !["metadata", "codes", "parameters", "tail"].contains(&b.name.as_str()))
    {
        return Err(Error::Corrupt("unsupported compact snapshot layout".into()));
    }
    let meta_bytes = bytes(snapshot, "metadata", "u8")?;
    if meta_bytes.is_empty() || meta_bytes.len() > MAX_META {
        return Err(Error::Corrupt("missing compact metadata".into()));
    }
    let meta: Metadata = serde_json::from_slice(meta_bytes)?;
    if meta.version != 1 {
        return Err(Error::Corrupt("unsupported compact version".into()));
    }
    let (group_axis, token_axis) = meta.policy.validate(&meta.shape, &meta.axes)?;
    let codes = bytes(snapshot, "codes", "u8")?;
    let params = bytes(snapshot, "parameters", "f32-le")?;
    let tail_buffer = snapshot.buffers.iter().find(|b| b.name == "tail");
    let (tail, tail_width) = match tail_buffer {
        Some(b) if b.dtype == "bf16-le" => (b.bytes.as_slice(), 2),
        Some(b) if b.dtype == "f32-le" => (b.bytes.as_slice(), 4),
        Some(_) => return Err(Error::Corrupt("compact tail dtype mismatch".into())),
        None => (&[][..], 4),
    };
    let mut values = vec![0.0; meta.shape.iter().product()];
    let (mut rank, mut parameter_offset, mut tail_offset) = (0usize, 0usize, 0usize);
    for indices in groups(&meta.shape, group_axis, meta.shape[group_axis]) {
        let non_tail = indices
            .iter()
            .filter(|&&i| !is_tail(i, &meta.shape, token_axis, meta.policy.residual_tokens))
            .count();
        let (a, b) = if non_tail > 0 {
            if parameter_offset + 8 > params.len() {
                return Err(Error::Corrupt("truncated compact parameters".into()));
            }
            let a = f32::from_le_bytes(
                params[parameter_offset..parameter_offset + 4]
                    .try_into()
                    .unwrap(),
            );
            let b = f32::from_le_bytes(
                params[parameter_offset + 4..parameter_offset + 8]
                    .try_into()
                    .unwrap(),
            );
            if !a.is_finite()
                || !b.is_finite()
                || (meta.policy.bits > 1 && b < 0.0)
                || (meta.policy.bits == 1 && a > b)
            {
                return Err(Error::Corrupt("invalid compact parameters".into()));
            }
            parameter_offset += 8;
            (a, b)
        } else {
            (0.0, 0.0)
        };
        for i in indices {
            let value = if is_tail(i, &meta.shape, token_axis, meta.policy.residual_tokens) {
                if tail_offset + tail_width > tail.len() {
                    return Err(Error::Corrupt("truncated compact tail".into()));
                }
                let value = if tail_width == 2 {
                    let bits =
                        u16::from_le_bytes(tail[tail_offset..tail_offset + 2].try_into().unwrap());
                    f32::from_bits((bits as u32) << 16)
                } else {
                    f32::from_le_bytes(tail[tail_offset..tail_offset + 4].try_into().unwrap())
                };
                tail_offset += tail_width;
                value
            } else {
                if rank * meta.policy.bits as usize / 8 >= codes.len() {
                    return Err(Error::Corrupt("truncated compact codes".into()));
                }
                let code = get_code(codes, rank, meta.policy.bits);
                rank += 1;
                if meta.policy.bits == 1 {
                    if code == 0 {
                        a
                    } else {
                        b
                    }
                } else {
                    (a as f64 + code as f64 * b as f64) as f32
                }
            };
            if !value.is_finite() {
                return Err(Error::Corrupt("nonfinite compact value".into()));
            }
            values[i] = value;
        }
    }
    let used_bits = rank * meta.policy.bits as usize;
    if codes.len() != used_bits.div_ceil(8)
        || params.len() != parameter_offset
        || tail.len() != tail_offset
        || (!used_bits.is_multiple_of(8) && codes.last().is_some_and(|b| b >> (used_bits % 8) != 0))
    {
        return Err(Error::Corrupt(
            "compact buffer length or padding mismatch".into(),
        ));
    }
    Ok(Tensor {
        shape: meta.shape,
        axes: meta.axes,
        values,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn compact_roundtrip_all_widths_and_axes() {
        let source = Tensor {
            shape: vec![7, 2, 5],
            axes: vec!["token".into(), "head".into(), "channel".into()],
            values: (0..70).map(|i| (i as f32 * 0.37).sin()).collect(),
        };
        for bits in [1, 2, 4] {
            for policy in [CompactPolicy::keys(bits), CompactPolicy::values(bits)] {
                let mut policy = policy;
                policy.residual_tokens = 2;
                let packed = encode_compact(&source, &policy).unwrap();
                let decoded = decode_compact(&packed).unwrap();
                assert_eq!(decoded.shape, source.shape);
                for t in 5..7 {
                    for h in 0..2 {
                        for c in 0..5 {
                            let i = (t * 2 + h) * 5 + c;
                            assert_eq!(decoded.values[i].to_bits(), source.values[i].to_bits());
                        }
                    }
                }
                assert!(decoded.values.iter().all(|x| x.is_finite()));
                let codes = bytes(&packed, "codes", "u8").unwrap();
                assert_eq!(codes.len(), (5 * 2 * 5 * bits as usize).div_ceil(8));
            }
        }
    }

    #[test]
    fn compact_packs_bf16_tail_exactly() {
        let source = Tensor {
            shape: vec![4, 2],
            axes: vec!["token".into(), "channel".into()],
            values: vec![1.0, -2.0, 3.0, 4.0, 5.0, -0.0, 7.0, 8.0],
        };
        let mut policy = CompactPolicy::values(1);
        policy.residual_tokens = 2;
        let packed = encode_compact(&source, &policy).unwrap();
        let tail = packed.buffers.iter().find(|b| b.name == "tail").unwrap();
        assert_eq!(tail.dtype, "bf16-le");
        assert_eq!(tail.bytes.len(), 8);
        let decoded = decode_compact(&packed).unwrap();
        for i in 4..8 {
            assert_eq!(decoded.values[i].to_bits(), source.values[i].to_bits());
        }
    }

    #[test]
    fn compact_rejects_bad_code_lengths() {
        let source = Tensor {
            shape: vec![5, 1],
            axes: vec!["token".into(), "channel".into()],
            values: vec![0.0, 1.0, 2.0, 3.0, 4.0],
        };
        let mut policy = CompactPolicy::keys(1);
        policy.residual_tokens = 1;
        let mut packed = encode_compact(&source, &policy).unwrap();
        packed
            .buffers
            .iter_mut()
            .find(|b| b.name == "codes")
            .unwrap()
            .bytes
            .push(0);
        packed
            .buffers
            .iter_mut()
            .find(|b| b.name == "codes")
            .unwrap()
            .shape[0] += 1;
        assert!(decode_compact(&packed).is_err());
    }
}
