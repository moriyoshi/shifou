//! Inspect persisted payloads; no inference or requantization.
use anyhow::{ensure, Context, Result};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use shifou::{Address, Cache};
use std::{collections::BTreeMap, fs, path::Path};
use tokenizers::Tokenizer;
use yesno_core::{roaring_format::serialize_u64, OrdSet};

fn roaring(bytes: &[u8], width: usize, transpose: bool) -> usize {
    let words = bytes.len() * 8 / width;
    let mut ordinals = Vec::new();
    for (i, &byte) in bytes.iter().enumerate() {
        for bit in 0..8 {
            if byte & (1 << bit) != 0 {
                let pos = i * 8 + bit;
                ordinals.push(if transpose {
                    ((pos % width) * words + pos / width) as u64
                } else {
                    pos as u64
                });
            }
        }
    }
    let mut set = OrdSet::from_iter_unsorted(ordinals);
    set.optimize();
    serialize_u64(&set).len()
}
#[derive(Default)]
struct Counts {
    bits: u64,
    ones: u64,
    plane_ones: Vec<u64>,
    words: u64,
    packed: usize,
    interleaved: usize,
    planes: usize,
    layer_densities: Vec<f64>,
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    ensure!(
        args.len() == 5,
        "usage: density MODEL_DIR CORPUS RESULT_DIR OUTPUT_JSON"
    );
    let root = Path::new(&args[3]);
    let config: Value =
        serde_json::from_slice(&fs::read(Path::new(&args[1]).join("config.json"))?)?;
    let layers = config["num_hidden_layers"]
        .as_u64()
        .context("layer count")? as u32;
    let tokenizer = Tokenizer::from_file(Path::new(&args[1]).join("tokenizer.json"))
        .map_err(anyhow::Error::msg)?;
    let text = fs::read_to_string(&args[2])?;
    let tokens = tokenizer
        .encode(text.clone(), false)
        .map_err(anyhow::Error::msg)?
        .get_ids()
        .to_vec();
    let corpus_hash = format!("{:x}", Sha256::digest(text.as_bytes()));
    let mut rows = Vec::new();
    for mode in ["auto", "turbo4", "turbo3"] {
        let result: Value = serde_json::from_slice(&fs::read(root.join(format!("{mode}.json")))?)?;
        ensure!(result["corpus_sha256"] == corpus_hash, "corpus mismatch");
        for row in result["cases"].as_array().context("cases")? {
            let case = row["case"].as_u64().unwrap();
            let prefix = row["prefix_tokens"].as_u64().unwrap() as usize;
            let offset = row["offset"].as_u64().unwrap() as usize;
            let label = row["mode"].as_str().unwrap();
            let mut address = Address {
                namespace: "xinfer-evaluation".into(),
                model_fingerprint: result["model_fingerprint"].as_str().unwrap().into(),
                prefix_fingerprint: format!(
                    "{:x}",
                    Sha256::digest(serde_json::to_vec(&tokens[offset..offset + prefix])?)
                ),
                layer: 0,
                slot: "native-snapshot".into(),
            };
            let cache = Cache::open(root.join(format!("cache-{case}-{label}")))?;
            if mode == "auto" {
                let (mut ones, mut bits, mut payload, mut packed, mut metadata) =
                    (0u64, 0u64, 0usize, 0u64, 0usize);
                let mut by_width = BTreeMap::<u8, usize>::new();
                for layer in 0..layers {
                    for slot in ["key", "value"] {
                        address.layer = layer;
                        address.slot = slot.into();
                        let r = cache.get(&address)?.context("missing affine tile")?.report;
                        ones += r.codec.payload_set_bits;
                        bits += r.codec.payload_bit_len;
                        payload += r.codec.payload_bytes;
                        packed += r.codec.packed_payload_bytes;
                        metadata += r.metadata_roaring_bytes;
                        for (width, count) in r.codec.groups_by_bits {
                            *by_width.entry(width).or_default() += count;
                        }
                    }
                }
                rows.push(json!({"case":case,"mode":label,"prefix_tokens":prefix,
                    "set_bits":ones,"logical_bits":bits,"density":ones as f64/bits as f64,
                    "payload_roaring_bytes":payload,"packed_payload_bytes":packed,
                    "metadata_roaring_bytes":metadata,"groups_by_bits":by_width}));
            } else {
                let snapshot = cache
                    .get_packed(&address, result["format"].as_str().unwrap())?
                    .context("missing packed snapshot")?;
                let mut groups = BTreeMap::<String, Counts>::new();
                for b in snapshot.buffers {
                    let kind = b.name.split('/').nth(1).unwrap().to_string();
                    let width = if kind == "k_quant" && mode == "turbo3" {
                        3
                    } else if kind.ends_with("quant") {
                        4
                    } else {
                        32
                    };
                    let c = groups.entry(kind).or_default();
                    if c.plane_ones.is_empty() {
                        c.plane_ones.resize(width, 0);
                    }
                    let words = b.bytes.len() * 8 / width;
                    let mut ones = 0u64;
                    for (i, &byte) in b.bytes.iter().enumerate() {
                        for bit in 0..8 {
                            if byte & (1 << bit) != 0 {
                                ones += 1;
                                c.plane_ones[(i * 8 + bit) % width] += 1;
                            }
                        }
                    }
                    c.layer_densities
                        .push(ones as f64 / (b.bytes.len() * 8) as f64);
                    c.ones += ones;
                    c.bits += (b.bytes.len() * 8) as u64;
                    c.words += words as u64;
                    c.packed += b.bytes.len();
                    c.interleaved += roaring(&b.bytes, width, false);
                    c.planes += roaring(&b.bytes, width, true);
                }
                for (kind, c) in groups {
                    rows.push(json!({"case":case,"mode":label,"prefix_tokens":prefix,"buffer":kind,
                        "set_bits":c.ones,"logical_bits":c.bits,"density":c.ones as f64/c.bits as f64,
                        "plane_densities":c.plane_ones.iter().map(|&x|x as f64/c.words as f64).collect::<Vec<_>>(),
                        "min_layer_density":c.layer_densities.iter().copied().fold(1.0,f64::min),
                        "max_layer_density":c.layer_densities.iter().copied().fold(0.0,f64::max),
                        "packed_bytes":c.packed,"per_buffer_roaring_bytes":c.interleaved,
                        "per_buffer_transposed_roaring_bytes":c.planes}));
                }
            }
        }
    }
    fs::write(&args[4], serde_json::to_vec_pretty(&rows)?)?;
    Ok(())
}
