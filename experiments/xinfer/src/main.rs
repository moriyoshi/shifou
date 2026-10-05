mod adaptive;
mod model;
mod onebit;
mod paper_quantizers;

use anyhow::{ensure, Context, Result};
use model::{Model, NativeCache};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use shifou::{Address, Cache, Policy, Tensor};
use std::{
    fs,
    io::{Read, Write},
    path::Path,
    time::Instant,
};
use tokenizers::Tokenizer;

const STEPS: usize = 32;
const GREEDY: usize = 16;

fn top(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0 as u32
}
fn teacher(
    model: &Model,
    cache: &NativeCache,
    suffix: &[u32],
    prefix: usize,
) -> Result<Vec<Vec<f32>>> {
    (0..STEPS)
        .map(|i| {
            let logits = model.forward(&suffix[i..i + 1], prefix + i, cache)?;
            ensure!(
                logits.iter().all(|v| v.is_finite()),
                "nonfinite model logits"
            );
            Ok(logits)
        })
        .collect()
}
fn greedy(model: &Model, cache: &NativeCache, seed: u32, prefix: usize) -> Result<Vec<u32>> {
    let mut token = seed;
    let mut result = Vec::new();
    for i in 0..GREEDY {
        let logits = model.forward(&[token], prefix + i, cache)?;
        ensure!(
            logits.iter().all(|v| v.is_finite()),
            "nonfinite greedy logits"
        );
        token = top(&logits);
        result.push(token);
    }
    Ok(result)
}
fn log_z(x: &[f32]) -> f64 {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    max + x.iter().map(|&v| (v as f64 - max).exp()).sum::<f64>().ln()
}
fn metrics(logits: &[Vec<f32>], reference: &[Vec<f32>], targets: &[u32]) -> Value {
    let (mut nll, mut kl, mut agree, mut max_delta) = (0.0, 0.0, 0usize, 0.0f32);
    for ((a, b), &target) in logits.iter().zip(reference).zip(targets) {
        let za = log_z(a);
        let zb = log_z(b);
        nll += za - a[target as usize] as f64;
        agree += usize::from(top(a) == top(b));
        for (&x, &y) in a.iter().zip(b) {
            let logp = y as f64 - zb;
            kl += logp.exp() * (logp - (x as f64 - za));
            max_delta = max_delta.max((x - y).abs());
        }
    }
    let count = logits.len() as f64;
    json!({"mean_nll": nll/count, "subset_perplexity": (nll/count).exp(),
        "kl_from_bf16": kl/count, "top1_agreement_with_bf16": agree as f64/count,
        "max_logit_delta_from_bf16": max_delta})
}
fn fingerprint(directory: &Path, format: &str) -> Result<String> {
    let mut hash = Sha256::new();
    for name in ["model.safetensors", "config.json", "tokenizer.json"] {
        hash.update(name.as_bytes());
        let mut file = fs::File::open(directory.join(name))?;
        let mut buf = vec![0u8; 1024 * 1024];
        loop {
            let n = file.read(&mut buf)?;
            if n == 0 {
                break;
            }
            hash.update(&buf[..n]);
        }
    }
    hash.update(format.as_bytes());
    Ok(format!("{:x}", hash.finalize()))
}
fn save_logits(path: &Path, logits: &[Vec<f32>]) -> Result<()> {
    let mut file = std::io::BufWriter::new(fs::File::create(path)?);
    for row in logits {
        for x in row {
            file.write_all(&x.to_le_bytes())?;
        }
    }
    file.flush()?;
    Ok(())
}
fn load_logits(path: &Path, vocab: usize) -> Result<Vec<Vec<f32>>> {
    let bytes = fs::read(path).context("run auto mode first to create the BF16 reference")?;
    ensure!(bytes.len() == STEPS * vocab * 4, "reference shape mismatch");
    Ok(bytes
        .chunks_exact(vocab * 4)
        .map(|row| {
            row.chunks_exact(4)
                .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
                .collect()
        })
        .collect())
}
fn physical(path: &Path) -> Result<(u64, u64)> {
    use std::os::unix::fs::MetadataExt;
    let (mut apparent, mut allocated) = (0, 0);
    for entry in fs::read_dir(path)? {
        let e = entry?;
        let m = e.metadata()?;
        let (a, b) = if m.is_dir() {
            physical(&e.path())?
        } else {
            (m.len(), m.blocks() * 512)
        };
        apparent += a;
        allocated += b;
    }
    Ok((apparent, allocated))
}
fn affine_restore(model: &Model, tiles: &[Tensor], capacity: usize) -> Result<NativeCache> {
    model.restore(tiles, capacity)
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    ensure!(
        args.len() == 5,
        "usage: shifou-xinfer MODEL_DIR CORPUS OUTPUT_DIR auto|turbo4|turbo3|adaptive|adaptive-joint|paper-quantizers|compact-onebit"
    );
    let directory = Path::new(&args[1]);
    let out = Path::new(&args[3]);
    let mode = &args[4];
    if mode == "compact-onebit" {
        return onebit::run(directory, Path::new(&args[2]), out);
    }
    if mode == "paper-quantizers" {
        return paper_quantizers::run(directory, Path::new(&args[2]), out);
    }
    if mode == "adaptive" || mode == "adaptive-joint" {
        return adaptive::run(
            directory,
            Path::new(&args[2]),
            out,
            mode == "adaptive-joint",
        );
    }
    fs::create_dir_all(out)?;
    ensure!(
        !out.join(format!("{mode}.json")).exists(),
        "results already exist; choose a new output directory"
    );
    let tokenizer =
        Tokenizer::from_file(directory.join("tokenizer.json")).map_err(anyhow::Error::msg)?;
    let text = fs::read_to_string(&args[2])?;
    let corpus_sha = format!("{:x}", Sha256::digest(text.as_bytes()));
    let tokens = tokenizer
        .encode(text, false)
        .map_err(anyhow::Error::msg)?
        .get_ids()
        .to_vec();
    let model = Model::load(directory, mode)?;
    if std::env::var_os("SHIFOU_SMOKE").is_some() {
        let cache = model.empty_cache(64)?;
        model.forward(&tokens[..32], 0, &cache)?;
        let tiles = if mode == "auto" {
            Some(model.export(&cache, 32)?)
        } else {
            None
        };
        let bf16_packed = if mode == "auto" {
            Some(
                model
                    .export_packed_bf16(&cache, 32)?
                    .into_iter()
                    .enumerate()
                    .map(|(index, snapshot)| {
                        (
                            Address {
                                namespace: "xinfer-smoke".into(),
                                model_fingerprint: "smoke-model".into(),
                                prefix_fingerprint: "smoke-prefix".into(),
                                layer: (index / 2) as u32,
                                slot: if index.is_multiple_of(2) {
                                    "key".into()
                                } else {
                                    "value".into()
                                },
                            },
                            snapshot,
                        )
                    })
                    .collect::<Vec<_>>(),
            )
        } else {
            None
        };
        let packed = if mode != "auto" {
            Some(model.export_tq(32)?)
        } else {
            None
        };
        let original = model.forward(&tokens[32..33], 32, &cache)?;
        let restored = if let Some(s) = packed {
            model.restore_tq(&s, 32, 64)?;
            ensure!(model.export_tq(32)? == s, "smoke packed bytes changed");
            model.duplicate(&cache)?
        } else {
            model.restore(tiles.as_ref().unwrap(), 64)?
        };
        let resumed = model.forward(&tokens[32..33], 32, &restored)?;
        ensure!(
            original == resumed && original.iter().all(|x| x.is_finite()),
            "smoke exact control failed"
        );
        if let Some(attention) = bf16_packed {
            let prepared = model.prepare_packed_bf16(&attention, 64)?;
            let prepared_cache = model.restore_prepared_bf16(&prepared)?;
            let packed_cache = model.restore_packed_bf16(&attention, 64)?;
            let native_chunk = model.forward_chunk(&tokens[32..34], 32, &cache)?;
            let restored_chunk = model.forward_chunk(&tokens[32..34], 32, &packed_cache)?;
            let prepared_chunk = model.forward_chunk(&tokens[32..34], 32, &prepared_cache)?;
            ensure!(
                native_chunk == restored_chunk && native_chunk == prepared_chunk,
                "smoke direct BF16 restore or batched append changed logits"
            );
        }
        println!("native {mode} export/restore: exact logits");
        return Ok(());
    }
    let format = model.format();
    let model_fingerprint = fingerprint(directory, &format)?;
    let comparison_fingerprint = fingerprint(directory, "bf16-comparison-v1")?;
    // Warm up loading and CUDA kernels; every measured case gets a fresh cache.
    let warm = model.empty_cache(64)?;
    model.forward(&tokens[..32], 0, &warm)?;
    model.forward(&tokens[32..33], 32, &warm)?;
    model.device.synchronize()?;
    let mut rows = Vec::new();
    for (case, (offset, prefix)) in [(0, 128), (4096, 128), (0, 512), (4096, 512)]
        .into_iter()
        .enumerate()
    {
        let input = &tokens[offset..offset + prefix];
        let suffix = &tokens[offset + prefix..offset + prefix + STEPS + 1];
        let capacity = prefix + STEPS;
        let address = Address {
            namespace: "xinfer-evaluation".into(),
            model_fingerprint: model_fingerprint.clone(),
            prefix_fingerprint: format!("{:x}", Sha256::digest(serde_json::to_vec(input)?)),
            layer: 0,
            slot: "native-snapshot".into(),
        };
        let native = model.empty_cache(capacity)?;
        model.device.synchronize()?;
        let now = Instant::now();
        model.forward(input, 0, &native)?;
        model.device.synchronize()?;
        let prefill_ms = now.elapsed().as_secs_f64() * 1000.0;
        let now = Instant::now();
        let tiles = if mode == "auto" {
            Some(model.export(&native, prefix)?)
        } else {
            None
        };
        let packed = if mode != "auto" {
            Some(model.export_tq(prefix)?)
        } else {
            None
        };
        let export_ms = now.elapsed().as_secs_f64() * 1000.0;
        let native_logits = teacher(&model, &native, suffix, prefix)?;
        // Replaying from the seed overwrites only continuation slots, preserving the prefix.
        let native_greedy = greedy(&model, &native, suffix[0], prefix)?;
        let reference_path = out.join(format!("bf16-{case}.bin"));
        let greedy_path = out.join(format!("bf16-{case}-greedy.json"));
        let manifest_path = out.join(format!("bf16-{case}-identity.json"));
        let comparison_identity = json!({"prefix":address.prefix_fingerprint,
            "corpus":corpus_sha,"model":comparison_fingerprint});
        let (reference, reference_greedy) = if mode == "auto" {
            save_logits(&reference_path, &native_logits)?;
            fs::write(&greedy_path, serde_json::to_vec(&native_greedy)?)?;
            fs::write(&manifest_path, serde_json::to_vec(&comparison_identity)?)?;
            (native_logits.clone(), native_greedy.clone())
        } else {
            ensure!(
                serde_json::from_slice::<Value>(&fs::read(&manifest_path)?)? == comparison_identity,
                "BF16 reference identity mismatch"
            );
            (
                load_logits(&reference_path, native_logits[0].len())?,
                serde_json::from_slice::<Vec<u32>>(&fs::read(&greedy_path)?)?,
            )
        };
        let errors: Vec<f64> = if mode == "auto" {
            vec![0.0, 0.05, 0.2]
        } else {
            vec![0.0]
        };
        for error in errors {
            let label = if mode == "auto" {
                format!("affine-{error}")
            } else {
                mode.clone()
            };
            let db_path = out.join(format!("cache-{case}-{label}"));
            ensure!(!db_path.exists(), "cache directory already exists");
            let now = Instant::now();
            let (logical, packed_bytes, codec_reports) = {
                let mut store = Cache::open(&db_path)?;
                if let Some(ref snapshot) = packed {
                    let r = store.put_packed(&address, snapshot)?;
                    (r.total_roaring_bytes, r.packed_bytes, json!(r))
                } else {
                    let mut reports = Vec::new();
                    let mut bytes = 0;
                    for (index, tile) in tiles.as_ref().unwrap().iter().enumerate() {
                        let mut a = address.clone();
                        a.layer = (index / 2) as u32;
                        a.slot = if index % 2 == 0 {
                            "key".into()
                        } else {
                            "value".into()
                        };
                        let mut p = if index % 2 == 0 {
                            Policy::keys(error)
                        } else {
                            Policy::values(error)
                        };
                        p.group_size = 64;
                        let r = store.put(&a, tile, &p)?;
                        bytes += r.total_roaring_bytes;
                        reports.push(r);
                    }
                    (bytes, 0, json!(reports))
                }
            };
            let persist_ms = now.elapsed().as_secs_f64() * 1000.0;
            let now = Instant::now();
            let (restored_tiles, restored_packed) = {
                let store = Cache::open(&db_path)?;
                if let Some(ref original) = packed {
                    let snapshot = store
                        .get_packed(&address, &format)?
                        .context("missing packed snapshot")?;
                    ensure!(&snapshot == original, "packed bytes changed after reopen");
                    (None, Some(snapshot))
                } else {
                    let mut result = Vec::new();
                    for index in 0..model.config.num_hidden_layers * 2 {
                        let mut a = address.clone();
                        a.layer = (index / 2) as u32;
                        a.slot = if index % 2 == 0 {
                            "key".into()
                        } else {
                            "value".into()
                        };
                        result.push(store.get(&a)?.context("missing tile")?.tensor);
                    }
                    if error == 0.0 {
                        ensure!(
                            result.iter().zip(tiles.as_ref().unwrap()).all(|(a, b)| a
                                .values
                                .iter()
                                .zip(&b.values)
                                .all(|(x, y)| x.to_bits() == y.to_bits())),
                            "exact tensor bits changed"
                        );
                    }
                    (Some(result), None)
                }
            };
            let reopen_read_ms = now.elapsed().as_secs_f64() * 1000.0;
            let now = Instant::now();
            let restored = if let Some(ref snapshot) = restored_packed {
                model.restore_tq(snapshot, prefix, capacity)?;
                ensure!(
                    model.export_tq(prefix)? == *snapshot,
                    "GPU TQ bytes changed"
                );
                model.duplicate(&native)?
            } else {
                affine_restore(&model, restored_tiles.as_ref().unwrap(), capacity)?
            };
            model.device.synchronize()?;
            let restore_ms = now.elapsed().as_secs_f64() * 1000.0;
            let logits = teacher(&model, &restored, suffix, prefix)?;
            let generated = greedy(&model, &restored, suffix[0], prefix)?;
            let exact = native_logits == logits && native_greedy == generated;
            if error == 0.0 {
                ensure!(
                    exact,
                    "native/restored inference differs for {label}, case {case}"
                );
            }
            let bf16_bytes = prefix
                * model.config.num_hidden_layers
                * model.config.num_key_value_heads
                * model.config.head_dim.unwrap()
                * 2
                * 2;
            let (apparent, allocated) = physical(&db_path)?;
            let row = json!({"case":case,"offset":offset,"prefix_tokens":prefix,"scored_tokens":STEPS,
                "mode":label,"error":error,"native_restore_exact":exact,
                "quality":metrics(&logits,&reference,&suffix[1..]),
                "native_quality":metrics(&native_logits,&reference,&suffix[1..]),
                "greedy_matches_bf16":generated.iter().zip(&reference_greedy).filter(|(a,b)|a==b).count(),
                "greedy_tokens":GREEDY,"greedy_text":tokenizer.decode(&generated,true).map_err(anyhow::Error::msg)?,
                "bf16_bytes":bf16_bytes,"native_packed_bytes":packed_bytes,
                "logical_roaring_bytes":logical,"bf16_over_logical":bf16_bytes as f64/logical as f64,
                "database_apparent_bytes":apparent,"database_allocated_bytes":allocated,
                "prefill_ms":prefill_ms,"export_ms":export_ms,"persist_ms":persist_ms,
                "reopen_read_ms":reopen_read_ms,"restore_ms":restore_ms,"storage":codec_reports});
            eprintln!(
                "case={case} mode={label} exact={exact} ratio={:.3} ppl={:.3}",
                bf16_bytes as f64 / logical as f64,
                row["quality"]["subset_perplexity"].as_f64().unwrap()
            );
            rows.push(row);
        }
    }
    let result = json!({"model_revision":serde_json::from_slice::<Value>(&fs::read(directory.join("revision.json"))?)?,
        "format":format,"model_fingerprint":model_fingerprint,"corpus_sha256":corpus_sha,
        "steps":STEPS,"greedy_steps":GREEDY,"cases":rows});
    fs::write(
        out.join(format!("{mode}.json")),
        serde_json::to_vec_pretty(&result)?,
    )?;
    Ok(())
}
