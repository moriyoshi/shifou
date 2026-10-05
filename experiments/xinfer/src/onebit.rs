//! Calibrate packed 1/2/4-bit KV tiles on two contexts, evaluate on disjoint text.
use std::{fs, path::Path, time::Instant};

use anyhow::{ensure, Context, Result};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use shifou::{
    decode_compact, encode_compact, Address, Cache, CompactPolicy, PackedSnapshot, Tensor,
};
use tokenizers::Tokenizer;

use crate::{
    fingerprint, greedy, log_z, metrics,
    model::{Model, NativeCache},
};

const BITS: [u8; 3] = [1, 2, 4];
const CAL_TOKENS: usize = 8;
const EVAL_TOKENS: usize = 32;
const CAL_OFFSETS: [usize; 2] = [0, 1024];
const HELDOUT_OFFSET: usize = 4096;

#[derive(Clone)]
struct Choice {
    bytes: usize,
    packed_bytes: usize,
    snapshot: PackedSnapshot,
    tile: Tensor,
}

struct TrainCase {
    rows: Vec<Vec<Tensor>>,
    suffix: Vec<u32>,
    reference: Vec<Vec<f32>>,
}

fn policy(index: usize, bits: u8) -> CompactPolicy {
    if index.is_multiple_of(2) {
        CompactPolicy::keys(bits)
    } else {
        CompactPolicy::values(bits)
    }
}

fn address(fp: &str, prefix: &[u32], index: usize) -> Result<Address> {
    Ok(Address {
        namespace: "xinfer-compact-1bit-v1".into(),
        model_fingerprint: fp.into(),
        prefix_fingerprint: format!("{:x}", Sha256::digest(serde_json::to_vec(prefix)?)),
        layer: (index / 2) as u32,
        slot: if index.is_multiple_of(2) {
            "key".into()
        } else {
            "value".into()
        },
    })
}

fn choice(source: &Tensor, index: usize, bits: u8, addr: &Address) -> Result<Choice> {
    let snapshot = encode_compact(source, &policy(index, bits))?;
    let report = Cache::estimate_packed(addr, &snapshot)?;
    let tile = decode_compact(&snapshot)?;
    Ok(Choice {
        bytes: report.total_roaring_bytes,
        packed_bytes: report.packed_bytes,
        snapshot,
        tile,
    })
}

fn teacher(
    model: &Model,
    cache: &NativeCache,
    suffix: &[u32],
    prefix: usize,
    n: usize,
) -> Result<Vec<Vec<f32>>> {
    (0..n)
        .map(|i| {
            let row = model.forward(&suffix[i..i + 1], prefix + i, cache)?;
            ensure!(row.iter().all(|x| x.is_finite()), "nonfinite logits");
            Ok(row)
        })
        .collect()
}

fn mean_kl(rows: &[Vec<f32>], reference: &[Vec<f32>]) -> f64 {
    let mut sum = 0.0;
    for (a, b) in rows.iter().zip(reference) {
        let za = log_z(a);
        let zb = log_z(b);
        for (&x, &y) in a.iter().zip(b) {
            let lp = y as f64 - zb;
            sum += lp.exp() * (lp - (x as f64 - za));
        }
    }
    (sum / rows.len() as f64).max(0.0)
}

fn bytes_for(rows: &[Vec<Choice>], picks: &[usize]) -> usize {
    rows.iter().zip(picks).map(|(r, &j)| r[j].bytes).sum()
}

fn packed_for(rows: &[Vec<Choice>], picks: &[usize]) -> usize {
    rows.iter()
        .zip(picks)
        .map(|(r, &j)| r[j].packed_bytes)
        .sum()
}

fn allocate(
    rows: &[Vec<Choice>],
    scores: &[[f64; 3]],
    budget: usize,
    value_only_one: bool,
) -> Vec<usize> {
    let allowed = |i: usize, j: usize| !value_only_one || !i.is_multiple_of(2) || j >= 1;
    let mut selected: Vec<usize> = rows
        .iter()
        .enumerate()
        .map(|(i, r)| {
            (0..3)
                .filter(|&j| allowed(i, j))
                .min_by_key(|&j| r[j].bytes)
                .unwrap()
        })
        .collect();
    let mut used = bytes_for(rows, &selected);
    assert!(used <= budget, "budget below cheapest allowed policy");
    loop {
        let mut best: Option<(usize, usize, f64, usize)> = None;
        for (i, row) in rows.iter().enumerate() {
            let old = selected[i];
            for j in 0..3 {
                if !allowed(i, j) || row[j].bytes < row[old].bytes || scores[i][j] >= scores[i][old]
                {
                    continue;
                }
                let extra = row[j].bytes - row[old].bytes;
                if extra > budget - used {
                    continue;
                }
                let benefit = if extra == 0 {
                    f64::INFINITY
                } else {
                    (scores[i][old] - scores[i][j]) / extra as f64
                };
                if best.as_ref().is_none_or(|(_, _, prior, cost)| {
                    benefit > *prior || (benefit == *prior && extra < *cost)
                }) {
                    best = Some((i, j, benefit, extra));
                }
            }
        }
        if let Some((i, j, _, extra)) = best {
            selected[i] = j;
            used += extra;
        } else {
            break;
        }
    }
    selected
}

fn dispersed(rows: &[Vec<Choice>], budget: usize) -> Vec<usize> {
    let mut picks = vec![0; rows.len()];
    let mut used = bytes_for(rows, &picks);
    assert!(used <= budget);
    let mut order: Vec<usize> = (0..rows.len()).collect();
    order.sort_by_key(|&i| (i * 17) % rows.len());
    for i in order {
        if rows[i][1].bytes <= rows[i][0].bytes {
            picks[i] = 1;
        } else {
            let extra = rows[i][1].bytes - rows[i][0].bytes;
            if extra <= budget - used {
                picks[i] = 1;
                used += extra;
            }
        }
    }
    picks
}

/// Admit a few one-bit tiles from a four-bit base under a fixed calibration KL cap.
/// Candidate bytes come from the held-out prefix, while quality uses calibration
/// contexts only. The held-out logits are never used to choose tiles.
fn sparse_onebit(
    model: &Model,
    cases: &[TrainCase],
    rows: &[Vec<Choice>],
    scores: &[[f64; 3]],
    baseline_kl: f64,
) -> Result<(Vec<usize>, f64, usize)> {
    const MAX_TILES: usize = 6;
    const SHORTLIST: usize = 20;
    const EXTRA_KL: f64 = 0.01;
    let mut picks = vec![2; rows.len()];
    let mut current_kl = baseline_kl;
    let mut trials = 0;
    for _ in 0..MAX_TILES {
        let mut candidates: Vec<usize> = (0..rows.len())
            .filter(|&i| picks[i] == 2 && rows[i][0].bytes < rows[i][2].bytes)
            .collect();
        candidates.sort_by(|&a, &b| {
            let proxy = |i: usize| {
                (scores[i][0] - scores[i][2]).max(0.0)
                    / (rows[i][2].bytes - rows[i][0].bytes) as f64
            };
            proxy(a).total_cmp(&proxy(b))
        });
        let mut best: Option<(usize, usize, f64)> = None;
        for &i in candidates.iter().take(SHORTLIST) {
            picks[i] = 0;
            let kl = joint_kl(model, cases, &picks)?;
            trials += 1;
            picks[i] = 2;
            if kl <= baseline_kl + EXTRA_KL {
                let saved = rows[i][2].bytes - rows[i][0].bytes;
                if best.as_ref().is_none_or(|(_, prior_saved, prior_kl)| {
                    saved > *prior_saved || (saved == *prior_saved && kl < *prior_kl)
                }) {
                    best = Some((i, saved, kl));
                }
            }
        }
        if let Some((i, _, kl)) = best {
            picks[i] = 0;
            current_kl = kl;
        } else {
            break;
        }
    }
    Ok((picks, current_kl, trials))
}

fn joint_kl(model: &Model, cases: &[TrainCase], picks: &[usize]) -> Result<f64> {
    let mut total = 0.0;
    for case in cases {
        let tiles: Vec<Tensor> = case
            .rows
            .iter()
            .zip(picks)
            .map(|(r, &j)| r[j].clone())
            .collect();
        let restored = model.restore(&tiles, 128 + EVAL_TOKENS)?;
        let logits = teacher(model, &restored, &case.suffix, 128, CAL_TOKENS)?;
        total += mean_kl(&logits, &case.reference);
    }
    Ok(total / cases.len() as f64)
}

fn eval(
    model: &Model,
    rows: &[Vec<Choice>],
    picks: &[usize],
    prefix: usize,
    suffix: &[u32],
    reference: &[Vec<f32>],
    reference_greedy: &[u32],
) -> Result<(Value, Vec<Vec<f32>>, Vec<u32>)> {
    let tiles: Vec<Tensor> = rows
        .iter()
        .zip(picks)
        .map(|(r, &j)| r[j].tile.clone())
        .collect();
    let restored = model.restore(&tiles, prefix + EVAL_TOKENS)?;
    let logits = teacher(model, &restored, suffix, prefix, EVAL_TOKENS)?;
    let generated = greedy(model, &restored, suffix[0], prefix)?;
    let matches = generated
        .iter()
        .zip(reference_greedy)
        .filter(|(a, b)| a == b)
        .count();
    Ok((
        json!({"quality":metrics(&logits,reference,&suffix[1..]),"greedy_matches_bf16":matches}),
        logits,
        generated,
    ))
}

struct PersistCase<'a> {
    out: &'a Path,
    name: &'a str,
    model: &'a Model,
    fp: &'a str,
    prefix: &'a [u32],
    suffix: &'a [u32],
    rows: &'a [Vec<Choice>],
    picks: &'a [usize],
    expected_logits: &'a [Vec<f32>],
    expected_greedy: &'a [u32],
}

fn persist(case: &PersistCase<'_>) -> Result<Value> {
    let db = case
        .out
        .join(format!("{}-cache-{}", case.name, case.prefix.len()));
    ensure!(!db.exists(), "compact cache directory exists");
    let started = Instant::now();
    let mut stored_bytes = 0;
    {
        let mut store = Cache::open(&db)?;
        // **One batch for the whole cache.** Writing tile by tile paid a commit,
        // a visibility wait and a checkpoint apiece -- 56 of each -- and left the
        // cache partly persisted if anything failed part way, since each tile is
        // individually well formed and carries its own manifest. `put_packed_many`
        // is one collision view, one batch, one commit, one checkpoint, and is
        // all-or-nothing.
        let mut items = Vec::with_capacity(case.rows.len());
        for (index, row) in case.rows.iter().enumerate() {
            items.push((
                address(case.fp, case.prefix, index)?,
                &row[case.picks[index]].snapshot,
            ));
        }
        let reports = store.put_packed_many(&items)?;
        ensure!(
            reports.len() == case.rows.len(),
            "one report per tile, in order"
        );
        for (index, (row, report)) in case.rows.iter().zip(&reports).enumerate() {
            let option = &row[case.picks[index]];
            ensure!(
                report.total_roaring_bytes == option.bytes
                    && report.packed_bytes == option.packed_bytes,
                "packed estimate differs from stored"
            );
            stored_bytes += report.total_roaring_bytes;
        }
    }
    let persist_ms = started.elapsed().as_secs_f64() * 1000.0;
    ensure!(
        stored_bytes == bytes_for(case.rows, case.picks),
        "stored size differs"
    );
    let started = Instant::now();
    let tiles = {
        let store = Cache::open(&db)?;
        let mut tiles = Vec::new();
        for (index, row) in case.rows.iter().enumerate() {
            let addr = address(case.fp, case.prefix, index)?;
            let option = &row[case.picks[index]];
            let saved = store
                .get_packed(&addr, &option.snapshot.format)?
                .context("missing compact tile")?;
            let tile = decode_compact(&saved)?;
            ensure!(
                tile.shape == option.tile.shape
                    && tile.axes == option.tile.axes
                    && tile
                        .values
                        .iter()
                        .zip(&option.tile.values)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                "reopened compact tensor changed"
            );
            tiles.push(tile);
        }
        tiles
    };
    let reopen_ms = started.elapsed().as_secs_f64() * 1000.0;
    let restored = case
        .model
        .restore(&tiles, case.prefix.len() + EVAL_TOKENS)?;
    let logits = teacher(
        case.model,
        &restored,
        case.suffix,
        case.prefix.len(),
        EVAL_TOKENS,
    )?;
    let generated = greedy(case.model, &restored, case.suffix[0], case.prefix.len())?;
    ensure!(
        logits == case.expected_logits && generated == case.expected_greedy,
        "compact persistence changed inference"
    );
    Ok(json!({"logical_bytes":stored_bytes,"persist_ms":persist_ms,
        "reopen_read_ms":reopen_ms,"inference_exact_after_reopen":true}))
}

pub fn run(directory: &Path, corpus: &Path, out: &Path) -> Result<()> {
    ensure!(
        !out.join("compact-onebit.json").exists(),
        "compact 1-bit result already exists"
    );
    fs::create_dir_all(out)?;
    let text = fs::read_to_string(corpus)?;
    let corpus_sha = format!("{:x}", Sha256::digest(text.as_bytes()));
    let tokenizer =
        Tokenizer::from_file(directory.join("tokenizer.json")).map_err(anyhow::Error::msg)?;
    let tokens = tokenizer
        .encode(text, false)
        .map_err(anyhow::Error::msg)?
        .get_ids()
        .to_vec();
    ensure!(
        tokens.len() > HELDOUT_OFFSET + 512 + EVAL_TOKENS + 1,
        "corpus too short"
    );
    let model = Model::load(directory, "auto")?;
    let fp = fingerprint(directory, &format!("{};compact-kv-v1", model.format()))?;
    let started = Instant::now();
    let mut scores = vec![[0.0; 3]; model.config.num_hidden_layers * 2];
    let mut train_cases = Vec::new();
    for offset in CAL_OFFSETS {
        let prefix = &tokens[offset..offset + 128];
        let suffix = &tokens[offset + 128..offset + 128 + CAL_TOKENS];
        let native = model.empty_cache(128 + EVAL_TOKENS)?;
        model.forward(prefix, 0, &native)?;
        let source = model.export(&native, 128)?;
        let reference = teacher(&model, &native, suffix, 128, CAL_TOKENS)?;
        let mut rows = Vec::new();
        for (index, tile) in source.iter().enumerate() {
            let mut choices = Vec::new();
            for (j, &bits) in BITS.iter().enumerate() {
                let snapshot = encode_compact(tile, &policy(index, bits))?;
                let decoded = decode_compact(&snapshot)?;
                let changed = model.replace_tile(&native, index, &decoded, 128 + EVAL_TOKENS)?;
                let logits = teacher(&model, &changed, suffix, 128, CAL_TOKENS)?;
                scores[index][j] += mean_kl(&logits, &reference) / CAL_OFFSETS.len() as f64;
                choices.push(decoded);
            }
            rows.push(choices);
            if index % 8 == 7 {
                eprintln!(
                    "compact calibration offset {offset}: {}/{} tiles",
                    index + 1,
                    source.len()
                );
            }
        }
        train_cases.push(TrainCase {
            rows,
            suffix: suffix.to_vec(),
            reference,
        });
    }
    let calibration_ms = started.elapsed().as_secs_f64() * 1000.0;
    eprintln!(
        "compact calibration done in {:.1} s",
        calibration_ms / 1000.0
    );
    let mut heldout = Vec::new();
    for &prefix_len in &[128usize, 512] {
        let prefix = &tokens[HELDOUT_OFFSET..HELDOUT_OFFSET + prefix_len];
        let suffix =
            &tokens[HELDOUT_OFFSET + prefix_len..HELDOUT_OFFSET + prefix_len + EVAL_TOKENS + 1];
        let native = model.empty_cache(prefix_len + EVAL_TOKENS)?;
        model.forward(prefix, 0, &native)?;
        let source = model.export(&native, prefix_len)?;
        let reference = teacher(&model, &native, suffix, prefix_len, EVAL_TOKENS)?;
        let reference_greedy = greedy(&model, &native, suffix[0], prefix_len)?;
        let started = Instant::now();
        let mut rows = Vec::new();
        for (index, tile) in source.iter().enumerate() {
            let addr = address(&fp, prefix, index)?;
            rows.push(
                BITS.iter()
                    .map(|&bits| choice(tile, index, bits, &addr))
                    .collect::<Result<Vec<_>>>()?,
            );
            if index % 8 == 7 {
                eprintln!(
                    "compact prefix {prefix_len}: encoded {}/{} tiles",
                    index + 1,
                    source.len()
                );
            }
        }
        let candidate_ms = started.elapsed().as_secs_f64() * 1000.0;
        let uniform1 = vec![0; rows.len()];
        let uniform2 = vec![1; rows.len()];
        let uniform4 = vec![2; rows.len()];
        let k2v1: Vec<usize> = (0..rows.len())
            .map(|i| if i % 2 == 0 { 1 } else { 0 })
            .collect();
        let k4v1: Vec<usize> = (0..rows.len())
            .map(|i| if i % 2 == 0 { 2 } else { 0 })
            .collect();
        let budget2 = bytes_for(&rows, &uniform2);
        let mid_budget = bytes_for(&rows, &uniform1) + (budget2 - bytes_for(&rows, &uniform1)) / 2;
        let proposed_mid = allocate(&rows, &scores, mid_budget, false);
        let dispersed_mid = dispersed(&rows, mid_budget);
        let mid_proposed_kl = joint_kl(&model, &train_cases, &proposed_mid)?;
        let mid_dispersed_kl = joint_kl(&model, &train_cases, &dispersed_mid)?;
        let adaptive_mid = if mid_proposed_kl <= mid_dispersed_kl {
            proposed_mid
        } else {
            dispersed_mid.clone()
        };
        let proposed_u2 = allocate(&rows, &scores, budget2, false);
        let u2_uniform_kl = joint_kl(&model, &train_cases, &uniform2)?;
        let u2_proposed_kl = joint_kl(&model, &train_cases, &proposed_u2)?;
        let adaptive_u2 = if u2_proposed_kl <= u2_uniform_kl {
            proposed_u2
        } else {
            uniform2.clone()
        };
        let proposed_v = allocate(&rows, &scores, budget2, true);
        let v_proposed_kl = joint_kl(&model, &train_cases, &proposed_v)?;
        let adaptive_v = if v_proposed_kl <= u2_uniform_kl {
            proposed_v
        } else {
            uniform2.clone()
        };
        let uniform4_kl = joint_kl(&model, &train_cases, &uniform4)?;
        let (sparse_onebit, sparse_onebit_kl, sparse_onebit_trials) =
            sparse_onebit(&model, &train_cases, &rows, &scores, uniform4_kl)?;
        let selections = [
            ("uniform1", &uniform1, false),
            ("uniform2", &uniform2, false),
            ("uniform4", &uniform4, false),
            ("k2v1", &k2v1, true),
            ("k4v1", &k4v1, false),
            ("dispersed_mid", &dispersed_mid, false),
            ("adaptive_mid", &adaptive_mid, true),
            ("adaptive_u2", &adaptive_u2, true),
            ("adaptive_v", &adaptive_v, false),
            ("sparse_onebit", &sparse_onebit, true),
        ];
        let mut policies = serde_json::Map::new();
        for (name, picks, save) in selections {
            let (evaluation, logits, generated) = eval(
                &model,
                &rows,
                picks,
                prefix_len,
                suffix,
                &reference,
                &reference_greedy,
            )?;
            let persistence = if save {
                let case = PersistCase {
                    out,
                    name,
                    model: &model,
                    fp: &fp,
                    prefix,
                    suffix,
                    rows: &rows,
                    picks,
                    expected_logits: &logits,
                    expected_greedy: &generated,
                };
                Some(persist(&case)?)
            } else {
                None
            };
            let mut counts = [0usize; 3];
            for &j in picks {
                counts[j] += 1;
            }
            policies.insert(name.into(),json!({
                "complete_logical_bytes":bytes_for(&rows,picks),
                "packed_payload_bytes":packed_for(&rows,picks),
                "effective_stored_bits_per_source_value":8.0*bytes_for(&rows,picks) as f64
                    / (prefix_len*model.config.num_hidden_layers*model.config.num_key_value_heads
                        *model.config.head_dim.unwrap()*2) as f64,
                "choice_counts_1_2_4":counts,
                "selected_bits_by_layer_kv":picks.iter().map(|&j|BITS[j]).collect::<Vec<_>>(),
                "evaluation":evaluation,"persistence":persistence,
            }));
        }
        let row = json!({"offset":HELDOUT_OFFSET,"prefix_tokens":prefix_len,
            "bf16_cache_bytes":prefix_len*model.config.num_hidden_layers*
                model.config.num_key_value_heads*model.config.head_dim.unwrap()*4,
            "calibration_kl":{"uniform2":u2_uniform_kl,"mid_proposed":mid_proposed_kl,
                "mid_dispersed":mid_dispersed_kl,"u2_proposed":u2_proposed_kl,
                "v_proposed":v_proposed_kl,"uniform4":uniform4_kl,
                "sparse_onebit":sparse_onebit_kl,"sparse_onebit_trials":sparse_onebit_trials,
                "sparse_onebit_max_extra_kl":0.01},
            "budgets":{"uniform2":budget2,"midpoint_1_2":mid_budget},
            "candidate_ms":candidate_ms,"policies":policies});
        eprintln!(
            "compact prefix {prefix_len}: 1-bit KL={} K2V1 KL={} adaptive-mid KL={}",
            row["policies"]["uniform1"]["evaluation"]["quality"]["kl_from_bf16"],
            row["policies"]["k2v1"]["evaluation"]["quality"]["kl_from_bf16"],
            row["policies"]["adaptive_mid"]["evaluation"]["quality"]["kl_from_bf16"]
        );
        heldout.push(row);
    }
    let report = json!({"method":"packed 1/2/4-bit compact KV with shared group parameters and exact bf16/f32 tail",
        "limitations":"prefix-only codec; 1-bit uses two centroids per K channel or V token/head; no native low-bit attention kernel",
        "calibration":{"offsets":CAL_OFFSETS,"prefix_tokens":128,"scored_tokens":CAL_TOKENS,
            "calibration_ms":calibration_ms,"isolated_kl_1_2_4":scores},
        "model_revision":serde_json::from_slice::<Value>(&fs::read(directory.join("revision.json"))?)?,
        "model_fingerprint":fp,"corpus_sha256":corpus_sha,"heldout":heldout});
    fs::write(
        out.join("compact-onebit.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn dummy(bytes: usize) -> Choice {
        Choice {
            bytes,
            packed_bytes: bytes,
            snapshot: PackedSnapshot {
                format: "x".into(),
                buffers: vec![],
            },
            tile: Tensor {
                shape: vec![1],
                axes: vec!["token".into()],
                values: vec![0.0],
            },
        }
    }
    #[test]
    fn selection_obeys_budget_and_key_constraint() {
        let rows = vec![
            vec![dummy(1), dummy(2), dummy(4)],
            vec![dummy(1), dummy(2), dummy(4)],
        ];
        let scores = [[3.0, 1.0, 0.0], [2.0, 0.5, 0.0]];
        let selected = allocate(&rows, &scores, 3, false);
        assert!(bytes_for(&rows, &selected) <= 3);
        let restricted = allocate(&rows, &scores, 3, true);
        assert_eq!(restricted[0], 1);
    }
}
