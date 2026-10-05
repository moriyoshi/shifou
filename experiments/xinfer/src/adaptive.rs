//! Calibrate K/V tile precision on one prefix, then evaluate on disjoint text.
use std::{fs, path::Path, time::Instant};

use anyhow::{ensure, Context, Result};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use shifou::{decode, encode, Address, Cache, Policy, Tensor};
use tokenizers::Tokenizer;

use crate::{fingerprint, greedy, log_z, metrics, model::Model, model::NativeCache, physical};

const ERRORS: [f64; 4] = [0.0, 0.05, 0.2, 0.5];
const CAL_TOKENS: usize = 8;
const EVAL_TOKENS: usize = 32;

#[derive(Clone)]
struct Choice {
    error: f64,
    bytes: usize,
    sensitivity: f64,
    tile: Tensor,
}

fn policy(index: usize, error: f64) -> Policy {
    let mut p = if index.is_multiple_of(2) {
        Policy::keys(error)
    } else {
        Policy::values(error)
    };
    p.group_size = 64;
    p
}

fn address(model_fingerprint: &str, prefix: &[u32], index: usize) -> Result<Address> {
    Ok(Address {
        namespace: "xinfer-adaptive-v1".into(),
        model_fingerprint: model_fingerprint.into(),
        prefix_fingerprint: format!("{:x}", Sha256::digest(serde_json::to_vec(prefix)?)),
        layer: (index / 2) as u32,
        slot: if index.is_multiple_of(2) {
            "key".into()
        } else {
            "value".into()
        },
    })
}

fn teacher_n(
    model: &Model,
    cache: &NativeCache,
    suffix: &[u32],
    prefix: usize,
    count: usize,
) -> Result<Vec<Vec<f32>>> {
    ensure!(suffix.len() >= count, "short suffix");
    (0..count)
        .map(|i| {
            let logits = model.forward(&suffix[i..i + 1], prefix + i, cache)?;
            ensure!(logits.iter().all(|x| x.is_finite()), "nonfinite logits");
            Ok(logits)
        })
        .collect()
}

fn mean_kl(logits: &[Vec<f32>], reference: &[Vec<f32>]) -> Result<f64> {
    ensure!(
        !logits.is_empty() && logits.len() == reference.len(),
        "KL shape mismatch"
    );
    let mut total = 0.0;
    for (candidate, baseline) in logits.iter().zip(reference) {
        ensure!(candidate.len() == baseline.len(), "KL vocab mismatch");
        let za = log_z(candidate);
        let zb = log_z(baseline);
        for (&a, &b) in candidate.iter().zip(baseline) {
            let logp = b as f64 - zb;
            total += logp.exp() * (logp - (a as f64 - za));
        }
    }
    Ok((total / logits.len() as f64).max(0.0))
}

/// Each tile starts at its cheapest candidate. Buy the greatest calibrated KL
/// reduction per additional byte until no affordable positive gain remains.
fn select(choices: &[Vec<Choice>], budget: usize) -> Vec<usize> {
    let mut selected: Vec<usize> = choices
        .iter()
        .map(|variants| {
            variants
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    a.bytes
                        .cmp(&b.bytes)
                        .then_with(|| a.sensitivity.total_cmp(&b.sensitivity))
                })
                .unwrap()
                .0
        })
        .collect();
    let mut used: usize = selected
        .iter()
        .enumerate()
        .map(|(i, &j)| choices[i][j].bytes)
        .sum();
    assert!(used <= budget, "uniform budget below minimum possible size");
    loop {
        let mut best: Option<(usize, usize, f64, usize)> = None;
        for (i, variants) in choices.iter().enumerate() {
            let current = &variants[selected[i]];
            for (j, next) in variants.iter().enumerate() {
                if next.bytes <= current.bytes
                    || next.bytes - current.bytes > budget - used
                    || next.sensitivity >= current.sensitivity
                {
                    continue;
                }
                let extra = next.bytes - current.bytes;
                let value = (current.sensitivity - next.sensitivity) / extra as f64;
                if best.as_ref().is_none_or(|(_, _, old, old_extra)| {
                    value > *old || (value == *old && extra < *old_extra)
                }) {
                    best = Some((i, j, value, extra));
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

fn candidate(
    source: &Tensor,
    index: usize,
    error: f64,
    addr: &Address,
    sensitivity: f64,
) -> Result<Choice> {
    let encoded = encode(source, &policy(index, error))?;
    let bytes = Cache::estimate_encoded(addr, &encoded)?.total_roaring_bytes;
    let tile = decode(&encoded)?;
    Ok(Choice {
        error,
        bytes,
        sensitivity,
        tile,
    })
}

fn eval_choice(
    model: &Model,
    choices: &[Vec<Choice>],
    indexes: &[usize],
    prefix: usize,
    suffix: &[u32],
) -> Result<(Vec<Vec<f32>>, Vec<u32>)> {
    let tiles: Vec<_> = choices
        .iter()
        .zip(indexes)
        .map(|(row, &j)| row[j].tile.clone())
        .collect();
    let cache = model.restore(&tiles, prefix + EVAL_TOKENS)?;
    let logits = teacher_n(model, &cache, suffix, prefix, EVAL_TOKENS)?;
    let generated = greedy(model, &cache, suffix[0], prefix)?;
    Ok((logits, generated))
}

fn joint_kl(
    model: &Model,
    train_variants: &[Vec<Tensor>],
    indexes: &[usize],
    suffix: &[u32],
    reference: &[Vec<f32>],
) -> Result<f64> {
    let tiles: Vec<_> = train_variants
        .iter()
        .zip(indexes)
        .map(|(row, &j)| row[j].clone())
        .collect();
    let cache = model.restore(&tiles, 128 + EVAL_TOKENS)?;
    let logits = teacher_n(model, &cache, suffix, 128, CAL_TOKENS)?;
    mean_kl(&logits, reference)
}

struct JointCalibration<'a> {
    model: &'a Model,
    variants: &'a [Vec<Tensor>],
    suffix: &'a [u32],
    reference: &'a [Vec<f32>],
}

impl JointCalibration<'_> {
    fn score(&self, indexes: &[usize]) -> Result<f64> {
        joint_kl(
            self.model,
            self.variants,
            indexes,
            self.suffix,
            self.reference,
        )
    }
}

/// Shortlist changes by isolated KL, but accept only if whole-cache calibration KL improves.
fn refine_joint(
    calibration: &JointCalibration<'_>,
    choices: &[Vec<Choice>],
    proposed: Vec<usize>,
    uniform: &[usize],
    budget: usize,
) -> Result<(Vec<usize>, f64, f64, usize)> {
    let uniform_kl = calibration.score(uniform)?;
    let proposed_kl = calibration.score(&proposed)?;
    let (mut selected, mut current_kl) = if proposed_kl < uniform_kl {
        (proposed, proposed_kl)
    } else {
        (uniform.to_vec(), uniform_kl)
    };
    let mut trials = 2usize;
    for _ in 0..2 {
        let used: usize = selected
            .iter()
            .enumerate()
            .map(|(i, &j)| choices[i][j].bytes)
            .sum();
        let mut proposals = Vec::new();
        for i in 0..choices.len() {
            for a in 0..choices[i].len() {
                if a == selected[i] {
                    continue;
                }
                let old_i = &choices[i][selected[i]];
                let next_i = &choices[i][a];
                let new_size_i = used - old_i.bytes + next_i.bytes;
                let gain_i = old_i.sensitivity - next_i.sensitivity;
                if new_size_i <= budget {
                    proposals.push((gain_i, i, a, None));
                }
                for j in i + 1..choices.len() {
                    let old_j = &choices[j][selected[j]];
                    for (b, next_j) in choices[j].iter().enumerate() {
                        if b == selected[j] {
                            continue;
                        }
                        let new_size = new_size_i - old_j.bytes + next_j.bytes;
                        if new_size <= budget {
                            proposals.push((
                                gain_i + old_j.sensitivity - next_j.sensitivity,
                                i,
                                a,
                                Some((j, b)),
                            ));
                        }
                    }
                }
            }
        }
        proposals.sort_by(|a, b| b.0.total_cmp(&a.0));
        let mut best = None;
        for (_, i, a, other) in proposals.into_iter().take(40) {
            let mut trial = selected.clone();
            trial[i] = a;
            if let Some((j, b)) = other {
                trial[j] = b;
            }
            let score = calibration.score(&trial)?;
            trials += 1;
            if score + 1e-8 < current_kl
                && best
                    .as_ref()
                    .is_none_or(|(_, best_score)| score < *best_score)
            {
                best = Some((trial, score));
            }
        }
        if let Some((trial, score)) = best {
            selected = trial;
            current_kl = score;
        } else {
            break;
        }
    }
    Ok((selected, uniform_kl, current_kl, trials))
}

pub fn run(directory: &Path, corpus: &Path, out: &Path, joint: bool) -> Result<()> {
    let result_name = if joint {
        "adaptive-joint.json"
    } else {
        "adaptive.json"
    };
    ensure!(
        !out.join(result_name).exists(),
        "adaptive result already exists"
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
        tokens.len() > 4096 + 512 + EVAL_TOKENS + 1,
        "corpus too short"
    );
    let model = Model::load(directory, "auto")?;
    let calibration_format = if joint {
        "adaptive-calibration-v1;joint=true"
    } else {
        "adaptive-calibration-v1"
    };
    let fingerprint = fingerprint(
        directory,
        &format!("{};{calibration_format}", model.format()),
    )?;
    // All precision decisions use offset 0. Offsets 4096+ are evaluation only.
    let train_prefix = &tokens[..128];
    let train_suffix = &tokens[128..128 + CAL_TOKENS];
    let train_cache = model.empty_cache(128 + EVAL_TOKENS)?;
    model.forward(train_prefix, 0, &train_cache)?;
    let train_tiles = model.export(&train_cache, 128)?;
    let train_reference = teacher_n(&model, &train_cache, train_suffix, 128, CAL_TOKENS)?;
    let calibration_start = Instant::now();
    let mut sensitivities = Vec::new();
    let mut train_variants = Vec::new();
    for (index, source) in train_tiles.iter().enumerate() {
        let mut row = Vec::new();
        let mut variants = Vec::new();
        for &error in &ERRORS {
            if error == 0.0 {
                row.push(0.0);
                variants.push(source.clone());
                continue;
            }
            let encoded = encode(source, &policy(index, error))?;
            let decoded = decode(&encoded)?;
            let perturbed = model.replace_tile(&train_cache, index, &decoded, 128 + EVAL_TOKENS)?;
            let logits = teacher_n(&model, &perturbed, train_suffix, 128, CAL_TOKENS)?;
            row.push(mean_kl(&logits, &train_reference)?);
            variants.push(decoded);
        }
        eprintln!("calibrated {}/{} tiles", index + 1, train_tiles.len());
        sensitivities.push(row);
        train_variants.push(variants);
    }
    let calibration_ms = calibration_start.elapsed().as_secs_f64() * 1000.0;
    let uniform_index = ERRORS.iter().position(|&x| x == 0.2).unwrap();
    let mut rows = Vec::new();
    for &prefix in &[128usize, 512] {
        let offset = 4096;
        let current = &tokens[offset..offset + prefix];
        let suffix = &tokens[offset + prefix..offset + prefix + EVAL_TOKENS + 1];
        let native = model.empty_cache(prefix + EVAL_TOKENS)?;
        model.forward(current, 0, &native)?;
        let native_tiles = model.export(&native, prefix)?;
        let reference = teacher_n(&model, &native, suffix, prefix, EVAL_TOKENS)?;
        let reference_greedy = greedy(&model, &native, suffix[0], prefix)?;
        let candidates_start = Instant::now();
        let mut choices = Vec::new();
        for (index, source) in native_tiles.iter().enumerate() {
            let addr = address(&fingerprint, current, index)?;
            let row = ERRORS
                .iter()
                .enumerate()
                .map(|(j, &error)| candidate(source, index, error, &addr, sensitivities[index][j]))
                .collect::<Result<Vec<_>>>()?;
            choices.push(row);
        }
        let uniform = vec![uniform_index; choices.len()];
        let budget: usize = choices.iter().map(|row| row[uniform_index].bytes).sum();
        let proxy_selected = select(&choices, budget);
        let (selected, joint_uniform_kl, joint_selected_kl, joint_trials) = if joint {
            let calibration = JointCalibration {
                model: &model,
                variants: &train_variants,
                suffix: train_suffix,
                reference: &train_reference,
            };
            let (selected, uniform_kl, chosen_kl, trials) =
                refine_joint(&calibration, &choices, proxy_selected, &uniform, budget)?;
            (selected, Some(uniform_kl), Some(chosen_kl), trials)
        } else {
            (proxy_selected, None, None, 0)
        };
        let selected_bytes: usize = selected
            .iter()
            .enumerate()
            .map(|(i, &j)| choices[i][j].bytes)
            .sum();
        ensure!(
            selected_bytes <= budget,
            "selector exceeded exact byte budget"
        );
        let candidate_ms = candidates_start.elapsed().as_secs_f64() * 1000.0;
        let (uniform_logits, uniform_greedy) =
            eval_choice(&model, &choices, &uniform, prefix, suffix)?;
        let (selected_logits, selected_greedy) =
            eval_choice(&model, &choices, &selected, prefix, suffix)?;
        let db_path = out.join(format!(
            "{}-cache-{prefix}",
            if joint { "adaptive-joint" } else { "adaptive" }
        ));
        ensure!(!db_path.exists(), "adaptive cache directory exists");
        let now = Instant::now();
        let mut actual_bytes = 0usize;
        {
            let mut store = Cache::open(&db_path)?;
            for (index, source) in native_tiles.iter().enumerate() {
                let addr = address(&fingerprint, current, index)?;
                let r = store.put(
                    &addr,
                    source,
                    &policy(index, choices[index][selected[index]].error),
                )?;
                ensure!(
                    r.total_roaring_bytes == choices[index][selected[index]].bytes,
                    "estimated and stored sizes disagree"
                );
                actual_bytes += r.total_roaring_bytes;
            }
        }
        ensure!(
            actual_bytes == selected_bytes,
            "sum of persisted bytes differs"
        );
        let persist_ms = now.elapsed().as_secs_f64() * 1000.0;
        let now = Instant::now();
        let stored_tiles = {
            let store = Cache::open(&db_path)?;
            let mut tiles = Vec::new();
            for index in 0..choices.len() {
                let addr = address(&fingerprint, current, index)?;
                let got = store.get(&addr)?.context("missing calibrated tile")?.tensor;
                ensure!(
                    got.values
                        .iter()
                        .zip(&choices[index][selected[index]].tile.values)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "reopened calibrated tile changed"
                );
                tiles.push(got);
            }
            tiles
        };
        let reopen_read_ms = now.elapsed().as_secs_f64() * 1000.0;
        let restored = model.restore(&stored_tiles, prefix + EVAL_TOKENS)?;
        let stored_logits = teacher_n(&model, &restored, suffix, prefix, EVAL_TOKENS)?;
        let stored_greedy = greedy(&model, &restored, suffix[0], prefix)?;
        ensure!(
            stored_logits == selected_logits && stored_greedy == selected_greedy,
            "persistence changed calibrated inference"
        );
        let (apparent, allocated) = physical(&db_path)?;
        let bf16_bytes = prefix
            * model.config.num_hidden_layers
            * model.config.num_key_value_heads
            * model.config.head_dim.unwrap()
            * 4;
        let mut choices_summary = Vec::new();
        for (index, &j) in selected.iter().enumerate() {
            choices_summary.push(
                json!({"layer":index/2,"slot":if index%2==0{"key"}else{"value"},
                "error":choices[index][j].error,"bytes":choices[index][j].bytes,
                "calibration_kl":choices[index][j].sensitivity,
                "uniform_kl":choices[index][uniform_index].sensitivity}),
            );
        }
        let proxy_selected: f64 = selected
            .iter()
            .enumerate()
            .map(|(i, &j)| choices[i][j].sensitivity)
            .sum();
        let proxy_uniform: f64 = choices
            .iter()
            .map(|row| row[uniform_index].sensitivity)
            .sum();
        let row = json!({"offset":offset,"prefix_tokens":prefix,"scored_tokens":EVAL_TOKENS,
            "byte_budget":budget,"adaptive_bytes":selected_bytes,
            "bf16_bytes":bf16_bytes,"bf16_over_adaptive":bf16_bytes as f64/selected_bytes as f64,
            "bf16_over_uniform":bf16_bytes as f64/budget as f64,
            "uniform_error":0.2,
            "uniform_quality":metrics(&uniform_logits,&reference,&suffix[1..]),
            "adaptive_quality":metrics(&selected_logits,&reference,&suffix[1..]),
            "uniform_greedy_matches_bf16":uniform_greedy.iter().zip(&reference_greedy).filter(|(a,b)|a==b).count(),
            "adaptive_greedy_matches_bf16":selected_greedy.iter().zip(&reference_greedy).filter(|(a,b)|a==b).count(),
            "adaptive_persistence_exact":true,
            "proxy_kl_uniform_sum":proxy_uniform,"proxy_kl_selected_sum":proxy_selected,
            "joint_calibration_uniform_kl":joint_uniform_kl,
            "joint_calibration_selected_kl":joint_selected_kl,
            "joint_calibration_trials":joint_trials,
            "candidate_ms":candidate_ms,"persist_ms":persist_ms,"reopen_read_ms":reopen_read_ms,
            "database_apparent_bytes":apparent,"database_allocated_bytes":allocated,
            "choices":choices_summary});
        eprintln!(
            "adaptive prefix={prefix} bytes={selected_bytes}/{budget} KL={} uniform_KL={}",
            row["adaptive_quality"]["kl_from_bf16"], row["uniform_quality"]["kl_from_bf16"]
        );
        rows.push(row);
    }
    let report = json!({"method":if joint {
        "single-tile KL sensitivity; greedy byte-budget upgrades; joint calibration search"
        } else { "single-tile KL sensitivity; greedy byte-budget upgrades" },
        "calibration":{"offset":0,"prefix_tokens":128,"scored_tokens":CAL_TOKENS,
            "calibration_ms":calibration_ms,"sensitivities":sensitivities},
        "model_revision":serde_json::from_slice::<Value>(&fs::read(directory.join("revision.json"))?)?,
        "model_fingerprint":fingerprint,"corpus_sha256":corpus_sha,"heldout":rows});
    fs::write(out.join(result_name), serde_json::to_vec_pretty(&report)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn choice(bytes: usize, sensitivity: f64) -> Choice {
        Choice {
            error: 0.0,
            bytes,
            sensitivity,
            tile: Tensor {
                shape: vec![1],
                axes: vec!["channel".into()],
                values: vec![0.0],
            },
        }
    }
    #[test]
    fn selector_respects_exact_budget_and_buys_best_gain() {
        let rows = vec![
            vec![choice(2, 1.0), choice(4, 0.0)],
            vec![choice(2, 0.4), choice(4, 0.0)],
        ];
        assert_eq!(select(&rows, 6), vec![1, 0]);
        assert_eq!(select(&rows, 4), vec![0, 0]);
    }
}
