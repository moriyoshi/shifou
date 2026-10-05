//! Small paper-inspired affine KV comparison on one pinned xinfer model.
//! These are controlled approximations, not authors' kernels or search code.
use std::{fs, path::Path, time::Instant};

use anyhow::{ensure, Context, Result};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use shifou::{decode, encode, Address, Cache, Policy, Tensor};
use tokenizers::Tokenizer;

use crate::{
    fingerprint, greedy, log_z, metrics,
    model::{Model, NativeCache},
};

const BITS: [u8; 3] = [2, 4, 8];
const CAL_TOKENS: usize = 8;
const EVAL_TOKENS: usize = 32;
const TAIL: usize = 32;
const GROUP: usize = 32;

#[derive(Clone)]
struct Choice {
    bits: u8,
    bytes: usize,
    mse: f64,
    tile: Tensor,
}

fn policy(tile_index: usize, bits: u8) -> Policy {
    let mut p = if tile_index.is_multiple_of(2) {
        Policy::keys(f64::MAX)
    } else {
        Policy::values(f64::MAX)
    };
    p.group_size = GROUP;
    p.residual_tokens = TAIL;
    p.outlier_fraction = 0.0;
    p.candidate_bits = vec![bits];
    p
}

fn address(fp: &str, prefix: &[u32], tile_index: usize, head: Option<usize>) -> Result<Address> {
    let slot = if tile_index.is_multiple_of(2) {
        "key"
    } else {
        "value"
    };
    Ok(Address {
        namespace: "xinfer-paper-quantizers-v1".into(),
        model_fingerprint: fp.into(),
        prefix_fingerprint: format!("{:x}", Sha256::digest(serde_json::to_vec(prefix)?)),
        layer: (tile_index / 2) as u32,
        slot: match head {
            Some(h) => format!("{slot}-head-{h}"),
            None => slot.into(),
        },
    })
}

fn choice(source: &Tensor, tile_index: usize, bits: u8, addr: &Address) -> Result<Choice> {
    let encoded = encode(source, &policy(tile_index, bits))?;
    let bytes = Cache::estimate_encoded(addr, &encoded)?.total_roaring_bytes;
    let tile = decode(&encoded)?;
    let mse = source
        .values
        .iter()
        .zip(&tile.values)
        .map(|(&a, &b)| {
            let d = a as f64 - b as f64;
            d * d
        })
        .sum::<f64>()
        / source.values.len() as f64;
    Ok(Choice {
        bits,
        bytes,
        mse,
        tile,
    })
}

fn head_tensor(tile: &Tensor, head: usize) -> Tensor {
    let tokens = tile.shape[0];
    let heads = tile.shape[1];
    let dim = tile.shape[2];
    let mut values = Vec::with_capacity(tokens * dim);
    for t in 0..tokens {
        let start = (t * heads + head) * dim;
        values.extend_from_slice(&tile.values[start..start + dim]);
    }
    Tensor {
        shape: vec![tokens, 1, dim],
        axes: tile.axes.clone(),
        values,
    }
}

fn replace_head(tile: &mut Tensor, head: usize, replacement: &Tensor) {
    let tokens = tile.shape[0];
    let heads = tile.shape[1];
    let dim = tile.shape[2];
    assert_eq!(replacement.shape, [tokens, 1, dim]);
    for t in 0..tokens {
        let start = (t * heads + head) * dim;
        tile.values[start..start + dim]
            .copy_from_slice(&replacement.values[t * dim..(t + 1) * dim]);
    }
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

fn chosen_bytes(rows: &[Vec<Choice>], selected: &[usize]) -> usize {
    rows.iter()
        .zip(selected)
        .map(|(row, &j)| row[j].bytes)
        .sum()
}

/// Independent sensitivity or weighted distortion proxy; exact record bytes.
fn allocate(rows: &[Vec<Choice>], scores: &[[f64; 3]], budget: usize) -> Vec<usize> {
    let mut selected: Vec<usize> = rows
        .iter()
        .map(|r| (0..BITS.len()).min_by_key(|&j| r[j].bytes).unwrap())
        .collect();
    let mut used = chosen_bytes(rows, &selected);
    assert!(used <= budget, "uniform four-bit budget is below minimum");
    loop {
        let mut best: Option<(usize, usize, f64, usize)> = None;
        for (i, row) in rows.iter().enumerate() {
            let old = selected[i];
            for j in 0..BITS.len() {
                if row[j].bytes < row[old].bytes || scores[i][j] >= scores[i][old] {
                    continue;
                }
                let extra = row[j].bytes - row[old].bytes;
                if extra > budget - used {
                    continue;
                }
                let gain = if extra == 0 {
                    f64::INFINITY
                } else {
                    (scores[i][old] - scores[i][j]) / extra as f64
                };
                if best.as_ref().is_none_or(|(_, _, old_gain, old_extra)| {
                    gain > *old_gain || (gain == *old_gain && extra < *old_extra)
                }) {
                    best = Some((i, j, gain, extra));
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

fn joint_score(
    model: &Model,
    rows: &[Vec<Choice>],
    selected: &[usize],
    suffix: &[u32],
    reference: &[Vec<f32>],
) -> Result<f64> {
    let tiles = make_tiles(rows, selected, None);
    let cache = model.restore(&tiles, 128 + EVAL_TOKENS)?;
    Ok(mean_kl(
        &teacher(model, &cache, suffix, 128, CAL_TOKENS)?,
        reference,
    ))
}

/// Screen layer K/V precision changes by isolated KL, then accept only an
/// improvement of the complete cache on the calibration continuation.
struct LayerSearch<'a> {
    model: &'a Model,
    train_rows: &'a [Vec<Choice>],
    heldout_rows: &'a [Vec<Choice>],
    isolated: &'a [[f64; 3]],
    suffix: &'a [u32],
    reference: &'a [Vec<f32>],
}

type LayerProposal = (f64, usize, usize, Option<(usize, usize)>);

fn refine_layer_policy(
    ctx: &LayerSearch<'_>,
    uniform: &[usize],
    greedy: &[usize],
    budget: usize,
) -> Result<(Vec<usize>, f64, f64, usize)> {
    let uniform_kl = joint_score(
        ctx.model,
        ctx.train_rows,
        uniform,
        ctx.suffix,
        ctx.reference,
    )?;
    let greedy_kl = joint_score(ctx.model, ctx.train_rows, greedy, ctx.suffix, ctx.reference)?;
    let (mut selected, mut current_kl) = if greedy_kl < uniform_kl {
        (greedy.to_vec(), greedy_kl)
    } else {
        (uniform.to_vec(), uniform_kl)
    };
    let mut trials = 2;
    for _ in 0..2 {
        let used = chosen_bytes(ctx.heldout_rows, &selected);
        let mut proposals: Vec<LayerProposal> = Vec::new();
        for i in 0..selected.len() {
            let old = selected[i];
            for a in 0..3 {
                if a == old {
                    continue;
                }
                let size_i = used - ctx.heldout_rows[i][old].bytes + ctx.heldout_rows[i][a].bytes;
                let gain_i = ctx.isolated[i][old] - ctx.isolated[i][a];
                if size_i <= budget {
                    proposals.push((gain_i, i, a, None));
                }
                for (k, &old_k) in selected.iter().enumerate().skip(i + 1) {
                    for b in 0..3 {
                        if b == old_k {
                            continue;
                        }
                        let size = size_i - ctx.heldout_rows[k][selected[k]].bytes
                            + ctx.heldout_rows[k][b].bytes;
                        if size <= budget {
                            let gain = gain_i + ctx.isolated[k][old_k] - ctx.isolated[k][b];
                            proposals.push((gain, i, a, Some((k, b))));
                        }
                    }
                }
            }
        }
        proposals.sort_by(|a, b| b.0.total_cmp(&a.0));
        let mut best: Option<(Vec<usize>, f64)> = None;
        for (_, i, a, second) in proposals.into_iter().take(40) {
            let mut trial = selected.clone();
            trial[i] = a;
            if let Some((k, b)) = second {
                trial[k] = b;
            }
            let score = joint_score(ctx.model, ctx.train_rows, &trial, ctx.suffix, ctx.reference)?;
            trials += 1;
            if score + 1e-8 < current_kl && best.as_ref().is_none_or(|(_, prev)| score < *prev) {
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

/// Spend the same budget on four-bit heads in a fixed, dispersed order.
fn dispersed_mix(rows: &[Vec<Choice>], budget: usize) -> Vec<usize> {
    let mut selected = vec![0; rows.len()];
    let mut used = chosen_bytes(rows, &selected);
    assert!(used <= budget);
    let mut order: Vec<_> = (0..rows.len()).collect();
    order.sort_by_key(|&i| (i * 73) % rows.len());
    for i in order {
        if rows[i][1].bytes <= rows[i][0].bytes {
            selected[i] = 1;
        } else {
            let extra = rows[i][1].bytes - rows[i][0].bytes;
            if extra <= budget - used {
                selected[i] = 1;
                used += extra;
            }
        }
    }
    selected
}

fn make_tiles(rows: &[Vec<Choice>], selected: &[usize], heads: Option<usize>) -> Vec<Tensor> {
    match heads {
        None => rows
            .iter()
            .zip(selected)
            .map(|(r, &j)| r[j].tile.clone())
            .collect(),
        Some(h) => rows
            .chunks_exact(h)
            .zip(selected.chunks_exact(h))
            .map(|(block, picks)| {
                let mut full = Tensor {
                    shape: vec![
                        block[0][picks[0]].tile.shape[0],
                        h,
                        block[0][picks[0]].tile.shape[2],
                    ],
                    axes: block[0][picks[0]].tile.axes.clone(),
                    values: vec![0.0; block[0][picks[0]].tile.values.len() * h],
                };
                for (head, (row, &j)) in block.iter().zip(picks).enumerate() {
                    replace_head(&mut full, head, &row[j].tile);
                }
                full
            })
            .collect(),
    }
}

fn fit_beta(rows: &[Vec<Choice>], slot: usize) -> (f64, [f64; 3]) {
    let mut means = [0.0; 3];
    let mut count = 0.0;
    for (i, row) in rows.iter().enumerate() {
        if (i / 8) % 2 != slot {
            continue;
        }
        for j in 0..3 {
            means[j] += row[j].mse;
        }
        count += 1.0;
    }
    for x in &mut means {
        *x /= count;
    }
    let xs = [2.0, 4.0, 8.0];
    let avg_x = xs.iter().sum::<f64>() / 3.0;
    let ys = means.map(|x| x.max(1e-30).ln());
    let avg_y = ys.iter().sum::<f64>() / 3.0;
    let covariance = xs
        .iter()
        .zip(ys)
        .map(|(&x, y)| (x - avg_x) * (y - avg_y))
        .sum::<f64>();
    let variance = xs.iter().map(|&x| (x - avg_x).powi(2)).sum::<f64>();
    ((-covariance / variance).exp().max(1.000001), means)
}

fn score_tiles(
    model: &Model,
    tiles: &[Tensor],
    prefix: usize,
    suffix: &[u32],
    n: usize,
) -> Result<(Vec<Vec<f32>>, Vec<u32>)> {
    let cache = model.restore(tiles, prefix + EVAL_TOKENS)?;
    Ok((
        teacher(model, &cache, suffix, prefix, n)?,
        greedy(model, &cache, suffix[0], prefix)?,
    ))
}

struct PersistContext<'a> {
    out: &'a Path,
    model: &'a Model,
    fp: &'a str,
    prefix_ids: &'a [u32],
    source_tiles: &'a [Tensor],
    suffix: &'a [u32],
}

fn persist_and_verify(
    ctx: &PersistContext<'_>,
    name: &str,
    rows: &[Vec<Choice>],
    selected: &[usize],
    heads: Option<usize>,
    expected_logits: &[Vec<f32>],
    expected_greedy: &[u32],
) -> Result<Value> {
    let db = ctx
        .out
        .join(format!("{name}-cache-{}", ctx.prefix_ids.len()));
    ensure!(!db.exists(), "cache path already exists");
    let started = Instant::now();
    let mut actual = 0usize;
    {
        let mut store = Cache::open(&db)?;
        for (i, row) in rows.iter().enumerate() {
            let tile_index = heads.map_or(i, |h| i / h);
            let head = heads.map(|h| i % h);
            let source = head.map_or_else(
                || ctx.source_tiles[i].clone(),
                |h| head_tensor(&ctx.source_tiles[tile_index], h),
            );
            let addr = address(ctx.fp, ctx.prefix_ids, tile_index, head)?;
            let saved = store.put(&addr, &source, &policy(tile_index, row[selected[i]].bits))?;
            ensure!(
                saved.total_roaring_bytes == row[selected[i]].bytes,
                "stored cost differs from estimate"
            );
            actual += saved.total_roaring_bytes;
        }
    }
    let persist_ms = started.elapsed().as_secs_f64() * 1000.0;
    ensure!(
        actual == chosen_bytes(rows, selected),
        "stored byte sum differs"
    );
    let started = Instant::now();
    let restored_units = {
        let store = Cache::open(&db)?;
        let mut units = Vec::with_capacity(rows.len());
        for (i, row) in rows.iter().enumerate() {
            let tile_index = heads.map_or(i, |h| i / h);
            let head = heads.map(|h| i % h);
            let addr = address(ctx.fp, ctx.prefix_ids, tile_index, head)?;
            let got = store
                .get(&addr)?
                .context("missing paper-comparison tile")?
                .tensor;
            ensure!(
                got.values.len() == row[selected[i]].tile.values.len(),
                "restored shape differs"
            );
            ensure!(
                got.values
                    .iter()
                    .zip(&row[selected[i]].tile.values)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "reopened decoded bits differ"
            );
            units.push(got);
        }
        units
    };
    let read_ms = started.elapsed().as_secs_f64() * 1000.0;
    let restored_tiles = match heads {
        None => restored_units,
        Some(h) => {
            let mut full = ctx.source_tiles.to_vec();
            for (i, unit) in restored_units.iter().enumerate() {
                replace_head(&mut full[i / h], i % h, unit);
            }
            full
        }
    };
    let (logits, generated) = score_tiles(
        ctx.model,
        &restored_tiles,
        ctx.prefix_ids.len(),
        ctx.suffix,
        EVAL_TOKENS,
    )?;
    ensure!(
        logits == expected_logits && generated == expected_greedy,
        "persistence changed inference"
    );
    Ok(
        json!({"logical_bytes":actual,"persist_ms":persist_ms,"reopen_read_ms":read_ms,"inference_exact_after_reopen":true}),
    )
}

fn evaluate(
    model: &Model,
    tiles: &[Tensor],
    prefix: usize,
    suffix: &[u32],
    reference: &[Vec<f32>],
    reference_greedy: &[u32],
) -> Result<(Value, Vec<Vec<f32>>, Vec<u32>)> {
    let (logits, generated) = score_tiles(model, tiles, prefix, suffix, EVAL_TOKENS)?;
    let matches = generated
        .iter()
        .zip(reference_greedy)
        .filter(|(a, b)| a == b)
        .count();
    Ok((
        json!({"quality":metrics(&logits, reference, &suffix[1..]),
        "greedy_matches_bf16":matches}),
        logits,
        generated,
    ))
}

pub fn run(directory: &Path, corpus: &Path, out: &Path) -> Result<()> {
    ensure!(
        !out.join("paper-quantizers.json").exists(),
        "paper comparison result already exists"
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
    let heads = model.config.num_key_value_heads;
    ensure!(heads == 8, "this comparison expects eight KV heads");
    let fp = fingerprint(
        directory,
        &format!("{};paper-quantizers-v1", model.format()),
    )?;
    let train_prefix = &tokens[..128];
    let train_suffix = &tokens[128..128 + CAL_TOKENS];
    let train_cache = model.empty_cache(128 + EVAL_TOKENS)?;
    model.forward(train_prefix, 0, &train_cache)?;
    let train_tiles = model.export(&train_cache, 128)?;
    let train_reference = teacher(&model, &train_cache, train_suffix, 128, CAL_TOKENS)?;
    let started = Instant::now();
    let mut layer_scores: Vec<[f64; 3]> = Vec::new();
    let mut layer_train_rows = Vec::new();
    let mut head_scores_2bit: Vec<f64> = Vec::new();
    let mut head_train_rows = Vec::new();
    for (index, source) in train_tiles.iter().enumerate() {
        let addr = address(&fp, train_prefix, index, None)?;
        let mut scores = [0.0; 3];
        let mut layer_row = Vec::new();
        for (j, &bits) in BITS.iter().enumerate() {
            let option = choice(source, index, bits, &addr)?;
            let changed =
                model.replace_tile(&train_cache, index, &option.tile, 128 + EVAL_TOKENS)?;
            let logits = teacher(&model, &changed, train_suffix, 128, CAL_TOKENS)?;
            scores[j] = mean_kl(&logits, &train_reference);
            layer_row.push(option);
        }
        layer_scores.push(scores);
        layer_train_rows.push(layer_row);
        for head in 0..heads {
            let source_head = head_tensor(source, head);
            let addr = address(&fp, train_prefix, index, Some(head))?;
            let row = BITS
                .iter()
                .map(|&bits| choice(&source_head, index, bits, &addr))
                .collect::<Result<Vec<_>>>()?;
            let mut changed_tile = source.clone();
            replace_head(&mut changed_tile, head, &row[0].tile);
            let changed =
                model.replace_tile(&train_cache, index, &changed_tile, 128 + EVAL_TOKENS)?;
            let logits = teacher(&model, &changed, train_suffix, 128, CAL_TOKENS)?;
            head_scores_2bit.push(mean_kl(&logits, &train_reference));
            head_train_rows.push(row);
        }
        eprintln!(
            "paper calibration {}/{} tiles",
            index + 1,
            train_tiles.len()
        );
    }
    let calibration_ms = started.elapsed().as_secs_f64() * 1000.0;
    let (beta_k, mse_k) = fit_beta(&head_train_rows, 0);
    let (beta_v, mse_v) = fit_beta(&head_train_rows, 1);
    let rate_scores: Vec<[f64; 3]> = head_scores_2bit
        .iter()
        .enumerate()
        .map(|(i, &s)| {
            let beta = if (i / heads) % 2 == 0 { beta_k } else { beta_v };
            BITS.map(|b| s * beta.powf(-(b as f64 - 2.0)))
        })
        .collect();
    eprintln!(
        "paper calibration done in {:.1} s; beta K={beta_k:.3}, V={beta_v:.3}",
        calibration_ms / 1000.0
    );

    let mut heldout = Vec::new();
    for &prefix in &[128usize, 512] {
        let ids = &tokens[4096..4096 + prefix];
        let suffix = &tokens[4096 + prefix..4096 + prefix + EVAL_TOKENS + 1];
        let native = model.empty_cache(prefix + EVAL_TOKENS)?;
        model.forward(ids, 0, &native)?;
        let source_tiles = model.export(&native, prefix)?;
        let reference = teacher(&model, &native, suffix, prefix, EVAL_TOKENS)?;
        let reference_greedy = greedy(&model, &native, suffix[0], prefix)?;
        let started = Instant::now();
        let mut layer_rows = Vec::new();
        let mut head_rows = Vec::new();
        for (index, source) in source_tiles.iter().enumerate() {
            let addr = address(&fp, ids, index, None)?;
            layer_rows.push(
                BITS.iter()
                    .map(|&bits| choice(source, index, bits, &addr))
                    .collect::<Result<Vec<_>>>()?,
            );
            for head in 0..heads {
                let source_head = head_tensor(source, head);
                let addr = address(&fp, ids, index, Some(head))?;
                head_rows.push(
                    BITS.iter()
                        .map(|&bits| choice(&source_head, index, bits, &addr))
                        .collect::<Result<Vec<_>>>()?,
                );
            }
            if index % 8 == 7 {
                eprintln!(
                    "paper prefix {prefix}: encoded {}/{} tiles",
                    index + 1,
                    source_tiles.len()
                );
            }
        }
        let encode_ms = started.elapsed().as_secs_f64() * 1000.0;
        let layer_uniform2 = vec![0; layer_rows.len()];
        let layer_uniform4 = vec![1; layer_rows.len()];
        let layer_budget = chosen_bytes(&layer_rows, &layer_uniform4);
        let layer_greedy = allocate(&layer_rows, &layer_scores, layer_budget);
        let layer_search = LayerSearch {
            model: &model,
            train_rows: &layer_train_rows,
            heldout_rows: &layer_rows,
            isolated: &layer_scores,
            suffix: train_suffix,
            reference: &train_reference,
        };
        let (layer_selected, layer_uniform_kl, layer_selected_kl, layer_trials) =
            refine_layer_policy(&layer_search, &layer_uniform4, &layer_greedy, layer_budget)?;
        let k4v2: Vec<usize> = (0..layer_rows.len())
            .map(|i| if i % 2 == 0 { 1 } else { 0 })
            .collect();
        let head_uniform4 = vec![1; head_rows.len()];
        let head_budget = chosen_bytes(&head_rows, &head_uniform4);
        let head_selected = allocate(&head_rows, &rate_scores, head_budget);
        let head_2bit_bytes: usize = head_rows.iter().map(|row| row[0].bytes).sum();
        let head_mid_budget = head_2bit_bytes + (head_budget - head_2bit_bytes) / 2;
        let head_mid_selected = allocate(&head_rows, &rate_scores, head_mid_budget);
        let dispersed_selected = dispersed_mix(&head_rows, head_mid_budget);
        let bf16_bytes = prefix * source_tiles.len() * heads * source_tiles[0].shape[2] * 2;

        let (kivi_eval, kivi_logits, kivi_greedy) = evaluate(
            &model,
            &make_tiles(&layer_rows, &layer_uniform2, None),
            prefix,
            suffix,
            &reference,
            &reference_greedy,
        )?;
        let (uniform4_eval, _, _) = evaluate(
            &model,
            &make_tiles(&layer_rows, &layer_uniform4, None),
            prefix,
            suffix,
            &reference,
            &reference_greedy,
        )?;
        let (k4v2_eval, _, _) = evaluate(
            &model,
            &make_tiles(&layer_rows, &k4v2, None),
            prefix,
            suffix,
            &reference,
            &reference_greedy,
        )?;
        let (kvtuner_eval, kvtuner_logits, kvtuner_greedy) = evaluate(
            &model,
            &make_tiles(&layer_rows, &layer_selected, None),
            prefix,
            suffix,
            &reference,
            &reference_greedy,
        )?;
        let (head4_eval, _, _) = evaluate(
            &model,
            &make_tiles(&head_rows, &head_uniform4, Some(heads)),
            prefix,
            suffix,
            &reference,
            &reference_greedy,
        )?;
        let (dispersed_eval, _, _) = evaluate(
            &model,
            &make_tiles(&head_rows, &dispersed_selected, Some(heads)),
            prefix,
            suffix,
            &reference,
            &reference_greedy,
        )?;
        let (rate_mid_eval, rate_mid_logits, rate_mid_greedy) = evaluate(
            &model,
            &make_tiles(&head_rows, &head_mid_selected, Some(heads)),
            prefix,
            suffix,
            &reference,
            &reference_greedy,
        )?;
        let (rate_eval, rate_logits, rate_greedy) = evaluate(
            &model,
            &make_tiles(&head_rows, &head_selected, Some(heads)),
            prefix,
            suffix,
            &reference,
            &reference_greedy,
        )?;
        let persist = PersistContext {
            out,
            model: &model,
            fp: &fp,
            prefix_ids: ids,
            source_tiles: &source_tiles,
            suffix,
        };
        let kivi_stored = persist_and_verify(
            &persist,
            "kivi2",
            &layer_rows,
            &layer_uniform2,
            None,
            &kivi_logits,
            &kivi_greedy,
        )?;
        let kvtuner_stored = persist_and_verify(
            &persist,
            "kvtuner",
            &layer_rows,
            &layer_selected,
            None,
            &kvtuner_logits,
            &kvtuner_greedy,
        )?;
        let rate_mid_stored = persist_and_verify(
            &persist,
            "ratequant-mid",
            &head_rows,
            &head_mid_selected,
            Some(heads),
            &rate_mid_logits,
            &rate_mid_greedy,
        )?;
        let rate_stored = persist_and_verify(
            &persist,
            "ratequant",
            &head_rows,
            &head_selected,
            Some(heads),
            &rate_logits,
            &rate_greedy,
        )?;
        let counts = |picks: &[usize]| -> [usize; 3] {
            let mut counts = [0; 3];
            for &j in picks {
                counts[j] += 1;
            }
            counts
        };
        let row = json!({"offset":4096,"prefix_tokens":prefix,"scored_tokens":EVAL_TOKENS,
            "bf16_cache_bytes":bf16_bytes,"encode_candidates_ms":encode_ms,
            "kivi2":{"requested_bits":2,"complete_logical_bytes":chosen_bytes(&layer_rows,&layer_uniform2),
                "choice_counts_2_4_8":counts(&layer_uniform2),
                "candidate_total_bytes_2_4_8":(0..3).map(|j| layer_rows.iter().map(|r| r[j].bytes).sum::<usize>()).collect::<Vec<_>>(),"evaluation":kivi_eval,"persistence":kivi_stored},
            "uniform4":{"requested_bits":4,"complete_logical_bytes":layer_budget,"evaluation":uniform4_eval},
            "k4v2":{"complete_logical_bytes":chosen_bytes(&layer_rows,&k4v2),"evaluation":k4v2_eval},
            "kvtuner_inspired":{"complete_logical_bytes":chosen_bytes(&layer_rows,&layer_selected),
                "joint_calibration_uniform_kl":layer_uniform_kl,
                "joint_calibration_selected_kl":layer_selected_kl,"joint_trials":layer_trials,
                "budget_bytes":layer_budget,"choice_counts_2_4_8":counts(&layer_selected),
                "selected_bits_by_layer_kv":layer_selected.iter().map(|&j| BITS[j]).collect::<Vec<_>>(),
                "evaluation":kvtuner_eval,"persistence":kvtuner_stored},
            "head_uniform4":{"complete_logical_bytes":head_budget,
                "candidate_total_bytes_2_4_8":(0..3).map(|j| head_rows.iter().map(|r| r[j].bytes).sum::<usize>()).collect::<Vec<_>>(),
                "equal_cost_heads_2_4":head_rows.iter().filter(|r| r[0].bytes == r[1].bytes).count(),
                "equal_cost_heads_4_8":head_rows.iter().filter(|r| r[1].bytes == r[2].bytes).count(),
                "evaluation":head4_eval},
            "head_dispersed_mid":{"complete_logical_bytes":chosen_bytes(&head_rows,&dispersed_selected),
                "budget_bytes":head_mid_budget,"choice_counts_2_4_8":counts(&dispersed_selected),
                "evaluation":dispersed_eval},
            "ratequant_mid":{"complete_logical_bytes":chosen_bytes(&head_rows,&head_mid_selected),
                "budget_bytes":head_mid_budget,"choice_counts_2_4_8":counts(&head_mid_selected),
                "selected_bits_by_layer_kv_head":head_mid_selected.iter().map(|&j| BITS[j]).collect::<Vec<_>>(),
                "evaluation":rate_mid_eval,"persistence":rate_mid_stored},
            "ratequant_inspired":{"complete_logical_bytes":chosen_bytes(&head_rows,&head_selected),
                "budget_bytes":head_budget,"choice_counts_2_4_8":counts(&head_selected),
                "selected_bits_by_layer_kv_head":head_selected.iter().map(|&j| BITS[j]).collect::<Vec<_>>(),
                "evaluation":rate_eval,"persistence":rate_stored}});
        eprintln!(
            "paper prefix {prefix}: KIVI KL={} KVTuner KL={} RateQuant KL={}",
            row["kivi2"]["evaluation"]["quality"]["kl_from_bf16"],
            row["kvtuner_inspired"]["evaluation"]["quality"]["kl_from_bf16"],
            row["ratequant_inspired"]["evaluation"]["quality"]["kl_from_bf16"]
        );
        heldout.push(row);
    }
    let report = json!({"method":"KIVI fixed 2-bit; KVTuner layerwise isolated-KL allocation with joint calibration search; RateQuant headwise calibrated rate-distortion proxy",
        "limitations":"paper-inspired approximations: affine shifou codec, forward-KL sensitivity instead of RateQuant gradients, no KVTuner MOEA/D or kernel reproduction",
        "calibration":{"offset":0,"prefix_tokens":128,"scored_tokens":CAL_TOKENS,"calibration_ms":calibration_ms,
            "layer_scores_2_4_8":layer_scores,"head_kl_2bit":head_scores_2bit,
            "rate_fit_beta_k":beta_k,"rate_fit_beta_v":beta_v,
            "rate_mean_mse_k_2_4_8":mse_k,"rate_mean_mse_v_2_4_8":mse_v},
        "model_revision":serde_json::from_slice::<Value>(&fs::read(directory.join("revision.json"))?)?,
        "model_fingerprint":fp,"corpus_sha256":corpus_sha,"heldout":heldout});
    fs::write(
        out.join("paper-quantizers.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn dummy(bits: u8, bytes: usize) -> Choice {
        Choice {
            bits,
            bytes,
            mse: 0.0,
            tile: Tensor {
                shape: vec![1],
                axes: vec!["channel".into()],
                values: vec![0.0],
            },
        }
    }
    #[test]
    fn allocator_respects_budget() {
        let rows = vec![
            vec![dummy(2, 2), dummy(4, 4), dummy(8, 8)],
            vec![dummy(2, 2), dummy(4, 4), dummy(8, 8)],
        ];
        let selected = allocate(&rows, &[[4.0, 1.0, 0.0], [1.0, 0.5, 0.0]], 8);
        assert_eq!(chosen_bytes(&rows, &selected), 8);
        assert_eq!(selected, vec![1, 1]);
    }
    #[test]
    fn free_quality_upgrade() {
        let rows = vec![vec![dummy(2, 4), dummy(4, 4), dummy(8, 4)]];
        assert_eq!(allocate(&rows, &[[4.0, 1.0, 0.1]], 4), vec![2]);
    }
    #[test]
    fn head_roundtrip() {
        let tile = Tensor {
            shape: vec![2, 2, 2],
            axes: vec!["token".into(), "head".into(), "channel".into()],
            values: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        };
        let mut rebuilt = tile.clone();
        for head in 0..2 {
            let part = head_tensor(&tile, head);
            replace_head(&mut rebuilt, head, &part);
        }
        assert_eq!(rebuilt.values, tile.values);
    }
}
