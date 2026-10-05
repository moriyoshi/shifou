#[allow(dead_code)]
mod model {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/model.rs"));
}

use anyhow::{ensure, Result};
use model::{Model, NativeCache};
use serde_json::json;
use sha2::{Digest, Sha256};
use shifou::{
    Address, Cache, CacheReader, PrefillBundle, PrefixScope, ReadVerification, SessionKey,
};
use std::{io::Read, path::Path, time::Instant};
use tokenizers::Tokenizer;

const SUFFIX: usize = 32;
const STATE_FORMAT: &str = "qwen3/no-recurrent-state/v1";

fn elapsed(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

fn fingerprint_model(dir: &Path) -> Result<String> {
    let mut hash = Sha256::new();
    for name in ["config.json", "model.safetensors"] {
        let mut file = std::fs::File::open(dir.join(name))?;
        let mut block = vec![0u8; 4 * 1024 * 1024];
        loop {
            let n = file.read(&mut block)?;
            if n == 0 {
                break;
            }
            hash.update(&block[..n]);
        }
    }
    hash.update(b"xinfer-17499e4/attention-c0f19f2/bf16-block16");
    Ok(format!("sha256:{:x}", hash.finalize()))
}

fn snapshot(model: &Model, native: &NativeCache, tokens: &[u32]) -> Result<PrefillBundle> {
    let attention = model
        .export_packed_bf16(native, tokens.len())?
        .into_iter()
        .enumerate()
        .map(|(i, snapshot)| {
            (
                Address {
                    namespace: "probe-input".into(),
                    model_fingerprint: "probe-input".into(),
                    prefix_fingerprint: "probe-input".into(),
                    layer: (i / 2) as u32,
                    slot: if i % 2 == 0 { "key" } else { "value" }.into(),
                },
                snapshot,
            )
        })
        .collect();
    Ok(PrefillBundle {
        prefix_tokens: tokens.len() as u64,
        token_ids: tokens.to_vec(),
        state_format: STATE_FORMAT.into(),
        state_bytes: vec![0],
        attention,
    })
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    ensure!(
        args.len() == 5 || args.len() == 6,
        "MODEL_DIR CORPUS DB_DIR RESULT_JSON [PREFIX_TOKENS]"
    );
    let prefix_tokens: usize = args.get(5).map(|v| v.parse()).transpose()?.unwrap_or(512);
    ensure!(prefix_tokens > 0, "empty prefix");
    let verification = if std::env::var("SESSION_TRUSTED_LOCAL").as_deref() == Ok("1") {
        ReadVerification::StorageOnly
    } else {
        ReadVerification::Full
    };
    let model_dir = Path::new(&args[1]);
    let db_dir = Path::new(&args[3]);
    let result = Path::new(&args[4]);
    ensure!(
        !db_dir.exists() && !result.exists(),
        "output already exists"
    );
    let tokenizer =
        Tokenizer::from_file(model_dir.join("tokenizer.json")).map_err(anyhow::Error::msg)?;
    let text = std::fs::read_to_string(&args[2])?;
    let tokens = tokenizer
        .encode(text, false)
        .map_err(anyhow::Error::msg)?
        .get_ids()
        .to_vec();
    ensure!(tokens.len() >= prefix_tokens + SUFFIX + 2, "short corpus");
    let prefix = &tokens[..prefix_tokens];
    let suffix = &tokens[prefix_tokens..prefix_tokens + SUFFIX];
    let total = prefix_tokens + SUFFIX;
    let model = Model::load(model_dir, "auto")?;
    let capacity = total + 8;
    let native = model.empty_cache(capacity)?;
    let start = Instant::now();
    model.forward(prefix, 0, &native)?;
    model.device.synchronize()?;
    let prefix_prefill_ms = elapsed(start);
    let prepared_bundle = snapshot(&model, &native, prefix)?;
    let model_fingerprint = fingerprint_model(model_dir)?;
    let scope = PrefixScope {
        namespace: "qwen-session-workflow-probe".into(),
        model_fingerprint: model_fingerprint.clone(),
        context_fingerprint: "tenant-a/wikitext-tokenizer+template/none".into(),
    };
    let session_key = SessionKey {
        namespace: scope.namespace.clone(),
        model_fingerprint,
        session_id: "session-a".into(),
    };
    let mut writer = Cache::open(db_dir)?;
    let start = Instant::now();
    let prepared_address = writer.put_prepared_prefix(&scope, &prepared_bundle)?;
    let publish_prepared_ms = elapsed(start);
    let reader = CacheReader::open(db_dir)?;
    let start = Instant::now();
    let hit = reader
        .find_longest_prepared_address(&scope, &tokens[..total])?
        .ok_or_else(|| anyhow::anyhow!("prepared prefix not found"))?;
    let lookup_ms = elapsed(start);
    ensure!(
        hit.prefix_tokens == prefix_tokens && hit.address == prepared_address,
        "wrong prefix hit"
    );
    let start = Instant::now();
    let fetched = reader
        .get_prefill_bundle_with_verification(&hit.address, STATE_FORMAT, verification)?
        .ok_or_else(|| anyhow::anyhow!("missing prepared state"))?;
    let read_prepared_ms = elapsed(start);
    let target = Model::load(model_dir, "auto")?;
    let start = Instant::now();
    let restored = target.restore_packed_bf16(&fetched.attention, capacity)?;
    let restore_prepared_ms = elapsed(start);
    let start = Instant::now();
    let expected = model.forward_chunk(suffix, prefix_tokens, &native)?;
    model.device.synchronize()?;
    let source_suffix_ms = elapsed(start);
    let start = Instant::now();
    let actual = target.forward_chunk(suffix, prefix_tokens, &restored)?;
    target.device.synchronize()?;
    let restored_suffix_ms = elapsed(start);
    let prepared_logit_diff = max_abs(&expected, &actual);
    ensure!(
        prepared_logit_diff <= 1e-3,
        "prepared-prefix continuation changed logits"
    );

    let session_bundle = snapshot(&model, &native, &tokens[..total])?;
    let start = Instant::now();
    writer.put_session_checkpoint(
        &session_key,
        1,
        &session_bundle,
        "workflow/v1",
        b"turn=1;tool=idle",
    )?;
    let publish_session_ms = elapsed(start);
    let session_reader = CacheReader::open(db_dir)?;
    let start = Instant::now();
    let resumed = session_reader
        .get_session_checkpoint_with_verification(&session_key, STATE_FORMAT, verification)?
        .ok_or_else(|| anyhow::anyhow!("missing session"))?;
    let read_session_ms = elapsed(start);
    ensure!(
        resumed.generation == 1 && resumed.workflow_bytes == b"turn=1;tool=idle",
        "workflow state changed"
    );
    let start = Instant::now();
    let resumed_cache = target.restore_packed_bf16(&resumed.bundle.attention, capacity)?;
    let restore_session_ms = elapsed(start);
    let next = tokens[total];
    let source_logits = model.forward(&[next], total, &native)?;
    let target_logits = target.forward(&[next], total, &resumed_cache)?;
    let session_logit_diff = max_abs(&source_logits, &target_logits);
    ensure!(
        session_logit_diff <= 1e-3,
        "session continuation changed logits"
    );

    let generation_two_bundle = snapshot(&model, &native, &tokens[..total + 1])?;
    let start = Instant::now();
    writer.put_session_append_checkpoint(
        &session_key,
        2,
        &generation_two_bundle,
        "workflow/v1",
        b"turn=2;tool=idle",
    )?;
    let publish_append_ms = elapsed(start);
    let latest = CacheReader::open(db_dir)?
        .get_session_checkpoint_with_verification(&session_key, STATE_FORMAT, verification)?
        .ok_or_else(|| anyhow::anyhow!("missing generation two"))?;
    ensure!(
        latest.generation == 2 && latest.bundle.token_ids == tokens[..total + 1],
        "generation did not advance"
    );
    let latest_cache = target.restore_packed_bf16(&latest.bundle.attention, capacity)?;
    let source_two = model.forward(&[tokens[total + 1]], total + 1, &native)?;
    let target_two = target.forward(&[tokens[total + 1]], total + 1, &latest_cache)?;
    let generation_two_logit_diff = max_abs(&source_two, &target_two);
    ensure!(
        generation_two_logit_diff <= 1e-3,
        "generation two changed logits"
    );
    let mut warm_fresh_startup_ms = Vec::new();
    let mut warm_cached_startup_ms = Vec::new();
    let mut warm_fresh_resume_ms = Vec::new();
    let mut warm_cached_resume_ms = Vec::new();
    let prepare_host_start = Instant::now();
    let prepared_host = target.prepare_packed_bf16(&fetched.attention, capacity)?;
    let prepare_host_ms = elapsed(prepare_host_start);
    let prepare_session_host_start = Instant::now();
    let session_host = target.prepare_packed_bf16(&latest.bundle.attention, capacity)?;
    let prepare_session_host_ms = elapsed(prepare_session_host_start);
    let mut warm_host_startup_ms = Vec::new();
    let mut warm_host_resume_ms = Vec::new();
    for _ in 0..3 {
        let fresh = model.empty_cache(capacity)?;
        let start = Instant::now();
        model.forward(&tokens[..total], 0, &fresh)?;
        model.device.synchronize()?;
        warm_fresh_startup_ms.push(elapsed(start));

        let start = Instant::now();
        let prepared_hit = reader
            .find_longest_prepared_address(&scope, &tokens[..total])?
            .ok_or_else(|| anyhow::anyhow!("prepared hit disappeared"))?;
        let prepared = reader
            .get_prefill_bundle_with_verification(
                &prepared_hit.address,
                STATE_FORMAT,
                verification,
            )?
            .ok_or_else(|| anyhow::anyhow!("prepared bundle disappeared"))?;
        let cache = target.restore_packed_bf16(&prepared.attention, capacity)?;
        target.forward_chunk(suffix, prefix_tokens, &cache)?;
        target.device.synchronize()?;
        warm_cached_startup_ms.push(elapsed(start));

        let start = Instant::now();
        let cache = target.restore_prepared_bf16(&prepared_host)?;
        let logits = target.forward_chunk(suffix, prefix_tokens, &cache)?;
        target.device.synchronize()?;
        ensure!(
            max_abs(&expected, &logits) <= 1e-3,
            "prepared-host startup changed logits"
        );
        warm_host_startup_ms.push(elapsed(start));

        let fresh = model.empty_cache(capacity)?;
        let start = Instant::now();
        model.forward(&tokens[..total + 2], 0, &fresh)?;
        model.device.synchronize()?;
        warm_fresh_resume_ms.push(elapsed(start));

        let start = Instant::now();
        let current = CacheReader::open(db_dir)?
            .get_session_checkpoint_with_verification(&session_key, STATE_FORMAT, verification)?
            .ok_or_else(|| anyhow::anyhow!("session disappeared"))?;
        let cache = target.restore_packed_bf16(&current.bundle.attention, capacity)?;
        target.forward(&[tokens[total + 1]], total + 1, &cache)?;
        target.device.synchronize()?;
        warm_cached_resume_ms.push(elapsed(start));

        let start = Instant::now();
        let cache = target.restore_prepared_bf16(&session_host)?;
        let logits = target.forward(&[tokens[total + 1]], total + 1, &cache)?;
        target.device.synchronize()?;
        ensure!(
            max_abs(&source_two, &logits) <= 1e-3,
            "prepared-host resume changed logits"
        );
        warm_host_resume_ms.push(elapsed(start));
    }
    std::fs::write(
        result,
        serde_json::to_vec_pretty(&json!({
            "model":"Qwen3-0.6B BF16 KV", "prefix_tokens":prefix_tokens,"suffix_tokens":SUFFIX,
            "verification": format!("{verification:?}"),
            "prefix_prefill_ms":prefix_prefill_ms,"prepared_publish_ms":publish_prepared_ms,
            "prepared_lookup_ms":lookup_ms,"prepared_read_ms":read_prepared_ms,
            "prepared_gpu_restore_ms":restore_prepared_ms,"source_suffix_ms":source_suffix_ms,
            "restored_suffix_ms":restored_suffix_ms,"prepared_max_abs_logit_diff":prepared_logit_diff,
            "session_publish_ms":publish_session_ms,"session_read_ms":read_session_ms,
            "session_gpu_restore_ms":restore_session_ms,"session_max_abs_logit_diff":session_logit_diff,
            "generation_two_max_abs_logit_diff":generation_two_logit_diff,
            "generation_two_append_publish_ms":publish_append_ms,
            "warm_fresh_startup_ms":warm_fresh_startup_ms,
            "warm_cached_startup_ms":warm_cached_startup_ms,
            "warm_fresh_resume_ms":warm_fresh_resume_ms,
            "warm_cached_resume_ms":warm_cached_resume_ms,
            "prepare_host_ms":prepare_host_ms,
            "prepare_session_host_ms":prepare_session_host_ms,
            "warm_host_startup_ms":warm_host_startup_ms,
            "warm_host_resume_ms":warm_host_resume_ms,
            "notes":"Model loading excluded; local warm yesno DB; both workflows use full BF16 state; source and target are distinct model instances on one GB10."
        }))?,
    )?;
    Ok(())
}
