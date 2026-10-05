#[allow(dead_code)]
mod model {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/model.rs"));
}

use anyhow::{ensure, Result};
use model::{Model, PreparedBf16};
use serde_json::json;
use sha2::{Digest, Sha256};
use shifou::{Address, BoundedHostTier, HostAdmission, PeerBundleReader, PrefixScope, SessionKey};
use std::{
    io::Read,
    path::{Path, PathBuf},
    thread,
    time::Instant,
};
use tokenizers::Tokenizer;
use xinfer::utils::config::Config;

const STATE_FORMAT: &str = "qwen3/no-recurrent-state/v1";
const SUFFIX: usize = 32;
type Warmed = (
    Option<(Address, PreparedBf16)>,
    Option<(Address, PreparedBf16)>,
    f64,
    f64,
    f64,
);

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

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn reader(socket: &Path, bundle_limit: usize) -> Result<PeerBundleReader> {
    Ok(PeerBundleReader::connect(socket)?
        .with_max_bundle_bytes(bundle_limit)?
        .with_parallel_verification(true)
        .with_pipelined_verification(true))
}

fn geometry_bytes(model_dir: &Path, capacity: usize) -> Result<usize> {
    let config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(model_dir.join("config.json"))?)?;
    let field = |name: &str| -> Result<usize> {
        Ok(config[name]
            .as_u64()
            .ok_or_else(|| anyhow::anyhow!("missing {name}"))? as usize)
    };
    Ok(capacity.div_ceil(16)
        * 16
        * field("num_hidden_layers")?
        * 2
        * field("num_key_value_heads")?
        * field("head_dim")?
        * 2)
}

fn admission(kind: &str, reuses: u64) -> HostAdmission {
    // Seeded from earlier Qwen cold peer and prepared-host observations.
    let (fallback, resident) = if kind == "prepared" {
        (718.0, 40.0)
    } else {
        (706.0, 23.0)
    };
    HostAdmission {
        fallback_first_token_ms: fallback,
        resident_first_token_ms: resident,
        fill_ms: 550.0,
        expected_reuses: reuses,
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    ensure!(
        args.len() == 7,
        "MODEL_DIR CORPUS PEER_SOCKET RESULT_JSON PREFIX_TOKENS MIX"
    );
    let model_dir = Path::new(&args[1]);
    let corpus = Path::new(&args[2]);
    let socket = PathBuf::from(&args[3]);
    let result = Path::new(&args[4]);
    ensure!(!result.exists(), "result already exists");
    let prefix_tokens: usize = args[5].parse()?;
    let mix = args[6].as_str();
    let (budget_mib, prepared_reuses, session_reuses) = match mix {
        "none" => (0_usize, 4_u64, 2_u64),
        "prepared" => (256, 4, 1),
        "session" => (256, 1, 4),
        "both" => (512, 4, 2),
        _ => anyhow::bail!("unknown mix"),
    };
    let tokenizer =
        Tokenizer::from_file(model_dir.join("tokenizer.json")).map_err(anyhow::Error::msg)?;
    let text = std::fs::read_to_string(corpus)?;
    let all_tokens = tokenizer.encode(text, false).map_err(anyhow::Error::msg)?;
    let all_tokens = all_tokens.get_ids();
    ensure!(
        all_tokens.len() >= prefix_tokens + SUFFIX + 2,
        "short corpus"
    );
    let tokens = all_tokens[..prefix_tokens + SUFFIX + 1].to_vec();
    let suffix = &tokens[prefix_tokens..prefix_tokens + SUFFIX];
    let capacity = tokens.len() + 8;
    let declared_bytes = geometry_bytes(model_dir, capacity)?;
    let fingerprint = fingerprint_model(model_dir)?;
    let key = SessionKey {
        namespace: "qwen-session-workflow-probe".into(),
        model_fingerprint: fingerprint.clone(),
        session_id: "session-a".into(),
    };
    let scope = PrefixScope {
        namespace: key.namespace.clone(),
        model_fingerprint: fingerprint,
        context_fingerprint: "tenant-a/wikitext-tokenizer+template/none".into(),
    };

    // The policy uses demand estimates and exact host payload bytes; it sees
    // neither physical file offsets nor future requests. A unit-valued tier
    // selects the bounded work before any peer bytes are fetched.
    let mut planner = BoundedHostTier::<&'static str, ()>::new(budget_mib * 1024 * 1024, 2);
    planner.offer(
        "prepared",
        (),
        declared_bytes,
        admission("prepared", prepared_reuses),
    )?;
    planner.offer(
        "session",
        (),
        declared_bytes,
        admission("session", session_reuses),
    )?;
    let warm_prepared = planner.get(&"prepared").is_some();
    let warm_session = planner.get(&"session").is_some();

    let startup_started = Instant::now();
    let bundle_limit = declared_bytes + 1024 * 1024;
    let worker = if warm_prepared || warm_session {
        let worker_socket = socket.clone();
        let worker_key = key.clone();
        let worker_scope = scope.clone();
        let worker_tokens = tokens.clone();
        let worker_model_dir = model_dir.to_path_buf();
        Some(thread::spawn(move || -> Result<_> {
            let started = Instant::now();
            let config: Config =
                serde_json::from_slice(&std::fs::read(worker_model_dir.join("config.json"))?)?;
            let mut peer = reader(&worker_socket, bundle_limit)?;
            let mut fetch_ms = 0.0;
            let mut prepare_ms = 0.0;
            let prepared = if warm_prepared {
                let fetching = Instant::now();
                let hit = peer
                    .find_longest_prepared_address(&worker_scope, &worker_tokens)?
                    .ok_or_else(|| anyhow::anyhow!("missing prepared prefix"))?;
                ensure!(hit.prefix_tokens == prefix_tokens, "wrong prepared prefix");
                let bundle = peer
                    .get_prefill_bundle(&hit.address, STATE_FORMAT)?
                    .ok_or_else(|| anyhow::anyhow!("missing prepared bundle"))?;
                fetch_ms += elapsed(fetching);
                ensure!(
                    bundle.token_ids == worker_tokens[..prefix_tokens],
                    "prepared tokens changed"
                );
                let preparing = Instant::now();
                let host = PreparedBf16::from_packed_offline(&config, &bundle.attention, capacity)?;
                prepare_ms += elapsed(preparing);
                ensure!(
                    host.resident_bytes() <= declared_bytes,
                    "prepared allocation exceeds declared bytes"
                );
                Some((hit.address, host))
            } else {
                None
            };
            let session = if warm_session {
                let fetching = Instant::now();
                let checkpoint = peer
                    .get_session_checkpoint(&worker_key, STATE_FORMAT)?
                    .ok_or_else(|| anyhow::anyhow!("missing session"))?;
                fetch_ms += elapsed(fetching);
                ensure!(
                    checkpoint.generation == 2 && checkpoint.bundle.token_ids == worker_tokens,
                    "session generation or tokens changed"
                );
                let preparing = Instant::now();
                let host = PreparedBf16::from_packed_offline(
                    &config,
                    &checkpoint.bundle.attention,
                    capacity,
                )?;
                prepare_ms += elapsed(preparing);
                ensure!(
                    host.resident_bytes() <= declared_bytes,
                    "session allocation exceeds declared bytes"
                );
                Some((checkpoint.checkpoint_address, host))
            } else {
                None
            };
            Ok((prepared, session, fetch_ms, prepare_ms, elapsed(started)))
        }))
    } else {
        None
    };
    let model_load_started = Instant::now();
    let target = Model::load(model_dir, "auto")?;
    let model_load_ms = elapsed(model_load_started);
    let wait_started = Instant::now();
    let (prepared, session, background_fetch_ms, background_prepare_ms, background_total_ms): Warmed = if let Some(worker) = worker {
        worker
            .join()
            .map_err(|_| anyhow::anyhow!("prefetch worker panicked"))??
    } else {
        (None, None, 0.0, 0.0, 0.0)
    };
    let worker_wait_ms = elapsed(wait_started);
    let mut tier = BoundedHostTier::<Address, PreparedBf16>::new(budget_mib * 1024 * 1024, 2);
    let host_admit_started = Instant::now();
    let mut prepared_address = None;
    if let Some((address, host)) = prepared {
        ensure!(
            tier.offer(
                address.clone(),
                host,
                declared_bytes,
                admission("prepared", prepared_reuses)
            )?,
            "prepared admission changed"
        );
        prepared_address = Some(address);
    }
    let mut session_address = None;
    if let Some((address, host)) = session {
        ensure!(
            tier.offer(
                address.clone(),
                host,
                declared_bytes,
                admission("session", session_reuses)
            )?,
            "session admission changed"
        );
        session_address = Some(address);
    }
    let host_admit_ms = elapsed(host_admit_started);
    let startup_total_ms = elapsed(startup_started);

    let mut peer = reader(&socket, bundle_limit)?;
    let prepared_started = Instant::now();
    let prepared_hit = prepared_address.is_some();
    let prepared_cache = if let Some(address) = &prepared_address {
        target.restore_prepared_bf16(tier.get(address).unwrap())?
    } else {
        let hit = peer
            .find_longest_prepared_address(&scope, &tokens)?
            .ok_or_else(|| anyhow::anyhow!("missing prepared prefix"))?;
        ensure!(hit.prefix_tokens == prefix_tokens, "wrong prepared prefix");
        let bundle = peer
            .get_prefill_bundle(&hit.address, STATE_FORMAT)?
            .ok_or_else(|| anyhow::anyhow!("missing prepared bundle"))?;
        ensure!(
            bundle.token_ids == tokens[..prefix_tokens],
            "prepared tokens changed"
        );
        target.restore_packed_bf16_parallel(&bundle.attention, capacity)?
    };
    let prepared_logits = target.forward_chunk(suffix, prefix_tokens, &prepared_cache)?;
    target.device.synchronize()?;
    let prepared_request_ms = elapsed(prepared_started);

    let session_started = Instant::now();
    let session_hit = session_address.is_some();
    let session_cache = if let Some(address) = &session_address {
        target.restore_prepared_bf16(tier.get(address).unwrap())?
    } else {
        let session = peer
            .get_session_checkpoint(&key, STATE_FORMAT)?
            .ok_or_else(|| anyhow::anyhow!("missing session"))?;
        ensure!(
            session.generation == 2 && session.bundle.token_ids == tokens,
            "session generation or tokens changed"
        );
        target.restore_packed_bf16_parallel(&session.bundle.attention, capacity)?
    };
    let session_logits = target.forward(&[42], tokens.len(), &session_cache)?;
    target.device.synchronize()?;
    let session_request_ms = elapsed(session_started);

    // The second model exists only for the accuracy oracle. Run it after
    // both timed requests so it cannot warm the serving GPU path.
    let oracle_load_started = Instant::now();
    let source = Model::load(model_dir, "auto")?;
    let oracle_model_load_ms = elapsed(oracle_load_started);
    let source_cache = source.empty_cache(capacity)?;
    let oracle_started = Instant::now();
    source.forward(&tokens[..prefix_tokens], 0, &source_cache)?;
    let prepared_expected = source.forward_chunk(suffix, prefix_tokens, &source_cache)?;
    source.forward(
        &[tokens[prefix_tokens + SUFFIX]],
        prefix_tokens + SUFFIX,
        &source_cache,
    )?;
    let session_expected = source.forward(&[42], tokens.len(), &source_cache)?;
    source.device.synchronize()?;
    let oracle_ms = elapsed(oracle_started);

    let prepared_diff = max_abs(&prepared_expected, &prepared_logits);
    let session_diff = max_abs(&session_expected, &session_logits);
    ensure!(prepared_diff <= 1e-3, "prepared logits changed");
    ensure!(session_diff <= 1e-3, "session logits changed");

    let output = json!({
        "model": "Qwen3-0.6B NVFP4 weights, exact BF16 KV",
        "transport": "separate yesnod Unix peer, full pipelined verification",
        "mix": mix, "budget_mib": budget_mib,
        "prepared_expected_reuses": prepared_reuses,
        "session_expected_reuses": session_reuses,
        "declared_host_bytes_per_entry": declared_bytes,
        "peer_bundle_limit_bytes": bundle_limit,
        "warm_prepared": warm_prepared, "warm_session": warm_session,
        "resident_bytes": tier.resident_bytes(),
        "model_load_ms": model_load_ms,
        "background_fetch_ms": background_fetch_ms,
        "background_prepare_ms": background_prepare_ms,
        "background_total_ms": background_total_ms,
        "worker_wait_ms": worker_wait_ms,
        "host_admit_ms": host_admit_ms,
        "startup_total_ms": startup_total_ms,
        "oracle_model_load_ms": oracle_model_load_ms,
        "oracle_ms": oracle_ms,
        "prepared_host_hit": prepared_hit,
        "prepared_request_ms": prepared_request_ms,
        "session_host_hit": session_hit,
        "session_request_ms": session_request_ms,
        "prepared_logit_diff": prepared_diff,
        "session_logit_diff": session_diff,
        "note": "Owner-coordinated immutable generation-2 session; serving request timers exclude model load, oracle computation, and background fetch."
    });
    std::fs::write(result, serde_json::to_vec_pretty(&output)?)?;
    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}
