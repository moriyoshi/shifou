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
    sync::mpsc::{self, Receiver},
    thread,
    time::Instant,
};
use tokenizers::Tokenizer;
use xinfer::utils::config::Config;

const STATE_FORMAT: &str = "qwen3/no-recurrent-state/v1";
const SUFFIX: usize = 32;
type WarmTimes = (f64, f64, f64);

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

fn percentile(values: &[f64], q: f64) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[((sorted.len() as f64 * q).ceil() as usize)
        .saturating_sub(1)
        .min(sorted.len() - 1)]
}

struct WarmEvent {
    kind: &'static str,
    address: Address,
    host: PreparedBf16,
    ready_ms: f64,
}

struct ReadyState {
    tier: BoundedHostTier<Address, PreparedBf16>,
    prepared_address: Option<Address>,
    session_address: Option<Address>,
    prepared_ready_ms: Option<f64>,
    session_ready_ms: Option<f64>,
    declared_bytes: usize,
    prepared_reuses: u64,
    session_reuses: u64,
}

impl ReadyState {
    fn new(
        budget_bytes: usize,
        declared_bytes: usize,
        prepared_reuses: u64,
        session_reuses: u64,
    ) -> Self {
        Self {
            tier: BoundedHostTier::new(budget_bytes, 2),
            prepared_address: None,
            session_address: None,
            prepared_ready_ms: None,
            session_ready_ms: None,
            declared_bytes,
            prepared_reuses,
            session_reuses,
        }
    }

    fn drain(&mut self, receiver: &Receiver<WarmEvent>) -> Result<()> {
        while let Ok(event) = receiver.try_recv() {
            let reuses = if event.kind == "prepared" {
                self.prepared_reuses
            } else {
                self.session_reuses
            };
            ensure!(
                self.tier.offer(
                    event.address.clone(),
                    event.host,
                    self.declared_bytes,
                    admission(event.kind, reuses)
                )?,
                "ready entry was not admitted"
            );
            if event.kind == "prepared" {
                self.prepared_address = Some(event.address);
                self.prepared_ready_ms = Some(event.ready_ms);
            } else {
                self.session_address = Some(event.address);
                self.session_ready_ms = Some(event.ready_ms);
            }
        }
        Ok(())
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    ensure!(
        args.len() == 8,
        "MODEL_DIR CORPUS PEER_SOCKET RESULT_JSON PREFIX_TOKENS MIX ACTIVE_REQUESTS"
    );
    let model_dir = Path::new(&args[1]);
    let corpus = Path::new(&args[2]);
    let socket = PathBuf::from(&args[3]);
    let result = Path::new(&args[4]);
    ensure!(!result.exists(), "result already exists");
    let prefix_tokens: usize = args[5].parse()?;
    let mix = args[6].as_str();
    let iterations: usize = args[7].parse()?;
    ensure!(iterations > 0, "active request count must be positive");
    let (budget_mib, prepared_reuses, session_reuses) = match mix {
        "none" => (0_usize, 4_u64, 2_u64),
        "prepared" => (256, 4, 1),
        "session" => (256, 1, 4),
        "both" | "both_parallel" => (512, 4, 2),
        _ => anyhow::bail!("unknown mix"),
    };
    let tokenizer =
        Tokenizer::from_file(model_dir.join("tokenizer.json")).map_err(anyhow::Error::msg)?;
    let text = std::fs::read_to_string(corpus)?;
    let encoded = tokenizer.encode(text, false).map_err(anyhow::Error::msg)?;
    let all_tokens = encoded.get_ids();
    let fixture_len = prefix_tokens + SUFFIX + 1;
    ensure!(all_tokens.len() >= fixture_len, "short corpus");
    let tokens = all_tokens[..fixture_len].to_vec();
    let mut active_tokens = tokens.clone();
    active_tokens.rotate_left(31);
    let suffix = &tokens[prefix_tokens..prefix_tokens + SUFFIX];
    let capacity = fixture_len + 8;
    let declared_bytes = geometry_bytes(model_dir, capacity)?;
    let bundle_limit = declared_bytes + 1024 * 1024;
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
    let model = Model::load(model_dir, "auto")?;

    // Establish exact oracles and a distinct hot session without reading yesno.
    // All setup and GPU warm-up finish before the timed contention window.
    let fixture_cache = model.empty_cache(capacity)?;
    model.forward(&tokens[..prefix_tokens], 0, &fixture_cache)?;
    let prepared_expected = model.forward_chunk(suffix, prefix_tokens, &fixture_cache)?;
    model.forward(
        &[tokens[prefix_tokens + SUFFIX]],
        prefix_tokens + SUFFIX,
        &fixture_cache,
    )?;
    let session_expected = model.forward(&[42], tokens.len(), &fixture_cache)?;
    model.device.synchronize()?;
    drop(fixture_cache);
    let active_cache = model.empty_cache(capacity)?;
    model.forward(&active_tokens[..prefix_tokens], 0, &active_cache)?;
    model.forward_chunk(
        &active_tokens[prefix_tokens..prefix_tokens + SUFFIX],
        prefix_tokens,
        &active_cache,
    )?;
    model.forward(
        &[active_tokens[prefix_tokens + SUFFIX]],
        prefix_tokens + SUFFIX,
        &active_cache,
    )?;
    let attention: Vec<_> = model
        .export_packed_bf16(&active_cache, fixture_len)?
        .into_iter()
        .enumerate()
        .map(|(index, snapshot)| {
            (
                Address {
                    namespace: "active-unpublished".into(),
                    model_fingerprint: "active-unpublished".into(),
                    prefix_fingerprint: "active-unpublished".into(),
                    layer: (index / 2) as u32,
                    slot: if index.is_multiple_of(2) {
                        "key"
                    } else {
                        "value"
                    }
                    .into(),
                },
                snapshot,
            )
        })
        .collect();
    let active_host = model.prepare_packed_bf16_parallel(&attention, capacity)?;
    ensure!(
        active_host.resident_bytes() <= declared_bytes,
        "active host bytes exceed estimate"
    );
    drop(attention);
    drop(active_cache);
    let cache = model.restore_prepared_bf16(&active_host)?;
    let active_expected = model.forward(&[42], fixture_len, &cache)?;
    model.device.synchronize()?;
    drop(cache);

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
    let (sender, receiver) = mpsc::channel::<WarmEvent>();
    let mut ready = ReadyState::new(
        budget_mib * 1024 * 1024,
        declared_bytes,
        prepared_reuses,
        session_reuses,
    );
    let background_origin = Instant::now();
    let spawn_worker = |warm_prepared: bool,
                        warm_session: bool,
                        sender: mpsc::Sender<WarmEvent>| {
        let worker_socket = socket.clone();
        let worker_model_dir = model_dir.to_path_buf();
        let worker_scope = scope.clone();
        let worker_key = key.clone();
        let worker_tokens = tokens.clone();
        thread::spawn(move || -> Result<WarmTimes> {
            let started = background_origin;
            let config: Config =
                serde_json::from_slice(&std::fs::read(worker_model_dir.join("config.json"))?)?;
            let mut peer = reader(&worker_socket, bundle_limit)?;
            let mut fetch_ms = 0.0;
            let mut prepare_ms = 0.0;
            if warm_prepared {
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
                    "prepared host bytes exceed estimate"
                );
                sender
                    .send(WarmEvent {
                        kind: "prepared",
                        address: hit.address,
                        host,
                        ready_ms: elapsed(started),
                    })
                    .map_err(|_| anyhow::anyhow!("ready receiver closed"))?;
            }
            if warm_session {
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
                    "session host bytes exceed estimate"
                );
                sender
                    .send(WarmEvent {
                        kind: "session",
                        address: checkpoint.checkpoint_address,
                        host,
                        ready_ms: elapsed(started),
                    })
                    .map_err(|_| anyhow::anyhow!("ready receiver closed"))?;
            }
            Ok((fetch_ms, prepare_ms, elapsed(started)))
        })
    };
    let workers = if mix == "both_parallel" {
        vec![
            spawn_worker(true, false, sender.clone()),
            spawn_worker(false, true, sender),
        ]
    } else if warm_prepared || warm_session {
        vec![spawn_worker(warm_prepared, warm_session, sender)]
    } else {
        Vec::new()
    };
    let worker_count = workers.len();

    let active_started = Instant::now();
    let mut active_rows = Vec::with_capacity(iterations);
    let mut active_ms = Vec::with_capacity(iterations);
    let mut overlap_count = 0;
    for index in 0..iterations {
        let overlapped = workers.iter().any(|handle| !handle.is_finished());
        if overlapped {
            overlap_count += 1;
        }
        let started = Instant::now();
        let cache = model.restore_prepared_bf16(&active_host)?;
        let actual = model.forward(&[42], fixture_len, &cache)?;
        model.device.synchronize()?;
        let ms = elapsed(started);
        let diff = max_abs(&active_expected, &actual);
        ensure!(diff <= 1e-3, "active logits changed");
        active_rows
            .push(json!({"request":index,"ms":ms,"overlapped":overlapped,"logit_diff":diff}));
        active_ms.push(ms);
        ready.drain(&receiver)?;
    }
    let active_loop_ms = elapsed(active_started);
    ready.drain(&receiver)?;

    let mut peer = reader(&socket, bundle_limit)?;
    let prepared_started = Instant::now();
    let prepared_hit = ready.prepared_address.is_some();
    let cache = if let Some(address) = &ready.prepared_address {
        model.restore_prepared_bf16(ready.tier.get(address).unwrap())?
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
        model.restore_packed_bf16_parallel(&bundle.attention, capacity)?
    };
    let prepared_logits = model.forward_chunk(suffix, prefix_tokens, &cache)?;
    model.device.synchronize()?;
    let prepared_request_ms = elapsed(prepared_started);
    let prepared_diff = max_abs(&prepared_expected, &prepared_logits);
    ensure!(prepared_diff <= 1e-3, "prepared logits changed");

    ready.drain(&receiver)?;
    let session_started = Instant::now();
    let session_hit = ready.session_address.is_some();
    let cache = if let Some(address) = &ready.session_address {
        model.restore_prepared_bf16(ready.tier.get(address).unwrap())?
    } else {
        let checkpoint = peer
            .get_session_checkpoint(&key, STATE_FORMAT)?
            .ok_or_else(|| anyhow::anyhow!("missing session"))?;
        ensure!(
            checkpoint.generation == 2 && checkpoint.bundle.token_ids == tokens,
            "session generation or tokens changed"
        );
        model.restore_packed_bf16_parallel(&checkpoint.bundle.attention, capacity)?
    };
    let session_logits = model.forward(&[42], fixture_len, &cache)?;
    model.device.synchronize()?;
    let session_request_ms = elapsed(session_started);
    let session_diff = max_abs(&session_expected, &session_logits);
    ensure!(session_diff <= 1e-3, "session logits changed");

    let resident_bytes_at_requests = ready.tier.resident_bytes();
    let waiting = Instant::now();
    let mut background_fetch_ms = 0.0;
    let mut background_prepare_ms = 0.0;
    let mut background_total_ms: f64 = 0.0;
    for worker in workers {
        let (fetch, prepare, total) = worker
            .join()
            .map_err(|_| anyhow::anyhow!("background worker panicked"))??;
        background_fetch_ms += fetch;
        background_prepare_ms += prepare;
        background_total_ms = background_total_ms.max(total);
    }
    let worker_wait_ms = elapsed(waiting);
    ready.drain(&receiver)?;

    let output = json!({
        "model": "Qwen3-0.6B NVFP4 weights, BF16 KV",
        "transport": "separate yesnod Unix peer, full pipelined verification",
        "mode": mix, "active_requests": iterations,
        "active_tokens": "fixture token sequence rotated left 31",
        "budget_mib": budget_mib,
        "resident_bytes_at_requests": resident_bytes_at_requests,
        "resident_bytes_after_worker": ready.tier.resident_bytes(),
        "prepared_ready_ms": ready.prepared_ready_ms,
        "session_ready_ms": ready.session_ready_ms,
        "active_host_bytes_outside_speculative_budget": active_host.resident_bytes(),
        "peer_bundle_limit_bytes": bundle_limit,
        "warm_prepared": warm_prepared, "warm_session": warm_session,
        "background_workers": worker_count,
        "active_loop_ms": active_loop_ms,
        "active_p50_ms": percentile(&active_ms, 0.5),
        "active_p95_ms": percentile(&active_ms, 0.95),
        "active_overlap_count": overlap_count,
        "active_rows": active_rows,
        "worker_wait_ms": worker_wait_ms,
        "background_fetch_ms": background_fetch_ms,
        "background_prepare_ms": background_prepare_ms,
        "background_total_ms": background_total_ms,
        "prepared_host_hit": prepared_hit,
        "prepared_request_ms": prepared_request_ms,
        "session_host_hit": session_hit,
        "session_request_ms": session_request_ms,
        "prepared_logit_diff": prepared_diff,
        "session_logit_diff": session_diff,
        "note": "Active requests use a separate freshly computed session. The speculative tier budget excludes that already-hot host state."
    });
    std::fs::write(result, serde_json::to_vec_pretty(&output)?)?;
    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}
