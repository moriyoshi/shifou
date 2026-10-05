#[allow(dead_code)]
mod model {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/model.rs"));
}

use anyhow::{ensure, Result};
use model::Model;
use rayon::prelude::*;
use serde_json::json;
use sha2::{Digest, Sha256};
use shifou::{
    Address, BoundedHostTier, HostAdmission, PackedSnapshot, PeerBundleReader, PrefixScope,
    SessionKey,
};
use std::{
    io::Read,
    os::unix::net::UnixListener,
    path::{Path, PathBuf},
    sync::{Arc, RwLock},
    time::{Instant, SystemTime, UNIX_EPOCH},
};
use yesno_core::{Db, DbOptions};
use yesno_plugin::{
    abi::Role,
    channel::{send_fd, serve_blocking, Arena, Limits, Session},
    Host,
};

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

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn attention_hashes(attention: &[(Address, PackedSnapshot)], parallel: bool) -> Vec<[u8; 32]> {
    let hash = |(_, snapshot): &(Address, PackedSnapshot)| {
        let mut digest = Sha256::new();
        for buffer in &snapshot.buffers {
            digest.update(&buffer.bytes);
        }
        digest.finalize().into()
    };
    if parallel {
        attention.par_iter().map(hash).collect()
    } else {
        attention.iter().map(hash).collect()
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    ensure!(
        args.len() == 4 || args.len() == 5,
        "MODEL_DIR DB_DIR RESULT_JSON [PREFIX_TOKENS]"
    );
    let prefix_tokens: usize = args.get(4).map(|v| v.parse()).transpose()?.unwrap_or(1980);
    let model_dir = Path::new(&args[1]);
    let db_dir = Path::new(&args[2]);
    let result_path = Path::new(&args[3]);
    ensure!(!result_path.exists(), "result already exists");

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
    let external_socket = std::env::var_os("PEER_SOCKET").map(PathBuf::from);
    let (socket, server) = if let Some(socket) = external_socket {
        (socket, None)
    } else {
        let db = Db::open_with(
            db_dir.join("data"),
            DbOptions {
                shards: 1,
                ..DbOptions::default()
            },
        )?;
        let host = Host::new(Arc::new(RwLock::new(Some(Arc::new(db)))), 1, Role::Leader);
        let socket_root = std::fs::canonicalize(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../yesno/.agents-workspace/tmp"),
        )?;
        let socket = socket_root.join(format!(
            "p-{}-{}.sock",
            std::process::id(),
            SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()
        ));
        let listener = UnixListener::bind(&socket)?;
        let server = std::thread::spawn(move || -> Result<()> {
            let (stream, _) = listener.accept()?;
            let limits = Limits {
                max_handles: 2,
                max_lanes: 16,
                max_blocks: 16,
                max_snapshots: 2,
                max_writes: 1,
            };
            let mut session = Session::new(host, Arena::new(limits.arena_bytes())?, limits);
            send_fd(
                &stream,
                session
                    .arena_fd()
                    .ok_or_else(|| anyhow::anyhow!("missing peer arena"))?,
            )?;
            serve_blocking(&mut session, stream)?;
            Ok(())
        });
        (socket, Some(server))
    };
    let parallel_peer_verification = std::env::var("SHIFOU_PARALLEL_VERIFY").as_deref() == Ok("1");
    let pipelined_peer_verification = std::env::var("SHIFOU_PIPELINE_VERIFY").as_deref() == Ok("1");
    let mut reader = PeerBundleReader::connect(&socket)?
        .with_parallel_verification(parallel_peer_verification)
        .with_pipelined_verification(pipelined_peer_verification);

    let start = Instant::now();
    let session = reader
        .get_session_checkpoint(&key, STATE_FORMAT)?
        .ok_or_else(|| anyhow::anyhow!("missing session"))?;
    let session_read_ms = elapsed(start);
    ensure!(session.generation == 2, "wrong session generation");
    let tokens = &session.bundle.token_ids;
    ensure!(
        tokens.len() == prefix_tokens + 33,
        "unexpected session token stream"
    );

    let start = Instant::now();
    let hit = reader
        .find_longest_prepared_address(&scope, tokens)?
        .ok_or_else(|| anyhow::anyhow!("missing prepared prefix"))?;
    let prepared_lookup_ms = elapsed(start);
    ensure!(hit.prefix_tokens == prefix_tokens, "wrong prepared prefix");
    let start = Instant::now();
    let prepared = reader
        .get_prefill_bundle(&hit.address, STATE_FORMAT)?
        .ok_or_else(|| anyhow::anyhow!("missing prepared bundle"))?;
    let prepared_read_ms = elapsed(start);
    ensure!(
        prepared.token_ids == tokens[..prefix_tokens],
        "wrong prefix tokens"
    );

    let source = Model::load(model_dir, "auto")?;
    let target = Model::load(model_dir, "auto")?;
    let capacity = tokens.len() + 8;
    let source_prepared = source.empty_cache(capacity)?;
    let start = Instant::now();
    source.forward(&prepared.token_ids, 0, &source_prepared)?;
    source.device.synchronize()?;
    let fresh_prepared_prefill_ms = elapsed(start);
    let start = Instant::now();
    let parallel_restore = std::env::var("SHIFOU_PARALLEL_RESTORE").as_deref() == Ok("1");
    let restored_prepared = if parallel_restore {
        target.restore_packed_bf16_parallel(&prepared.attention, capacity)?
    } else {
        target.restore_packed_bf16(&prepared.attention, capacity)?
    };
    target.device.synchronize()?;
    let prepared_gpu_restore_ms = elapsed(start);
    let suffix = &tokens[prefix_tokens..prefix_tokens + 32];
    let start = Instant::now();
    let prepared_expected = source.forward_chunk(suffix, prefix_tokens, &source_prepared)?;
    source.device.synchronize()?;
    let fresh_suffix_ms = elapsed(start);
    let start = Instant::now();
    let actual = target.forward_chunk(suffix, prefix_tokens, &restored_prepared)?;
    target.device.synchronize()?;
    let restored_suffix_ms = elapsed(start);
    let prepared_logit_diff = max_abs(&prepared_expected, &actual);
    ensure!(prepared_logit_diff <= 1e-3, "prepared logits changed");

    let source_session = source.empty_cache(capacity)?;
    let start = Instant::now();
    source.forward(&tokens[..prefix_tokens], 0, &source_session)?;
    source.forward_chunk(suffix, prefix_tokens, &source_session)?;
    source.forward(
        &[tokens[prefix_tokens + 32]],
        prefix_tokens + 32,
        &source_session,
    )?;
    source.device.synchronize()?;
    let fresh_session_prefill_ms = elapsed(start);
    let start = Instant::now();
    let restored_session = if parallel_restore {
        target.restore_packed_bf16_parallel(&session.bundle.attention, capacity)?
    } else {
        target.restore_packed_bf16(&session.bundle.attention, capacity)?
    };
    target.device.synchronize()?;
    let session_gpu_restore_ms = elapsed(start);
    let next = 42;
    let start = Instant::now();
    let expected = source.forward(&[next], tokens.len(), &source_session)?;
    source.device.synchronize()?;
    let fresh_next_ms = elapsed(start);
    let start = Instant::now();
    let actual = target.forward(&[next], tokens.len(), &restored_session)?;
    target.device.synchronize()?;
    let restored_next_ms = elapsed(start);
    let session_logit_diff = max_abs(&expected, &actual);
    ensure!(
        session_logit_diff <= 1e-3,
        "session logits changed: {session_logit_diff}"
    );

    let start = Instant::now();
    let prepared_host = target.prepare_packed_bf16(&prepared.attention, capacity)?;
    let prepared_host_setup_ms = elapsed(start);
    let start = Instant::now();
    let cache = target.restore_prepared_bf16(&prepared_host)?;
    let host_prepared_logits = target.forward_chunk(suffix, prefix_tokens, &cache)?;
    target.device.synchronize()?;
    let prepared_host_hit_ms = elapsed(start);
    ensure!(
        max_abs(&host_prepared_logits, &prepared_expected) <= 1e-3,
        "prepared host logits changed"
    );

    let start = Instant::now();
    let session_host = target.prepare_packed_bf16(&session.bundle.attention, capacity)?;
    let session_host_setup_ms = elapsed(start);
    let start = Instant::now();
    let cache = target.restore_prepared_bf16(&session_host)?;
    let host_session_logits = target.forward(&[next], tokens.len(), &cache)?;
    target.device.synchronize()?;
    let session_host_hit_ms = elapsed(start);
    ensure!(
        max_abs(&host_session_logits, &expected) <= 1e-3,
        "session host logits changed"
    );

    let mut serial_decode_ms = Vec::new();
    let mut parallel_decode_ms = Vec::new();
    if std::env::var("SHIFOU_BENCH_DECODE").as_deref() == Ok("1") {
        for iteration in 0..6 {
            if iteration % 2 == 0 {
                let start = Instant::now();
                let prepared = target.prepare_packed_bf16(&session.bundle.attention, capacity)?;
                serial_decode_ms.push(elapsed(start));
                drop(std::hint::black_box(prepared));
            }
            let start = Instant::now();
            let prepared =
                target.prepare_packed_bf16_parallel(&session.bundle.attention, capacity)?;
            parallel_decode_ms.push(elapsed(start));
            drop(std::hint::black_box(prepared));
            if iteration % 2 != 0 {
                let start = Instant::now();
                let prepared = target.prepare_packed_bf16(&session.bundle.attention, capacity)?;
                serial_decode_ms.push(elapsed(start));
                drop(std::hint::black_box(prepared));
            }
        }
    }

    let mut serial_hash_ms = Vec::new();
    let mut parallel_hash_ms = Vec::new();
    if std::env::var("SHIFOU_BENCH_HASH").as_deref() == Ok("1") {
        for iteration in 0..6 {
            let serial;
            let parallel;
            if iteration % 2 == 0 {
                let start = Instant::now();
                serial = attention_hashes(&session.bundle.attention, false);
                serial_hash_ms.push(elapsed(start));
                let start = Instant::now();
                parallel = attention_hashes(&session.bundle.attention, true);
                parallel_hash_ms.push(elapsed(start));
            } else {
                let start = Instant::now();
                parallel = attention_hashes(&session.bundle.attention, true);
                parallel_hash_ms.push(elapsed(start));
                let start = Instant::now();
                serial = attention_hashes(&session.bundle.attention, false);
                serial_hash_ms.push(elapsed(start));
            }
            ensure!(serial == parallel, "parallel record hashes changed");
        }
    }

    // This optional probe keeps the request path and asynchronous fill cost
    // separate. Keys are scoped to the immutable checkpoint observed above.
    let bounded_tier = if std::env::var("SHIFOU_BENCH_BOUNDED_TIER").as_deref() == Ok("1") {
        let prepared_bytes = target.prepared_bf16_bytes(&prepared.attention, capacity)?;
        let session_bytes = target.prepared_bf16_bytes(&session.bundle.attention, capacity)?;
        let mut budgets = Vec::new();
        let mut limits = [0_usize, 256, 512];
        let rotation = std::env::var("SHIFOU_BOUNDED_ROTATION")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(0)
            % limits.len();
        limits.rotate_left(rotation);
        for budget_mib in limits {
            let mut tier =
                BoundedHostTier::<Address, model::PreparedBf16>::new(budget_mib * 1024 * 1024, 2);
            let mut rows = Vec::new();
            for kind in [
                "prepared", "session", "prepared", "prepared", "session", "prepared", "session",
            ] {
                let cache_key = if kind == "prepared" {
                    hit.address.clone()
                } else {
                    session.checkpoint_address.clone()
                };
                let started = Instant::now();
                let resident = tier.get(&cache_key);
                let host_hit = resident.is_some();
                let mut peer_read_ms = 0.0;
                let mut gpu_restore_ms = 0.0;
                let mut fill_ms = 0.0;
                let (logit_diff, admitted) = if let Some(host) = resident {
                    let cache = target.restore_prepared_bf16(host)?;
                    let logits = if kind == "prepared" {
                        target.forward_chunk(suffix, prefix_tokens, &cache)?
                    } else {
                        target.forward(&[next], tokens.len(), &cache)?
                    };
                    target.device.synchronize()?;
                    let expected_logits = if kind == "prepared" {
                        &prepared_expected
                    } else {
                        &expected
                    };
                    (max_abs(expected_logits, &logits), false)
                } else if kind == "prepared" {
                    let read_started = Instant::now();
                    let lookup = reader
                        .find_longest_prepared_address(&scope, tokens)?
                        .ok_or_else(|| anyhow::anyhow!("missing prepared lookup"))?;
                    ensure!(
                        lookup.address == hit.address,
                        "prepared publication changed"
                    );
                    let bundle = reader
                        .get_prefill_bundle(&lookup.address, STATE_FORMAT)?
                        .ok_or_else(|| anyhow::anyhow!("missing prepared bundle"))?;
                    peer_read_ms = elapsed(read_started);
                    let restore_started = Instant::now();
                    let cache = if parallel_restore {
                        target.restore_packed_bf16_parallel(&bundle.attention, capacity)?
                    } else {
                        target.restore_packed_bf16(&bundle.attention, capacity)?
                    };
                    target.device.synchronize()?;
                    gpu_restore_ms = elapsed(restore_started);
                    let logits = target.forward_chunk(suffix, prefix_tokens, &cache)?;
                    target.device.synchronize()?;
                    let diff = max_abs(&prepared_expected, &logits);
                    ensure!(diff <= 1e-3, "bounded prepared logits changed");
                    let request_ms = elapsed(started);
                    let fill_started = Instant::now();
                    let admitted = tier.offer_with(
                        cache_key.clone(),
                        prepared_bytes,
                        HostAdmission {
                            fallback_first_token_ms: prepared_lookup_ms
                                + prepared_read_ms
                                + prepared_gpu_restore_ms
                                + restored_suffix_ms,
                            resident_first_token_ms: prepared_host_hit_ms,
                            fill_ms: prepared_host_setup_ms,
                            expected_reuses: 4,
                        },
                        || {
                            target
                                .prepare_packed_bf16_parallel(&bundle.attention, capacity)
                                .map_err(|error| shifou::Error::Invalid(error.to_string()))
                        },
                    )?;
                    fill_ms = elapsed(fill_started);
                    if admitted {
                        ensure!(
                            tier.get(&cache_key).unwrap().resident_bytes() <= prepared_bytes,
                            "declared prepared bytes undercount allocation"
                        );
                    }
                    rows.push(json!({"kind":kind,"host_hit":false,"admitted":admitted,
                        "request_ms":request_ms,"peer_read_ms":peer_read_ms,
                        "gpu_restore_ms":gpu_restore_ms,"fill_ms":fill_ms,"logit_diff":diff,
                        "resident_bytes":tier.resident_bytes()}));
                    continue;
                } else {
                    let read_started = Instant::now();
                    let checkpoint = reader
                        .get_session_checkpoint(&key, STATE_FORMAT)?
                        .ok_or_else(|| anyhow::anyhow!("missing session checkpoint"))?;
                    ensure!(
                        checkpoint.generation == session.generation,
                        "session publication changed"
                    );
                    peer_read_ms = elapsed(read_started);
                    let restore_started = Instant::now();
                    let cache = if parallel_restore {
                        target
                            .restore_packed_bf16_parallel(&checkpoint.bundle.attention, capacity)?
                    } else {
                        target.restore_packed_bf16(&checkpoint.bundle.attention, capacity)?
                    };
                    target.device.synchronize()?;
                    gpu_restore_ms = elapsed(restore_started);
                    let logits = target.forward(&[next], tokens.len(), &cache)?;
                    target.device.synchronize()?;
                    let diff = max_abs(&expected, &logits);
                    ensure!(diff <= 1e-3, "bounded session logits changed");
                    let request_ms = elapsed(started);
                    let fill_started = Instant::now();
                    let admitted = tier.offer_with(
                        cache_key.clone(),
                        session_bytes,
                        HostAdmission {
                            fallback_first_token_ms: session_read_ms
                                + session_gpu_restore_ms
                                + restored_next_ms,
                            resident_first_token_ms: session_host_hit_ms,
                            fill_ms: session_host_setup_ms,
                            expected_reuses: 2,
                        },
                        || {
                            target
                                .prepare_packed_bf16_parallel(
                                    &checkpoint.bundle.attention,
                                    capacity,
                                )
                                .map_err(|error| shifou::Error::Invalid(error.to_string()))
                        },
                    )?;
                    fill_ms = elapsed(fill_started);
                    if admitted {
                        ensure!(
                            tier.get(&cache_key).unwrap().resident_bytes() <= session_bytes,
                            "declared session bytes undercount allocation"
                        );
                    }
                    rows.push(json!({"kind":kind,"host_hit":false,"admitted":admitted,
                        "request_ms":request_ms,"peer_read_ms":peer_read_ms,
                        "gpu_restore_ms":gpu_restore_ms,"fill_ms":fill_ms,"logit_diff":diff,
                        "resident_bytes":tier.resident_bytes()}));
                    continue;
                };
                ensure!(logit_diff <= 1e-3, "bounded tier logits changed");
                rows.push(json!({"kind":kind,"host_hit":host_hit,"admitted":admitted,
                    "request_ms":elapsed(started),"peer_read_ms":peer_read_ms,
                    "gpu_restore_ms":gpu_restore_ms,"fill_ms":fill_ms,"logit_diff":logit_diff,
                    "resident_bytes":tier.resident_bytes()}));
            }
            budgets.push(json!({"budget_mib":budget_mib,"rows":rows,
                "final_resident_bytes":tier.resident_bytes()}));
        }
        Some(json!({"prepared_declared_bytes":prepared_bytes,
            "session_declared_bytes":session_bytes,
            "budgets":budgets,
            "note":"Warm peer, immutable generation-2 checkpoint; prior-run measured costs seed admission. Request time excludes later host fill."}))
    } else {
        None
    };

    let output = json!({
        "transport": "yesno peer socket with shared-memory arena, full verification",
        "bounded_tier": bounded_tier,
        "parallel_cpu_restore": parallel_restore,
        "serial_decode_ms": serial_decode_ms,
        "parallel_decode_ms": parallel_decode_ms,
        "serial_hash_ms": serial_hash_ms,
        "parallel_hash_ms": parallel_hash_ms,
        "external_yesnod": server.is_none(),
        "parallel_peer_verification": parallel_peer_verification,
        "pipelined_peer_verification": pipelined_peer_verification,
        "prefix_tokens": prefix_tokens,
        "session_tokens": tokens.len(),
        "prepared_attention_records": prepared.attention.len(),
        "prepared_attention_bytes": prepared.attention.iter().flat_map(|(_, snapshot)| &snapshot.buffers).map(|buffer| buffer.bytes.len()).sum::<usize>(),
        "session_attention_bytes": session.bundle.attention.iter().flat_map(|(_, snapshot)| &snapshot.buffers).map(|buffer| buffer.bytes.len()).sum::<usize>(),
        "prepared_lookup_ms": prepared_lookup_ms,
        "prepared_read_ms": prepared_read_ms,
        "prepared_gpu_restore_ms": prepared_gpu_restore_ms,
        "fresh_prepared_prefill_ms": fresh_prepared_prefill_ms,
        "fresh_suffix_ms": fresh_suffix_ms,
        "restored_suffix_ms": restored_suffix_ms,
        "prepared_fresh_total_ms": fresh_prepared_prefill_ms + fresh_suffix_ms,
        "prepared_hit_total_ms": prepared_lookup_ms + prepared_read_ms + prepared_gpu_restore_ms + restored_suffix_ms,
        "prepared_logit_diff": prepared_logit_diff,
        "session_read_ms": session_read_ms,
        "session_gpu_restore_ms": session_gpu_restore_ms,
        "fresh_session_prefill_ms": fresh_session_prefill_ms,
        "fresh_next_ms": fresh_next_ms,
        "restored_next_ms": restored_next_ms,
        "session_fresh_total_ms": fresh_session_prefill_ms + fresh_next_ms,
        "session_hit_total_ms": session_read_ms + session_gpu_restore_ms + restored_next_ms,
        "session_logit_diff": session_logit_diff,
        "prepared_host_setup_ms": prepared_host_setup_ms,
        "prepared_host_hit_ms": prepared_host_hit_ms,
        "session_host_setup_ms": session_host_setup_ms,
        "session_host_hit_ms": session_host_hit_ms,
        "note": "One GB10; two model instances; server and client share a host; model load and server startup excluded."
    });
    std::fs::write(result_path, serde_json::to_vec_pretty(&output)?)?;
    println!("{}", serde_json::to_string_pretty(&output)?);
    drop(reader);
    if let Some(server) = server {
        server
            .join()
            .map_err(|_| anyhow::anyhow!("peer server panicked"))??;
        std::fs::remove_file(socket)?;
    }
    Ok(())
}
