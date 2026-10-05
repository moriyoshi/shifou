use shifou::{
    Address, AdmissionEstimate, Cache, CacheReader, CacheTier, Error, HitChoice, PackedBuffer,
    PackedSnapshot, PrefillBundle, PrefixScope, SessionKey, TierCost,
};
use std::sync::atomic::{AtomicUsize, Ordering};

fn directory() -> std::path::PathBuf {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(".agents-workspace/tmp/tests");
    std::fs::create_dir_all(&root).unwrap();
    let path = root.join(format!(
        "sessions-{}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir(&path).unwrap();
    path
}

fn bundle(tokens: &[u32]) -> PrefillBundle {
    PrefillBundle {
        prefix_tokens: tokens.len() as u64,
        token_ids: tokens.to_vec(),
        state_format: "engine-state/v1".into(),
        state_bytes: vec![0x55, 0, 0xaa, tokens.len() as u8],
        attention: vec![(
            Address {
                namespace: "input".into(),
                model_fingerprint: "input".into(),
                prefix_fingerprint: "input".into(),
                layer: 0,
                slot: "attention-k".into(),
            },
            PackedSnapshot {
                format: "engine-kv/v1".into(),
                buffers: vec![PackedBuffer {
                    name: "k".into(),
                    dtype: "u8".into(),
                    shape: vec![3],
                    bytes: vec![1, 0, tokens.len() as u8],
                }],
            },
        )],
    }
}

fn append_bundle(tokens: &[u32]) -> PrefillBundle {
    let mut value = bundle(tokens);
    let buffer = &mut value.attention[0].1.buffers[0];
    buffer.bytes = tokens.iter().map(|token| *token as u8).collect();
    buffer.shape = vec![buffer.bytes.len()];
    value
}

#[test]
fn append_checkpoint_survives_reopen_and_protects_only_live_base() {
    let path = directory();
    let mut writer = Cache::open(&path).unwrap();
    let key = SessionKey {
        namespace: "tenant-a".into(),
        model_fingerprint: "weights+execution".into(),
        session_id: "append-session".into(),
    };
    let one = append_bundle(&[1, 2, 3]);
    let two = append_bundle(&[1, 2, 3, 4]);
    let three = append_bundle(&[1, 2, 3, 4, 5]);
    writer
        .put_session_checkpoint(&key, 1, &one, "workflow/v1", b"turn=1")
        .unwrap();
    writer
        .put_session_append_checkpoint(&key, 2, &two, "workflow/v1", b"turn=2")
        .unwrap();
    assert!(matches!(
        writer.prune_session_checkpoint(&key, 1, &one.token_ids),
        Err(Error::Invalid(_))
    ));
    let restored_two = writer
        .get_session_checkpoint(&key, "engine-state/v1")
        .unwrap()
        .unwrap();
    assert_eq!(restored_two.bundle.token_ids, two.token_ids);
    assert_eq!(restored_two.bundle.state_bytes, two.state_bytes);
    assert_eq!(restored_two.bundle.attention[0].1, two.attention[0].1);
    let mut mutated = three.clone();
    mutated.attention[0].1.buffers[0].bytes[0] ^= 1;
    assert!(matches!(
        writer.put_session_append_checkpoint(&key, 3, &mutated, "workflow/v1", b"bad"),
        Err(Error::Invalid(_))
    ));
    assert_eq!(
        writer
            .get_session_checkpoint(&key, "engine-state/v1")
            .unwrap()
            .unwrap()
            .generation,
        2
    );
    writer
        .put_session_append_checkpoint(&key, 3, &three, "workflow/v1", b"turn=3")
        .unwrap();
    assert!(writer
        .prune_session_checkpoint(&key, 2, &two.token_ids)
        .unwrap());
    drop(writer);
    let reader = CacheReader::open(&path).unwrap();
    let restored = reader
        .get_session_checkpoint(&key, "engine-state/v1")
        .unwrap()
        .unwrap();
    assert_eq!(restored.generation, 3);
    assert_eq!(restored.bundle.token_ids, three.token_ids);
    assert_eq!(restored.bundle.state_bytes, three.state_bytes);
    assert_eq!(restored.bundle.attention[0].1, three.attention[0].1);
    assert_eq!(restored.workflow_bytes, b"turn=3");
    drop(reader);

    let mut writer = Cache::open(&path).unwrap();
    let four = append_bundle(&[1, 2, 3, 4, 5, 6]);
    writer
        .put_session_checkpoint(&key, 4, &four, "workflow/v1", b"turn=4")
        .unwrap();
    assert!(writer
        .prune_session_checkpoint(&key, 1, &one.token_ids)
        .unwrap());
}

#[test]
fn append_checks_the_complete_committed_multibuffer_payload() {
    let path = directory();
    let mut writer = Cache::open(&path).unwrap();
    let key = SessionKey {
        namespace: "tenant-a".into(),
        model_fingerprint: "weights+execution".into(),
        session_id: "multibuffer-session".into(),
    };
    let mut first = append_bundle(&[1, 2, 3]);
    first.attention[0].1.buffers.push(PackedBuffer {
        name: "v".into(),
        dtype: "u8".into(),
        shape: vec![3],
        bytes: vec![8, 9, 10],
    });
    let mut second = first.clone();
    second.token_ids.push(4);
    second.prefix_tokens += 1;
    second.state_bytes.push(4);
    for (index, buffer) in second.attention[0].1.buffers.iter_mut().enumerate() {
        buffer.bytes.push((index + 4) as u8);
        buffer.shape = vec![4];
    }
    writer
        .put_session_checkpoint(&key, 1, &first, "workflow/v1", b"turn=1")
        .unwrap();
    let mut bad = second.clone();
    bad.attention[0].1.buffers[1].bytes[0] ^= 1;
    assert!(matches!(
        writer.put_session_append_checkpoint(&key, 2, &bad, "workflow/v1", b"bad"),
        Err(Error::Invalid(_))
    ));
    writer
        .put_session_append_checkpoint(&key, 2, &second, "workflow/v1", b"turn=2")
        .unwrap();
    let restored = writer
        .get_session_checkpoint(&key, "engine-state/v1")
        .unwrap()
        .unwrap();
    assert_eq!(restored.bundle.attention[0].1, second.attention[0].1);
}

#[test]
fn prepared_prefix_finds_longest_exact_match_and_isolates_scope() {
    let path = directory();
    let mut writer = Cache::open(&path).unwrap();
    let scope = PrefixScope {
        namespace: "tenant-a".into(),
        model_fingerprint: "weights+execution".into(),
        context_fingerprint: "template+tools+salt".into(),
    };
    let first = [10, 20, 30, 40];
    let longer = [10, 20, 30, 40, 50, 60, 70];
    let first_address = writer.put_prepared_prefix(&scope, &bundle(&first)).unwrap();
    let longer_address = writer
        .put_prepared_prefix(&scope, &bundle(&longer))
        .unwrap();
    assert_ne!(first_address, longer_address);
    let reader = CacheReader::open(&path).unwrap();
    let metadata_hit = reader
        .find_longest_prepared_address(&scope, &[10, 20, 30, 40, 50, 60, 70, 80])
        .unwrap()
        .unwrap();
    assert_eq!(metadata_hit.prefix_tokens, 7);
    assert_eq!(metadata_hit.address, longer_address);
    let (hit, state) = reader
        .find_longest_prepared_prefix(&scope, &[10, 20, 30, 40, 50, 60, 70, 80], "engine-state/v1")
        .unwrap()
        .unwrap();
    assert_eq!(hit, longer_address);
    assert_eq!(state.token_ids, longer);
    assert_eq!(
        state.attention[0].0.prefix_fingerprint,
        hit.prefix_fingerprint
    );
    let (hit, state) = reader
        .find_longest_prepared_prefix(&scope, &[10, 20, 30, 40, 99, 60, 70], "engine-state/v1")
        .unwrap()
        .unwrap();
    assert_eq!(hit, first_address);
    assert_eq!(state.token_ids, first);
    let other = PrefixScope {
        context_fingerprint: "another-tenant".into(),
        ..scope.clone()
    };
    assert!(reader
        .find_longest_prepared_prefix(&other, &longer, "engine-state/v1")
        .unwrap()
        .is_none());
    assert!(reader
        .find_longest_prepared_prefix(&scope, &[11, 20, 30, 40], "engine-state/v1")
        .unwrap()
        .is_none());
}

#[test]
fn admission_uses_measured_first_token_cost_and_publication_payback() {
    let mut estimate = AdmissionEstimate {
        recompute_first_token_ms: 100.0,
        publish_ms: 200.0,
        expected_future_hits: 3,
        available_tiers: vec![
            TierCost {
                tier: CacheTier::LocalYesno,
                first_token_ms: 80.0,
            },
            TierCost {
                tier: CacheTier::PreparedHost,
                first_token_ms: 20.0,
            },
        ],
    };
    assert_eq!(
        estimate.choose_hit().unwrap(),
        HitChoice::Restore(estimate.available_tiers[1])
    );
    assert!(estimate.should_publish().unwrap());
    estimate.expected_future_hits = 2;
    assert!(!estimate.should_publish().unwrap());
    estimate.available_tiers[1].first_token_ms = 120.0;
    assert_eq!(
        estimate.choose_hit().unwrap(),
        HitChoice::Restore(estimate.available_tiers[0])
    );
    estimate.available_tiers[0].first_token_ms = 100.0;
    assert_eq!(estimate.choose_hit().unwrap(), HitChoice::Recompute);
    estimate.publish_ms = f64::NAN;
    assert!(matches!(estimate.choose_hit(), Err(Error::Invalid(_))));
}

#[test]
fn session_generation_moves_atomically_and_rejected_save_keeps_previous() {
    let path = directory();
    let mut writer = Cache::open(&path).unwrap();
    let key = SessionKey {
        namespace: "tenant-a".into(),
        model_fingerprint: "weights+execution".into(),
        session_id: "opaque-session-17".into(),
    };
    let one = bundle(&[1, 2, 3, 4]);
    let first_address = writer
        .put_session_checkpoint(&key, 1, &one, "workflow/v1", b"turn=1")
        .unwrap();
    let reader = CacheReader::open(&path).unwrap();
    let initial = reader
        .get_session_checkpoint(&key, "engine-state/v1")
        .unwrap()
        .unwrap();
    assert_eq!(initial.generation, 1);
    assert_eq!(initial.bundle.token_ids, one.token_ids);
    assert_eq!(initial.workflow_bytes, b"turn=1");

    let two = bundle(&[1, 2, 3, 4, 5]);
    writer
        .put_session_checkpoint(&key, 2, &two, "workflow/v1", b"turn=2")
        .unwrap();
    assert!(matches!(
        writer.prune_session_checkpoint(&key, 2, &two.token_ids),
        Err(Error::Invalid(_))
    ));
    assert!(matches!(
        writer.put_session_checkpoint(&key, 2, &two, "workflow/v1", b"stale"),
        Err(Error::Invalid(_))
    ));
    let mut bad = bundle(&[1, 2, 3, 4, 5, 6]);
    bad.attention.push(bad.attention[0].clone());
    assert!(writer
        .put_session_checkpoint(&key, 3, &bad, "workflow/v1", b"bad")
        .is_err());
    assert_eq!(
        writer
            .get_session_checkpoint(&key, "engine-state/v1")
            .unwrap()
            .unwrap()
            .generation,
        2
    );
    assert_eq!(
        writer
            .get_prefill_bundle(&first_address, "engine-state/v1")
            .unwrap()
            .unwrap()
            .token_ids,
        one.token_ids
    );
    assert!(writer
        .prune_session_checkpoint(&key, 1, &one.token_ids)
        .unwrap());
    assert!(writer
        .get_prefill_bundle(&first_address, "engine-state/v1")
        .unwrap()
        .is_none());
    drop(reader);
    drop(writer);
    let reopened = CacheReader::open(&path).unwrap();
    let restored = reopened
        .get_session_checkpoint(&key, "engine-state/v1")
        .unwrap()
        .unwrap();
    assert_eq!(restored.generation, 2);
    assert_eq!(restored.bundle.token_ids, two.token_ids);
    assert_eq!(restored.workflow_bytes, b"turn=2");
}
