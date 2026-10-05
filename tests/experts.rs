use shifou::{Cache, CacheReader, ExpertSnapshotKey, ReadVerification};
use std::sync::atomic::{AtomicUsize, Ordering};

fn directory() -> std::path::PathBuf {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join(".agents-workspace/tmp/tests")
        .join(format!(
            "experts-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ))
}

fn key(layer: u32, expert: u32) -> ExpertSnapshotKey {
    ExpertSnapshotKey {
        model_fingerprint: [7; 32],
        format: "xinfer-nemotron-expert/v1".into(),
        layer,
        expert,
    }
}

#[test]
fn batch_is_selective_and_survives_reader_reopen() {
    let path = directory();
    let mut cache = Cache::open(&path).unwrap();
    let a = vec![0x55; 8193];
    let b = vec![0x80, 0, 0, 0];
    let reports = cache
        .put_expert_snapshots(&[(key(2, 5), &a), (key(2, 6), &b)])
        .unwrap();
    assert_eq!(reports.len(), 2);
    let reader = CacheReader::open(&path).unwrap();
    assert_eq!(reader.get_expert_snapshot(&key(2, 5)).unwrap(), Some(a));
    assert_eq!(
        reader
            .get_expert_snapshot_with_verification(&key(2, 6), ReadVerification::StorageOnly)
            .unwrap(),
        Some(b)
    );
    assert_eq!(reader.get_expert_snapshot(&key(2, 7)).unwrap(), None);
    let shared = std::sync::Arc::new(reader);
    let concurrent = (0..4)
        .map(|_| {
            let shared = shared.clone();
            std::thread::spawn(move || shared.get_expert_snapshot(&key(2, 6)).unwrap())
        })
        .collect::<Vec<_>>();
    for result in concurrent {
        assert_eq!(result.join().unwrap(), Some(vec![0x80, 0, 0, 0]));
    }
    let reader = shared;
    let mut other_model = key(2, 5);
    other_model.model_fingerprint[0] ^= 1;
    assert_eq!(reader.get_expert_snapshot(&other_model).unwrap(), None);
    let mut other_format = key(2, 5);
    other_format.format = "another-engine/v1".into();
    assert_eq!(reader.get_expert_snapshot(&other_format).unwrap(), None);
    assert!(cache.remove_expert_snapshot(&key(2, 5)).unwrap());
    let reader = CacheReader::open(&path).unwrap();
    assert_eq!(reader.get_expert_snapshot(&key(2, 5)).unwrap(), None);
    assert_eq!(
        reader.get_expert_snapshot(&key(2, 6)).unwrap(),
        Some(vec![0x80, 0, 0, 0])
    );
}

#[test]
fn invalid_or_duplicate_group_publishes_nothing() {
    let path = directory();
    let mut cache = Cache::open(&path).unwrap();
    let a = [1, 2, 3];
    assert!(cache
        .put_expert_snapshots(&[(key(0, 1), &a), (key(0, 1), &a)])
        .is_err());
    assert_eq!(cache.get_expert_snapshot(&key(0, 1)).unwrap(), None);
    assert!(cache
        .put_expert_snapshots(&[(key(0, 1), &a), (key(0, 2), &[])])
        .is_err());
    assert_eq!(cache.get_expert_snapshot(&key(0, 1)).unwrap(), None);
    let mut invalid = key(0, 1);
    invalid.model_fingerprint = [0; 32];
    assert!(cache.put_expert_snapshot(&invalid, &a).is_err());
    assert!(cache.put_expert_snapshot(&key(0, 1), &[]).is_err());
}
