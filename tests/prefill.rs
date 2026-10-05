use shifou::{
    Address, Cache, CacheReader, Error, PackedBuffer, PackedSnapshot, PrefillBundle,
    ReadVerification,
};
use std::sync::atomic::{AtomicUsize, Ordering};

fn directory() -> std::path::PathBuf {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(".agents-workspace/tmp/tests");
    std::fs::create_dir_all(&root).unwrap();
    let path = root.join(format!(
        "prefill-{}-{}-{}",
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

fn address(slot: &str) -> Address {
    Address {
        namespace: "prefill-test".into(),
        model_fingerprint: "weights+execution".into(),
        prefix_fingerprint: "exact-causal-input".into(),
        layer: 0,
        slot: slot.into(),
    }
}

fn bundle(state: Vec<u8>) -> PrefillBundle {
    PrefillBundle {
        prefix_tokens: 4,
        token_ids: vec![17, 0, u32::MAX, 23],
        state_format: "xinfer-gdn/v1".into(),
        state_bytes: state,
        attention: vec![(
            address("attention-k"),
            PackedSnapshot {
                format: "compact4/v1".into(),
                buffers: vec![PackedBuffer {
                    name: "codes".into(),
                    dtype: "u8".into(),
                    shape: vec![5],
                    bytes: vec![0x55, 0, 0xff, 0, 0],
                }],
            },
        )],
    }
}

#[test]
fn large_state_pages_and_attention_round_trip_through_foreign_reader() {
    let path = directory();
    let expected = bundle(vec![0x55; 8 * 1024 * 1024 + 17]);
    let mut writer = Cache::open(&path).unwrap();
    let report = writer
        .put_prefill_bundle(&address("manifest"), &expected)
        .unwrap();
    assert_eq!(report.len(), 5); // tokens, two state pages, manifest, attention
    let reader = CacheReader::open(&path).unwrap();
    assert_eq!(
        reader
            .get_prefill_bundle(&address("manifest"), "xinfer-gdn/v1")
            .unwrap(),
        Some(expected.clone())
    );
    assert_eq!(
        reader
            .get_prefill_bundle_with_verification(
                &address("manifest"),
                "xinfer-gdn/v1",
                ReadVerification::StorageOnly,
            )
            .unwrap(),
        Some(expected.clone())
    );
    assert!(matches!(
        reader.get_prefill_bundle(&address("manifest"), "wrong-format"),
        Err(Error::Invalid(_))
    ));
    drop(reader);
    drop(writer);
    assert_eq!(
        CacheReader::open(&path)
            .unwrap()
            .get_prefill_bundle(&address("manifest"), "xinfer-gdn/v1")
            .unwrap(),
        Some(expected)
    );
}

#[test]
fn rejected_bundle_publishes_no_manifest() {
    let path = directory();
    let mut writer = Cache::open(&path).unwrap();
    let mut invalid = bundle(vec![1, 2, 3]);
    invalid.attention[0].0.prefix_fingerprint = "different-prefix".into();
    assert!(matches!(
        writer.put_prefill_bundle(&address("manifest"), &invalid),
        Err(Error::Invalid(_))
    ));
    assert!(writer
        .get_prefill_bundle(&address("manifest"), "xinfer-gdn/v1")
        .unwrap()
        .is_none());
    invalid.attention[0].0 = address("manifest:state:0");
    assert!(writer
        .put_prefill_bundle(&address("manifest"), &invalid)
        .is_err());
    assert!(writer
        .get_prefill_bundle(&address("manifest"), "xinfer-gdn/v1")
        .unwrap()
        .is_none());
}

#[test]
fn bundle_removal_clears_manifest_and_owned_records() {
    let path = directory();
    let mut writer = Cache::open(&path).unwrap();
    let expected = bundle(vec![0x55; 8193]);
    writer
        .put_prefill_bundle(&address("manifest"), &expected)
        .unwrap();
    assert!(writer.remove_prefill_bundle(&address("manifest")).unwrap());
    assert!(!writer.remove_prefill_bundle(&address("manifest")).unwrap());
    assert!(CacheReader::open(&path)
        .unwrap()
        .get_prefill_bundle(&address("manifest"), "xinfer-gdn/v1")
        .unwrap()
        .is_none());
    assert!(writer
        .get_packed(&address("manifest:state:0"), "shifou-prefill-state-page/v1")
        .unwrap()
        .is_none());
    assert!(writer
        .get_packed(&address("attention-k"), "compact4/v1")
        .unwrap()
        .is_none());
}

#[test]
fn published_manifest_rejects_a_missing_state_page() {
    let path = directory();
    let mut writer = Cache::open(&path).unwrap();
    writer
        .put_prefill_bundle(&address("manifest"), &bundle(vec![0x55; 8193]))
        .unwrap();
    assert!(writer.remove_packed(&address("manifest:state:0")).unwrap());
    assert!(matches!(
        CacheReader::open(&path)
            .unwrap()
            .get_prefill_bundle(&address("manifest"), "xinfer-gdn/v1"),
        Err(Error::Corrupt(_))
    ));
}
