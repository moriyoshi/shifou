use proptest::prelude::*;
use shifou::{Address, Cache, PackedBuffer, PackedSnapshot, ReadVerification};
use std::sync::atomic::{AtomicUsize, Ordering};

fn address() -> Address {
    Address {
        namespace: "packed-test".into(),
        model_fingerprint: "model+engine".into(),
        prefix_fingerprint: "all-prefix-tokens".into(),
        layer: 0,
        slot: "snapshot".into(),
    }
}
fn directory() -> std::path::PathBuf {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join(".agents-workspace/tmp/tests")
        .join(format!(
            "packed-{}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ))
}
fn snapshot(bytes: Vec<u8>) -> PackedSnapshot {
    PackedSnapshot {
        format: "test-codec/v1".into(),
        buffers: vec![
            PackedBuffer {
                name: "codes".into(),
                dtype: "u8".into(),
                shape: vec![bytes.len()],
                bytes,
            },
            PackedBuffer {
                name: "scales".into(),
                dtype: "f32-le".into(),
                shape: vec![2],
                bytes: [0u32, 0x80000000]
                    .into_iter()
                    .flat_map(u32::to_le_bytes)
                    .collect(),
            },
        ],
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(24))]
    #[test]
    fn arbitrary_packed_bytes_survive_reopen_and_overwrite(mut bytes in prop::collection::vec(any::<u8>(), 1..20000)) {
        bytes.extend_from_slice(&[0, 0]);
        let expected = snapshot(bytes);
        let path = directory();
        {
            let mut store = Cache::open(&path).unwrap();
            store.put_packed(&address(), &expected).unwrap();
        }
        {
            let mut store = Cache::open(&path).unwrap();
            prop_assert_eq!(store.get_packed(&address(), &expected.format).unwrap(), Some(expected.clone()));
            prop_assert!(store.get_packed(&address(), "wrong-version").is_err());
            let mut other = address(); other.model_fingerprint.push('x');
            prop_assert!(store.get_packed(&other, &expected.format).unwrap().is_none());
            store.put_packed(&address(), &snapshot(vec![0; 11])).unwrap();
        }
        prop_assert_eq!(Cache::open(&path).unwrap().get_packed(&address(), &expected.format).unwrap(),
            Some(snapshot(vec![0; 11])));
    }
}
#[test]
fn malformed_packed_buffers_are_rejected() {
    let mut s = snapshot(vec![0; 3]);
    s.buffers[0].shape = vec![usize::MAX, 2];
    assert!(s.validate().is_err());
    s = snapshot(vec![0; 3]);
    s.buffers[0].dtype = "unknown".into();
    assert!(s.validate().is_err());
    s = snapshot(vec![0; 3]);
    s.buffers[1].name = s.buffers[0].name.clone();
    assert!(s.validate().is_err());
}

#[test]
fn packed_removal_survives_reopen() {
    let path = directory();
    {
        let mut store = Cache::open(&path).unwrap();
        store
            .put_packed(&address(), &snapshot(vec![0; 17]))
            .unwrap();
        assert!(store.remove_packed(&address()).unwrap());
        assert!(!store.remove_packed(&address()).unwrap());
    }
    assert!(Cache::open(&path)
        .unwrap()
        .get_packed(&address(), "test-codec/v1")
        .unwrap()
        .is_none());
}

#[test]
fn storage_only_read_keeps_format_and_address_checks() {
    let path = directory();
    let expected = snapshot(vec![0x55; 8193]);
    let mut store = Cache::open(&path).unwrap();
    store.put_packed(&address(), &expected).unwrap();
    assert_eq!(
        store
            .get_packed_with_verification(
                &address(),
                &expected.format,
                ReadVerification::StorageOnly
            )
            .unwrap(),
        Some(expected.clone())
    );
    assert!(store
        .get_packed_with_verification(&address(), "wrong-version", ReadVerification::StorageOnly)
        .is_err());
    let mut wrong = address();
    wrong.model_fingerprint.push('x');
    assert!(store
        .get_packed_with_verification(&wrong, &expected.format, ReadVerification::StorageOnly)
        .unwrap()
        .is_none());
}
