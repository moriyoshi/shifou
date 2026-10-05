use shifou::{Address, Cache, Error, Policy, Tensor};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

fn directory() -> PathBuf {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join(".agents-workspace/tmp/tests")
        .join(format!(
            "{}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
    std::fs::create_dir_all(&path).unwrap();
    path
}

fn address() -> Address {
    Address {
        namespace: "test".into(),
        model_fingerprint: "weights/config/adapter-v1".into(),
        prefix_fingerprint: "whole-prefix-v1".into(),
        layer: 7,
        slot: "key".into(),
    }
}

fn tensor() -> Tensor {
    Tensor {
        shape: vec![64, 8],
        axes: vec!["token".into(), "channel".into()],
        values: (0..512).map(|x| (x as f32 * 0.17).sin()).collect(),
    }
}

#[test]
fn survives_reopen_and_replacement_without_old_bits() {
    let path = directory();
    let original = tensor();
    let policy = Policy::keys(0.02);
    {
        let mut cache = Cache::open(&path).unwrap();
        cache.put(&address(), &original, &policy).unwrap();
        assert!(matches!(Cache::open(&path), Err(Error::Busy(_))));
    }
    {
        let mut cache = Cache::open(&path).unwrap();
        let read = cache.get(&address()).unwrap().unwrap();
        assert!(read
            .tensor
            .values
            .iter()
            .zip(&original.values)
            .all(|(&a, &b)| (a as f64 - b as f64).abs() <= 0.02));
        let replacement = Tensor {
            values: vec![0.0; 512],
            ..original.clone()
        };
        cache.put(&address(), &replacement, &policy).unwrap();
    }
    {
        let mut cache = Cache::open(&path).unwrap();
        assert_eq!(
            cache.get(&address()).unwrap().unwrap().tensor.values,
            vec![0.0; 512]
        );
        assert!(cache.remove(&address()).unwrap());
        assert!(!cache.remove(&address()).unwrap());
    }
    assert!(Cache::open(&path)
        .unwrap()
        .get(&address())
        .unwrap()
        .is_none());
}

#[test]
fn namespaces_models_prefixes_layers_and_slots_do_not_alias() {
    let path = directory();
    let mut addresses = vec![address(); 6];
    addresses[1].namespace = "other-tenant".into();
    addresses[2].model_fingerprint = "other-model".into();
    addresses[3].prefix_fingerprint = "other-prefix".into();
    addresses[4].layer += 1;
    addresses[5].slot = "value".into();
    {
        let mut cache = Cache::open(&path).unwrap();
        for (i, address) in addresses.iter().enumerate() {
            cache
                .put(
                    address,
                    &Tensor {
                        values: vec![i as f32; 512],
                        ..tensor()
                    },
                    &Policy::values(0.0),
                )
                .unwrap();
        }
    }
    let cache = Cache::open(&path).unwrap();
    for (i, address) in addresses.iter().enumerate() {
        assert_eq!(
            cache.get(address).unwrap().unwrap().tensor.values,
            vec![i as f32; 512]
        );
    }
}

#[test]
fn cli_put_and_get_across_processes() {
    let path = directory();
    let request = path.join("request.json");
    let lookup = path.join("address.json");
    std::fs::write(
        &request,
        serde_json::to_vec(&serde_json::json!({
            "address": address(), "tensor": tensor(), "policy": Policy::values(0.02)
        }))
        .unwrap(),
    )
    .unwrap();
    std::fs::write(&lookup, serde_json::to_vec(&address()).unwrap()).unwrap();
    let binary = env!("CARGO_BIN_EXE_shifou");
    let write = std::process::Command::new(binary)
        .arg("put")
        .arg(path.join("cache"))
        .arg(request)
        .output()
        .unwrap();
    assert!(
        write.status.success(),
        "{}",
        String::from_utf8_lossy(&write.stderr)
    );
    let read = std::process::Command::new(binary)
        .arg("get")
        .arg(path.join("cache"))
        .arg(lookup)
        .output()
        .unwrap();
    assert!(
        read.status.success(),
        "{}",
        String::from_utf8_lossy(&read.stderr)
    );
    let result: shifou::Retrieved = serde_json::from_slice(&read.stdout).unwrap();
    assert_eq!(result.tensor.shape, tensor().shape);
    assert!(result.report.codec.max_abs_error <= 0.02);
}

#[test]
fn estimate_matches_persisted_bytes() {
    let path = directory();
    let mut cache = Cache::open(path).unwrap();
    let tensor = tensor();
    for error in [0.0, 0.05, 0.2] {
        let policy = Policy::keys(error);
        let expected = Cache::estimate(&address(), &tensor, &policy).unwrap();
        let actual = cache.put(&address(), &tensor, &policy).unwrap();
        assert_eq!(expected.total_roaring_bytes, actual.total_roaring_bytes);
        assert_eq!(
            expected.metadata_roaring_bytes,
            actual.metadata_roaring_bytes
        );
        assert_eq!(expected.codec.payload_bytes, actual.codec.payload_bytes);
    }
}
