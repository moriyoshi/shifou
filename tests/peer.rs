#![cfg(feature = "peer")]

use sha2::{Digest, Sha256};
use shifou::{
    Address, Cache, Error, PackedBuffer, PackedSnapshot, PeerBundleReader, PrefillBundle,
    PrefixScope, SessionKey,
};
use std::{
    io::Write,
    os::unix::net::UnixListener,
    path::PathBuf,
    sync::{Arc, RwLock},
    time::Duration,
};
use yesno_core::{Db, DbOptions, OrdSet};
use yesno_plugin::{
    abi::Role,
    channel::{read_frame, send_fd, serve_blocking, Arena, Limits, Session},
    ipc::Frame,
    Host,
};

struct Clean(PathBuf);

impl Drop for Clean {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn directory(tag: &str) -> (Clean, PathBuf) {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(".agents-workspace/tmp/tests");
    std::fs::create_dir_all(&root).unwrap();
    let path = root.join(format!(
        "peer-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir(&path).unwrap();
    (Clean(path.clone()), path)
}

fn address(slot: &str) -> Address {
    Address {
        namespace: "peer-test".into(),
        model_fingerprint: "weights+execution".into(),
        prefix_fingerprint: "exact-prefix".into(),
        layer: 0,
        slot: slot.into(),
    }
}

fn record_key(address: &Address) -> u64 {
    let digest = Sha256::digest(serde_json::to_vec(address).unwrap());
    u64::from_le_bytes(digest[..8].try_into().unwrap()) & !1
}

fn serve(
    data_dir: &std::path::Path,
    socket: &std::path::Path,
    inline: bool,
) -> std::thread::JoinHandle<()> {
    let db = Db::open_with(
        data_dir,
        DbOptions {
            shards: 1,
            ..DbOptions::default()
        },
    )
    .unwrap();
    let slot = Arc::new(RwLock::new(Some(Arc::new(db))));
    let host = Host::new(slot, 1, Role::Leader);
    let listener = UnixListener::bind(socket).unwrap();
    std::thread::spawn(move || {
        let (stream, _) = listener.accept().unwrap();
        let limits = Limits {
            max_handles: 2,
            max_lanes: 3,
            max_blocks: 16,
            max_snapshots: 2,
            max_writes: 1,
        };
        let mut session = if inline {
            Session::new_inline(host, limits)
        } else {
            Session::new(host, Arena::new(limits.arena_bytes()).unwrap(), limits)
        };
        if let Some(fd) = session.arena_fd() {
            send_fd(&stream, fd).unwrap();
        }
        serve_blocking(&mut session, stream).unwrap();
    })
}

#[test]
fn session_head_and_checkpoint_round_trip_over_peer_snapshot() {
    let (_clean, path) = directory("session");
    let key = SessionKey {
        namespace: "peer-test".into(),
        model_fingerprint: "weights+execution".into(),
        session_id: "session-1".into(),
    };
    let first = PrefillBundle {
        prefix_tokens: 2,
        token_ids: vec![3, 5],
        state_format: "state/v1".into(),
        state_bytes: vec![0x55; 8193],
        attention: Vec::new(),
    };
    let mut second = first.clone();
    second.prefix_tokens = 3;
    second.token_ids.push(7);
    second.state_bytes.push(0xaa);
    let mut writer = Cache::open(&path).unwrap();
    writer
        .put_session_checkpoint(&key, 1, &first, "workflow/v1", b"turn=1")
        .unwrap();
    writer
        .put_session_checkpoint(&key, 2, &second, "workflow/v1", b"turn=2")
        .unwrap();
    let expected = writer
        .get_session_checkpoint(&key, "state/v1")
        .unwrap()
        .unwrap();
    drop(writer);

    for inline in [false, true] {
        let socket = path.parent().unwrap().join(format!(
            "peer-test-{}-session-{inline}.sock",
            std::process::id()
        ));
        let server = serve(&path.join("data"), &socket, inline);
        let mut reader = PeerBundleReader::connect(&socket).unwrap();
        assert_eq!(
            reader.get_session_checkpoint(&key, "state/v1").unwrap(),
            Some(expected.clone())
        );
        assert!(reader
            .get_session_checkpoint(
                &SessionKey {
                    session_id: "absent".into(),
                    ..key.clone()
                },
                "state/v1"
            )
            .unwrap()
            .is_none());
        assert!(matches!(
            reader.get_session_checkpoint(&key, "wrong/v1"),
            Err(Error::Invalid(_))
        ));
        drop(reader);
        server.join().unwrap();
        std::fs::remove_file(socket).unwrap();
    }
}

#[test]
fn appended_session_round_trips_over_arena_and_inline_peer() {
    let (_clean, path) = directory("append-session");
    let key = SessionKey {
        namespace: "peer-test".into(),
        model_fingerprint: "weights+execution".into(),
        session_id: "append-session".into(),
    };
    let first = PrefillBundle {
        prefix_tokens: 2,
        token_ids: vec![3, 5],
        state_format: "state/v1".into(),
        state_bytes: vec![0x55; 8193],
        attention: vec![(
            address("input-k"),
            PackedSnapshot {
                format: "kv/v1".into(),
                buffers: vec![PackedBuffer {
                    name: "k".into(),
                    dtype: "u8".into(),
                    shape: vec![2],
                    bytes: vec![3, 5],
                }],
            },
        )],
    };
    let mut second = first.clone();
    second.prefix_tokens = 3;
    second.token_ids.push(7);
    second.state_bytes[0] = 0xaa;
    second.attention[0].1.buffers[0].bytes.push(7);
    second.attention[0].1.buffers[0].shape = vec![3];
    let mut writer = Cache::open(&path).unwrap();
    writer
        .put_session_checkpoint(&key, 1, &first, "workflow/v1", b"turn=1")
        .unwrap();
    writer
        .put_session_append_checkpoint(&key, 2, &second, "workflow/v1", b"turn=2")
        .unwrap();
    let expected = writer
        .get_session_checkpoint(&key, "state/v1")
        .unwrap()
        .unwrap();
    drop(writer);
    for (inline, parallel, pipeline) in [
        (false, false, false),
        (false, true, false),
        (false, true, true),
        (true, false, false),
        (true, true, false),
        (true, true, true),
    ] {
        let socket = path.parent().unwrap().join(format!(
            "peer-test-{}-append-{inline}-{parallel}-{pipeline}.sock",
            std::process::id()
        ));
        let server = serve(&path.join("data"), &socket, inline);
        let mut reader = PeerBundleReader::connect(&socket)
            .unwrap()
            .with_parallel_verification(parallel)
            .with_pipelined_verification(pipeline);
        assert_eq!(
            reader.get_session_checkpoint(&key, "state/v1").unwrap(),
            Some(expected.clone())
        );
        drop(reader);
        server.join().unwrap();
        std::fs::remove_file(socket).unwrap();
    }
}

#[test]
fn prepared_prefix_lookup_finds_longest_exact_peer_hit() {
    let (_clean, path) = directory("prepared");
    let scope = PrefixScope {
        namespace: "peer-test".into(),
        model_fingerprint: "weights+execution".into(),
        context_fingerprint: "template+tools+tenant".into(),
    };
    let mut writer = Cache::open(&path).unwrap();
    let mut first = PrefillBundle {
        prefix_tokens: 2,
        token_ids: vec![11, 13],
        state_format: "state/v1".into(),
        state_bytes: vec![0x55; 8193],
        attention: Vec::new(),
    };
    let first_address = writer.put_prepared_prefix(&scope, &first).unwrap();
    first.prefix_tokens = 4;
    first.token_ids.extend([17, 19]);
    first.state_bytes.push(0xaa);
    let longer_address = writer.put_prepared_prefix(&scope, &first).unwrap();
    drop(writer);

    for inline in [false, true] {
        let socket = path.parent().unwrap().join(format!(
            "peer-test-{}-prepared-{inline}.sock",
            std::process::id()
        ));
        let server = serve(&path.join("data"), &socket, inline);
        let mut reader = PeerBundleReader::connect(&socket).unwrap();
        assert_eq!(
            reader
                .find_longest_prepared_address(&scope, &[11, 13, 17, 19, 23])
                .unwrap()
                .unwrap()
                .address,
            longer_address
        );
        assert_eq!(
            reader
                .find_longest_prepared_prefix(&scope, &[11, 13, 17, 99], "state/v1")
                .unwrap()
                .unwrap()
                .0,
            first_address
        );
        assert!(reader
            .find_longest_prepared_address(&scope, &[11, 99, 17, 19])
            .unwrap()
            .is_none());
        drop(reader);
        server.join().unwrap();
        std::fs::remove_file(socket).unwrap();
    }
}

#[test]
fn full_bundle_round_trips_over_arena_and_inline_peer_sockets() {
    let (_clean, path) = directory("roundtrip");
    let mut expected = PrefillBundle {
        prefix_tokens: 4,
        token_ids: vec![17, 0, u32::MAX, 23],
        state_format: "xinfer-gdn/v1".into(),
        state_bytes: (0..8 * 1024 * 1024 + 17)
            .map(|index| (index as u8).wrapping_mul(31))
            .collect(),
        attention: vec![(
            address("attention-k"),
            PackedSnapshot {
                format: "compact4/v1".into(),
                buffers: vec![
                    PackedBuffer {
                        name: "codes".into(),
                        dtype: "u8".into(),
                        shape: vec![5],
                        bytes: vec![0x55, 0, 0xff, 0, 0],
                    },
                    PackedBuffer {
                        name: "scales".into(),
                        dtype: "u8".into(),
                        shape: vec![3],
                        bytes: vec![1, 2, 3],
                    },
                ],
            },
        )],
    };
    let extra = expected.attention[0].1.clone();
    for slot in ["attention-v", "attention-out"] {
        expected.attention.push((address(slot), extra.clone()));
    }
    let mut writer = Cache::open(&path).unwrap();
    writer
        .put_prefill_bundle(&address("manifest"), &expected)
        .unwrap();
    drop(writer);

    for (inline, parallel, pipeline) in [
        (false, false, false),
        (false, true, false),
        (false, true, true),
        (true, false, false),
        (true, true, false),
        (true, true, true),
    ] {
        let socket = path.parent().unwrap().join(format!(
            "peer-test-{}-{inline}-{parallel}-{pipeline}.sock",
            std::process::id()
        ));
        let server = serve(&path.join("data"), &socket, inline);
        let mut reader = PeerBundleReader::connect(&socket)
            .unwrap()
            .with_parallel_verification(parallel)
            .with_pipelined_verification(pipeline);
        assert_eq!(
            reader
                .get_prefill_bundle(&address("manifest"), "xinfer-gdn/v1")
                .unwrap(),
            Some(expected.clone())
        );
        assert!(reader
            .get_prefill_bundle(&address("missing"), "xinfer-gdn/v1")
            .unwrap()
            .is_none());
        let mut limited = reader.with_max_bundle_bytes(8 * 1024 * 1024).unwrap();
        assert!(matches!(
            limited.get_prefill_bundle(&address("manifest"), "xinfer-gdn/v1"),
            Err(Error::Invalid(_))
        ));
        drop(limited);
        server.join().unwrap();
        std::fs::remove_file(socket).unwrap();
    }
}

#[test]
fn pipelined_read_rejects_a_corrupt_record_from_the_previous_batch() {
    let (_clean, path) = directory("pipelined-corrupt");
    let snapshot = PackedSnapshot {
        format: "kv/v1".into(),
        buffers: vec![PackedBuffer {
            name: "k".into(),
            dtype: "u8".into(),
            shape: vec![2],
            bytes: vec![3, 5],
        }],
    };
    let expected = PrefillBundle {
        prefix_tokens: 1,
        token_ids: vec![42],
        state_format: "state/v1".into(),
        state_bytes: vec![0x55; 100],
        attention: ["attention-k", "attention-v", "attention-out"]
            .into_iter()
            .map(|slot| (address(slot), snapshot.clone()))
            .collect(),
    };
    let mut writer = Cache::open(&path).unwrap();
    writer
        .put_prefill_bundle(&address("manifest"), &expected)
        .unwrap();
    drop(writer);

    // Keep the child metadata intact so the first batch reaches the delayed
    // checksum check while the peer scans the next batch.
    let mut ordinals = Vec::new();
    for (index, byte) in [9u8, 5, 1].into_iter().enumerate() {
        for bit in 0..8 {
            if byte & (1 << bit) != 0 {
                ordinals.push((index * 8 + bit) as u64);
            }
        }
    }
    let db = Db::open_with(
        path.join("data"),
        DbOptions {
            shards: 1,
            ..DbOptions::default()
        },
    )
    .unwrap();
    let mut batch = db.batch();
    batch.store_set(
        record_key(&address("attention-k")),
        &OrdSet::from_sorted_slice(&ordinals),
    );
    let committed = batch.commit().unwrap();
    db.wait_visible(committed.version, Duration::from_secs(30))
        .unwrap();
    drop(db);

    let socket = path.parent().unwrap().join(format!(
        "peer-test-{}-pipelined-corrupt.sock",
        std::process::id()
    ));
    let server = serve(&path.join("data"), &socket, false);
    let mut reader = PeerBundleReader::connect(&socket)
        .unwrap()
        .with_pipelined_verification(true);
    assert!(matches!(
        reader.get_prefill_bundle(&address("manifest"), "state/v1"),
        Err(Error::Corrupt(_))
    ));
    drop(reader);
    server.join().unwrap();
    std::fs::remove_file(socket).unwrap();
}

#[test]
fn missing_published_page_fails_peer_read() {
    let (_clean, path) = directory("missing");
    let expected = PrefillBundle {
        prefix_tokens: 1,
        token_ids: vec![42],
        state_format: "state/v1".into(),
        state_bytes: vec![0x55; 8193],
        attention: Vec::new(),
    };
    let mut writer = Cache::open(&path).unwrap();
    writer
        .put_prefill_bundle(&address("manifest"), &expected)
        .unwrap();
    writer.remove_packed(&address("manifest:state:0")).unwrap();
    drop(writer);
    let socket = path
        .parent()
        .unwrap()
        .join(format!("peer-test-{}-missing.sock", std::process::id()));
    let server = serve(&path.join("data"), &socket, false);
    let mut reader = PeerBundleReader::connect(&socket).unwrap();
    assert!(matches!(
        reader.get_prefill_bundle(&address("manifest"), "state/v1"),
        Err(Error::Corrupt(_))
    ));
    drop(reader);
    server.join().unwrap();
    std::fs::remove_file(socket).unwrap();
}

#[test]
fn checkpoint_after_snapshot_open_cannot_split_a_bundle() {
    let (_clean, path) = directory("checkpoint");
    let expected = PrefillBundle {
        prefix_tokens: 2,
        token_ids: vec![42, 99],
        state_format: "state/v1".into(),
        state_bytes: vec![0x55; 8193],
        attention: Vec::new(),
    };
    let mut writer = Cache::open(&path).unwrap();
    writer
        .put_prefill_bundle(&address("manifest"), &expected)
        .unwrap();
    drop(writer);

    let db = Arc::new(
        Db::open_with(
            path.join("data"),
            DbOptions {
                shards: 1,
                ..DbOptions::default()
            },
        )
        .unwrap(),
    );
    let host = Host::new(Arc::new(RwLock::new(Some(db.clone()))), 1, Role::Leader);
    let socket = path
        .parent()
        .unwrap()
        .join(format!("peer-test-{}-checkpoint.sock", std::process::id()));
    let listener = UnixListener::bind(&socket).unwrap();
    let server = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let limits = Limits {
            max_handles: 2,
            max_lanes: 4,
            max_blocks: 16,
            max_snapshots: 2,
            max_writes: 1,
        };
        let mut session = Session::new(host, Arena::new(limits.arena_bytes()).unwrap(), limits);
        send_fd(&stream, session.arena_fd().unwrap()).unwrap();
        stream
            .write_all(&session.hello().encode().unwrap())
            .unwrap();
        let mut input = Vec::new();
        let mut mutated = false;
        while let Some(request) = read_frame(&mut stream, &mut input).unwrap() {
            let opening = matches!(request, Frame::SnapshotOpen);
            let reply = session.handle(request);
            stream.write_all(&reply.encode().unwrap()).unwrap();
            if opening && !mutated {
                mutated = true;
                let key = record_key(&address("manifest:state:0"));
                let mut batch = db.batch();
                batch.delete_key(key).delete_key(key | 1);
                let committed = batch.commit().unwrap();
                db.wait_visible(committed.version, Duration::from_secs(30))
                    .unwrap();
                db.checkpoint().unwrap();
            }
        }
        assert!(mutated, "fixture never checkpointed after snapshot open");
    });

    let mut reader = PeerBundleReader::connect(&socket).unwrap();
    assert_eq!(
        reader
            .get_prefill_bundle(&address("manifest"), "state/v1")
            .unwrap(),
        Some(expected)
    );
    assert!(matches!(
        reader.get_prefill_bundle(&address("manifest"), "state/v1"),
        Err(Error::Corrupt(_))
    ));
    drop(reader);
    server.join().unwrap();
    std::fs::remove_file(socket).unwrap();
}
