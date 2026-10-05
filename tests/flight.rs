#![cfg(feature = "flight")]

use arrow_flight::flight_service_server::FlightServiceServer;
use shifou::{
    Address, Cache, Error, FlightBundleReader, PackedBuffer, PackedSnapshot, PrefillBundle,
};
use std::{
    path::PathBuf,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::Duration,
};
use tokio_stream::wrappers::TcpListenerStream;
use tonic::transport::Server;
use yesno_core::{Db, DbOptions};
use yesno_flight::YesnoFlightService;

fn directory() -> PathBuf {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(".agents-workspace/tmp/tests");
    std::fs::create_dir_all(&root).unwrap();
    let path = root.join(format!(
        "flight-{}-{}-{}",
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
        namespace: "flight-test".into(),
        model_fingerprint: "weights+execution".into(),
        prefix_fingerprint: "exact-prefix".into(),
        layer: 0,
        slot: slot.into(),
    }
}

async fn serve(
    path: &std::path::Path,
) -> (
    String,
    tokio::task::JoinHandle<Result<(), tonic::transport::Error>>,
) {
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
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        Server::builder()
            .add_service(FlightServiceServer::new(YesnoFlightService::new(db)))
            .serve_with_incoming(TcpListenerStream::new(listener))
            .await
    });
    (endpoint, server)
}

#[tokio::test]
async fn full_bundle_round_trips_across_three_parallel_state_pages() {
    let path = directory();
    let state_bytes = (0..2 * 8 * 1024 * 1024 + 17)
        .map(|index| (index as u8).wrapping_mul(31))
        .collect();
    let expected = PrefillBundle {
        prefix_tokens: 4,
        token_ids: vec![17, 0, u32::MAX, 23],
        state_format: "xinfer-gdn/v1".into(),
        state_bytes,
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
    };
    let mut writer = Cache::open(&path).unwrap();
    writer
        .put_prefill_bundle(&address("manifest"), &expected)
        .unwrap();
    drop(writer);
    let (endpoint, server) = serve(&path).await;
    let reader = FlightBundleReader::connect(&endpoint).await.unwrap();
    assert_eq!(
        reader
            .get_prefill_bundle(&address("manifest"), "xinfer-gdn/v1")
            .await
            .unwrap(),
        Some(expected)
    );
    assert!(reader
        .get_prefill_bundle(&address("missing"), "xinfer-gdn/v1")
        .await
        .unwrap()
        .is_none());
    assert!(matches!(
        reader
            .get_prefill_bundle(&address("manifest"), "wrong-format")
            .await,
        Err(Error::Invalid(_))
    ));
    assert!(matches!(
        reader
            .with_max_state_bytes(8 * 1024 * 1024)
            .unwrap()
            .get_prefill_bundle(&address("manifest"), "xinfer-gdn/v1")
            .await,
        Err(Error::Invalid(_))
    ));
    let budget_reader = FlightBundleReader::connect(&endpoint)
        .await
        .unwrap()
        .with_max_bundle_bytes(8 * 1024 * 1024)
        .unwrap();
    assert!(matches!(
        budget_reader
            .get_prefill_bundle(&address("manifest"), "xinfer-gdn/v1")
            .await,
        Err(Error::Invalid(_))
    ));
    server.abort();
}

#[tokio::test]
async fn missing_published_page_fails_the_flight_bundle_read() {
    let path = directory();
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
    let (endpoint, server) = serve(&path).await;
    let reader = FlightBundleReader::connect(&endpoint).await.unwrap();
    assert!(matches!(
        reader
            .get_prefill_bundle(&address("manifest"), "state/v1")
            .await,
        Err(Error::Corrupt(_))
    ));
    server.abort();
}

#[tokio::test]
async fn reclaimed_first_version_restarts_the_whole_bundle() {
    let path = directory();
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
    let initial = db.snapshot().unwrap().version();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let calls = Arc::new(AtomicUsize::new(0));
    let interceptor_db = db.clone();
    let interceptor_calls = calls.clone();
    let service = YesnoFlightService::with_ticket_lease(db, Duration::ZERO);
    let server = tokio::spawn(async move {
        Server::builder()
            .add_service(FlightServiceServer::with_interceptor(
                service,
                move |request: tonic::Request<()>| {
                    if interceptor_calls.fetch_add(1, Ordering::Relaxed) == 1 {
                        interceptor_db
                            .insert(0xdead_beef, 1)
                            .map_err(|error| tonic::Status::internal(error.to_string()))?;
                        interceptor_db
                            .checkpoint()
                            .map_err(|error| tonic::Status::internal(error.to_string()))?;
                        if interceptor_db.read_floor() <= initial {
                            return Err(tonic::Status::internal(
                                "fixture did not reclaim the first version",
                            ));
                        }
                    }
                    Ok(request)
                },
            ))
            .serve_with_incoming(TcpListenerStream::new(listener))
            .await
    });
    let reader = FlightBundleReader::connect(&endpoint).await.unwrap();
    assert_eq!(
        reader
            .get_prefill_bundle(&address("manifest"), "state/v1")
            .await
            .unwrap(),
        Some(expected)
    );
    assert!(calls.load(Ordering::Relaxed) > 2);
    server.abort();
}
