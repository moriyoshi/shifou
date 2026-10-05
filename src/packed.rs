//! Lossless persistence for engine-owned quantized buffers.
//! The engine owns codec semantics; shifou preserves bytes and validates identity.
use crate::store::{bytes_to_set, set_to_bytes};
use crate::{Address, Cache, CacheReader, Error, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{HashMap, HashSet},
    time::Duration,
};
use yesno_core::{roaring_format::serialize_u64, OrdSet, Snapshot};

const LIMIT: usize = 64 * 1024 * 1024;

/// Controls end-to-end payload checks on reads. `StorageOnly` keeps the
/// metadata, address, format, length and yesno storage checks, but does not
/// recompute shifou's payload SHA-256. Use it only for a trusted local cache.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReadVerification {
    Full,
    StorageOnly,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PackedBuffer {
    pub name: String,
    /// Explicit byte order: "u8", "f32-le", or "bf16-le".
    pub dtype: String,
    pub shape: Vec<usize>,
    pub bytes: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PackedSnapshot {
    /// Versioned engine codec identifier, including kernel revision and layout.
    pub format: String,
    pub buffers: Vec<PackedBuffer>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PackedStorageReport {
    pub packed_bytes: usize,
    pub payload_roaring_bytes: usize,
    pub metadata_roaring_bytes: usize,
    /// Logical portable bytes only; excludes WAL, indexes and slab allocation.
    pub total_roaring_bytes: usize,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Layout {
    pub(crate) name: String,
    pub(crate) dtype: String,
    pub(crate) shape: Vec<usize>,
    pub(crate) length: usize,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Manifest {
    pub(crate) address: Address,
    pub(crate) format: String,
    pub(crate) buffers: Vec<Layout>,
    pub(crate) payload_sha256: [u8; 32],
}

impl PackedSnapshot {
    pub fn validate(&self) -> Result<()> {
        if self.format.is_empty()
            || self.format.len() > 4096
            || self.buffers.is_empty()
            || self.buffers.len() > 4096
        {
            return Err(Error::Invalid(
                "invalid packed format or buffer count".into(),
            ));
        }
        let mut names = HashSet::new();
        let mut total = 0usize;
        for b in &self.buffers {
            let width = match b.dtype.as_str() {
                "u8" => 1usize,
                "bf16-le" => 2,
                "f32-le" => 4,
                _ => return Err(Error::Invalid("unsupported packed dtype".into())),
            };
            if b.name.is_empty()
                || b.name.len() > 256
                || !names.insert(&b.name)
                || b.shape.is_empty()
                || b.shape.len() > 8
                || b.shape.contains(&0)
            {
                return Err(Error::Invalid("invalid packed buffer geometry/name".into()));
            }
            let size = b.shape.iter().try_fold(width, |n, &d| n.checked_mul(d));
            if size != Some(b.bytes.len()) {
                return Err(Error::Invalid("packed shape/byte length mismatch".into()));
            }
            total = total
                .checked_add(b.bytes.len())
                .ok_or_else(|| Error::Invalid("packed length overflow".into()))?;
            if total > LIMIT {
                return Err(Error::Invalid("packed snapshot exceeds 64 MiB".into()));
            }
        }
        Ok(())
    }
}

fn manifest_bytes(manifest: &Manifest) -> Result<Vec<u8>> {
    let body = serde_json::to_vec(manifest)?;
    let mut bytes = b"SHIFOU02".to_vec();
    bytes.extend_from_slice(&Sha256::digest(&body));
    bytes.extend_from_slice(&body);
    Ok(bytes)
}

pub(crate) fn read_manifest(bytes: &[u8]) -> Result<Manifest> {
    if bytes.len() < 40 || &bytes[..8] != b"SHIFOU02" {
        return Err(Error::Corrupt("not a supported packed snapshot".into()));
    }
    if Sha256::digest(&bytes[40..])[..] != bytes[8..40] {
        return Err(Error::Corrupt("packed manifest checksum mismatch".into()));
    }
    Ok(serde_json::from_slice(&bytes[40..])?)
}

fn prepare_snapshot(
    address: &Address,
    snapshot: &PackedSnapshot,
) -> Result<(OrdSet, OrdSet, PackedStorageReport)> {
    snapshot.validate()?;
    address.key()?;
    let payload: Vec<u8> = snapshot
        .buffers
        .iter()
        .flat_map(|b| b.bytes.iter().copied())
        .collect();
    let manifest = Manifest {
        address: address.clone(),
        format: snapshot.format.clone(),
        buffers: snapshot
            .buffers
            .iter()
            .map(|b| Layout {
                name: b.name.clone(),
                dtype: b.dtype.clone(),
                shape: b.shape.clone(),
                length: b.bytes.len(),
            })
            .collect(),
        payload_sha256: Sha256::digest(&payload).into(),
    };
    let metadata_bytes = manifest_bytes(&manifest)?;
    if metadata_bytes.len() > LIMIT {
        return Err(Error::Invalid("packed metadata exceeds 64 MiB".into()));
    }
    let data = bytes_to_set(&payload);
    let metadata = bytes_to_set(&metadata_bytes);
    let payload_roaring_bytes = serialize_u64(&data).len();
    let metadata_roaring_bytes = serialize_u64(&metadata).len();
    let report = PackedStorageReport {
        packed_bytes: payload.len(),
        payload_roaring_bytes,
        metadata_roaring_bytes,
        total_roaring_bytes: payload_roaring_bytes + metadata_roaring_bytes,
    };
    Ok((data, metadata, report))
}

impl Cache {
    /// Exact logical Roaring size and packed payload bytes without writing.
    pub fn estimate_packed(
        address: &Address,
        snapshot: &PackedSnapshot,
    ) -> Result<PackedStorageReport> {
        Ok(prepare_snapshot(address, snapshot)?.2)
    }

    /// Persist all buffers atomically without applying another lossy codec.
    /// Use separate addresses for packed snapshots and affine tensor tiles.
    pub fn put_packed(
        &mut self,
        address: &Address,
        snapshot: &PackedSnapshot,
    ) -> Result<PackedStorageReport> {
        snapshot.validate()?;
        let key = address.key()?;
        {
            let view = self.db.snapshot()?;
            let previous = view.load(key | 1)?;
            if !previous.is_empty() && read_manifest(&set_to_bytes(&previous)?)?.address != *address
            {
                return Err(Error::Collision);
            }
        }
        let (data, metadata, report) = prepare_snapshot(address, snapshot)?;
        let mut batch = self.db.batch();
        batch.store_set(key, &data).store_set(key | 1, &metadata);
        let committed = batch.commit()?;
        self.db
            .wait_visible(committed.version, Duration::from_secs(30))?;
        self.db.checkpoint()?;
        Ok(report)
    }

    /// Persist many snapshots as **one** atomic unit.
    ///
    /// # Why this exists beside `put_packed`
    ///
    /// A Qwen3 cache is 56 tiles, and writing them one at a time pays a commit, a
    /// visibility wait and a **checkpoint** apiece. Two things follow, and only
    /// one of them is about speed.
    ///
    /// **Atomicity.** Fifty-six separate commits are fifty-six separate durable
    /// points, so a failure at tile 30 leaves 29 tiles of one logical cache
    /// persisted and 27 absent -- a state no reader can detect, because each tile
    /// is individually well formed and carries its own manifest. One batch makes
    /// the whole cache all-or-nothing, which is what "persist a cache" should have
    /// meant from the start.
    ///
    /// **Cost.** Measured on the real `compact-onebit` path, `persist_ms` fell
    /// from 1375-2038 to 267-733 across eight policy and prefix combinations --
    /// see `.agents/docs/write-strategies.md`, which also records why the
    /// synthetic decomposition behind it supports the direction and not a split
    /// between the checkpoints and the rest.
    ///
    /// What *is* established without measurement is that the per-tile checkpoint
    /// was not buying durability. yesno's commit is durable once its WAL is
    /// synchronized, and its checkpoint policy already fires on dirty bytes, WAL
    /// bytes and elapsed time, so an explicit checkpoint per tile adds only the
    /// cold-copy property -- and adds it 56 times over.
    ///
    /// # Collisions are all checked before anything is written
    ///
    /// Against **one** snapshot of the database, so every address is judged at the
    /// same instant, and a batch that would collide is rejected whole rather than
    /// leaving part of itself behind.
    /// The snapshots are **borrowed**, not owned. Taking them by value would
    /// make every caller clone its whole cache to call this -- 17 MB of copying
    /// at a 512-token prefix, which the per-tile loop it replaces never did. The
    /// address is owned because callers compute it per tile anyway and it is five
    /// short strings.
    pub fn put_packed_many(
        &mut self,
        items: &[(Address, &PackedSnapshot)],
    ) -> Result<Vec<PackedStorageReport>> {
        if items.is_empty() {
            return Ok(Vec::new());
        }
        // One view for every check: `put_packed` opens a snapshot per tile, which
        // is both slower and weaker -- the tiles are judged at different instants.
        {
            let view = self.db.snapshot()?;
            for (address, _) in items {
                let key = address.key()?;
                let previous = view.load(key | 1)?;
                if !previous.is_empty()
                    && read_manifest(&set_to_bytes(&previous)?)?.address != *address
                {
                    return Err(Error::Collision);
                }
            }
        }

        // **Two entries on one key are refused, not silently merged.**
        //
        // `store_set` emits a `DeleteKey` before its chunks, so a repeated key in
        // one batch is last-wins: the second entry erases the first and this
        // method would still return a report for both, telling the caller that 56
        // tiles were stored when 55 were. That is a wrong answer, not a lost
        // optimisation.
        //
        // It also closes a hole the per-tile path did not have. `put_packed`
        // checks each address against what is already stored, so a second address
        // landing on the same 63-bit key is caught -- the first write is visible
        // by the time the second is checked. Inside one batch neither is stored
        // yet, so both would pass the check above and one would silently
        // overwrite the other. Batching must not weaken that, so the same
        // condition is detected here, against the batch itself.
        let mut seen: std::collections::HashMap<u64, &Address> =
            HashMap::with_capacity(items.len());
        let mut prepared = Vec::with_capacity(items.len());
        for (address, snapshot) in items {
            snapshot.validate()?;
            let key = address.key()?;
            if let Some(previous) = seen.insert(key, address) {
                return Err(repeated_key(previous, address));
            }
            prepared.push((key, prepare_snapshot(address, snapshot)?));
        }

        let mut batch = self.db.batch();
        for (key, (data, metadata, _)) in &prepared {
            batch.store_set(*key, data).store_set(*key | 1, metadata);
        }
        let committed = batch.commit()?;
        self.db
            .wait_visible(committed.version, Duration::from_secs(30))?;
        // One checkpoint for the whole cache. Not for durability -- the commit
        // above already has that -- but so a cold copy of the data directory is
        // usable without replaying this batch's WAL.
        self.db.checkpoint()?;
        Ok(prepared.into_iter().map(|(_, (_, _, r))| r).collect())
    }

    /// Reject an incompatible format before allocating/uploading GPU buffers.
    pub fn get_packed(
        &self,
        address: &Address,
        expected_format: &str,
    ) -> Result<Option<PackedSnapshot>> {
        let view = self.db.snapshot()?;
        get_packed_in_view(&view, address, expected_format, ReadVerification::Full)
    }

    pub fn get_packed_with_verification(
        &self,
        address: &Address,
        expected_format: &str,
        verification: ReadVerification,
    ) -> Result<Option<PackedSnapshot>> {
        let view = self.db.snapshot()?;
        get_packed_in_view(&view, address, expected_format, verification)
    }
}

impl CacheReader {
    pub fn get_packed(
        &self,
        address: &Address,
        expected_format: &str,
    ) -> Result<Option<PackedSnapshot>> {
        let view = self.db.snapshot()?;
        get_packed_in_view(&view, address, expected_format, ReadVerification::Full)
    }

    pub fn get_packed_with_verification(
        &self,
        address: &Address,
        expected_format: &str,
        verification: ReadVerification,
    ) -> Result<Option<PackedSnapshot>> {
        let view = self.db.snapshot()?;
        get_packed_in_view(&view, address, expected_format, verification)
    }
}

pub(crate) fn stored_address_in_view(view: &Snapshot, key: u64) -> Result<Option<Address>> {
    let metadata = view.load(key | 1)?;
    if metadata.is_empty() {
        return Ok(None);
    }
    Ok(Some(read_manifest(&set_to_bytes(&metadata)?)?.address))
}

pub(crate) fn packed_format_in_view(view: &Snapshot, address: &Address) -> Result<Option<String>> {
    Ok(packed_manifest_in_view(view, address)?.map(|manifest| manifest.format))
}

pub(crate) fn packed_manifest_in_view(
    view: &Snapshot,
    address: &Address,
) -> Result<Option<Manifest>> {
    let metadata = view.load(address.key()? | 1)?;
    if metadata.is_empty() {
        return Ok(None);
    }
    let manifest = read_manifest(&set_to_bytes(&metadata)?)?;
    if manifest.address != *address {
        return Ok(None);
    }
    Ok(Some(manifest))
}

pub(crate) fn get_packed_in_view(
    view: &Snapshot,
    address: &Address,
    expected_format: &str,
    verification: ReadVerification,
) -> Result<Option<PackedSnapshot>> {
    let key = address.key()?;
    let metadata = view.load(key | 1)?;
    if metadata.is_empty() {
        return Ok(None);
    }
    let manifest = read_manifest(&set_to_bytes(&metadata)?)?;
    if manifest.address != *address {
        return Ok(None);
    }
    if manifest.format != expected_format {
        return Err(Error::Invalid("packed engine format mismatch".into()));
    }
    let payload = set_to_bytes(&view.load(key)?)?;
    if verification == ReadVerification::Full
        && Sha256::digest(&payload)[..] != manifest.payload_sha256
    {
        return Err(Error::Corrupt("packed payload checksum mismatch".into()));
    }
    let mut offset = 0usize;
    let mut buffers = Vec::new();
    for b in manifest.buffers {
        let end = offset
            .checked_add(b.length)
            .filter(|&end| end <= payload.len())
            .ok_or_else(|| Error::Corrupt("packed buffer length overflow".into()))?;
        buffers.push(PackedBuffer {
            name: b.name,
            dtype: b.dtype,
            shape: b.shape,
            bytes: payload[offset..end].to_vec(),
        });
        offset = end;
    }
    if offset != payload.len() {
        return Err(Error::Corrupt("unclaimed packed payload bytes".into()));
    }
    let snapshot = PackedSnapshot {
        format: manifest.format,
        buffers,
    };
    snapshot.validate()?;
    Ok(Some(snapshot))
}

impl Cache {
    pub fn remove_packed(&mut self, address: &Address) -> Result<bool> {
        let key = address.key()?;
        {
            let view = self.db.snapshot()?;
            let metadata = view.load(key | 1)?;
            if metadata.is_empty() || read_manifest(&set_to_bytes(&metadata)?)?.address != *address
            {
                return Ok(false);
            }
        }
        let mut batch = self.db.batch();
        batch.delete_key(key).delete_key(key | 1);
        let committed = batch.commit()?;
        self.db
            .wait_visible(committed.version, Duration::from_secs(30))?;
        self.db.checkpoint()?;
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn packed_checksum_rejects_corruption_and_address_collision() {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join(".agents-workspace/tmp/tests")
            .join(format!(
                "packed-integrity-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
        let mut cache = Cache::open(path).unwrap();
        let address = Address {
            namespace: "test".into(),
            model_fingerprint: "weights".into(),
            prefix_fingerprint: "prefix".into(),
            layer: 0,
            slot: "snapshot".into(),
        };
        let snapshot = PackedSnapshot {
            format: "test/v1".into(),
            buffers: vec![PackedBuffer {
                name: "codes".into(),
                dtype: "u8".into(),
                shape: vec![3],
                bytes: vec![1, 2, 0],
            }],
        };
        cache.put_packed(&address, &snapshot).unwrap();
        let key = address.key().unwrap();
        let mut batch = cache.db.batch();
        batch.store_set(key, &bytes_to_set(&[9, 2, 0]));
        let committed = batch.commit().unwrap();
        cache
            .db
            .wait_visible(committed.version, Duration::from_secs(30))
            .unwrap();
        assert!(matches!(
            cache.get_packed(&address, "test/v1"),
            Err(Error::Corrupt(_))
        ));
        cache.put_packed(&address, &snapshot).unwrap();
        let mut manifest = {
            let view = cache.db.snapshot().unwrap();
            read_manifest(&set_to_bytes(&view.load(key | 1).unwrap()).unwrap()).unwrap()
        };
        manifest.payload_sha256[0] ^= 1;
        let metadata = bytes_to_set(&manifest_bytes(&manifest).unwrap());
        let mut batch = cache.db.batch();
        batch.store_set(key | 1, &metadata);
        let committed = batch.commit().unwrap();
        cache
            .db
            .wait_visible(committed.version, Duration::from_secs(30))
            .unwrap();
        assert!(matches!(
            cache.get_packed(&address, "test/v1"),
            Err(Error::Corrupt(_))
        ));
        assert_eq!(
            cache
                .get_packed_with_verification(&address, "test/v1", ReadVerification::StorageOnly,)
                .unwrap(),
            Some(snapshot.clone())
        );
        cache.put_packed(&address, &snapshot).unwrap();
        let mut manifest = {
            let view = cache.db.snapshot().unwrap();
            read_manifest(&set_to_bytes(&view.load(key | 1).unwrap()).unwrap()).unwrap()
        };
        manifest.address.namespace = "another-tenant".into();
        let metadata = bytes_to_set(&manifest_bytes(&manifest).unwrap());
        let mut batch = cache.db.batch();
        batch.store_set(key | 1, &metadata);
        let committed = batch.commit().unwrap();
        cache
            .db
            .wait_visible(committed.version, Duration::from_secs(30))
            .unwrap();
        assert!(cache.get_packed(&address, "test/v1").unwrap().is_none());
        assert!(matches!(
            cache.put_packed(&address, &snapshot),
            Err(Error::Collision)
        ));
        assert!(!cache.remove_packed(&address).unwrap());
    }
}

/// What a second entry on an already-used key means.
///
/// Extracted so **both** arms can be tested. The distinct-address arm needs two
/// addresses sharing a 63-bit key, which cannot be produced in a unit test -- a
/// first attempt searched 200 000 candidates, found none, printed that it had
/// skipped, and passed. Deciding on the inputs rather than on a database makes
/// the branch reachable.
fn repeated_key(previous: &Address, current: &Address) -> Error {
    if previous == current {
        Error::Invalid(format!(
            "address {}/{}/{} layer {} slot {} appears twice in one batch; a batch stores each address once",
            current.namespace,
            current.model_fingerprint,
            current.prefix_fingerprint,
            current.layer,
            current.slot
        ))
    } else {
        // Distinct addresses, one key: exactly what `put_packed` reports across
        // two calls, and it must not become a silent overwrite here.
        Error::Collision
    }
}

#[cfg(test)]
mod atomicity {
    use super::*;

    fn address(i: usize, prefix: &str) -> Address {
        Address {
            namespace: "atomicity".into(),
            model_fingerprint: "qwen3-0.6b".into(),
            prefix_fingerprint: prefix.into(),
            layer: (i / 2) as u32,
            slot: if i.is_multiple_of(2) {
                "k".into()
            } else {
                "v".into()
            },
        }
    }

    fn tile(i: usize) -> PackedSnapshot {
        PackedSnapshot {
            format: "atomicity-v1".into(),
            buffers: vec![PackedBuffer {
                name: "codes".into(),
                dtype: "u8".into(),
                shape: vec![8],
                bytes: vec![i as u8; 8],
            }],
        }
    }

    /// A directory this test **created**, removed only if the creation succeeded.
    ///
    /// # Why not `temp_dir()` with a predictable name
    ///
    /// The first version of these tests built
    /// `temp_dir()/shifou-atomicity-<tag>-<pid>-<thread>` and began with
    /// `remove_dir_all` on it. A pid is reused and a name is guessable, so that
    /// is a recursive delete of a path the test did not create and cannot
    /// identify -- if anything already occupied it, the test destroyed it. The
    /// odds are small and the failure is unbounded, which is the wrong trade for
    /// a convenience.
    ///
    /// `create_dir` rather than `create_dir_all`: it **fails** if the path
    /// exists, so reaching the body at all proves this run made it, and the
    /// cleanup is then provably removing only what this run owns.
    struct Scratch(std::path::PathBuf);

    impl Scratch {
        fn new(tag: &str) -> Scratch {
            let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join(".agents-workspace")
                .join("tmp");
            std::fs::create_dir_all(&root).expect("scratch root");
            for attempt in 0..64 {
                let path = root.join(format!(
                    "atomicity-{tag}-{}-{:?}-{attempt}",
                    std::process::id(),
                    std::thread::current().id()
                ));
                if std::fs::create_dir(&path).is_ok() {
                    return Scratch(path);
                }
            }
            panic!("no free scratch directory for {tag}");
        }

        fn path(&self) -> &std::path::Path {
            &self.0
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            // Only ever this run's own directory, and only on the way out, so a
            // panicking test still cleans up.
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// Per-tile writes leave a cache that is **partly** persisted, and nothing
    /// can tell.
    ///
    /// This is the cost of one commit per tile, and it is a correctness property
    /// rather than a performance one. Each tile is individually well formed and
    /// carries its own manifest, so a reader that finds 30 of 56 sees 30 valid
    /// tiles -- not a torn write, not a checksum failure, just a cache that
    /// silently is not the cache anybody stored.
    #[test]
    fn per_tile_writes_can_leave_a_partial_cache() {
        let scratch = Scratch::new("partial");
        let d = scratch.path();
        {
            let mut cache = Cache::open(d).unwrap();
            // Interrupted after 30 of 56, which is what a crash, an OOM kill or a
            // returned error in the middle of the loop produces.
            for i in 0..30 {
                cache.put_packed(&address(i, "p"), &tile(i)).unwrap();
            }
        }
        let cache = Cache::open(d).unwrap();
        let present = (0..56)
            .filter(|i| {
                cache
                    .get_packed(&address(*i, "p"), "atomicity-v1")
                    .unwrap()
                    .is_some()
            })
            .count();
        assert_eq!(
            present, 30,
            "30 tiles survived an interrupted write and 26 did not, which is a \
             half-written cache that reads as a whole one"
        );
    }

    /// One batch is all-or-nothing: a rejected write leaves **nothing** behind.
    ///
    /// The collision is planted on the 40th tile's key, so the batch is refused
    /// after 39 perfectly good tiles have been prepared. Under the per-tile path
    /// those 39 would already be durable; here none of them is.
    ///
    /// The collision is forged the way `store`'s own collision test forges one --
    /// by storing a manifest whose recorded address is not the address that maps
    /// to its key -- because two different addresses do not otherwise share a
    /// key. An earlier version of this test assumed they would and asserted it;
    /// the assertion failed, which is the only reason this is a real collision
    /// rather than a test that passes because nothing collided.
    #[test]
    fn a_rejected_batch_writes_nothing_at_all() {
        let scratch = Scratch::new("allornothing");
        let d = scratch.path();
        {
            let mut cache = Cache::open(d).unwrap();
            let victim = address(40, "p");
            let key = victim.key().unwrap();
            cache.put_packed(&victim, &tile(40)).unwrap();

            // Rewrite the stored manifest so its address no longer matches the
            // key it lives under. Any later write to this key must be refused.
            let view = cache.db.snapshot().unwrap();
            let stored = view.load(key | 1).unwrap();
            let mut manifest = read_manifest(&set_to_bytes(&stored).unwrap()).unwrap();
            drop(view);
            manifest.address.model_fingerprint = "a-different-model".into();
            let forged = bytes_to_set(&manifest_bytes(&manifest).unwrap());
            let mut batch = cache.db.batch();
            batch.store_set(key | 1, &forged);
            let version = batch.commit().unwrap().version;
            cache
                .db
                .wait_visible(version, Duration::from_secs(30))
                .unwrap();

            let tiles: Vec<PackedSnapshot> = (0..56).map(tile).collect();
            let items: Vec<(Address, &PackedSnapshot)> =
                (0..56).map(|i| (address(i, "p"), &tiles[i])).collect();
            assert!(
                matches!(cache.put_packed_many(&items), Err(Error::Collision)),
                "the batch must be refused"
            );
        }
        let cache = Cache::open(d).unwrap();
        // Every tile except the forged one: none may have been written.
        let present = (0..56)
            .filter(|i| *i != 40)
            .filter(|i| {
                cache
                    .get_packed(&address(*i, "p"), "atomicity-v1")
                    .unwrap()
                    .is_some()
            })
            .count();
        assert_eq!(
            present, 0,
            "a refused batch must leave nothing, not the 39 tiles it had already \
             prepared before reaching the collision"
        );
    }

    /// The same address twice in one batch is refused.
    ///
    /// Not merged, and not last-wins. `store_set` emits a `DeleteKey` before its
    /// chunks, so the second entry would erase the first while this method
    /// returned a report for both -- telling the caller 56 tiles were stored when
    /// 55 were. Measured before the check existed: the call returned `Ok` with
    /// two reports and the database held only the second tile.
    #[test]
    fn a_duplicate_address_in_one_batch_is_refused() {
        let scratch = Scratch::new("duplicate");
        let mut cache = Cache::open(scratch.path()).unwrap();
        let a = address(7, "p");
        let (t1, t2) = (tile(1), tile(2));
        let items = vec![(a.clone(), &t1), (a.clone(), &t2)];
        let e = cache.put_packed_many(&items);
        assert!(
            matches!(&e, Err(Error::Invalid(m)) if m.contains("twice in one batch")),
            "expected a duplicate refusal, got {e:?}"
        );
        assert!(
            cache.get_packed(&a, "atomicity-v1").unwrap().is_none(),
            "and nothing may have been written"
        );
    }

    /// The decision a repeated key triggers, on both arms.
    ///
    /// Asserted on the inputs rather than through a database, because the
    /// distinct-address arm needs two addresses sharing a 63-bit key and that
    /// cannot be produced in a test: an earlier version searched 200 000
    /// candidates, found none, printed "skipped" and passed green. A branch only
    /// a birthday collision can reach still has to be right, so it is checked
    /// where it can be.
    #[test]
    fn a_repeated_key_is_a_duplicate_or_a_collision() {
        let a = address(7, "p");
        let same = a.clone();
        let other = address(8, "p");
        assert!(
            matches!(repeated_key(&a, &same), Error::Invalid(m) if m.contains("twice in one batch")),
            "the same address twice is a caller mistake, named as one"
        );
        assert!(
            matches!(repeated_key(&other, &a), Error::Collision),
            "two addresses on one key is the collision put_packed would report"
        );
    }

    /// And the batched path stores what it claims to.
    #[test]
    fn a_whole_cache_batch_round_trips() {
        let scratch = Scratch::new("roundtrip");
        let d = scratch.path();
        {
            let mut cache = Cache::open(d).unwrap();
            let tiles: Vec<PackedSnapshot> = (0..56).map(tile).collect();
            let items: Vec<(Address, &PackedSnapshot)> =
                (0..56).map(|i| (address(i, "p"), &tiles[i])).collect();
            let reports = cache.put_packed_many(&items).unwrap();
            assert_eq!(reports.len(), 56, "one report per tile, in order");
        }
        let cache = Cache::open(d).unwrap();
        for i in 0..56 {
            let got = cache
                .get_packed(&address(i, "p"), "atomicity-v1")
                .unwrap()
                .unwrap_or_else(|| panic!("tile {i} missing after reopen"));
            assert_eq!(got.buffers[0].bytes, vec![i as u8; 8], "tile {i} contents");
        }
    }
}
