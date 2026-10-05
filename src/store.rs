use std::fs::{File, OpenOptions};
use std::path::Path;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use yesno_core::{
    container::{BitmapContainer, Container},
    roaring_format::serialize_u64,
    Db, DbOptions, OrdSet,
};

use crate::codec::{Descriptor, Group};
use crate::{decode, encode, EncodedTile, Error, Policy, Report, Result, Tensor};

/// Adapters must fingerprint weights, adapters, position/attention semantics,
/// and numerical execution compatibility in model_fingerprint. Prefix identity
/// covers the complete causal prefix and any non-text inputs, not just this tile.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Address {
    pub namespace: String,
    pub model_fingerprint: String,
    pub prefix_fingerprint: String,
    pub layer: u32,
    pub slot: String,
}

impl Address {
    fn validate(&self) -> Result<()> {
        if [
            &self.namespace,
            &self.model_fingerprint,
            &self.prefix_fingerprint,
            &self.slot,
        ]
        .iter()
        .any(|s| s.is_empty() || s.len() > 4096)
        {
            return Err(Error::Invalid(
                "address strings must contain 1..=4096 bytes".into(),
            ));
        }
        Ok(())
    }

    pub(crate) fn key(&self) -> Result<u64> {
        self.validate()?;
        let digest = Sha256::digest(serde_json::to_vec(self)?);
        let mut prefix = [0; 8];
        prefix.copy_from_slice(&digest[..8]);
        Ok(u64::from_le_bytes(prefix) & !1)
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Record {
    address: Address,
    descriptor: Descriptor,
    payload_sha256: [u8; 32],
}

#[derive(Debug, Serialize, Deserialize)]
pub struct StorageReport {
    pub codec: Report,
    pub metadata_roaring_bytes: usize,
    /// Logical portable serializations; excludes WAL, indexes and allocated slabs.
    pub total_roaring_bytes: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Retrieved {
    pub tensor: Tensor,
    pub report: StorageReport,
}

/// One exclusive process-local owner per directory. Mutations require &mut self.
/// Metadata and payload are replaced in one yesno transaction, then checkpointed.
pub struct Cache {
    pub(crate) db: Db,
    _lock: File,
}

/// A checkpoint-visible, read-only handle that can coexist with a writer in
/// another process. Reopen it to observe a later checkpoint.
pub struct CacheReader {
    pub(crate) db: Db,
}

impl CacheReader {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Ok(Self {
            db: Db::open_reader(path.as_ref().join("data"))?,
        })
    }
}

const MAX_METADATA_BYTES: usize = 64 * 1024 * 1024;

pub(crate) fn bytes_to_set(bytes: &[u8]) -> OrdSet {
    // One Roaring chunk represents 8 KiB of input bytes. Build its bitmap
    // words directly; dense quantized payloads must not stage one u64 ordinal
    // for every one bit. The final bit records the byte length, including
    // trailing zeros. An aligned length puts that bit in a new chunk.
    const CHUNK_BYTES: usize = 8192;
    const WORDS: usize = CHUNK_BYTES / 8;
    let mut chunks = Vec::with_capacity(bytes.len() / CHUNK_BYTES + 1);
    for index in 0..=bytes.len() / CHUNK_BYTES {
        let start = index * CHUNK_BYTES;
        let end = (start + CHUNK_BYTES).min(bytes.len());
        let mut words = vec![0u64; WORDS];
        for (word, part) in bytes[start..end].chunks(8).enumerate() {
            let mut little_endian = [0u8; 8];
            little_endian[..part.len()].copy_from_slice(part);
            words[word] = u64::from_le_bytes(little_endian);
        }
        if index == bytes.len() / CHUNK_BYTES {
            let bit = (end - start) * 8;
            words[bit / 64] |= 1u64 << (bit % 64);
        }
        let cardinality = words.iter().map(|word| word.count_ones()).sum();
        if cardinality > 0 {
            chunks.push((
                index as u64,
                Container::Bitmap(BitmapContainer::from_words(words, cardinality)),
            ));
        }
    }
    let mut set = OrdSet::from_chunks(chunks);
    set.optimize();
    set
}

pub(crate) fn set_to_bytes(set: &OrdSet) -> Result<Vec<u8>> {
    let sentinel = set
        .max()
        .ok_or_else(|| Error::Corrupt("missing metadata".into()))?;
    if sentinel % 8 != 0 || sentinel / 8 > MAX_METADATA_BYTES as u64 {
        return Err(Error::Corrupt("invalid metadata length marker".into()));
    }
    let mut bytes = vec![0; (sentinel / 8) as usize];
    for (prefix, container) in set.chunks() {
        let base = prefix << 16;
        match container {
            Container::Bitmap(bitmap) if base < sentinel => {
                let start = (base / 8) as usize;
                let remaining = (bytes.len() - start).min(8192);
                let mut copied = [0u64; 1024];
                let words = if let Some(words) = bitmap.try_words() {
                    words
                } else {
                    bitmap.copy_words_into(&mut copied);
                    &copied
                };
                for (i, word) in words.iter().enumerate().take(remaining.div_ceil(8)) {
                    let offset = start + i * 8;
                    let count = (bytes.len() - offset).min(8);
                    bytes[offset..offset + count].copy_from_slice(&word.to_le_bytes()[..count]);
                }
            }
            _ => {
                for ordinal in container.iter() {
                    let ordinal = base | ordinal as u64;
                    if ordinal < sentinel {
                        bytes[(ordinal / 8) as usize] |= 1 << (ordinal % 8);
                    }
                }
            }
        }
    }
    Ok(bytes)
}

// A small JSON header keeps model metadata extensible. Group descriptors are
// fixed 29-byte records so per-group field names cannot dominate tensor storage.
fn write_record(mut record: Record) -> Result<Vec<u8>> {
    let groups = std::mem::take(&mut record.descriptor.groups);
    let header = serde_json::to_vec(&record)?;
    let mut body = (header.len() as u32).to_le_bytes().to_vec();
    body.extend_from_slice(&header);
    for g in groups {
        body.extend_from_slice(&g.offset.to_le_bytes());
        body.push(g.bits);
        body.extend_from_slice(&g.minimum.to_le_bytes());
        body.extend_from_slice(&g.step.to_le_bytes());
        body.extend_from_slice(&(g.exceptions as u32).to_le_bytes());
    }
    let mut bytes = b"SHIFOU01".to_vec();
    bytes.extend_from_slice(&Sha256::digest(&body));
    bytes.extend_from_slice(&body);
    Ok(bytes)
}

fn read_record(metadata: &OrdSet) -> Result<Record> {
    let bytes = set_to_bytes(metadata)?;
    if bytes.len() < 44 || &bytes[..8] != b"SHIFOU01" {
        return Err(Error::Corrupt(
            "unsupported storage version or truncated record".into(),
        ));
    }
    let digest: [u8; 32] = Sha256::digest(&bytes[40..]).into();
    if digest != bytes[8..40] {
        return Err(Error::Corrupt("descriptor checksum mismatch".into()));
    }
    let header_len = u32::from_le_bytes(bytes[40..44].try_into().unwrap()) as usize;
    if header_len > bytes.len() - 44 || !(bytes.len() - 44 - header_len).is_multiple_of(29) {
        return Err(Error::Corrupt("invalid descriptor lengths".into()));
    }
    let mut record: Record = serde_json::from_slice(&bytes[44..44 + header_len])?;
    if !record.descriptor.groups.is_empty() {
        return Err(Error::Corrupt("duplicate group descriptors".into()));
    }
    for chunk in bytes[44 + header_len..].chunks_exact(29) {
        record.descriptor.groups.push(Group {
            offset: u64::from_le_bytes(chunk[..8].try_into().unwrap()),
            bits: chunk[8],
            minimum: f64::from_le_bytes(chunk[9..17].try_into().unwrap()),
            step: f64::from_le_bytes(chunk[17..25].try_into().unwrap()),
            exceptions: u32::from_le_bytes(chunk[25..29].try_into().unwrap()) as usize,
        });
    }
    Ok(record)
}

fn prepare_tile(address: &Address, tile: &EncodedTile) -> Result<(OrdSet, StorageReport)> {
    address.validate()?;
    let codec = tile.report()?;
    let record = Record {
        address: address.clone(),
        descriptor: tile.descriptor.clone(),
        payload_sha256: Sha256::digest(serialize_u64(&tile.payload)).into(),
    };
    let bytes = write_record(record)?;
    if bytes.len() > MAX_METADATA_BYTES {
        return Err(Error::Invalid("descriptor exceeds 64 MiB".into()));
    }
    let metadata = bytes_to_set(&bytes);
    let metadata_roaring_bytes = serialize_u64(&metadata).len();
    let report = StorageReport {
        total_roaring_bytes: codec.payload_bytes + metadata_roaring_bytes,
        codec,
        metadata_roaring_bytes,
    };
    Ok((metadata, report))
}

impl Cache {
    /// Exact portable bytes for this tile and address, before any database write.
    /// Useful for selecting a precision policy under a storage budget.
    pub fn estimate(address: &Address, tensor: &Tensor, policy: &Policy) -> Result<StorageReport> {
        let tile = encode(tensor, policy)?;
        Self::estimate_encoded(address, &tile)
    }

    /// Estimate an already encoded tile without quantizing it a second time.
    pub fn estimate_encoded(address: &Address, tile: &EncodedTile) -> Result<StorageReport> {
        Ok(prepare_tile(address, tile)?.1)
    }

    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        std::fs::create_dir_all(path)?;
        let lock = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path.join("shifou.lock"))?;
        lock.try_lock().map_err(|e| Error::Busy(e.to_string()))?;
        let db = Db::open_with(
            path.join("data"),
            DbOptions {
                shards: 1,
                ..DbOptions::default()
            },
        )?;
        Ok(Self { db, _lock: lock })
    }

    /// Replaces this exact address, including all old bits. A hash collision
    /// with another full address is an explicit error, never an alias or overwrite.
    pub fn put(
        &mut self,
        address: &Address,
        tensor: &Tensor,
        policy: &Policy,
    ) -> Result<StorageReport> {
        let key = address.key()?;
        {
            let snapshot = self.db.snapshot()?;
            let previous = snapshot.load(key | 1)?;
            if !previous.is_empty() && read_record(&previous)?.address != *address {
                return Err(Error::Collision);
            }
        }
        let tile = encode(tensor, policy)?;
        let (metadata, report) = prepare_tile(address, &tile)?;
        let mut batch = self.db.batch();
        batch
            .store_set(key, &tile.payload)
            .store_set(key | 1, &metadata);
        let committed = batch.commit()?;
        self.db
            .wait_visible(committed.version, Duration::from_secs(30))?;
        self.db.checkpoint()?;
        Ok(report)
    }

    pub fn get(&self, address: &Address) -> Result<Option<Retrieved>> {
        let key = address.key()?;
        let snapshot = self.db.snapshot()?;
        let metadata = snapshot.load(key | 1)?;
        if metadata.is_empty() {
            return Ok(None);
        }
        let record = read_record(&metadata)?;
        if record.address != *address {
            return Ok(None);
        }
        let payload = snapshot.load(key)?;
        let digest: [u8; 32] = Sha256::digest(serialize_u64(&payload)).into();
        if digest != record.payload_sha256 {
            return Err(Error::Corrupt("payload checksum mismatch".into()));
        }
        let tile = EncodedTile {
            descriptor: record.descriptor,
            payload,
        };
        let tensor = decode(&tile)?;
        let codec = tile.report()?;
        let metadata_roaring_bytes = serialize_u64(&metadata).len();
        Ok(Some(Retrieved {
            tensor,
            report: StorageReport {
                total_roaring_bytes: codec.payload_bytes + metadata_roaring_bytes,
                codec,
                metadata_roaring_bytes,
            },
        }))
    }

    pub fn remove(&mut self, address: &Address) -> Result<bool> {
        let key = address.key()?;
        {
            let snapshot = self.db.snapshot()?;
            let metadata = snapshot.load(key | 1)?;
            if metadata.is_empty() || read_record(&metadata)?.address != *address {
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
    fn metadata_preserves_zero_bytes_and_detects_bad_length() {
        assert_eq!(
            set_to_bytes(&bytes_to_set(&[0, 3, 0])).unwrap(),
            vec![0, 3, 0]
        );
        assert!(set_to_bytes(&OrdSet::from_iter_unsorted([9])).is_err());
    }

    /// Compare the word-oriented constructor with an independent ordinal
    /// oracle across chunk boundaries and dense and sparse bytes.
    #[test]
    fn word_oriented_constructor_matches_bit_oracle() {
        for bytes in [
            vec![],
            vec![0],
            vec![0xff],
            vec![0xff; 8192],
            vec![0; 8192],
            vec![0x55; 8193],
            (0..=255u8).cycle().take(16_385).collect(),
        ] {
            let mut ordinals = Vec::new();
            for (i, &byte) in bytes.iter().enumerate() {
                for bit in 0..8 {
                    if byte & (1 << bit) != 0 {
                        ordinals.push((i * 8 + bit) as u64);
                    }
                }
            }
            ordinals.push(bytes.len() as u64 * 8);
            assert_eq!(bytes_to_set(&bytes), OrdSet::from_sorted_slice(&ordinals));
            assert_eq!(set_to_bytes(&bytes_to_set(&bytes)).unwrap(), bytes);
        }
    }

    /// Bytes survive the set round trip, trailing zeros included.
    ///
    /// The existing case above is three bytes. Trailing zeros are the interesting
    /// input because they contribute **no** ordinal at all -- the length is
    /// carried only by the sentinel -- so a payload ending in zeros is exactly
    /// where a length bug hides, and a fixed fixture is unlikely to contain a long
    /// enough run of them to notice.
    #[test]
    fn arbitrary_bytes_round_trip_including_trailing_zeros() {
        let cases: Vec<Vec<u8>> = vec![
            vec![],
            vec![0],
            vec![0, 0, 0, 0, 0, 0, 0, 0],
            vec![0xff, 0, 0, 0],
            vec![0, 0, 0, 0xff],
            vec![0x01, 0x80, 0x00, 0x00, 0x00],
            (0..=255u8).collect(),
            {
                let mut v: Vec<u8> = (0..64u8).collect();
                v.extend(std::iter::repeat_n(0u8, 64));
                v
            },
        ];
        for bytes in cases {
            let back = set_to_bytes(&bytes_to_set(&bytes)).unwrap();
            assert_eq!(back, bytes, "round trip lost {} bytes", bytes.len());
        }
    }

    fn fixture() -> (Address, Tensor, Policy) {
        (
            Address {
                namespace: "integrity".into(),
                model_fingerprint: "model-a".into(),
                prefix_fingerprint: "prefix".into(),
                layer: 0,
                slot: "key".into(),
            },
            Tensor {
                shape: vec![4],
                axes: vec!["channel".into()],
                values: vec![0.0, 1.0, 2.0, 3.0],
            },
            Policy {
                group_axis: "channel".into(),
                group_size: 4,
                candidate_bits: vec![2],
                max_abs_error: 0.0,
                outlier_fraction: 0.0,
                residual_tokens: 0,
                token_axis: "token".into(),
            },
        )
    }

    fn directory(label: &str) -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join(".agents-workspace/tmp/tests")
            .join(format!(
                "{label}-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ))
    }

    #[test]
    fn modified_payload_is_detected_through_public_read() {
        let (address, tensor, policy) = fixture();
        let mut cache = Cache::open(directory("corrupt")).unwrap();
        cache.put(&address, &tensor, &policy).unwrap();
        let mut batch = cache.db.batch();
        batch.store_set(address.key().unwrap(), &OrdSet::new());
        let version = batch.commit().unwrap().version;
        cache
            .db
            .wait_visible(version, Duration::from_secs(5))
            .unwrap();
        assert!(matches!(cache.get(&address), Err(Error::Corrupt(_))));
    }

    #[test]
    fn forced_address_collision_is_a_miss_and_cannot_overwrite() {
        let (address, tensor, policy) = fixture();
        let mut cache = Cache::open(directory("collision")).unwrap();
        cache.put(&address, &tensor, &policy).unwrap();
        let snapshot = cache.db.snapshot().unwrap();
        let metadata = snapshot.load(address.key().unwrap() | 1).unwrap();
        let mut record = read_record(&metadata).unwrap();
        record.address.model_fingerprint = "different-full-identity".into();
        let metadata = bytes_to_set(&write_record(record).unwrap());
        drop(snapshot);
        let mut batch = cache.db.batch();
        batch.store_set(address.key().unwrap() | 1, &metadata);
        let version = batch.commit().unwrap().version;
        cache
            .db
            .wait_visible(version, Duration::from_secs(5))
            .unwrap();
        assert!(cache.get(&address).unwrap().is_none());
        assert!(matches!(
            cache.put(&address, &tensor, &policy),
            Err(Error::Collision)
        ));
        assert!(!cache.remove(&address).unwrap());
    }

    #[test]
    fn descriptor_corruption_and_truncation_are_errors() {
        let (address, tensor, policy) = fixture();
        let tile = encode(&tensor, &policy).unwrap();
        let bytes = write_record(Record {
            address,
            descriptor: tile.descriptor,
            payload_sha256: Sha256::digest(serialize_u64(&tile.payload)).into(),
        })
        .unwrap();
        assert!(read_record(&bytes_to_set(&bytes)).is_ok());
        for n in 0..bytes.len() {
            assert!(read_record(&bytes_to_set(&bytes[..n])).is_err());
        }
        for index in [0, 8, 43, bytes.len() - 1] {
            let mut modified = bytes.clone();
            modified[index] ^= 1;
            assert!(read_record(&bytes_to_set(&modified)).is_err());
        }
    }
}
