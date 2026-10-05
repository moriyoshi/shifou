//! Lossless, prefix-independent storage for routed-expert snapshots.
//! The model engine owns the snapshot envelope and validates it after reading.

use crate::{
    Address, Cache, CacheReader, Error, PackedBuffer, PackedSnapshot, PackedStorageReport,
    ReadVerification, Result,
};

const NAMESPACE: &str = "shifou-expert-snapshot/v1";
const PREFIX: &str = "model-weights";
const BUFFER: &str = "expert";

/// One expert in one exact model and engine snapshot format.
///
/// `model_fingerprint` must identify weight values, adapters, and numerical
/// settings. A tensor-layout digest alone is insufficient. The key does not
/// contain a prompt prefix because weights can be shared by all requests for
/// the same model. The engine must still validate its own snapshot envelope.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ExpertSnapshotKey {
    pub model_fingerprint: [u8; 32],
    pub format: String,
    pub layer: u32,
    pub expert: u32,
}

impl ExpertSnapshotKey {
    pub fn address(&self) -> Result<Address> {
        if self.model_fingerprint == [0; 32] || self.format.is_empty() || self.format.len() > 256 {
            return Err(Error::Invalid("invalid expert snapshot identity".into()));
        }
        let mut fingerprint = String::with_capacity(64);
        for byte in self.model_fingerprint {
            use std::fmt::Write;
            write!(&mut fingerprint, "{byte:02x}").unwrap();
        }
        let address = Address {
            namespace: NAMESPACE.into(),
            model_fingerprint: fingerprint,
            prefix_fingerprint: PREFIX.into(),
            layer: self.layer,
            slot: format!("{}:{}", self.format, self.expert),
        };
        address.key()?;
        Ok(address)
    }
}

fn packed(key: &ExpertSnapshotKey, bytes: &[u8]) -> Result<(Address, PackedSnapshot)> {
    if bytes.is_empty() {
        return Err(Error::Invalid("empty expert snapshot".into()));
    }
    let address = key.address()?;
    let snapshot = PackedSnapshot {
        format: key.format.clone(),
        buffers: vec![PackedBuffer {
            name: BUFFER.into(),
            dtype: "u8".into(),
            shape: vec![bytes.len()],
            bytes: bytes.to_vec(),
        }],
    };
    snapshot.validate()?;
    Ok((address, snapshot))
}

fn unpack(snapshot: PackedSnapshot) -> Result<Vec<u8>> {
    let mut buffers = snapshot.buffers.into_iter();
    let buffer = buffers
        .next()
        .ok_or_else(|| Error::Corrupt("expert snapshot has no payload".into()))?;
    if buffers.next().is_some() || buffer.name != BUFFER || buffer.dtype != "u8" {
        return Err(Error::Corrupt("invalid expert snapshot payload".into()));
    }
    Ok(buffer.bytes)
}

impl Cache {
    /// Store one opaque engine snapshot. Use `put_expert_snapshots` to amortize
    /// commits and checkpointing across several experts.
    pub fn put_expert_snapshot(
        &mut self,
        key: &ExpertSnapshotKey,
        bytes: &[u8],
    ) -> Result<PackedStorageReport> {
        let (address, snapshot) = packed(key, bytes)?;
        self.put_packed(&address, &snapshot)
    }

    /// Publish a bounded group of experts in one atomic yesno transaction.
    /// Callers should choose a moderate group size rather than staging a whole
    /// multi-GiB model at once. Duplicate keys and address collisions fail
    /// before any member of the group is written.
    pub fn put_expert_snapshots(
        &mut self,
        items: &[(ExpertSnapshotKey, &[u8])],
    ) -> Result<Vec<PackedStorageReport>> {
        let prepared = items
            .iter()
            .map(|(key, bytes)| packed(key, bytes))
            .collect::<Result<Vec<_>>>()?;
        let borrowed = prepared
            .iter()
            .map(|(address, snapshot)| (address.clone(), snapshot))
            .collect::<Vec<_>>();
        self.put_packed_many(&borrowed)
    }

    pub fn get_expert_snapshot(&self, key: &ExpertSnapshotKey) -> Result<Option<Vec<u8>>> {
        self.get_expert_snapshot_with_verification(key, ReadVerification::Full)
    }

    /// `StorageOnly` is useful when the engine validates its own complete
    /// snapshot envelope immediately after reading these bytes.
    pub fn get_expert_snapshot_with_verification(
        &self,
        key: &ExpertSnapshotKey,
        verification: ReadVerification,
    ) -> Result<Option<Vec<u8>>> {
        let address = key.address()?;
        self.get_packed_with_verification(&address, &key.format, verification)?
            .map(unpack)
            .transpose()
    }

    pub fn remove_expert_snapshot(&mut self, key: &ExpertSnapshotKey) -> Result<bool> {
        self.remove_packed(&key.address()?)
    }
}

impl CacheReader {
    pub fn get_expert_snapshot(&self, key: &ExpertSnapshotKey) -> Result<Option<Vec<u8>>> {
        self.get_expert_snapshot_with_verification(key, ReadVerification::Full)
    }

    /// Keep `Full` for general callers. Use `StorageOnly` only when the engine
    /// immediately checks its own envelope and payload digest before GPU use.
    pub fn get_expert_snapshot_with_verification(
        &self,
        key: &ExpertSnapshotKey,
        verification: ReadVerification,
    ) -> Result<Option<Vec<u8>>> {
        let address = key.address()?;
        self.get_packed_with_verification(&address, &key.format, verification)?
            .map(unpack)
            .transpose()
    }
}
