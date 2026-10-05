//! Atomic, model-agnostic prefill bundles built from packed yesno records.
//! The engine owns GDN and attention codecs; this module validates identity,
//! token boundary, checksums and publication as one durable transaction.

use crate::packed::{
    get_packed_in_view, packed_manifest_in_view, stored_address_in_view,
    Manifest as PackedManifest, ReadVerification,
};
use crate::{
    Address, Cache, CacheReader, Error, PackedBuffer, PackedSnapshot, PackedStorageReport, Result,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::time::Duration;
use yesno_core::{Db, Snapshot};

#[cfg(feature = "flight")]
mod flight;
#[cfg(feature = "flight")]
pub use flight::FlightBundleReader;
#[cfg(feature = "peer")]
mod peer;
#[cfg(feature = "peer")]
pub use peer::PeerBundleReader;

pub(crate) const MANIFEST_FORMAT: &str = "shifou-prefill-manifest/v1";
const TOKENS_FORMAT: &str = "shifou-prefill-tokens/v1";
const STATE_PAGE_FORMAT: &str = "shifou-prefill-state-page/v1";
pub(crate) const STATE_PAGE_BYTES: usize = 8 * 1024 * 1024;
pub(crate) const MAX_STATE_BYTES: usize = 2 * 1024 * 1024 * 1024;

/// One exact causal token boundary. `state_bytes` can hold xinfer's portable
/// GDN snapshot or another engine's opaque recurrent state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrefillBundle {
    pub prefix_tokens: u64,
    pub token_ids: Vec<u32>,
    pub state_format: String,
    pub state_bytes: Vec<u8>,
    pub attention: Vec<(Address, PackedSnapshot)>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct AttentionEntry {
    address: Address,
    format: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    version: u32,
    address: Address,
    prefix_tokens: u64,
    token_sha256: [u8; 32],
    state_format: String,
    state_bytes: usize,
    state_sha256: [u8; 32],
    state_pages: usize,
    attention: Vec<AttentionEntry>,
}

pub(crate) fn child_address(base: &Address, suffix: &str) -> Result<Address> {
    let mut address = base.clone();
    address.slot = format!("{}:{suffix}", base.slot);
    address.key()?;
    Ok(address)
}

pub(crate) fn raw_snapshot(format: &str, name: &str, bytes: Vec<u8>) -> PackedSnapshot {
    PackedSnapshot {
        format: format.into(),
        buffers: vec![PackedBuffer {
            name: name.into(),
            dtype: "u8".into(),
            shape: vec![bytes.len()],
            bytes,
        }],
    }
}

pub(crate) fn one_buffer(snapshot: PackedSnapshot, name: &str) -> Result<Vec<u8>> {
    if snapshot.buffers.len() != 1
        || snapshot.buffers[0].name != name
        || snapshot.buffers[0].dtype != "u8"
    {
        return Err(Error::Corrupt(format!("invalid prefill {name} record")));
    }
    Ok(snapshot.buffers.into_iter().next().unwrap().bytes)
}

fn same_identity(base: &Address, part: &Address) -> bool {
    base.namespace == part.namespace
        && base.model_fingerprint == part.model_fingerprint
        && base.prefix_fingerprint == part.prefix_fingerprint
}

pub(crate) fn prepare_prefill_bundle(
    address: &Address,
    bundle: &PrefillBundle,
    attention_addresses: &[Address],
) -> Result<Vec<(Address, PackedSnapshot)>> {
    address.key()?;
    if bundle.token_ids.is_empty()
        || bundle.prefix_tokens != bundle.token_ids.len() as u64
        || bundle.state_format.is_empty()
        || bundle.state_format.len() > 4096
        || bundle.state_bytes.is_empty()
        || bundle.state_bytes.len() > MAX_STATE_BYTES
        || bundle.attention.len() > 4096
        || attention_addresses.len() != bundle.attention.len()
    {
        return Err(Error::Invalid(
            "invalid prefill boundary or state size".into(),
        ));
    }
    for (part, (_, snapshot)) in attention_addresses.iter().zip(&bundle.attention) {
        part.key()?;
        snapshot.validate()?;
        if !same_identity(address, part) {
            return Err(Error::Invalid(
                "attention identity differs from prefill".into(),
            ));
        }
    }
    let token_bytes = bundle
        .token_ids
        .iter()
        .flat_map(|id| id.to_le_bytes())
        .collect::<Vec<_>>();
    let pages = bundle.state_bytes.chunks(STATE_PAGE_BYTES).count();
    let manifest = Manifest {
        version: 1,
        address: address.clone(),
        prefix_tokens: bundle.prefix_tokens,
        token_sha256: Sha256::digest(&token_bytes).into(),
        state_format: bundle.state_format.clone(),
        state_bytes: bundle.state_bytes.len(),
        state_sha256: Sha256::digest(&bundle.state_bytes).into(),
        state_pages: pages,
        attention: attention_addresses
            .iter()
            .zip(&bundle.attention)
            .map(|(part, (_, snapshot))| AttentionEntry {
                address: part.clone(),
                format: snapshot.format.clone(),
            })
            .collect(),
    };
    let mut generated = Vec::with_capacity(pages + 2);
    generated.push((
        child_address(address, "tokens")?,
        raw_snapshot(TOKENS_FORMAT, "tokens", token_bytes),
    ));
    for (index, page) in bundle.state_bytes.chunks(STATE_PAGE_BYTES).enumerate() {
        generated.push((
            child_address(address, &format!("state:{index}"))?,
            raw_snapshot(STATE_PAGE_FORMAT, "state", page.to_vec()),
        ));
    }
    generated.push((
        address.clone(),
        raw_snapshot(MANIFEST_FORMAT, "manifest", serde_json::to_vec(&manifest)?),
    ));
    Ok(generated)
}

pub(crate) fn remap_attention_addresses(address: &Address, bundle: &PrefillBundle) -> Vec<Address> {
    bundle
        .attention
        .iter()
        .map(|(part, _)| Address {
            namespace: address.namespace.clone(),
            model_fingerprint: address.model_fingerprint.clone(),
            prefix_fingerprint: address.prefix_fingerprint.clone(),
            layer: part.layer,
            slot: part.slot.clone(),
        })
        .collect()
}

impl Cache {
    /// Publish tokens, opaque recurrent state and every attention snapshot in
    /// one yesno batch. A failure before commit leaves no visible bundle.
    pub fn put_prefill_bundle(
        &mut self,
        address: &Address,
        bundle: &PrefillBundle,
    ) -> Result<Vec<PackedStorageReport>> {
        if self.get_packed(address, MANIFEST_FORMAT)?.is_some() {
            return Err(Error::Invalid("prefill address already published".into()));
        }
        let addresses = bundle
            .attention
            .iter()
            .map(|(a, _)| a.clone())
            .collect::<Vec<_>>();
        let items = prepare_prefill_bundle(address, bundle, &addresses)?;
        let mut borrowed = items
            .iter()
            .map(|(a, s)| (a.clone(), s))
            .collect::<Vec<_>>();
        borrowed.extend(
            addresses
                .iter()
                .zip(&bundle.attention)
                .map(|(a, (_, s))| (a.clone(), s)),
        );
        self.put_packed_many(&borrowed)
    }

    /// Remove the manifest and every owned record in one durable batch.
    pub fn remove_prefill_bundle(&mut self, address: &Address) -> Result<bool> {
        let view = self.db.snapshot()?;
        let Some(manifest) = load_manifest(&view, address)? else {
            return Ok(false);
        };
        let mut addresses = Vec::with_capacity(manifest.state_pages + manifest.attention.len() + 2);
        addresses.push(address.clone());
        addresses.push(child_address(address, "tokens")?);
        for index in 0..manifest.state_pages {
            addresses.push(child_address(address, &format!("state:{index}"))?);
        }
        for entry in manifest.attention {
            if !same_identity(address, &entry.address) {
                return Err(Error::Corrupt("prefill attention identity mismatch".into()));
            }
            addresses.push(entry.address);
        }
        let mut keys = HashSet::with_capacity(addresses.len());
        for part in &addresses {
            let key = part.key()?;
            if !keys.insert(key) {
                return Err(Error::Corrupt("duplicate prefill record key".into()));
            }
            if stored_address_in_view(&view, key)?.is_some_and(|stored| stored != *part) {
                return Err(Error::Collision);
            }
        }
        drop(view);
        let mut batch = self.db.batch();
        for key in keys {
            batch.delete_key(key).delete_key(key | 1);
        }
        let committed = batch.commit()?;
        self.db
            .wait_visible(committed.version, Duration::from_secs(30))?;
        self.db.checkpoint()?;
        Ok(true)
    }

    pub fn get_prefill_bundle(
        &self,
        address: &Address,
        expected_state_format: &str,
    ) -> Result<Option<PrefillBundle>> {
        get_prefill_bundle(
            &self.db,
            address,
            expected_state_format,
            ReadVerification::Full,
        )
    }

    pub fn get_prefill_bundle_with_verification(
        &self,
        address: &Address,
        expected_state_format: &str,
        verification: ReadVerification,
    ) -> Result<Option<PrefillBundle>> {
        get_prefill_bundle(&self.db, address, expected_state_format, verification)
    }
}

impl CacheReader {
    /// Reads one checkpoint-visible version. Reopen this handle for a newer
    /// checkpoint published after it was opened.
    pub fn get_prefill_bundle(
        &self,
        address: &Address,
        expected_state_format: &str,
    ) -> Result<Option<PrefillBundle>> {
        get_prefill_bundle(
            &self.db,
            address,
            expected_state_format,
            ReadVerification::Full,
        )
    }

    pub fn get_prefill_bundle_with_verification(
        &self,
        address: &Address,
        expected_state_format: &str,
        verification: ReadVerification,
    ) -> Result<Option<PrefillBundle>> {
        get_prefill_bundle(&self.db, address, expected_state_format, verification)
    }
}

fn load_manifest(view: &Snapshot, address: &Address) -> Result<Option<Manifest>> {
    let Some(snapshot) =
        get_packed_in_view(view, address, MANIFEST_FORMAT, ReadVerification::Full)?
    else {
        return Ok(None);
    };
    let manifest: Manifest = serde_json::from_slice(&one_buffer(snapshot, "manifest")?)?;
    validate_manifest(&manifest, address)?;
    Ok(Some(manifest))
}

pub(crate) struct BaseDescription {
    pub(crate) token_ids: Vec<u32>,
    pub(crate) state_format: String,
    pub(crate) attention: Vec<(Address, PackedManifest)>,
}

/// Read the committed base identity and attention digests without transferring
/// its KV payload. Full payload verification still occurs on restore.
pub(crate) fn describe_base_in_view(
    view: &Snapshot,
    address: &Address,
    expected_state_format: &str,
) -> Result<BaseDescription> {
    let manifest = load_manifest(view, address)?
        .ok_or_else(|| Error::Corrupt("append base checkpoint is missing".into()))?;
    if manifest.state_format != expected_state_format {
        return Err(Error::Invalid("prefill state format mismatch".into()));
    }
    let token_bytes = one_buffer(
        get_packed_in_view(
            view,
            &child_address(address, "tokens")?,
            TOKENS_FORMAT,
            ReadVerification::Full,
        )?
        .ok_or_else(|| Error::Corrupt("append base tokens are missing".into()))?,
        "tokens",
    )?;
    if usize::try_from(manifest.prefix_tokens)
        .ok()
        .and_then(|count| count.checked_mul(4))
        != Some(token_bytes.len())
        || Sha256::digest(&token_bytes).as_slice() != manifest.token_sha256
    {
        return Err(Error::Corrupt("append base token checksum mismatch".into()));
    }
    let token_ids = token_bytes
        .chunks_exact(4)
        .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
        .collect();
    let mut seen = HashSet::new();
    let mut attention = Vec::with_capacity(manifest.attention.len());
    for entry in manifest.attention {
        if !same_identity(address, &entry.address) || !seen.insert(entry.address.key()?) {
            return Err(Error::Corrupt(
                "append base attention identity mismatch".into(),
            ));
        }
        let packed = packed_manifest_in_view(view, &entry.address)?
            .ok_or_else(|| Error::Corrupt("append base attention is missing".into()))?;
        if packed.format != entry.format {
            return Err(Error::Corrupt(
                "append base attention format mismatch".into(),
            ));
        }
        attention.push((entry.address, packed));
    }
    Ok(BaseDescription {
        token_ids,
        state_format: manifest.state_format,
        attention,
    })
}

fn validate_manifest(manifest: &Manifest, address: &Address) -> Result<()> {
    if manifest.version != 1
        || manifest.address != *address
        || manifest.prefix_tokens == 0
        || manifest.state_bytes == 0
        || manifest.state_bytes > MAX_STATE_BYTES
        || manifest.state_format.is_empty()
        || manifest.state_format.len() > 4096
        || manifest.state_pages != manifest.state_bytes.div_ceil(STATE_PAGE_BYTES)
        || manifest.attention.len() > 4096
    {
        return Err(Error::Corrupt("invalid prefill manifest".into()));
    }
    Ok(())
}

fn required(
    view: &yesno_core::Snapshot,
    address: &Address,
    format: &str,
    verification: ReadVerification,
) -> Result<PackedSnapshot> {
    get_packed_in_view(view, address, format, verification)?
        .ok_or_else(|| Error::Corrupt("incomplete published prefill bundle".into()))
}

fn get_prefill_bundle(
    db: &Db,
    address: &Address,
    expected_state_format: &str,
    verification: ReadVerification,
) -> Result<Option<PrefillBundle>> {
    let view = db.snapshot()?;
    get_prefill_bundle_in_view(&view, address, expected_state_format, verification)
}

pub(crate) fn get_prefill_bundle_in_view(
    view: &Snapshot,
    address: &Address,
    expected_state_format: &str,
    verification: ReadVerification,
) -> Result<Option<PrefillBundle>> {
    let Some(manifest) = load_manifest(view, address)? else {
        return Ok(None);
    };
    if manifest.state_format != expected_state_format {
        return Err(Error::Invalid("prefill state format mismatch".into()));
    }
    let token_bytes = one_buffer(
        required(
            view,
            &child_address(address, "tokens")?,
            TOKENS_FORMAT,
            verification,
        )?,
        "tokens",
    )?;
    if usize::try_from(manifest.prefix_tokens)
        .ok()
        .and_then(|count| count.checked_mul(4))
        != Some(token_bytes.len())
        || (verification == ReadVerification::Full
            && Sha256::digest(&token_bytes).as_slice() != manifest.token_sha256)
    {
        return Err(Error::Corrupt(
            "prefill token checksum or length mismatch".into(),
        ));
    }
    let token_ids = token_bytes
        .chunks_exact(4)
        .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
        .collect();
    let mut state_bytes = Vec::with_capacity(manifest.state_bytes);
    for index in 0..manifest.state_pages {
        let page = one_buffer(
            required(
                view,
                &child_address(address, &format!("state:{index}"))?,
                STATE_PAGE_FORMAT,
                verification,
            )?,
            "state",
        )?;
        let expected = (manifest.state_bytes - index * STATE_PAGE_BYTES).min(STATE_PAGE_BYTES);
        if page.len() != expected {
            return Err(Error::Corrupt("prefill state page length mismatch".into()));
        }
        state_bytes.extend_from_slice(&page);
    }
    if verification == ReadVerification::Full
        && Sha256::digest(&state_bytes).as_slice() != manifest.state_sha256
    {
        return Err(Error::Corrupt("prefill state checksum mismatch".into()));
    }
    let mut attention = Vec::with_capacity(manifest.attention.len());
    for entry in manifest.attention {
        if !same_identity(address, &entry.address) {
            return Err(Error::Corrupt("prefill attention identity mismatch".into()));
        }
        let snapshot = required(view, &entry.address, &entry.format, verification)?;
        attention.push((entry.address, snapshot));
    }
    Ok(Some(PrefillBundle {
        prefix_tokens: manifest.prefix_tokens,
        token_ids,
        state_format: manifest.state_format,
        state_bytes,
        attention,
    }))
}
