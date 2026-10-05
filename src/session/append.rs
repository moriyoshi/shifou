//! One full base checkpoint plus a cumulative append-only attention tail.
//! Every delta remains independently readable after intermediate generations
//! are pruned. A new full checkpoint resets the base before tails grow large.

use super::{head_snapshot, read_head, Head, SessionKey};
use crate::packed::{
    get_packed_in_view, packed_format_in_view, stored_address_in_view, ReadVerification,
};
use crate::prefill::{
    child_address, describe_base_in_view, get_prefill_bundle_in_view, one_buffer, raw_snapshot,
    BaseDescription, PrefillBundle, MANIFEST_FORMAT, MAX_STATE_BYTES, STATE_PAGE_BYTES,
};
use crate::{Address, Cache, Error, PackedSnapshot, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::time::Duration;
use yesno_core::Snapshot;

pub(crate) const APPEND_FORMAT: &str = "shifou-session-append/v1";
pub(crate) const STATE_FORMAT: &str = "shifou-session-append-state/v1";
pub(crate) const TAIL_FORMAT: &str = "shifou-session-append-tails/v1";
const MAX_TAIL_BYTES: usize = 64 * 1024 * 1024;

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct BufferPlan {
    pub(crate) name: String,
    pub(crate) dtype: String,
    pub(crate) shape: Vec<usize>,
    pub(crate) base_len: usize,
    pub(crate) tail_len: usize,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct AttentionPlan {
    pub(crate) layer: u32,
    pub(crate) slot: String,
    pub(crate) format: String,
    pub(crate) buffers: Vec<BufferPlan>,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct AppendManifest {
    pub(crate) version: u32,
    pub(crate) address: Address,
    pub(crate) base_address: Address,
    pub(crate) base_generation: u64,
    pub(crate) token_ids: Vec<u32>,
    pub(crate) state_format: String,
    pub(crate) state_bytes: usize,
    pub(crate) state_sha256: [u8; 32],
    pub(crate) state_pages: usize,
    pub(crate) tail_bytes: usize,
    pub(crate) tail_sha256: [u8; 32],
    pub(crate) attention: Vec<AttentionPlan>,
}

pub(crate) fn state_address(address: &Address, index: usize) -> Result<Address> {
    child_address(address, &format!("append-state:{index}"))
}

pub(crate) fn tails_address(address: &Address) -> Result<Address> {
    child_address(address, "append-tails")
}

pub(crate) fn parse_manifest(
    address: &Address,
    snapshot: PackedSnapshot,
) -> Result<AppendManifest> {
    let bytes = one_buffer(snapshot, "manifest")?;
    let manifest: AppendManifest = serde_json::from_slice(&bytes)?;
    validate_manifest(&manifest, address)?;
    Ok(manifest)
}

pub(crate) fn validate_manifest(manifest: &AppendManifest, address: &Address) -> Result<()> {
    if manifest.version != 1
        || manifest.address != *address
        || manifest.base_generation == 0
        || manifest.base_address == *address
        || manifest.base_address.namespace != address.namespace
        || manifest.base_address.model_fingerprint != address.model_fingerprint
        || manifest.token_ids.is_empty()
        || manifest.state_format.is_empty()
        || manifest.state_format.len() > 4096
        || manifest.state_bytes == 0
        || manifest.state_bytes > MAX_STATE_BYTES
        || manifest.state_pages != manifest.state_bytes.div_ceil(STATE_PAGE_BYTES)
        || manifest.tail_bytes > MAX_TAIL_BYTES
        || manifest.attention.len() > 4096
    {
        return Err(Error::Corrupt("invalid append checkpoint manifest".into()));
    }
    let mut tail_total = 0usize;
    for entry in &manifest.attention {
        if entry.slot.is_empty()
            || entry.slot.len() > 4096
            || entry.buffers.is_empty()
            || entry.buffers.len() > 4096
            || entry.format.is_empty()
            || entry.format.len() > 4096
        {
            return Err(Error::Corrupt("invalid append attention layout".into()));
        }
        for buffer in &entry.buffers {
            if buffer.name.is_empty()
                || buffer.name.len() > 256
                || !matches!(buffer.dtype.as_str(), "u8" | "bf16-le" | "f32-le")
                || buffer.base_len == 0
                || buffer.base_len.checked_add(buffer.tail_len).is_none()
            {
                return Err(Error::Corrupt("invalid append buffer layout".into()));
            }
            tail_total = tail_total
                .checked_add(buffer.tail_len)
                .ok_or_else(|| Error::Corrupt("append tail length overflow".into()))?;
        }
    }
    if tail_total != manifest.tail_bytes {
        return Err(Error::Corrupt("append tail length mismatch".into()));
    }
    Ok(())
}

pub(crate) fn read_manifest_in_view(
    view: &Snapshot,
    address: &Address,
) -> Result<Option<AppendManifest>> {
    get_packed_in_view(view, address, APPEND_FORMAT, ReadVerification::Full)?
        .map(|snapshot| parse_manifest(address, snapshot))
        .transpose()
}

fn required(
    view: &Snapshot,
    address: &Address,
    format: &str,
    verification: ReadVerification,
) -> Result<PackedSnapshot> {
    get_packed_in_view(view, address, format, verification)?
        .ok_or_else(|| Error::Corrupt("missing append checkpoint record".into()))
}

pub(crate) fn load_checkpoint_in_view(
    view: &Snapshot,
    key: &SessionKey,
    generation: u64,
    address: &Address,
    expected_state_format: &str,
    verification: ReadVerification,
) -> Result<Option<PrefillBundle>> {
    match packed_format_in_view(view, address)?.as_deref() {
        None => Ok(None),
        Some(MANIFEST_FORMAT) => {
            get_prefill_bundle_in_view(view, address, expected_state_format, verification)
        }
        Some(APPEND_FORMAT) => {
            let snapshot = required(view, address, APPEND_FORMAT, verification)?;
            let manifest = parse_manifest(address, snapshot)?;
            if manifest.state_format != expected_state_format {
                return Err(Error::Invalid("prefill state format mismatch".into()));
            }
            let base = get_prefill_bundle_in_view(
                view,
                &manifest.base_address,
                expected_state_format,
                verification,
            )?
            .ok_or_else(|| Error::Corrupt("append base checkpoint is missing".into()))?;
            let mut state = Vec::with_capacity(manifest.state_bytes);
            for index in 0..manifest.state_pages {
                let page = one_buffer(
                    required(
                        view,
                        &state_address(address, index)?,
                        STATE_FORMAT,
                        verification,
                    )?,
                    "state",
                )?;
                let expected =
                    (manifest.state_bytes - index * STATE_PAGE_BYTES).min(STATE_PAGE_BYTES);
                if page.len() != expected {
                    return Err(Error::Corrupt("append state page length mismatch".into()));
                }
                state.extend_from_slice(&page);
            }
            if verification == ReadVerification::Full
                && Sha256::digest(&state).as_slice() != manifest.state_sha256
            {
                return Err(Error::Corrupt("append state checksum mismatch".into()));
            }
            let tails = if manifest.tail_bytes == 0 {
                Vec::new()
            } else {
                one_buffer(
                    required(view, &tails_address(address)?, TAIL_FORMAT, verification)?,
                    "tails",
                )?
            };
            Ok(Some(apply_append(
                key,
                generation,
                address,
                manifest,
                base,
                state,
                tails,
                verification,
            )?))
        }
        Some(_) => Err(Error::Invalid(
            "unsupported session checkpoint format".into(),
        )),
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_append(
    key: &SessionKey,
    generation: u64,
    address: &Address,
    manifest: AppendManifest,
    mut base: PrefillBundle,
    state: Vec<u8>,
    tails: Vec<u8>,
    verification: ReadVerification,
) -> Result<PrefillBundle> {
    validate_manifest(&manifest, address)?;
    if generation <= manifest.base_generation
        || key.checkpoint_address(generation, &manifest.token_ids)? != *address
        || key.checkpoint_address(manifest.base_generation, &base.token_ids)?
            != manifest.base_address
        || !manifest.token_ids.starts_with(&base.token_ids)
        || manifest.token_ids.len() <= base.token_ids.len()
        || base.state_format != manifest.state_format
        || base.attention.len() != manifest.attention.len()
        || state.len() != manifest.state_bytes
        || tails.len() != manifest.tail_bytes
    {
        return Err(Error::Corrupt("append checkpoint identity mismatch".into()));
    }
    if verification == ReadVerification::Full {
        if Sha256::digest(&state).as_slice() != manifest.state_sha256 {
            return Err(Error::Corrupt("append state checksum mismatch".into()));
        }
        if Sha256::digest(&tails).as_slice() != manifest.tail_sha256 {
            return Err(Error::Corrupt("append tail checksum mismatch".into()));
        }
    }
    let mut offset: usize = 0;
    for ((part, snapshot), entry) in base.attention.iter_mut().zip(&manifest.attention) {
        if part.layer != entry.layer
            || part.slot != entry.slot
            || snapshot.format != entry.format
            || snapshot.buffers.len() != entry.buffers.len()
        {
            return Err(Error::Corrupt("append attention identity mismatch".into()));
        }
        for (buffer, plan) in snapshot.buffers.iter_mut().zip(&entry.buffers) {
            if buffer.name != plan.name
                || buffer.dtype != plan.dtype
                || buffer.bytes.len() != plan.base_len
            {
                return Err(Error::Corrupt("append buffer base mismatch".into()));
            }
            let end = offset
                .checked_add(plan.tail_len)
                .filter(|&end| end <= tails.len())
                .ok_or_else(|| Error::Corrupt("append tail offset overflow".into()))?;
            buffer.bytes.extend_from_slice(&tails[offset..end]);
            buffer.shape = plan.shape.clone();
            offset = end;
        }
        snapshot.validate()?;
        part.namespace = address.namespace.clone();
        part.model_fingerprint = address.model_fingerprint.clone();
        part.prefix_fingerprint = address.prefix_fingerprint.clone();
        part.key()?;
    }
    if offset != tails.len() {
        return Err(Error::Corrupt("unclaimed append tail bytes".into()));
    }
    Ok(PrefillBundle {
        prefix_tokens: manifest.token_ids.len() as u64,
        token_ids: manifest.token_ids,
        state_format: manifest.state_format,
        state_bytes: state,
        attention: base.attention,
    })
}

fn make_manifest(
    key: &SessionKey,
    generation: u64,
    address: &Address,
    base_address: Address,
    base_generation: u64,
    base: &BaseDescription,
    bundle: &PrefillBundle,
) -> Result<(AppendManifest, Vec<u8>)> {
    if generation <= base_generation
        || !bundle.token_ids.starts_with(&base.token_ids)
        || bundle.token_ids.len() <= base.token_ids.len()
        || bundle.state_format != base.state_format
        || bundle.attention.len() != base.attention.len()
        || bundle.state_bytes.is_empty()
        || bundle.state_bytes.len() > MAX_STATE_BYTES
        || key.checkpoint_address(base_generation, &base.token_ids)? != base_address
    {
        return Err(Error::Invalid(
            "session state cannot append to its base".into(),
        ));
    }
    let mut tails = Vec::new();
    let mut attention = Vec::with_capacity(bundle.attention.len());
    for ((old_address, old), (new_address, new)) in base.attention.iter().zip(&bundle.attention) {
        new.validate()?;
        if old_address.layer != new_address.layer
            || old_address.slot != new_address.slot
            || old.format != new.format
            || old.buffers.len() != new.buffers.len()
        {
            return Err(Error::Invalid("attention identity cannot append".into()));
        }
        let mut buffers = Vec::with_capacity(new.buffers.len());
        let mut base_hash = Sha256::new();
        for (before, after) in old.buffers.iter().zip(&new.buffers) {
            if before.name != after.name
                || before.dtype != after.dtype
                || after.bytes.len() < before.length
            {
                return Err(Error::Invalid("attention layout cannot append".into()));
            }
            base_hash.update(&after.bytes[..before.length]);
            let extra = &after.bytes[before.length..];
            if tails
                .len()
                .checked_add(extra.len())
                .is_none_or(|len| len > MAX_TAIL_BYTES)
            {
                return Err(Error::Invalid(
                    "append tail exceeds 64 MiB; publish a full checkpoint".into(),
                ));
            }
            tails.extend_from_slice(extra);
            buffers.push(BufferPlan {
                name: after.name.clone(),
                dtype: after.dtype.clone(),
                shape: after.shape.clone(),
                base_len: before.length,
                tail_len: extra.len(),
            });
        }
        if base_hash.finalize().as_slice() != old.payload_sha256 {
            return Err(Error::Invalid("attention buffer is not append-only".into()));
        }
        attention.push(AttentionPlan {
            layer: new_address.layer,
            slot: new_address.slot.clone(),
            format: new.format.clone(),
            buffers,
        });
    }
    let manifest = AppendManifest {
        version: 1,
        address: address.clone(),
        base_address,
        base_generation,
        token_ids: bundle.token_ids.clone(),
        state_format: bundle.state_format.clone(),
        state_bytes: bundle.state_bytes.len(),
        state_sha256: Sha256::digest(&bundle.state_bytes).into(),
        state_pages: bundle.state_bytes.len().div_ceil(STATE_PAGE_BYTES),
        tail_bytes: tails.len(),
        tail_sha256: Sha256::digest(&tails).into(),
        attention,
    };
    validate_manifest(&manifest, address)?;
    Ok((manifest, tails))
}

impl Cache {
    /// Store a cumulative append-only attention tail against the last full
    /// checkpoint. All records and the session head move in one yesno batch.
    /// If state is not append-only or the tail exceeds 64 MiB, publish a full
    /// checkpoint with `put_session_checkpoint` instead.
    pub fn put_session_append_checkpoint(
        &mut self,
        key: &SessionKey,
        generation: u64,
        bundle: &PrefillBundle,
        workflow_format: &str,
        workflow_bytes: &[u8],
    ) -> Result<Address> {
        if workflow_format.is_empty()
            || workflow_format.len() > 4096
            || workflow_bytes.is_empty()
            || workflow_bytes.len() > super::MAX_WORKFLOW_BYTES
            || bundle.prefix_tokens != bundle.token_ids.len() as u64
        {
            return Err(Error::Invalid(
                "invalid workflow state or token boundary".into(),
            ));
        }
        let session_address = key.head_address()?;
        let address = key.checkpoint_address(generation, &bundle.token_ids)?;
        let view = self.db.snapshot()?;
        let (previous, _) = read_head(&view, &session_address, ReadVerification::Full)?
            .ok_or_else(|| Error::Invalid("append checkpoint needs a full base".into()))?;
        if generation <= previous.generation {
            return Err(Error::Invalid("session generation must increase".into()));
        }
        if stored_address_in_view(&view, address.key()?)?.is_some() {
            return Err(Error::Invalid(
                "session checkpoint already published".into(),
            ));
        }
        let (base_address, base_generation, current_tokens, current_base) =
            match packed_format_in_view(&view, &previous.checkpoint_address)?.as_deref() {
                Some(MANIFEST_FORMAT) => {
                    let full = describe_base_in_view(
                        &view,
                        &previous.checkpoint_address,
                        &bundle.state_format,
                    )?;
                    (
                        previous.checkpoint_address.clone(),
                        previous.generation,
                        full.token_ids.clone(),
                        Some(full),
                    )
                }
                Some(APPEND_FORMAT) => {
                    let delta = read_manifest_in_view(&view, &previous.checkpoint_address)?
                        .ok_or_else(|| {
                            Error::Corrupt("current append manifest is missing".into())
                        })?;
                    (
                        delta.base_address,
                        delta.base_generation,
                        delta.token_ids,
                        None,
                    )
                }
                _ => {
                    return Err(Error::Invalid(
                        "unsupported current checkpoint format".into(),
                    ))
                }
            };
        if current_tokens.len() as u64 != previous.prefix_tokens
            || bundle.token_ids.len() <= current_tokens.len()
            || !bundle.token_ids.starts_with(&current_tokens)
        {
            return Err(Error::Invalid(
                "session tokens do not extend the current generation".into(),
            ));
        }
        let base = match current_base {
            Some(base) => base,
            None => describe_base_in_view(&view, &base_address, &bundle.state_format)?,
        };
        let (manifest, tails) = make_manifest(
            key,
            generation,
            &address,
            base_address,
            base_generation,
            &base,
            bundle,
        )?;
        drop(view);
        let mut items = Vec::with_capacity(manifest.state_pages + 3);
        for (index, page) in bundle.state_bytes.chunks(STATE_PAGE_BYTES).enumerate() {
            items.push((
                state_address(&address, index)?,
                raw_snapshot(STATE_FORMAT, "state", page.to_vec()),
            ));
        }
        if !tails.is_empty() {
            items.push((
                tails_address(&address)?,
                raw_snapshot(TAIL_FORMAT, "tails", tails),
            ));
        }
        items.push((
            address.clone(),
            raw_snapshot(APPEND_FORMAT, "manifest", serde_json::to_vec(&manifest)?),
        ));
        let head = Head {
            version: 1,
            session_address: session_address.clone(),
            generation,
            checkpoint_address: address.clone(),
            prefix_tokens: bundle.prefix_tokens,
            workflow_format: workflow_format.into(),
            workflow_sha256: Sha256::digest(workflow_bytes).into(),
        };
        items.push((session_address, head_snapshot(&head, workflow_bytes)?));
        let borrowed = items
            .iter()
            .map(|(a, s)| (a.clone(), s))
            .collect::<Vec<_>>();
        self.put_packed_many(&borrowed)?;
        Ok(address)
    }
}

pub(crate) fn remove_append_checkpoint(cache: &mut Cache, address: &Address) -> Result<bool> {
    let view = cache.db.snapshot()?;
    let Some(manifest) = read_manifest_in_view(&view, address)? else {
        return Ok(false);
    };
    let mut addresses = Vec::with_capacity(manifest.state_pages + 2);
    addresses.push(address.clone());
    for index in 0..manifest.state_pages {
        addresses.push(state_address(address, index)?);
    }
    if manifest.tail_bytes > 0 {
        addresses.push(tails_address(address)?);
    }
    let mut keys = std::collections::HashSet::new();
    for part in &addresses {
        let key = part.key()?;
        if !keys.insert(key) || stored_address_in_view(&view, key)?.as_ref() != Some(part) {
            return Err(Error::Corrupt(
                "append checkpoint record identity mismatch".into(),
            ));
        }
    }
    drop(view);
    let mut batch = cache.db.batch();
    for key in keys {
        batch.delete_key(key).delete_key(key | 1);
    }
    let committed = batch.commit()?;
    cache
        .db
        .wait_visible(committed.version, Duration::from_secs(30))?;
    cache.db.checkpoint()?;
    Ok(true)
}
