//! Atomic, exact session generations on top of prefill bundles.

use crate::packed::{
    get_packed_in_view, packed_format_in_view, stored_address_in_view, ReadVerification,
};
use crate::prefill::{prepare_prefill_bundle, remap_attention_addresses, PrefillBundle};
use crate::{Address, Cache, CacheReader, Error, PackedBuffer, PackedSnapshot, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use yesno_core::{Db, Snapshot};

pub(crate) mod append;

pub(crate) const HEAD_FORMAT: &str = "shifou-session-head/v1";
const MAX_WORKFLOW_BYTES: usize = 1024 * 1024;

/// The owner supplies a stable, access-controlled session ID. Model identity
/// includes weights, adapters and all execution settings affecting state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SessionKey {
    pub namespace: String,
    pub model_fingerprint: String,
    pub session_id: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RestoredSession {
    pub generation: u64,
    pub checkpoint_address: Address,
    pub bundle: PrefillBundle,
    pub workflow_format: String,
    pub workflow_bytes: Vec<u8>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Head {
    version: u32,
    session_address: Address,
    generation: u64,
    checkpoint_address: Address,
    prefix_tokens: u64,
    workflow_format: String,
    workflow_sha256: [u8; 32],
}

impl SessionKey {
    pub(crate) fn head_address(&self) -> Result<Address> {
        if self.session_id.is_empty() || self.session_id.len() > 4096 {
            return Err(Error::Invalid("invalid session ID".into()));
        }
        let mut h = Sha256::new();
        h.update(b"shifou-session/v1");
        h.update((self.session_id.len() as u64).to_le_bytes());
        h.update(self.session_id.as_bytes());
        let address = Address {
            namespace: self.namespace.clone(),
            model_fingerprint: self.model_fingerprint.clone(),
            prefix_fingerprint: format!("{:x}", h.finalize()),
            layer: 0,
            slot: "session-head".into(),
        };
        address.key()?;
        Ok(address)
    }

    pub(crate) fn checkpoint_address(&self, generation: u64, tokens: &[u32]) -> Result<Address> {
        if generation == 0 || tokens.is_empty() {
            return Err(Error::Invalid(
                "invalid session generation or boundary".into(),
            ));
        }
        let head = self.head_address()?;
        let mut h = Sha256::new();
        h.update(b"shifou-session-checkpoint/v1");
        h.update(head.prefix_fingerprint.as_bytes());
        h.update(generation.to_le_bytes());
        for token in tokens {
            h.update(token.to_le_bytes());
        }
        let address = Address {
            namespace: self.namespace.clone(),
            model_fingerprint: self.model_fingerprint.clone(),
            prefix_fingerprint: format!("{:x}", h.finalize()),
            layer: 0,
            slot: "session-checkpoint".into(),
        };
        address.key()?;
        Ok(address)
    }
}

fn head_snapshot(head: &Head, workflow_bytes: &[u8]) -> Result<PackedSnapshot> {
    let bytes = serde_json::to_vec(head)?;
    Ok(PackedSnapshot {
        format: HEAD_FORMAT.into(),
        buffers: vec![
            PackedBuffer {
                name: "header".into(),
                dtype: "u8".into(),
                shape: vec![bytes.len()],
                bytes,
            },
            PackedBuffer {
                name: "workflow".into(),
                dtype: "u8".into(),
                shape: vec![workflow_bytes.len()],
                bytes: workflow_bytes.to_vec(),
            },
        ],
    })
}

fn read_head(
    view: &Snapshot,
    address: &Address,
    verification: ReadVerification,
) -> Result<Option<(Head, Vec<u8>)>> {
    let Some(snapshot) = get_packed_in_view(view, address, HEAD_FORMAT, verification)? else {
        return Ok(None);
    };
    Ok(Some(parse_head_snapshot(address, snapshot)?))
}

pub(crate) fn parse_head_snapshot(
    address: &Address,
    snapshot: PackedSnapshot,
) -> Result<(Head, Vec<u8>)> {
    let [header, workflow] = snapshot.buffers.as_slice() else {
        return Err(Error::Corrupt("invalid session head buffers".into()));
    };
    if header.name != "header"
        || workflow.name != "workflow"
        || workflow.bytes.len() > MAX_WORKFLOW_BYTES
    {
        return Err(Error::Corrupt("invalid session head layout".into()));
    }
    let head: Head = serde_json::from_slice(&header.bytes)?;
    if head.version != 1
        || head.session_address != *address
        || head.generation == 0
        || head.prefix_tokens == 0
        || head.workflow_format.is_empty()
        || head.workflow_format.len() > 4096
        || Sha256::digest(&workflow.bytes).as_slice() != head.workflow_sha256
    {
        return Err(Error::Corrupt("invalid session head".into()));
    }
    Ok((head, workflow.bytes.clone()))
}

pub(crate) fn restored_session(
    key: &SessionKey,
    head: Head,
    workflow_bytes: Vec<u8>,
    bundle: PrefillBundle,
) -> Result<RestoredSession> {
    if bundle.prefix_tokens != head.prefix_tokens
        || head.checkpoint_address != key.checkpoint_address(head.generation, &bundle.token_ids)?
    {
        return Err(Error::Corrupt(
            "session checkpoint identity mismatch".into(),
        ));
    }
    Ok(RestoredSession {
        generation: head.generation,
        checkpoint_address: head.checkpoint_address,
        bundle,
        workflow_format: head.workflow_format,
        workflow_bytes,
    })
}

impl Head {
    #[cfg(feature = "peer")]
    pub(crate) fn checkpoint_address(&self) -> &Address {
        &self.checkpoint_address
    }

    #[cfg(feature = "peer")]
    pub(crate) fn generation(&self) -> u64 {
        self.generation
    }
}

impl Cache {
    /// Publish one immutable state bundle and move the session head in the
    /// same yesno transaction. Old generations remain available for recovery.
    pub fn put_session_checkpoint(
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
            || workflow_bytes.len() > MAX_WORKFLOW_BYTES
            || bundle.prefix_tokens != bundle.token_ids.len() as u64
        {
            return Err(Error::Invalid(
                "invalid workflow state or token boundary".into(),
            ));
        }
        let session_address = key.head_address()?;
        let checkpoint_address = key.checkpoint_address(generation, &bundle.token_ids)?;
        let view = self.db.snapshot()?;
        if read_head(&view, &session_address, ReadVerification::Full)?
            .is_some_and(|(previous, _)| generation <= previous.generation)
        {
            return Err(Error::Invalid("session generation must increase".into()));
        }
        if stored_address_in_view(&view, checkpoint_address.key()?)?.is_some() {
            return Err(Error::Invalid(
                "session checkpoint already published".into(),
            ));
        }
        drop(view);

        let addresses = remap_attention_addresses(&checkpoint_address, bundle);
        let mut items = prepare_prefill_bundle(&checkpoint_address, bundle, &addresses)?;
        let head = Head {
            version: 1,
            session_address: session_address.clone(),
            generation,
            checkpoint_address: checkpoint_address.clone(),
            prefix_tokens: bundle.prefix_tokens,
            workflow_format: workflow_format.into(),
            workflow_sha256: Sha256::digest(workflow_bytes).into(),
        };
        items.push((session_address, head_snapshot(&head, workflow_bytes)?));
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
        self.put_packed_many(&borrowed)?;
        Ok(checkpoint_address)
    }

    pub fn get_session_checkpoint(
        &self,
        key: &SessionKey,
        state_format: &str,
    ) -> Result<Option<RestoredSession>> {
        get_session(&self.db, key, state_format, ReadVerification::Full)
    }

    pub fn get_session_checkpoint_with_verification(
        &self,
        key: &SessionKey,
        state_format: &str,
        verification: ReadVerification,
    ) -> Result<Option<RestoredSession>> {
        get_session(&self.db, key, state_format, verification)
    }

    /// Remove a superseded generation after the application no longer needs
    /// rollback. The current head is protected from pruning.
    pub fn prune_session_checkpoint(
        &mut self,
        key: &SessionKey,
        generation: u64,
        tokens: &[u32],
    ) -> Result<bool> {
        let address = key.checkpoint_address(generation, tokens)?;
        let view = self.db.snapshot()?;
        if read_head(&view, &key.head_address()?, ReadVerification::Full)?
            .is_some_and(|(head, _)| head.checkpoint_address == address)
        {
            return Err(Error::Invalid(
                "cannot prune the current session generation".into(),
            ));
        }
        if let Some((head, _)) = read_head(&view, &key.head_address()?, ReadVerification::Full)? {
            if packed_format_in_view(&view, &head.checkpoint_address)?.as_deref()
                == Some(append::APPEND_FORMAT)
                && append::read_manifest_in_view(&view, &head.checkpoint_address)?
                    .is_some_and(|manifest| manifest.base_address == address)
            {
                return Err(Error::Invalid(
                    "cannot prune the base of the current session generation".into(),
                ));
            }
        }
        let format = packed_format_in_view(&view, &address)?;
        drop(view);
        if format.as_deref() == Some(append::APPEND_FORMAT) {
            append::remove_append_checkpoint(self, &address)
        } else {
            self.remove_prefill_bundle(&address)
        }
    }
}

impl CacheReader {
    pub fn get_session_checkpoint(
        &self,
        key: &SessionKey,
        state_format: &str,
    ) -> Result<Option<RestoredSession>> {
        get_session(&self.db, key, state_format, ReadVerification::Full)
    }

    pub fn get_session_checkpoint_with_verification(
        &self,
        key: &SessionKey,
        state_format: &str,
        verification: ReadVerification,
    ) -> Result<Option<RestoredSession>> {
        get_session(&self.db, key, state_format, verification)
    }
}

fn get_session(
    db: &Db,
    key: &SessionKey,
    state_format: &str,
    verification: ReadVerification,
) -> Result<Option<RestoredSession>> {
    let view = db.snapshot()?;
    let Some((head, workflow_bytes)) = read_head(&view, &key.head_address()?, verification)? else {
        return Ok(None);
    };
    let bundle = append::load_checkpoint_in_view(
        &view,
        key,
        head.generation,
        &head.checkpoint_address,
        state_format,
        verification,
    )?
    .ok_or_else(|| Error::Corrupt("session head points to a missing checkpoint".into()))?;
    Ok(Some(restored_session(key, head, workflow_bytes, bundle)?))
}
