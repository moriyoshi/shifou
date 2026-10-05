//! Exact causal-prefix addressing and lookup for prepared sessions.

use crate::packed::{get_packed_in_view, ReadVerification};
use crate::prefill::{
    get_prefill_bundle_in_view, prepare_prefill_bundle, remap_attention_addresses, PrefillBundle,
};
use crate::{Address, Cache, CacheReader, Error, PackedBuffer, PackedSnapshot, Result};
use sha2::{Digest, Sha256};
use yesno_core::{Db, Snapshot};

pub(crate) const INDEX_FORMAT: &str = "shifou-prepared-prefix-index/v1";
const BUNDLE_FORMAT: &str = "shifou-prefill-manifest/v1";
const MAX_LENGTHS: usize = 4096;

fn hex(digest: [u8; 32]) -> String {
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// All causal inputs outside the token IDs, including template, multimodal
/// content and tenant sharing scope, belong in `context_fingerprint`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrefixScope {
    pub namespace: String,
    pub model_fingerprint: String,
    pub context_fingerprint: String,
}

/// A metadata-only hit. The caller can compare restore and recompute costs
/// before fetching the potentially large state bundle.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrefixHit {
    pub prefix_tokens: usize,
    pub address: Address,
}

impl PrefixScope {
    fn seed(&self) -> Result<[u8; 32]> {
        if self.context_fingerprint.is_empty() || self.context_fingerprint.len() > 4096 {
            return Err(Error::Invalid("invalid causal context fingerprint".into()));
        }
        let mut h = Sha256::new();
        h.update(b"shifou-prefix-chain/v1");
        for part in [
            self.namespace.as_bytes(),
            self.model_fingerprint.as_bytes(),
            self.context_fingerprint.as_bytes(),
        ] {
            h.update((part.len() as u64).to_le_bytes());
            h.update(part);
        }
        let address = Address {
            namespace: self.namespace.clone(),
            model_fingerprint: self.model_fingerprint.clone(),
            prefix_fingerprint: format!("{:x}", h.clone().finalize()),
            layer: 0,
            slot: "prepared-index".into(),
        };
        address.key()?;
        Ok(h.finalize().into())
    }

    pub(crate) fn index_address(&self) -> Result<Address> {
        Ok(Address {
            namespace: self.namespace.clone(),
            model_fingerprint: self.model_fingerprint.clone(),
            prefix_fingerprint: hex(self.seed()?),
            layer: 0,
            slot: "prepared-index".into(),
        })
    }

    pub fn address_for_tokens(&self, tokens: &[u32]) -> Result<Address> {
        if tokens.is_empty() {
            return Err(Error::Invalid("empty prepared prefix".into()));
        }
        let mut digest = self.seed()?;
        for token in tokens {
            let mut h = Sha256::new();
            h.update(digest);
            h.update(token.to_le_bytes());
            digest = h.finalize().into();
        }
        let address = Address {
            namespace: self.namespace.clone(),
            model_fingerprint: self.model_fingerprint.clone(),
            prefix_fingerprint: hex(digest),
            layer: 0,
            slot: "prepared-manifest".into(),
        };
        address.key()?;
        Ok(address)
    }
}

fn index_snapshot(lengths: &[u32]) -> PackedSnapshot {
    let bytes = lengths
        .iter()
        .flat_map(|n| n.to_le_bytes())
        .collect::<Vec<_>>();
    PackedSnapshot {
        format: INDEX_FORMAT.into(),
        buffers: vec![PackedBuffer {
            name: "lengths".into(),
            dtype: "u8".into(),
            shape: vec![bytes.len()],
            bytes,
        }],
    }
}

fn read_lengths(view: &Snapshot, address: &Address) -> Result<Vec<u32>> {
    let Some(snapshot) = get_packed_in_view(view, address, INDEX_FORMAT, ReadVerification::Full)?
    else {
        return Ok(Vec::new());
    };
    parse_lengths(snapshot)
}

pub(crate) fn parse_lengths(snapshot: PackedSnapshot) -> Result<Vec<u32>> {
    let [buffer] = snapshot.buffers.as_slice() else {
        return Err(Error::Corrupt("invalid prepared-prefix index".into()));
    };
    if buffer.name != "lengths"
        || buffer.bytes.len() % 4 != 0
        || buffer.bytes.len() > MAX_LENGTHS * 4
    {
        return Err(Error::Corrupt("invalid prepared-prefix lengths".into()));
    }
    let lengths = buffer
        .bytes
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect::<Vec<_>>();
    if lengths.first().copied() == Some(0) || lengths.windows(2).any(|w| w[0] >= w[1]) {
        return Err(Error::Corrupt("unordered prepared-prefix lengths".into()));
    }
    Ok(lengths)
}

impl Cache {
    /// Publish a complete prepared prefix and its searchable length together.
    pub fn put_prepared_prefix(
        &mut self,
        scope: &PrefixScope,
        bundle: &PrefillBundle,
    ) -> Result<Address> {
        if bundle.prefix_tokens != bundle.token_ids.len() as u64 {
            return Err(Error::Invalid(
                "prepared-prefix token boundary differs".into(),
            ));
        }
        let address = scope.address_for_tokens(&bundle.token_ids)?;
        let index_address = scope.index_address()?;
        let view = self.db.snapshot()?;
        if get_packed_in_view(&view, &address, BUNDLE_FORMAT, ReadVerification::Full)?.is_some() {
            return Err(Error::Invalid("prepared prefix already published".into()));
        }
        let mut lengths = read_lengths(&view, &index_address)?;
        drop(view);
        let n = u32::try_from(bundle.token_ids.len())
            .map_err(|_| Error::Invalid("prepared prefix is too long".into()))?;
        if let Err(at) = lengths.binary_search(&n) {
            if lengths.len() == MAX_LENGTHS {
                return Err(Error::Invalid("prepared-prefix index is full".into()));
            }
            lengths.insert(at, n);
        }
        let addresses = remap_attention_addresses(&address, bundle);
        let mut items = prepare_prefill_bundle(&address, bundle, &addresses)?;
        items.push((index_address, index_snapshot(&lengths)));
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
        Ok(address)
    }

    pub fn find_longest_prepared_prefix(
        &self,
        scope: &PrefixScope,
        tokens: &[u32],
        state_format: &str,
    ) -> Result<Option<(Address, PrefillBundle)>> {
        find_longest(&self.db, scope, tokens, state_format)
    }

    pub fn find_longest_prepared_address(
        &self,
        scope: &PrefixScope,
        tokens: &[u32],
    ) -> Result<Option<PrefixHit>> {
        find_longest_address(&self.db, scope, tokens)
    }
}

impl CacheReader {
    pub fn find_longest_prepared_prefix(
        &self,
        scope: &PrefixScope,
        tokens: &[u32],
        state_format: &str,
    ) -> Result<Option<(Address, PrefillBundle)>> {
        find_longest(&self.db, scope, tokens, state_format)
    }

    pub fn find_longest_prepared_address(
        &self,
        scope: &PrefixScope,
        tokens: &[u32],
    ) -> Result<Option<PrefixHit>> {
        find_longest_address(&self.db, scope, tokens)
    }
}

fn candidate_addresses(
    view: &Snapshot,
    scope: &PrefixScope,
    tokens: &[u32],
) -> Result<Vec<PrefixHit>> {
    let lengths = read_lengths(view, &scope.index_address()?)?;
    candidate_addresses_for_lengths(scope, tokens, lengths)
}

pub(crate) fn candidate_addresses_for_lengths(
    scope: &PrefixScope,
    tokens: &[u32],
    lengths: Vec<u32>,
) -> Result<Vec<PrefixHit>> {
    let mut candidates = lengths
        .into_iter()
        .filter(|&n| n as usize <= tokens.len())
        .peekable();
    if candidates.peek().is_none() {
        return Ok(Vec::new());
    }
    let mut digest = scope.seed()?;
    let mut matches = Vec::new();
    let mut next = candidates.next().map(|n| n as usize);
    for (i, token) in tokens.iter().enumerate() {
        let mut h = Sha256::new();
        h.update(digest);
        h.update(token.to_le_bytes());
        digest = h.finalize().into();
        if next == Some(i + 1) {
            matches.push(PrefixHit {
                prefix_tokens: i + 1,
                address: Address {
                    namespace: scope.namespace.clone(),
                    model_fingerprint: scope.model_fingerprint.clone(),
                    prefix_fingerprint: hex(digest),
                    layer: 0,
                    slot: "prepared-manifest".into(),
                },
            });
            next = candidates.next().map(|n| n as usize);
            if next.is_none() {
                break;
            }
        }
    }
    Ok(matches)
}

fn find_longest_address(db: &Db, scope: &PrefixScope, tokens: &[u32]) -> Result<Option<PrefixHit>> {
    let view = db.snapshot()?;
    for hit in candidate_addresses(&view, scope, tokens)?.into_iter().rev() {
        if get_packed_in_view(&view, &hit.address, BUNDLE_FORMAT, ReadVerification::Full)?.is_some()
        {
            return Ok(Some(hit));
        }
    }
    Ok(None)
}

fn find_longest(
    db: &Db,
    scope: &PrefixScope,
    tokens: &[u32],
    state_format: &str,
) -> Result<Option<(Address, PrefillBundle)>> {
    let view = db.snapshot()?;
    for hit in candidate_addresses(&view, scope, tokens)?.into_iter().rev() {
        if let Some(bundle) =
            get_prefill_bundle_in_view(&view, &hit.address, state_format, ReadVerification::Full)?
        {
            if bundle.token_ids != tokens[..hit.prefix_tokens] {
                return Err(Error::Corrupt("prepared-prefix token mismatch".into()));
            }
            return Ok(Some((hit.address, bundle)));
        }
    }
    Ok(None)
}
