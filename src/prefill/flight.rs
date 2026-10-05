//! Full-verification prefill reads over yesno-flight.
//! A bundle is assembled at one database version; a reclaimed version restarts
//! the entire read, so pages from different publications cannot be combined.

use super::{
    child_address, one_buffer, same_identity, validate_manifest, Manifest, PrefillBundle,
    MANIFEST_FORMAT, MAX_STATE_BYTES, STATE_PAGE_BYTES, STATE_PAGE_FORMAT, TOKENS_FORMAT,
};
use crate::packed::{read_manifest as read_packed_manifest, Manifest as PackedManifest};
use crate::{Address, Error, PackedBuffer, PackedSnapshot, Result};
use arrow_array::BinaryArray;
use arrow_flight::{decode::FlightRecordBatchStream, error::FlightError};
use futures::{StreamExt, TryStreamExt};
use sha2::{Digest, Sha256};
use tonic::{
    transport::{Channel, Endpoint},
    Code,
};
use yesno_flight::{QueryInfo, SetWire, Ticket, YesnoClient};

const CHUNK_BYTES: usize = 8192;
const MAX_PACKED_BYTES: usize = 64 * 1024 * 1024;
const MAX_METADATA_CHUNKS: usize = MAX_PACKED_BYTES / CHUNK_BYTES + 1;

fn corrupt(message: &str) -> Error {
    Error::Corrupt(message.into())
}

fn sentinel_length(bytes: &[u8]) -> Result<usize> {
    let index = bytes
        .iter()
        .rposition(|&byte| byte != 0)
        .ok_or_else(|| corrupt("missing packed length marker"))?;
    if bytes[index] != 1 {
        return Err(corrupt("invalid packed length marker"));
    }
    Ok(index)
}

async fn collect_bits(mut stream: FlightRecordBatchStream, chunks: usize) -> Result<Vec<u8>> {
    let expected = chunks
        .checked_mul(CHUNK_BYTES)
        .ok_or_else(|| corrupt("bitvector window overflow"))?;
    let mut bytes = Vec::with_capacity(expected);
    while let Some(batch) = stream.next().await {
        let batch = batch?;
        if batch.num_columns() != 1 {
            return Err(corrupt("invalid Flight bitvector schema"));
        }
        let column = batch
            .column(0)
            .as_any()
            .downcast_ref::<BinaryArray>()
            .ok_or_else(|| corrupt("invalid Flight bitvector column"))?;
        for value in column.iter() {
            let chunk = value.ok_or_else(|| corrupt("null Flight bitvector chunk"))?;
            if chunk.len() != CHUNK_BYTES || chunk.len() > expected - bytes.len() {
                return Err(corrupt("invalid Flight bitvector chunk length"));
            }
            bytes.extend_from_slice(chunk);
        }
    }
    if bytes.len() != expected {
        return Err(corrupt("incomplete Flight bitvector"));
    }
    Ok(bytes)
}

async fn bits_at(
    client: &mut YesnoClient,
    version: u64,
    key: u64,
    chunks: usize,
) -> Result<Vec<u8>> {
    let mut ticket = Ticket::whole_key(version, key).with_wire(SetWire::Bitvector);
    ticket.prefix_hi = chunks as u64;
    collect_bits(client.fetch_ticket(ticket.encode()).await?, chunks).await
}

async fn metadata_at(
    client: &mut YesnoClient,
    version: u64,
    address: &Address,
    first_query: Option<&QueryInfo>,
) -> Result<PackedManifest> {
    let key = address.key()? | 1;
    let mut chunks = 1;
    loop {
        let bits = match first_query {
            Some(query) => {
                collect_bits(
                    client.fetch_bitvector(query, 0, chunks as u64).await?,
                    chunks,
                )
                .await?
            }
            None => bits_at(client, version, key, chunks).await?,
        };
        if chunks == 1 && bits.iter().all(|&byte| byte == 0) {
            return Err(corrupt("missing packed metadata"));
        }
        if let Ok(length) = sentinel_length(&bits) {
            if let Ok(manifest) = read_packed_manifest(&bits[..length]) {
                return Ok(manifest);
            }
        }
        if chunks == MAX_METADATA_CHUNKS {
            return Err(corrupt("packed metadata missing, corrupt, or too large"));
        }
        chunks = (chunks * 2).min(MAX_METADATA_CHUNKS);
    }
}

async fn record_at(
    client: &mut YesnoClient,
    version: u64,
    address: &Address,
    expected_format: &str,
    first_query: Option<&QueryInfo>,
) -> Result<PackedSnapshot> {
    let manifest = metadata_at(client, version, address, first_query).await?;
    if manifest.address != *address {
        return Err(Error::Collision);
    }
    if manifest.format != expected_format {
        return Err(Error::Invalid("packed engine format mismatch".into()));
    }
    if manifest.buffers.is_empty() || manifest.buffers.len() > 4096 {
        return Err(corrupt("invalid packed buffer count"));
    }
    let length = manifest
        .buffers
        .iter()
        .try_fold(0usize, |sum, buffer| sum.checked_add(buffer.length))
        .ok_or_else(|| corrupt("packed payload length overflow"))?;
    if length == 0 || length > MAX_PACKED_BYTES {
        return Err(corrupt("packed payload exceeds 64 MiB"));
    }
    let mut bits = bits_at(client, version, address.key()?, length / CHUNK_BYTES + 1).await?;
    if sentinel_length(&bits)? != length {
        return Err(corrupt("packed payload length mismatch"));
    }
    if Sha256::digest(&bits[..length]).as_slice() != manifest.payload_sha256 {
        return Err(corrupt("packed payload checksum mismatch"));
    }
    bits.truncate(length);
    let buffers = if manifest.buffers.len() == 1 {
        let layout = manifest.buffers.into_iter().next().unwrap();
        vec![PackedBuffer {
            name: layout.name,
            dtype: layout.dtype,
            shape: layout.shape,
            bytes: bits,
        }]
    } else {
        let mut offset = 0;
        let mut buffers = Vec::with_capacity(manifest.buffers.len());
        for layout in manifest.buffers {
            let end = offset + layout.length;
            buffers.push(PackedBuffer {
                name: layout.name,
                dtype: layout.dtype,
                shape: layout.shape,
                bytes: bits[offset..end].to_vec(),
            });
            offset = end;
        }
        buffers
    };
    let snapshot = PackedSnapshot {
        format: manifest.format,
        buffers,
    };
    snapshot.validate()?;
    Ok(snapshot)
}

/// A full-verification reader for published prefill bundles over yesno-flight.
/// Each call uses one database version. If checkpointing reclaims that version,
/// it discards all partial pages and retries the complete bundle once.
pub struct FlightBundleReader {
    channel: Channel,
    page_concurrency: usize,
    max_state_bytes: usize,
    max_bundle_bytes: usize,
}

impl FlightBundleReader {
    pub fn new(channel: Channel) -> Self {
        Self {
            channel,
            page_concurrency: 4,
            max_state_bytes: MAX_STATE_BYTES,
            max_bundle_bytes: MAX_STATE_BYTES,
        }
    }

    pub async fn connect(endpoint: &str) -> Result<Self> {
        let channel = Endpoint::new(endpoint.to_owned())?.connect().await?;
        Ok(Self::new(channel))
    }

    pub fn with_page_concurrency(mut self, concurrency: usize) -> Result<Self> {
        if !(1..=16).contains(&concurrency) {
            return Err(Error::Invalid(
                "Flight page concurrency must be 1..=16".into(),
            ));
        }
        self.page_concurrency = concurrency;
        Ok(self)
    }

    pub fn with_max_state_bytes(mut self, limit: usize) -> Result<Self> {
        if limit == 0 || limit > MAX_STATE_BYTES {
            return Err(Error::Invalid("invalid Flight state memory limit".into()));
        }
        self.max_state_bytes = limit;
        Ok(self)
    }

    pub fn with_max_bundle_bytes(mut self, limit: usize) -> Result<Self> {
        if limit == 0 {
            return Err(Error::Invalid("invalid Flight bundle memory limit".into()));
        }
        self.max_bundle_bytes = limit;
        Ok(self)
    }

    /// Return `None` for an absent or colliding top-level address. Every
    /// referenced child record must exist and pass full SHA-256 verification.
    pub async fn get_prefill_bundle(
        &self,
        address: &Address,
        expected_state_format: &str,
    ) -> Result<Option<PrefillBundle>> {
        address.key()?;
        for attempt in 0..2 {
            match self.read_once(address, expected_state_format).await {
                Err(Error::Flight(FlightError::Tonic(status)))
                    if attempt == 0 && status.code() == Code::FailedPrecondition =>
                {
                    continue
                }
                result => return result,
            }
        }
        unreachable!("bounded Flight read loop always returns")
    }

    async fn read_once(
        &self,
        address: &Address,
        expected_state_format: &str,
    ) -> Result<Option<PrefillBundle>> {
        let mut client = YesnoClient::new(self.channel.clone());
        let query = client.prepare_key(address.key()? | 1).await?;
        if query.total_records() == 0 {
            return Ok(None);
        }
        let version = query.version();
        if version == 0 {
            return Err(corrupt("Flight returned an unversioned bundle"));
        }
        let manifest_record =
            match record_at(&mut client, version, address, MANIFEST_FORMAT, Some(&query)).await {
                Err(Error::Collision) => return Ok(None),
                result => result?,
            };
        let manifest_bytes = one_buffer(manifest_record, "manifest")?;
        let manifest: Manifest = serde_json::from_slice(&manifest_bytes)?;
        validate_manifest(&manifest, address)?;
        if manifest.state_format != expected_state_format {
            return Err(Error::Invalid("prefill state format mismatch".into()));
        }
        if manifest.state_bytes > self.max_state_bytes {
            return Err(Error::Invalid(
                "prefill state exceeds Flight memory limit".into(),
            ));
        }
        let token_record = record_at(
            &mut client,
            version,
            &child_address(address, "tokens")?,
            TOKENS_FORMAT,
            None,
        )
        .await?;
        let token_bytes = one_buffer(token_record, "tokens")?;
        if usize::try_from(manifest.prefix_tokens)
            .ok()
            .and_then(|n| n.checked_mul(4))
            != Some(token_bytes.len())
            || Sha256::digest(&token_bytes).as_slice() != manifest.token_sha256
        {
            return Err(corrupt("prefill token checksum or length mismatch"));
        }
        let mut bundle_bytes = manifest
            .state_bytes
            .checked_add(token_bytes.len())
            .ok_or_else(|| corrupt("prefill bundle size overflow"))?;
        if bundle_bytes > self.max_bundle_bytes {
            return Err(Error::Invalid(
                "prefill bundle exceeds Flight memory limit".into(),
            ));
        }
        let token_ids = token_bytes
            .chunks_exact(4)
            .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
            .collect();

        let state_bytes = futures::stream::iter(0..manifest.state_pages)
            .map(|index| {
                let channel = self.channel.clone();
                async move {
                    let mut client = YesnoClient::new(channel);
                    let record = record_at(
                        &mut client,
                        version,
                        &child_address(address, &format!("state:{index}"))?,
                        STATE_PAGE_FORMAT,
                        None,
                    )
                    .await?;
                    Ok::<_, Error>((index, one_buffer(record, "state")?))
                }
            })
            .buffer_unordered(self.page_concurrency)
            .try_fold(
                vec![0u8; manifest.state_bytes],
                |mut state, (index, page)| async move {
                    let start = index * STATE_PAGE_BYTES;
                    let expected = (state.len() - start).min(STATE_PAGE_BYTES);
                    if page.len() != expected {
                        return Err(corrupt("prefill state page length mismatch"));
                    }
                    state[start..start + expected].copy_from_slice(&page);
                    Ok(state)
                },
            )
            .await?;
        if Sha256::digest(&state_bytes).as_slice() != manifest.state_sha256 {
            return Err(corrupt("prefill state checksum mismatch"));
        }

        let mut attention = Vec::with_capacity(manifest.attention.len());
        for entry in manifest.attention {
            if !same_identity(address, &entry.address) {
                return Err(corrupt("prefill attention identity mismatch"));
            }
            let snapshot =
                record_at(&mut client, version, &entry.address, &entry.format, None).await?;
            let attention_bytes = snapshot
                .buffers
                .iter()
                .try_fold(0usize, |sum, buffer| sum.checked_add(buffer.bytes.len()))
                .ok_or_else(|| corrupt("prefill attention size overflow"))?;
            bundle_bytes = bundle_bytes
                .checked_add(attention_bytes)
                .ok_or_else(|| corrupt("prefill bundle size overflow"))?;
            if bundle_bytes > self.max_bundle_bytes {
                return Err(Error::Invalid(
                    "prefill bundle exceeds Flight memory limit".into(),
                ));
            }
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
}
