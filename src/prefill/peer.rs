//! Full-verification prefill reads through yesno's Unix peer socket.
//! One server-owned snapshot covers every packed record in a bundle. Dense
//! state lanes are decoded directly into the final state allocation.

use super::{
    child_address, one_buffer, same_identity, validate_manifest, Manifest, PrefillBundle,
    MANIFEST_FORMAT, MAX_STATE_BYTES, STATE_PAGE_BYTES, STATE_PAGE_FORMAT, TOKENS_FORMAT,
};
use crate::packed::{read_manifest as read_packed_manifest, Manifest as PackedManifest};
use crate::prefix::{
    candidate_addresses_for_lengths, parse_lengths, PrefixHit, PrefixScope, INDEX_FORMAT,
};
use crate::session::{
    append::{self, AppendManifest},
    parse_head_snapshot, restored_session, RestoredSession, SessionKey, HEAD_FORMAT,
};
use crate::{Address, Error, PackedBuffer, PackedSnapshot, Result};
use memmap2::Mmap;
use rayon::prelude::*;
use sha2::{Digest, Sha256};
use std::{
    collections::{HashMap, HashSet},
    io::Write,
    os::{fd::AsRawFd, unix::net::UnixStream},
    path::Path,
};
use yesno_plugin::{
    channel::{read_frame, recv_fd},
    ipc::{Frame, Lane, LaneKind, LANE_BYTES, MAGIC, VERSION},
};

const CHUNK_BYTES: usize = 8192;
const MAX_PACKED_BYTES: usize = 64 * 1024 * 1024;

fn corrupt(message: &str) -> Error {
    Error::Corrupt(message.into())
}

#[derive(Clone, Copy)]
struct ScanSpec {
    key: u64,
    max_len: usize,
}

#[derive(Clone, Copy)]
struct PageTarget {
    start: usize,
    len: usize,
}

struct DirectState<'a> {
    bytes: &'a mut [u8],
    pages: &'a HashMap<u64, PageTarget>,
}

struct Peer {
    stream: UnixStream,
    arena: Option<Mmap>,
    input: Vec<u8>,
    max_lanes: usize,
    max_blocks: u32,
    parallel_verification: bool,
    pipelined_verification: bool,
}

impl Peer {
    fn connect(path: &Path) -> Result<Self> {
        let stream = UnixStream::connect(path)?;
        let first = loop {
            let mut byte = 0u8;
            // SAFETY: `byte` is one writable byte, `stream` is a live socket,
            // and MSG_PEEK leaves the descriptor-handoff byte available to recv_fd.
            let received = unsafe {
                libc::recv(
                    stream.as_raw_fd(),
                    (&mut byte as *mut u8).cast(),
                    1,
                    libc::MSG_PEEK,
                )
            };
            if received == 1 {
                break byte;
            }
            if received == 0 {
                return Err(Error::Peer("socket closed before greeting".into()));
            }
            let error = std::io::Error::last_os_error();
            if error.kind() != std::io::ErrorKind::Interrupted {
                return Err(error.into());
            }
        };
        let arena = if first == VERSION {
            let (fd, version) = recv_fd(&stream)?;
            if version != VERSION {
                return Err(Error::Peer("peer protocol version mismatch".into()));
            }
            let file = std::fs::File::from(fd);
            // SAFETY: yesnod seals the handed-off arena against shrinking before
            // sending its descriptor, so this mapping cannot be truncated below us.
            Some(unsafe { Mmap::map(&file) }?)
        } else if first == MAGIC[0] {
            None
        } else {
            return Err(Error::Peer("unrecognized peer greeting".into()));
        };
        let mut peer = Self {
            stream,
            arena,
            input: Vec::new(),
            max_lanes: 0,
            max_blocks: 0,
            parallel_verification: false,
            pipelined_verification: false,
        };
        loop {
            match peer.recv()? {
                Frame::ServerHello {
                    protocol,
                    arena_bytes,
                    max_lanes,
                    max_blocks,
                    ..
                } if protocol == VERSION as u32
                    && peer.arena.as_ref().map_or(0, |arena| arena.len())
                        == arena_bytes as usize
                    && max_lanes > 0
                    && max_blocks > 0 =>
                {
                    peer.max_lanes = max_lanes as usize;
                    peer.max_blocks = max_blocks;
                    break;
                }
                Frame::Available { .. } | Frame::RoleChanged { .. } => {}
                other => return Err(Error::Peer(format!("invalid peer greeting: {other:?}"))),
            }
        }
        match peer.ask(Frame::ClientHello {
            protocol: VERSION as u32,
            name: "shifou-prefill".into(),
        })? {
            Frame::Done => Ok(peer),
            other => Err(Error::Peer(format!("peer hello refused: {other:?}"))),
        }
    }

    fn recv(&mut self) -> Result<Frame> {
        read_frame(&mut self.stream, &mut self.input)?
            .ok_or_else(|| Error::Peer("peer socket closed".into()))
    }

    fn ask(&mut self, frame: Frame) -> Result<Frame> {
        self.stream
            .write_all(&frame.encode().map_err(|e| Error::Peer(e.to_string()))?)?;
        loop {
            match self.recv()? {
                Frame::Available { .. } | Frame::RoleChanged { .. } => {}
                Frame::Unavailable | Frame::GenerationChanged { .. } => {
                    return Err(Error::Peer(
                        "database became unavailable during read".into(),
                    ));
                }
                Frame::Fault { status, message } => {
                    return Err(Error::Peer(format!("server status {status}: {message}")));
                }
                response => return Ok(response),
            }
        }
    }

    fn open_snapshot(&mut self) -> Result<u64> {
        match self.ask(Frame::SnapshotOpen)? {
            Frame::SnapshotOpened { snapshot, .. } => Ok(snapshot),
            other => Err(Error::Peer(format!("expected snapshot: {other:?}"))),
        }
    }

    fn scan(
        &mut self,
        snapshot: u64,
        specs: &[ScanSpec],
        max_staging_bytes: usize,
        mut direct: Option<DirectState<'_>>,
    ) -> Result<HashMap<u64, Vec<u8>>> {
        if specs.is_empty() || specs.len() > self.max_lanes {
            return Err(Error::Invalid(
                "peer lane count exceeds server limit".into(),
            ));
        }
        let (lanes, arena_off) = match self.ask(Frame::LanesAcquire {
            snapshot,
            keys: specs.iter().map(|spec| spec.key).collect(),
        })? {
            Frame::LanesAcquired { lanes, arena_off } => (lanes, arena_off),
            other => return Err(Error::Peer(format!("lane acquire refused: {other:?}"))),
        };
        let mut images = vec![Vec::<u8>::new(); specs.len()];
        let mut allocated = 0usize;
        let mut sentinels = HashSet::new();
        let block_stride = self
            .max_lanes
            .checked_mul(LANE_BYTES)
            .ok_or_else(|| corrupt("peer arena stride overflow"))?;
        loop {
            let response = self.ask(Frame::BlockAdvanceMany {
                lanes,
                max_blocks: self.max_blocks,
            })?;
            let count = match response {
                Frame::Blocks { blocks } => {
                    let arena = self
                        .arena
                        .as_ref()
                        .ok_or_else(|| corrupt("arena blocks without an arena"))?;
                    for (block_index, block) in blocks.iter().enumerate() {
                        if block.lanes.len() != specs.len() {
                            return Err(corrupt("peer block lane count mismatch"));
                        }
                        for (lane_index, lane) in block.lanes.iter().enumerate() {
                            let start = usize::try_from(arena_off)
                                .ok()
                                .and_then(|base| {
                                    block_index
                                        .checked_mul(block_stride)
                                        .and_then(|offset| base.checked_add(offset))
                                })
                                .and_then(|base| {
                                    lane_index
                                        .checked_mul(LANE_BYTES)
                                        .and_then(|offset| base.checked_add(offset))
                                })
                                .ok_or_else(|| corrupt("peer arena offset overflow"))?;
                            let end = start
                                .checked_add(lane_bytes(lane)?)
                                .ok_or_else(|| corrupt("peer arena offset overflow"))?;
                            let payload = arena
                                .get(start..end)
                                .ok_or_else(|| corrupt("peer lane outside arena"))?;
                            consume_lane(
                                &specs[lane_index],
                                block.prefix,
                                lane,
                                payload,
                                &mut images[lane_index],
                                &mut allocated,
                                max_staging_bytes,
                                &mut direct,
                                &mut sentinels,
                            )?;
                        }
                    }
                    blocks.len()
                }
                Frame::BlocksInline { blocks, payload } => {
                    if self.arena.is_some() {
                        return Err(corrupt("inline blocks from arena peer"));
                    }
                    let mut offset = 0usize;
                    for block in &blocks {
                        if block.lanes.len() != specs.len() {
                            return Err(corrupt("peer block lane count mismatch"));
                        }
                        for (lane_index, lane) in block.lanes.iter().enumerate() {
                            let end = offset
                                .checked_add(lane_bytes(lane)?)
                                .ok_or_else(|| corrupt("peer inline offset overflow"))?;
                            let bytes = payload
                                .get(offset..end)
                                .ok_or_else(|| corrupt("short peer inline payload"))?;
                            consume_lane(
                                &specs[lane_index],
                                block.prefix,
                                lane,
                                bytes,
                                &mut images[lane_index],
                                &mut allocated,
                                max_staging_bytes,
                                &mut direct,
                                &mut sentinels,
                            )?;
                            offset = end;
                        }
                    }
                    if offset != payload.len() {
                        return Err(corrupt("extra peer inline payload"));
                    }
                    blocks.len()
                }
                other => return Err(Error::Peer(format!("block advance refused: {other:?}"))),
            };
            if count == 0 {
                break;
            }
        }
        if !matches!(self.ask(Frame::LanesRelease { lanes })?, Frame::Done) {
            return Err(corrupt("peer lane release failed"));
        }
        if let Some(state) = &direct {
            if sentinels.len() != state.pages.len() {
                return Err(corrupt("missing state page sentinel"));
            }
        }
        Ok(specs
            .iter()
            .zip(images)
            .filter(|(spec, _)| {
                !direct
                    .as_ref()
                    .is_some_and(|state| state.pages.contains_key(&spec.key))
            })
            .map(|(spec, bytes)| (spec.key, bytes))
            .collect())
    }
}

fn lane_bytes(lane: &Lane) -> Result<usize> {
    let valid = match lane.kind {
        LaneKind::Absent => lane.count == 0,
        LaneKind::Bitmap => lane.count == 1024,
        LaneKind::Array => lane.count <= 4096,
        LaneKind::Run => lane.count <= 2032,
    };
    if !valid {
        return Err(corrupt("invalid peer lane count"));
    }
    Ok(lane.kind.payload_bytes(lane.count))
}

fn decode_lane(lane: &Lane, payload: &[u8], output: &mut [u8]) -> Result<()> {
    if payload.len() != lane_bytes(lane)? || output.len() != CHUNK_BYTES {
        return Err(corrupt("invalid peer lane length"));
    }
    match lane.kind {
        LaneKind::Absent => {}
        LaneKind::Bitmap => output.copy_from_slice(payload),
        LaneKind::Array => {
            for word in payload.chunks_exact(2) {
                let bit = u16::from_le_bytes([word[0], word[1]]) as usize;
                output[bit / 8] |= 1 << (bit % 8);
            }
        }
        LaneKind::Run => {
            for pair in payload.chunks_exact(4) {
                let start = u16::from_le_bytes([pair[0], pair[1]]) as usize;
                let end = u16::from_le_bytes([pair[2], pair[3]]) as usize;
                if start > end {
                    return Err(corrupt("invalid peer run interval"));
                }
                for bit in start..=end {
                    output[bit / 8] |= 1 << (bit % 8);
                }
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn consume_lane(
    spec: &ScanSpec,
    prefix: u64,
    lane: &Lane,
    payload: &[u8],
    image: &mut Vec<u8>,
    allocated: &mut usize,
    max_staging_bytes: usize,
    direct: &mut Option<DirectState<'_>>,
    sentinels: &mut HashSet<u64>,
) -> Result<()> {
    if lane.kind == LaneKind::Absent {
        return Ok(());
    }
    let prefix = usize::try_from(prefix).map_err(|_| corrupt("peer prefix overflow"))?;
    if prefix > spec.max_len / CHUNK_BYTES {
        return Err(corrupt("peer bitvector exceeds expected length"));
    }
    if let Some(state) = direct.as_mut() {
        if let Some(target) = state.pages.get(&spec.key) {
            let marker_chunk = target.len / CHUNK_BYTES;
            if prefix > marker_chunk {
                return Err(corrupt("state bits after sentinel"));
            }
            let offset = target.start + prefix * CHUNK_BYTES;
            if prefix == marker_chunk {
                let marker_byte = target.len % CHUNK_BYTES;
                let mut chunk = [0u8; CHUNK_BYTES];
                decode_lane(lane, payload, &mut chunk)?;
                if chunk[marker_byte] != 1 || chunk[marker_byte + 1..].iter().any(|&byte| byte != 0)
                {
                    return Err(corrupt("state page sentinel mismatch"));
                }
                state.bytes[offset..offset + marker_byte].copy_from_slice(&chunk[..marker_byte]);
                sentinels.insert(spec.key);
            } else {
                decode_lane(
                    lane,
                    payload,
                    &mut state.bytes[offset..offset + CHUNK_BYTES],
                )?;
            }
            return Ok(());
        }
    }
    let end = (prefix + 1)
        .checked_mul(CHUNK_BYTES)
        .ok_or_else(|| corrupt("peer bitvector allocation overflow"))?;
    if end > image.len() {
        let new_total = allocated
            .checked_add(end - image.len())
            .ok_or_else(|| corrupt("peer staging budget overflow"))?;
        if new_total > max_staging_bytes {
            return Err(Error::Invalid("peer staging exceeds memory limit".into()));
        }
        image.resize(end, 0);
        *allocated = new_total;
    }
    decode_lane(lane, payload, &mut image[prefix * CHUNK_BYTES..end])
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

fn parse_metadata(bytes: &[u8], address: &Address, format: &str) -> Result<PackedManifest> {
    let len = sentinel_length(bytes)?;
    if len > MAX_PACKED_BYTES {
        return Err(corrupt("packed metadata exceeds 64 MiB"));
    }
    let manifest = read_packed_manifest(&bytes[..len])?;
    if manifest.address != *address {
        return Err(Error::Collision);
    }
    if manifest.format != format {
        return Err(Error::Invalid("packed engine format mismatch".into()));
    }
    let length = packed_length(&manifest)?;
    if length == 0 || length > MAX_PACKED_BYTES {
        return Err(corrupt("packed payload exceeds 64 MiB"));
    }
    Ok(manifest)
}

fn packed_length(manifest: &PackedManifest) -> Result<usize> {
    if manifest.buffers.is_empty() || manifest.buffers.len() > 4096 {
        return Err(corrupt("invalid packed buffer count"));
    }
    manifest
        .buffers
        .iter()
        .try_fold(0usize, |sum, buffer| sum.checked_add(buffer.length))
        .ok_or_else(|| corrupt("packed payload length overflow"))
}

fn unpack_record(manifest: PackedManifest, mut bits: Vec<u8>) -> Result<PackedSnapshot> {
    let length = packed_length(&manifest)?;
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

type PendingRecord = (usize, PackedManifest, Vec<u8>);
type VerifiedRecords = Vec<Result<(usize, PackedSnapshot)>>;

fn verify_records(pending: Vec<PendingRecord>, parallel: bool) -> VerifiedRecords {
    let verify =
        |(index, layout, payload)| unpack_record(layout, payload).map(|packed| (index, packed));
    if parallel {
        pending.into_par_iter().map(verify).collect()
    } else {
        pending.into_iter().map(verify).collect()
    }
}

fn append_verified(
    verified: VerifiedRecords,
    addresses: &[Address],
    token_len: usize,
    token_sha256: &[u8],
    token_ids: &mut Option<Vec<u32>>,
    attention: &mut Vec<(Address, PackedSnapshot)>,
) -> Result<()> {
    for item in verified {
        let (index, packed) = item?;
        if index == 0 {
            let bytes = one_buffer(packed, "tokens")?;
            if bytes.len() != token_len || Sha256::digest(&bytes).as_slice() != token_sha256 {
                return Err(corrupt("prefill token checksum mismatch"));
            }
            *token_ids = Some(
                bytes
                    .chunks_exact(4)
                    .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
                    .collect(),
            );
        } else {
            attention.push((addresses[index].clone(), packed));
        }
    }
    Ok(())
}

fn verify_state_page(manifest: &PackedManifest, page: &[u8]) -> Result<()> {
    if manifest.buffers.len() != 1 {
        return Err(corrupt("invalid state page buffer count"));
    }
    let layout = &manifest.buffers[0];
    if layout.name != "state"
        || layout.dtype != "u8"
        || layout.shape != [page.len()]
        || layout.length != page.len()
        || Sha256::digest(page).as_slice() != manifest.payload_sha256
    {
        return Err(corrupt("state page layout or checksum mismatch"));
    }
    Ok(())
}

/// A synchronous, full-verification reader for yesno's Unix peer socket.
/// The server owns the snapshot, so checkpointing cannot reclaim a version
/// between pages. A failed read closes this connection and releases its snapshot;
/// connect again to retry. One reader handles one call at a time.
pub struct PeerBundleReader {
    peer: Option<Peer>,
    max_state_bytes: usize,
    max_bundle_bytes: usize,
}

impl PeerBundleReader {
    pub fn connect(socket: impl AsRef<Path>) -> Result<Self> {
        Ok(Self {
            peer: Some(Peer::connect(socket.as_ref())?),
            max_state_bytes: MAX_STATE_BYTES,
            max_bundle_bytes: MAX_STATE_BYTES,
        })
    }

    /// Verify independent packed records concurrently after each peer scan.
    /// Socket reads and state page verification remain ordered.
    pub fn with_parallel_verification(mut self, enabled: bool) -> Self {
        if let Some(peer) = &mut self.peer {
            peer.parallel_verification = enabled;
        }
        self
    }

    /// Verify a completed batch while the peer fetches the next batch.
    /// This also enables parallel verification within each batch.
    pub fn with_pipelined_verification(mut self, enabled: bool) -> Self {
        if let Some(peer) = &mut self.peer {
            peer.pipelined_verification = enabled;
            if enabled {
                peer.parallel_verification = true;
            }
        }
        self
    }

    pub fn with_max_state_bytes(mut self, limit: usize) -> Result<Self> {
        if limit == 0 || limit > MAX_STATE_BYTES {
            return Err(Error::Invalid("invalid peer state memory limit".into()));
        }
        self.max_state_bytes = limit;
        Ok(self)
    }

    pub fn with_max_bundle_bytes(mut self, limit: usize) -> Result<Self> {
        if limit == 0 {
            return Err(Error::Invalid("invalid peer bundle memory limit".into()));
        }
        self.max_bundle_bytes = limit;
        Ok(self)
    }

    pub fn get_prefill_bundle(
        &mut self,
        address: &Address,
        expected_state_format: &str,
    ) -> Result<Option<PrefillBundle>> {
        address.key()?;
        let mut peer = self
            .peer
            .take()
            .ok_or_else(|| Error::Peer("connection closed after a failed read".into()))?;
        let result = Self::read_once(
            &mut peer,
            address,
            expected_state_format,
            self.max_state_bytes,
            self.max_bundle_bytes,
        );
        if result.is_ok() {
            self.peer = Some(peer);
        }
        result
    }

    /// Resolve the current session head and checkpoint inside one server-owned
    /// snapshot. A failed read closes the connection, as for bundle reads.
    pub fn get_session_checkpoint(
        &mut self,
        key: &SessionKey,
        expected_state_format: &str,
    ) -> Result<Option<RestoredSession>> {
        let address = key.head_address()?;
        let mut peer = self
            .peer
            .take()
            .ok_or_else(|| Error::Peer("connection closed after a failed read".into()))?;
        let result = Self::read_session_once(
            &mut peer,
            key,
            &address,
            expected_state_format,
            self.max_state_bytes,
            self.max_bundle_bytes,
        );
        if result.is_ok() {
            self.peer = Some(peer);
        }
        result
    }

    /// Find an exact prepared prefix through one server-owned snapshot
    /// without transferring its state bundle.
    pub fn find_longest_prepared_address(
        &mut self,
        scope: &PrefixScope,
        tokens: &[u32],
    ) -> Result<Option<PrefixHit>> {
        self.read_prepared(scope, tokens, None)
            .map(|hit| hit.map(|(hit, _)| hit))
    }

    /// Find and verify the longest exact prepared prefix in one snapshot.
    pub fn find_longest_prepared_prefix(
        &mut self,
        scope: &PrefixScope,
        tokens: &[u32],
        state_format: &str,
    ) -> Result<Option<(Address, PrefillBundle)>> {
        self.read_prepared(scope, tokens, Some(state_format))
            .and_then(|hit| {
                hit.map(|(hit, bundle)| {
                    bundle
                        .map(|bundle| (hit.address, bundle))
                        .ok_or_else(|| corrupt("missing requested prepared bundle"))
                })
                .transpose()
            })
    }

    fn read_prepared(
        &mut self,
        scope: &PrefixScope,
        tokens: &[u32],
        state_format: Option<&str>,
    ) -> Result<Option<(PrefixHit, Option<PrefillBundle>)>> {
        let address = scope.index_address()?;
        let mut peer = self
            .peer
            .take()
            .ok_or_else(|| Error::Peer("connection closed after a failed read".into()))?;
        let result = Self::read_prepared_once(
            &mut peer,
            scope,
            tokens,
            &address,
            state_format,
            self.max_state_bytes,
            self.max_bundle_bytes,
        );
        if result.is_ok() {
            self.peer = Some(peer);
        }
        result
    }

    fn read_prepared_once(
        peer: &mut Peer,
        scope: &PrefixScope,
        tokens: &[u32],
        address: &Address,
        state_format: Option<&str>,
        max_state_bytes: usize,
        max_bundle_bytes: usize,
    ) -> Result<Option<(PrefixHit, Option<PrefillBundle>)>> {
        let snapshot = peer.open_snapshot()?;
        let mut result = None;
        if let Some(index) = read_record_in_snapshot(peer, snapshot, address, INDEX_FORMAT)? {
            let lengths = parse_lengths(index)?;
            for hit in candidate_addresses_for_lengths(scope, tokens, lengths)?
                .into_iter()
                .rev()
            {
                if let Some(expected) = state_format {
                    let Some(bundle) = Self::read_in_snapshot(
                        peer,
                        snapshot,
                        &hit.address,
                        expected,
                        max_state_bytes,
                        max_bundle_bytes,
                    )?
                    else {
                        continue;
                    };
                    if bundle.token_ids != tokens[..hit.prefix_tokens] {
                        return Err(corrupt("prepared-prefix token mismatch"));
                    }
                    result = Some((hit, Some(bundle)));
                } else {
                    let Some(manifest_record) =
                        read_record_in_snapshot(peer, snapshot, &hit.address, MANIFEST_FORMAT)?
                    else {
                        continue;
                    };
                    let manifest_bytes = one_buffer(manifest_record, "manifest")?;
                    let manifest: Manifest = serde_json::from_slice(&manifest_bytes)?;
                    validate_manifest(&manifest, &hit.address)?;
                    result = Some((hit, None));
                }
                break;
            }
        }
        if !matches!(peer.ask(Frame::SnapshotClose { snapshot })?, Frame::Done) {
            return Err(corrupt("peer snapshot close failed"));
        }
        Ok(result)
    }

    fn read_session_once(
        peer: &mut Peer,
        key: &SessionKey,
        address: &Address,
        expected_state_format: &str,
        max_state_bytes: usize,
        max_bundle_bytes: usize,
    ) -> Result<Option<RestoredSession>> {
        let snapshot = peer.open_snapshot()?;
        let result = match read_record_in_snapshot(peer, snapshot, address, HEAD_FORMAT)? {
            None => None,
            Some(head_snapshot) => {
                let (head, workflow_bytes) = parse_head_snapshot(address, head_snapshot)?;
                let bundle = Self::read_session_checkpoint_in_snapshot(
                    peer,
                    snapshot,
                    key,
                    head.generation(),
                    head.checkpoint_address(),
                    expected_state_format,
                    max_state_bytes,
                    max_bundle_bytes,
                )?
                .ok_or_else(|| corrupt("session head points to a missing checkpoint"))?;
                Some(restored_session(key, head, workflow_bytes, bundle)?)
            }
        };
        if !matches!(peer.ask(Frame::SnapshotClose { snapshot })?, Frame::Done) {
            return Err(corrupt("peer snapshot close failed"));
        }
        Ok(result)
    }

    fn read_once(
        peer: &mut Peer,
        address: &Address,
        expected_state_format: &str,
        max_state_bytes: usize,
        max_bundle_bytes: usize,
    ) -> Result<Option<PrefillBundle>> {
        let snapshot = peer.open_snapshot()?;
        let result = Self::read_in_snapshot(
            peer,
            snapshot,
            address,
            expected_state_format,
            max_state_bytes,
            max_bundle_bytes,
        )?;
        if !matches!(peer.ask(Frame::SnapshotClose { snapshot })?, Frame::Done) {
            return Err(corrupt("peer snapshot close failed"));
        }
        Ok(result)
    }

    #[allow(clippy::too_many_arguments)]
    fn read_session_checkpoint_in_snapshot(
        peer: &mut Peer,
        snapshot: u64,
        key: &SessionKey,
        generation: u64,
        address: &Address,
        expected_state_format: &str,
        max_state_bytes: usize,
        max_bundle_bytes: usize,
    ) -> Result<Option<PrefillBundle>> {
        match record_format_in_snapshot(peer, snapshot, address)?.as_deref() {
            None => Ok(None),
            Some(MANIFEST_FORMAT) => Self::read_in_snapshot(
                peer,
                snapshot,
                address,
                expected_state_format,
                max_state_bytes,
                max_bundle_bytes,
            ),
            Some(append::APPEND_FORMAT) => {
                let root = read_record_in_snapshot(peer, snapshot, address, append::APPEND_FORMAT)?
                    .ok_or_else(|| corrupt("missing append checkpoint manifest"))?;
                let manifest = append::parse_manifest(address, root)?;
                if manifest.state_format != expected_state_format {
                    return Err(Error::Invalid("prefill state format mismatch".into()));
                }
                if manifest.state_bytes > max_state_bytes {
                    return Err(Error::Invalid(
                        "prefill state exceeds peer memory limit".into(),
                    ));
                }
                let logical_bytes = append_logical_bytes(&manifest)?;
                if logical_bytes > max_bundle_bytes {
                    return Err(Error::Invalid(
                        "prefill bundle exceeds peer memory limit".into(),
                    ));
                }
                let base = Self::read_in_snapshot(
                    peer,
                    snapshot,
                    &manifest.base_address,
                    expected_state_format,
                    max_state_bytes,
                    max_bundle_bytes,
                )?
                .ok_or_else(|| corrupt("append base checkpoint is missing"))?;
                let mut state = Vec::with_capacity(manifest.state_bytes);
                for index in 0..manifest.state_pages {
                    let page = read_record_in_snapshot(
                        peer,
                        snapshot,
                        &append::state_address(address, index)?,
                        append::STATE_FORMAT,
                    )?
                    .ok_or_else(|| corrupt("missing append state page"))?;
                    let bytes = one_buffer(page, "state")?;
                    let expected =
                        (manifest.state_bytes - index * STATE_PAGE_BYTES).min(STATE_PAGE_BYTES);
                    if bytes.len() != expected {
                        return Err(corrupt("append state page length mismatch"));
                    }
                    state.extend_from_slice(&bytes);
                }
                let tails = if manifest.tail_bytes == 0 {
                    Vec::new()
                } else {
                    one_buffer(
                        read_record_in_snapshot(
                            peer,
                            snapshot,
                            &append::tails_address(address)?,
                            append::TAIL_FORMAT,
                        )?
                        .ok_or_else(|| corrupt("missing append tails"))?,
                        "tails",
                    )?
                };
                Ok(Some(append::apply_append(
                    key,
                    generation,
                    address,
                    manifest,
                    base,
                    state,
                    tails,
                    crate::ReadVerification::Full,
                )?))
            }
            Some(_) => Err(Error::Invalid(
                "unsupported session checkpoint format".into(),
            )),
        }
    }

    fn read_in_snapshot(
        peer: &mut Peer,
        snapshot: u64,
        address: &Address,
        expected_state_format: &str,
        max_state_bytes: usize,
        max_bundle_bytes: usize,
    ) -> Result<Option<PrefillBundle>> {
        let key = address.key()?;
        let metadata = peer.scan(
            snapshot,
            &[ScanSpec {
                key: key | 1,
                max_len: MAX_PACKED_BYTES,
            }],
            MAX_PACKED_BYTES + CHUNK_BYTES,
            None,
        )?;
        let bits = metadata
            .get(&(key | 1))
            .ok_or_else(|| corrupt("missing top-level metadata lane"))?;
        if bits.is_empty() {
            return Ok(None);
        }
        let root = match parse_metadata(bits, address, MANIFEST_FORMAT) {
            Err(Error::Collision) => return Ok(None),
            result => result?,
        };
        let root_len = packed_length(&root)?;
        let mut root_bits = peer.scan(
            snapshot,
            &[ScanSpec {
                key,
                max_len: root_len,
            }],
            MAX_PACKED_BYTES + CHUNK_BYTES,
            None,
        )?;
        let root_payload = root_bits
            .remove(&key)
            .ok_or_else(|| corrupt("missing top-level payload lane"))?;
        let manifest_bytes = one_buffer(unpack_record(root, root_payload)?, "manifest")?;
        let manifest: Manifest = serde_json::from_slice(&manifest_bytes)?;
        validate_manifest(&manifest, address)?;
        if manifest.state_format != expected_state_format {
            return Err(Error::Invalid("prefill state format mismatch".into()));
        }
        if manifest.state_bytes > max_state_bytes {
            return Err(Error::Invalid(
                "prefill state exceeds peer memory limit".into(),
            ));
        }
        let token_len = usize::try_from(manifest.prefix_tokens)
            .ok()
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| corrupt("prefill token count overflow"))?;
        let mut logical_bytes = manifest
            .state_bytes
            .checked_add(token_len)
            .ok_or_else(|| corrupt("prefill bundle size overflow"))?;
        if logical_bytes > max_bundle_bytes {
            return Err(Error::Invalid(
                "prefill bundle exceeds peer memory limit".into(),
            ));
        }
        let mut addresses = Vec::with_capacity(manifest.state_pages + manifest.attention.len() + 1);
        addresses.push(child_address(address, "tokens")?);
        for index in 0..manifest.state_pages {
            addresses.push(child_address(address, &format!("state:{index}"))?);
        }
        for entry in &manifest.attention {
            if !same_identity(address, &entry.address) {
                return Err(corrupt("prefill attention identity mismatch"));
            }
            addresses.push(entry.address.clone());
        }
        let mut seen = HashSet::new();
        let mut keys = Vec::with_capacity(addresses.len());
        for part in &addresses {
            let key = part.key()?;
            if !seen.insert(key) {
                return Err(corrupt("duplicate packed child key"));
            }
            keys.push(key);
        }
        let mut layouts = Vec::with_capacity(addresses.len());
        for (base, group) in addresses.chunks(peer.max_lanes).enumerate() {
            let start = base * peer.max_lanes;
            let specs = keys[start..start + group.len()]
                .iter()
                .map(|&key| ScanSpec {
                    key: key | 1,
                    max_len: MAX_PACKED_BYTES,
                })
                .collect::<Vec<_>>();
            let mut images = peer.scan(
                snapshot,
                &specs,
                max_bundle_bytes.max(MAX_PACKED_BYTES),
                None,
            )?;
            for (index, part) in group.iter().enumerate() {
                let bits = images
                    .remove(&(keys[start + index] | 1))
                    .ok_or_else(|| corrupt("missing child metadata lane"))?;
                let format = if start + index == 0 {
                    TOKENS_FORMAT
                } else if start + index <= manifest.state_pages {
                    STATE_PAGE_FORMAT
                } else {
                    &manifest.attention[start + index - manifest.state_pages - 1].format
                };
                let layout = parse_metadata(&bits, part, format)?;
                let length = packed_length(&layout)?;
                if start + index == 0 && length != token_len {
                    return Err(corrupt("prefill token length mismatch"));
                }
                if (1..=manifest.state_pages).contains(&(start + index)) {
                    let page = start + index - 1;
                    let expected =
                        (manifest.state_bytes - page * STATE_PAGE_BYTES).min(STATE_PAGE_BYTES);
                    if layout.buffers.len() != 1
                        || layout.buffers[0].name != "state"
                        || layout.buffers[0].dtype != "u8"
                        || layout.buffers[0].shape != [expected]
                        || length != expected
                    {
                        return Err(corrupt("prefill state page layout mismatch"));
                    }
                } else if start + index > manifest.state_pages {
                    logical_bytes = logical_bytes
                        .checked_add(length)
                        .ok_or_else(|| corrupt("prefill bundle size overflow"))?;
                    if logical_bytes > max_bundle_bytes {
                        return Err(Error::Invalid(
                            "prefill bundle exceeds peer memory limit".into(),
                        ));
                    }
                }
                layouts.push(Some(layout));
            }
        }
        let mut state_bytes = vec![0u8; manifest.state_bytes];
        let mut token_ids = None;
        let mut attention = Vec::with_capacity(manifest.attention.len());
        std::thread::scope(|scope| -> Result<()> {
            let mut previous: Option<std::thread::ScopedJoinHandle<'_, VerifiedRecords>> = None;
            for start in (0..addresses.len()).step_by(peer.max_lanes) {
                let end = (start + peer.max_lanes).min(addresses.len());
                let mut pages = HashMap::new();
                let mut specs = Vec::with_capacity(end - start);
                for index in start..end {
                    let key = keys[index];
                    let layout = layouts[index]
                        .as_ref()
                        .ok_or_else(|| corrupt("missing packed child layout"))?;
                    let len = packed_length(layout)?;
                    if (1..=manifest.state_pages).contains(&index) {
                        let page = index - 1;
                        pages.insert(
                            key,
                            PageTarget {
                                start: page * STATE_PAGE_BYTES,
                                len,
                            },
                        );
                    }
                    specs.push(ScanSpec { key, max_len: len });
                }
                let mut images = peer.scan(
                    snapshot,
                    &specs,
                    max_bundle_bytes.saturating_add(specs.len() * CHUNK_BYTES),
                    Some(DirectState {
                        bytes: &mut state_bytes,
                        pages: &pages,
                    }),
                )?;
                let mut pending = Vec::with_capacity(end - start);
                for index in start..end {
                    if (1..=manifest.state_pages).contains(&index) {
                        let target = pages
                            .get(&keys[index])
                            .ok_or_else(|| corrupt("missing state page target"))?;
                        verify_state_page(
                            layouts[index]
                                .as_ref()
                                .ok_or_else(|| corrupt("missing state page layout"))?,
                            &state_bytes[target.start..target.start + target.len],
                        )?;
                        continue;
                    }
                    let payload = images
                        .remove(&keys[index])
                        .ok_or_else(|| corrupt("missing packed child payload"))?;
                    let layout = layouts[index]
                        .take()
                        .ok_or_else(|| corrupt("missing packed child layout"))?;
                    pending.push((index, layout, payload));
                }
                if let Some(job) = previous.take() {
                    let verified = job
                        .join()
                        .map_err(|_| Error::Peer("peer verification worker panicked".into()))?;
                    append_verified(
                        verified,
                        &addresses,
                        token_len,
                        &manifest.token_sha256,
                        &mut token_ids,
                        &mut attention,
                    )?;
                }
                if peer.pipelined_verification {
                    previous = Some(scope.spawn(move || verify_records(pending, true)));
                } else {
                    append_verified(
                        verify_records(pending, peer.parallel_verification),
                        &addresses,
                        token_len,
                        &manifest.token_sha256,
                        &mut token_ids,
                        &mut attention,
                    )?;
                }
            }
            if let Some(job) = previous {
                let verified = job
                    .join()
                    .map_err(|_| Error::Peer("peer verification worker panicked".into()))?;
                append_verified(
                    verified,
                    &addresses,
                    token_len,
                    &manifest.token_sha256,
                    &mut token_ids,
                    &mut attention,
                )?;
            }
            Ok(())
        })?;
        if Sha256::digest(&state_bytes).as_slice() != manifest.state_sha256 {
            return Err(corrupt("prefill state checksum mismatch"));
        }
        Ok(Some(PrefillBundle {
            prefix_tokens: manifest.prefix_tokens,
            token_ids: token_ids.ok_or_else(|| corrupt("missing token record"))?,
            state_format: manifest.state_format,
            state_bytes,
            attention,
        }))
    }
}

fn append_logical_bytes(manifest: &AppendManifest) -> Result<usize> {
    let token_bytes = manifest
        .token_ids
        .len()
        .checked_mul(4)
        .ok_or_else(|| corrupt("append token count overflow"))?;
    manifest.attention.iter().try_fold(
        manifest
            .state_bytes
            .checked_add(token_bytes)
            .ok_or_else(|| corrupt("append bundle size overflow"))?,
        |total, entry| {
            entry.buffers.iter().try_fold(total, |total, buffer| {
                total
                    .checked_add(buffer.base_len)
                    .and_then(|total| total.checked_add(buffer.tail_len))
                    .ok_or_else(|| corrupt("append bundle size overflow"))
            })
        },
    )
}

fn record_format_in_snapshot(
    peer: &mut Peer,
    snapshot: u64,
    address: &Address,
) -> Result<Option<String>> {
    let key = address.key()?;
    let metadata = peer.scan(
        snapshot,
        &[ScanSpec {
            key: key | 1,
            max_len: MAX_PACKED_BYTES,
        }],
        MAX_PACKED_BYTES + CHUNK_BYTES,
        None,
    )?;
    let bits = metadata
        .get(&(key | 1))
        .ok_or_else(|| corrupt("missing packed metadata lane"))?;
    if bits.is_empty() {
        return Ok(None);
    }
    let len = sentinel_length(bits)?;
    if len > MAX_PACKED_BYTES {
        return Err(corrupt("packed metadata exceeds 64 MiB"));
    }
    let manifest = read_packed_manifest(&bits[..len])?;
    if manifest.address != *address {
        return Ok(None);
    }
    Ok(Some(manifest.format))
}

fn read_record_in_snapshot(
    peer: &mut Peer,
    snapshot: u64,
    address: &Address,
    expected_format: &str,
) -> Result<Option<PackedSnapshot>> {
    let key = address.key()?;
    let metadata = peer.scan(
        snapshot,
        &[ScanSpec {
            key: key | 1,
            max_len: MAX_PACKED_BYTES,
        }],
        MAX_PACKED_BYTES + CHUNK_BYTES,
        None,
    )?;
    let bits = metadata
        .get(&(key | 1))
        .ok_or_else(|| corrupt("missing packed metadata lane"))?;
    if bits.is_empty() {
        return Ok(None);
    }
    let manifest = match parse_metadata(bits, address, expected_format) {
        Err(Error::Collision) => return Ok(None),
        result => result?,
    };
    let length = packed_length(&manifest)?;
    let mut images = peer.scan(
        snapshot,
        &[ScanSpec {
            key,
            max_len: length,
        }],
        MAX_PACKED_BYTES + CHUNK_BYTES,
        None,
    )?;
    let payload = images
        .remove(&key)
        .ok_or_else(|| corrupt("missing packed payload lane"))?;
    Ok(Some(unpack_record(manifest, payload)?))
}
