//! Exact-input tokenization reuse. The caller fingerprints every tokenizer
//! setting that can change token IDs; input bytes are hashed here.

use crate::{
    Address, Cache, CacheReader, Error, PackedBuffer, PackedSnapshot, PackedStorageReport, Result,
};
use sha2::{Digest, Sha256};

const FORMAT: &str = "shifou-token-ids-le-u32/v1";

pub fn token_address(tokenizer_fingerprint: &str, input: &[u8]) -> Result<Address> {
    let address = Address {
        namespace: "shifou-token-cache/v1".into(),
        model_fingerprint: tokenizer_fingerprint.into(),
        prefix_fingerprint: format!("{:x}", Sha256::digest(input)),
        layer: 0,
        slot: "ids".into(),
    };
    address.key()?;
    Ok(address)
}

fn decode_ids(snapshot: PackedSnapshot) -> Result<Vec<u32>> {
    if snapshot.buffers.len() != 1
        || snapshot.buffers[0].name != "ids"
        || snapshot.buffers[0].dtype != "u8"
        || !snapshot.buffers[0].bytes.len().is_multiple_of(4)
    {
        return Err(Error::Corrupt("invalid token ID record".into()));
    }
    Ok(snapshot.buffers[0]
        .bytes
        .chunks_exact(4)
        .map(|part| u32::from_le_bytes(part.try_into().unwrap()))
        .collect())
}

impl Cache {
    pub fn put_token_ids(
        &mut self,
        tokenizer_fingerprint: &str,
        input: &[u8],
        ids: &[u32],
    ) -> Result<PackedStorageReport> {
        if ids.is_empty() {
            return Err(Error::Invalid("empty token sequence".into()));
        }
        let address = token_address(tokenizer_fingerprint, input)?;
        let bytes = ids
            .iter()
            .flat_map(|id| id.to_le_bytes())
            .collect::<Vec<_>>();
        self.put_packed(
            &address,
            &PackedSnapshot {
                format: FORMAT.into(),
                buffers: vec![PackedBuffer {
                    name: "ids".into(),
                    dtype: "u8".into(),
                    shape: vec![bytes.len()],
                    bytes,
                }],
            },
        )
    }

    pub fn get_token_ids(
        &self,
        tokenizer_fingerprint: &str,
        input: &[u8],
    ) -> Result<Option<Vec<u32>>> {
        self.get_packed(&token_address(tokenizer_fingerprint, input)?, FORMAT)?
            .map(decode_ids)
            .transpose()
    }
}

impl CacheReader {
    pub fn get_token_ids(
        &self,
        tokenizer_fingerprint: &str,
        input: &[u8],
    ) -> Result<Option<Vec<u32>>> {
        self.get_packed(&token_address(tokenizer_fingerprint, input)?, FORMAT)?
            .map(decode_ids)
            .transpose()
    }
}
