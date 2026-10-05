//! Experimental tensor tiles encoded as persistent yesno bit planes.
//! Model adapters supply grouping, an error budget, and computation identity.

mod codec;
mod compact;
mod experts;
mod packed;
mod policy;
mod prefill;
mod prefix;
mod serving;
mod session;
mod store;
mod tensor;
mod tokens;

pub use codec::{decode, encode, EncodedTile, Policy, Report};
pub use compact::{decode_compact, encode_compact, CompactPolicy};
pub use experts::ExpertSnapshotKey;
pub use packed::{PackedBuffer, PackedSnapshot, PackedStorageReport, ReadVerification};
pub use policy::{AdmissionEstimate, CacheTier, HitChoice, TierCost};
#[cfg(feature = "flight")]
pub use prefill::FlightBundleReader;
#[cfg(feature = "peer")]
pub use prefill::PeerBundleReader;
pub use prefill::PrefillBundle;
pub use prefix::{PrefixHit, PrefixScope};
pub use serving::{BoundedHostTier, HostAdmission};
pub use session::{RestoredSession, SessionKey};
pub use store::{Address, Cache, CacheReader, Retrieved, StorageReport};
pub use tensor::Tensor;
pub use tokens::token_address;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid tensor or policy: {0}")]
    Invalid(String),
    #[error("invalid cached record: {0}")]
    Corrupt(String),
    #[error("cache address collides with a different full digest")]
    Collision,
    #[error("cache directory is already in use: {0}")]
    Busy(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Yesno(#[from] yesno_core::CodecError),
    #[cfg(feature = "flight")]
    #[error(transparent)]
    Flight(#[from] arrow_flight::error::FlightError),
    #[cfg(feature = "flight")]
    #[error(transparent)]
    FlightTransport(#[from] tonic::transport::Error),
    #[cfg(feature = "peer")]
    #[error("peer socket: {0}")]
    Peer(String),
}

pub type Result<T> = std::result::Result<T, Error>;
