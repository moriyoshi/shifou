//! Bounded admission and retention of engine-prepared host state.
//!
//! The caller owns cache coherence: use a generation and checkpoint address
//! in a session key, and invalidate prepared keys after a publication change
//! or removal. A hit here does not query yesno for current metadata.

use crate::{Error, Result};

/// All costs must describe the same model, request shape, and serving tier.
/// `fill_ms` is extra work to retain the entry. Include peer read time when
/// filling in the background; omit it when the miss already needed that read.
#[derive(Clone, Copy, Debug)]
pub struct HostAdmission {
    pub fallback_first_token_ms: f64,
    pub resident_first_token_ms: f64,
    pub fill_ms: f64,
    pub expected_reuses: u64,
}

impl HostAdmission {
    pub fn net_saving_ms(self) -> Result<f64> {
        let costs = [
            self.fallback_first_token_ms,
            self.resident_first_token_ms,
            self.fill_ms,
        ];
        if costs.iter().any(|value| !value.is_finite() || *value < 0.0) {
            return Err(Error::Invalid("invalid host admission cost".into()));
        }
        let saving = (self.fallback_first_token_ms - self.resident_first_token_ms)
            * self.expected_reuses as f64
            - self.fill_ms;
        if !saving.is_finite() {
            return Err(Error::Invalid("host admission saving overflow".into()));
        }
        Ok(saving)
    }
}

struct Entry<K, V> {
    key: K,
    value: V,
    bytes: usize,
    net_saving_ms: f64,
    last_use: u64,
}

/// Synchronous host tier with hard limits on declared resident bytes and
/// entries. It owns each value and only lends `&V`, so eviction cannot free a
/// value still borrowed by a request. The caller must include all engine
/// padding and allocations in `bytes`; the limit is not a process RSS cap.
pub struct BoundedHostTier<K, V> {
    max_bytes: usize,
    max_entries: usize,
    resident_bytes: usize,
    clock: u64,
    entries: Vec<Entry<K, V>>,
}

impl<K: PartialEq, V> BoundedHostTier<K, V> {
    pub fn new(max_bytes: usize, max_entries: usize) -> Self {
        Self {
            max_bytes,
            max_entries,
            resident_bytes: 0,
            clock: 0,
            entries: Vec::new(),
        }
    }

    pub fn resident_bytes(&self) -> usize {
        self.resident_bytes
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn get(&mut self, key: &K) -> Option<&V> {
        let index = self.entries.iter().position(|entry| entry.key == *key)?;
        self.clock = self.clock.saturating_add(1);
        self.entries[index].last_use = self.clock;
        Some(&self.entries[index].value)
    }

    pub fn invalidate(&mut self, key: &K) -> bool {
        if let Some(index) = self.entries.iter().position(|entry| entry.key == *key) {
            self.resident_bytes -= self.entries[index].bytes;
            self.entries.swap_remove(index);
            true
        } else {
            false
        }
    }

    /// Admit only a positive-net-value entry. When full, evict entries with
    /// lower expected saving per resident byte; oldest use breaks ties.
    /// A rejected offer leaves existing entries untouched.
    pub fn offer(
        &mut self,
        key: K,
        value: V,
        bytes: usize,
        admission: HostAdmission,
    ) -> Result<bool> {
        self.offer_with(key, bytes, admission, || Ok(value))
    }

    /// Plan admission before building an expensive engine representation.
    /// `build` is called only when the candidate can displace lower-value
    /// entries; a failed build leaves all resident entries untouched.
    pub fn offer_with<F>(
        &mut self,
        key: K,
        bytes: usize,
        admission: HostAdmission,
        build: F,
    ) -> Result<bool>
    where
        F: FnOnce() -> Result<V>,
    {
        if bytes == 0 {
            return Err(Error::Invalid("zero-sized host entry".into()));
        }
        let net_saving_ms = admission.net_saving_ms()?;
        if net_saving_ms <= 0.0
            || bytes > self.max_bytes
            || self.max_entries == 0
            || self.max_bytes == 0
        {
            return Ok(false);
        }
        let existing = self.entries.iter().position(|entry| entry.key == key);
        let existing_bytes = existing.map_or(0, |index| self.entries[index].bytes);
        let mut kept_bytes = self.resident_bytes - existing_bytes;
        let mut kept_count = self.entries.len() - usize::from(existing.is_some());
        let mut victims = Vec::new();
        let candidate_density = net_saving_ms / bytes as f64;
        while bytes > self.max_bytes - kept_bytes || kept_count + 1 > self.max_entries {
            let victim = self
                .entries
                .iter()
                .enumerate()
                .filter(|(index, _)| Some(*index) != existing && !victims.contains(index))
                .min_by(|(_, left), (_, right)| {
                    (left.net_saving_ms / left.bytes as f64)
                        .total_cmp(&(right.net_saving_ms / right.bytes as f64))
                        .then_with(|| left.last_use.cmp(&right.last_use))
                });
            let Some((index, victim)) = victim else {
                return Ok(false);
            };
            if victim.net_saving_ms / victim.bytes as f64 > candidate_density {
                return Ok(false);
            }
            kept_bytes -= victim.bytes;
            kept_count -= 1;
            victims.push(index);
        }
        let displaced_saving_ms: f64 = victims
            .iter()
            .map(|index| self.entries[*index].net_saving_ms)
            .sum();
        if net_saving_ms < displaced_saving_ms {
            return Ok(false);
        }
        if let Some(index) = existing {
            victims.push(index);
        }
        let value = build()?;
        victims.sort_unstable_by(|left, right| right.cmp(left));
        for index in victims {
            self.entries.swap_remove(index);
        }
        self.clock = self.clock.saturating_add(1);
        self.resident_bytes = kept_bytes + bytes;
        self.entries.push(Entry {
            key,
            value,
            bytes,
            net_saving_ms,
            last_use: self.clock,
        });
        Ok(true)
    }
}
