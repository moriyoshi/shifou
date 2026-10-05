//! Measured-cost admission for exact prefill reuse.

use crate::{Error, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CacheTier {
    Gpu,
    PreparedHost,
    LocalYesno,
    Peer,
    Flight,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TierCost {
    pub tier: CacheTier,
    /// Full hit cost through the first new token, including suffix prefill.
    pub first_token_ms: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum HitChoice {
    Recompute,
    Restore(TierCost),
}

/// Supply measurements for the same model, prompt shape and serving tier.
/// Memory and storage budgets are enforced by the caller before admitting a
/// tier; this estimator only compares observed latency.
#[derive(Clone, Debug)]
pub struct AdmissionEstimate {
    pub recompute_first_token_ms: f64,
    pub publish_ms: f64,
    pub expected_future_hits: u64,
    pub available_tiers: Vec<TierCost>,
}

impl AdmissionEstimate {
    fn validate(&self) -> Result<()> {
        let all = std::iter::once(self.recompute_first_token_ms)
            .chain(std::iter::once(self.publish_ms))
            .chain(self.available_tiers.iter().map(|cost| cost.first_token_ms));
        if all
            .into_iter()
            .any(|value| !value.is_finite() || value < 0.0)
        {
            return Err(Error::Invalid("invalid measured cache cost".into()));
        }
        Ok(())
    }

    pub fn choose_hit(&self) -> Result<HitChoice> {
        self.validate()?;
        let best = self
            .available_tiers
            .iter()
            .copied()
            .min_by(|a, b| a.first_token_ms.total_cmp(&b.first_token_ms));
        Ok(match best {
            Some(cost) if cost.first_token_ms < self.recompute_first_token_ms => {
                HitChoice::Restore(cost)
            }
            _ => HitChoice::Recompute,
        })
    }

    pub fn should_publish(&self) -> Result<bool> {
        let HitChoice::Restore(best) = self.choose_hit()? else {
            return Ok(false);
        };
        let saving = self.recompute_first_token_ms - best.first_token_ms;
        Ok((self.expected_future_hits as f64) * saving > self.publish_ms)
    }
}
