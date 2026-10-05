use shifou::{BoundedHostTier, Error, HostAdmission};

fn admission(net_ms: f64) -> HostAdmission {
    HostAdmission {
        fallback_first_token_ms: net_ms + 1.0,
        resident_first_token_ms: 1.0,
        fill_ms: 0.0,
        expected_reuses: 1,
    }
}

#[test]
fn admission_respects_bytes_and_rejects_without_eviction() {
    let mut tier = BoundedHostTier::new(10, 2);
    assert!(tier.offer("a", vec![1; 6], 6, admission(60.0)).unwrap());
    assert!(tier.offer("b", vec![2; 4], 4, admission(4.0)).unwrap());
    assert_eq!(tier.resident_bytes(), 10);

    // C displaces only the lower-value B.
    assert!(tier.offer("c", vec![3; 4], 4, admission(20.0)).unwrap());
    assert_eq!(tier.resident_bytes(), 10);
    assert!(tier.get(&"a").is_some());
    assert!(tier.get(&"b").is_none());
    assert!(tier.get(&"c").is_some());

    // D would require displacing A as well, which has higher value per byte.
    assert!(!tier.offer("d", vec![4; 6], 6, admission(30.0)).unwrap());
    assert_eq!(tier.resident_bytes(), 10);
    assert_eq!(tier.len(), 2);
    assert!(tier.get(&"a").is_some());
    assert!(tier.get(&"c").is_some());
}

#[test]
fn no_reuse_or_oversize_does_not_consume_budget() {
    let mut tier = BoundedHostTier::new(10, 1);
    assert!(!tier
        .offer(
            "never",
            vec![1; 4],
            4,
            HostAdmission {
                expected_reuses: 0,
                ..admission(10.0)
            },
        )
        .unwrap());
    assert!(!tier
        .offer("too-large", vec![1; 11], 11, admission(1000.0))
        .unwrap());
    assert!(tier.is_empty());
    assert_eq!(tier.resident_bytes(), 0);
    assert!(matches!(
        tier.offer("zero", Vec::<u8>::new(), 0, admission(1.0)),
        Err(Error::Invalid(_))
    ));
}

#[test]
fn equal_value_evicts_least_recently_used() {
    let mut tier = BoundedHostTier::new(10, 2);
    tier.offer("a", 1, 5, admission(10.0)).unwrap();
    tier.offer("b", 2, 5, admission(10.0)).unwrap();
    assert_eq!(tier.get(&"a"), Some(&1));
    assert!(tier.offer("c", 3, 5, admission(10.0)).unwrap());
    assert!(tier.get(&"b").is_none());
    assert_eq!(tier.get(&"a"), Some(&1));
    assert_eq!(tier.get(&"c"), Some(&3));
}

#[test]
fn generation_scoped_identity_and_invalidation_prevent_stale_hits() {
    let mut tier = BoundedHostTier::new(10, 2);
    tier.offer(("session-a", 1_u64), vec![1], 1, admission(10.0))
        .unwrap();
    assert!(tier.get(&("session-a", 2)).is_none());
    assert!(tier.invalidate(&("session-a", 1)));
    assert!(tier.get(&("session-a", 1)).is_none());
    assert_eq!(tier.resident_bytes(), 0);
}

#[test]
fn invalid_costs_cannot_enter_the_tier() {
    let mut tier = BoundedHostTier::new(10, 1);
    assert!(matches!(
        tier.offer(
            "bad",
            1,
            1,
            HostAdmission {
                fallback_first_token_ms: f64::NAN,
                ..admission(10.0)
            },
        ),
        Err(Error::Invalid(_))
    ));
    assert!(tier.is_empty());
}

#[test]
fn rejected_offer_does_not_build_and_failed_build_preserves_residents() {
    let mut tier = BoundedHostTier::new(10, 1);
    tier.offer("first", 1, 10, admission(100.0)).unwrap();
    let mut built = false;
    assert!(!tier
        .offer_with("weak", 10, admission(1.0), || {
            built = true;
            Ok(2)
        })
        .unwrap());
    assert!(!built);
    assert!(matches!(
        tier.offer_with("strong", 10, admission(200.0), || {
            Err(Error::Invalid("build failed".into()))
        }),
        Err(Error::Invalid(_))
    ));
    assert_eq!(tier.get(&"first"), Some(&1));
    assert_eq!(tier.resident_bytes(), 10);
}

#[test]
fn aggregate_eviction_must_increase_expected_savings() {
    let mut tier = BoundedHostTier::new(14, 2);
    tier.offer("a", 1, 7, admission(60.0)).unwrap();
    tier.offer("b", 2, 7, admission(60.0)).unwrap();
    // This candidate has higher value per byte than either resident, but
    // replacing both would lose 10 ms of total expected saving.
    assert!(!tier.offer("c", 3, 12, admission(110.0)).unwrap());
    assert_eq!(tier.resident_bytes(), 14);
    assert_eq!(tier.get(&"a"), Some(&1));
    assert_eq!(tier.get(&"b"), Some(&2));
}
