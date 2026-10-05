use proptest::prelude::*;
use shifou::{decode, encode, Policy, Tensor};

fn policy(axis: &str, size: usize, error: f64) -> Policy {
    Policy {
        group_axis: axis.into(),
        group_size: size,
        candidate_bits: vec![2, 4, 8],
        max_abs_error: error,
        outlier_fraction: 0.05,
        residual_tokens: 0,
        token_axis: "token".into(),
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(72))]
    #[test]
    fn bounded_roundtrip_for_both_grouping_axes(
        rows in 1usize..20, columns in 1usize..20, group in 1usize..20,
        seed in any::<u64>(), transpose in any::<bool>(), error in 0.0f64..0.4,
    ) {
        let mut state = seed;
        let values = (0..rows * columns).map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 32) as i32) as f32 / i32::MAX as f32 * 3.0
        }).collect();
        let tensor = Tensor { shape: vec![rows, columns], axes: vec!["token".into(), "channel".into()], values };
        let p = policy(if transpose { "token" } else { "channel" }, group, error);
        let encoded = encode(&tensor, &p).unwrap();
        let decoded = decode(&encoded).unwrap();
        prop_assert_eq!(&decoded.shape, &tensor.shape);
        prop_assert_eq!(&decoded.axes, &tensor.axes);
        let measured = tensor.values.iter().zip(decoded.values)
            .map(|(&a,b)| (a as f64 - b as f64).abs()).fold(0.0, f64::max);
        prop_assert!(measured <= error);
        prop_assert_eq!(measured, encoded.report().unwrap().max_abs_error);
    }

    #[test]
    fn exact_mode_preserves_all_finite_float_bits(words in prop::collection::vec(any::<u32>(), 1..100)) {
        let values: Vec<_> = words.into_iter().map(f32::from_bits).filter(|x| x.is_finite()).collect();
        prop_assume!(!values.is_empty());
        let tensor = Tensor { shape: vec![values.len()], axes: vec!["channel".into()], values };
        let restored = decode(&encode(&tensor, &policy("channel", 32, 0.0)).unwrap()).unwrap();
        prop_assert_eq!(tensor.values.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            restored.values.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
    }
}

#[test]
fn tail_is_exact_for_both_axis_orders() {
    for (shape, axes) in [
        (vec![12, 5], vec!["token", "channel"]),
        (vec![5, 12], vec!["channel", "token"]),
    ] {
        let tensor = Tensor {
            shape,
            axes: axes.iter().map(|s| s.to_string()).collect(),
            values: (0..60).map(|i| (i as f32 * 0.137).sin()).collect(),
        };
        for group_axis in ["token", "channel"] {
            let mut p = policy(group_axis, 4, 1.0);
            p.residual_tokens = 3;
            let restored = decode(&encode(&tensor, &p).unwrap()).unwrap();
            for i in 0..60 {
                let token = if axes[0] == "token" { i / 5 } else { i % 12 };
                if token >= 9 {
                    assert_eq!(tensor.values[i].to_bits(), restored.values[i].to_bits());
                }
            }
        }
    }
}

#[test]
fn sparse_outliers_are_kept_exactly() {
    let mut values = vec![0.0; 256];
    for (i, v) in values.iter_mut().enumerate() {
        *v = (i % 4) as f32;
    }
    values[100] = 10000.0;
    let tensor = Tensor {
        shape: vec![256],
        axes: vec!["channel".into()],
        values,
    };
    let mut p = policy("channel", 256, 0.01);
    p.candidate_bits = vec![2];
    let tile = encode(&tensor, &p).unwrap();
    assert!(tile.report().unwrap().exception_values > 0);
    assert_eq!(
        decode(&tile).unwrap().values[100].to_bits(),
        10000f32.to_bits()
    );
}

#[test]
fn low_precision_and_raw_fallback_both_execute() {
    let p = policy("channel", 256, 0.0);
    let easy = Tensor {
        shape: vec![256],
        axes: vec!["channel".into()],
        values: vec![0.0; 256],
    };
    assert!(encode(&easy, &p)
        .unwrap()
        .report()
        .unwrap()
        .groups_by_bits
        .contains_key(&2));
    let hard = Tensor {
        values: (0..256).map(|i| (i as f32).sin()).collect(),
        ..easy
    };
    assert_eq!(encode(&hard, &p).unwrap().report().unwrap().raw_values, 256);
}

#[test]
fn malformed_inputs_are_rejected() {
    let valid = Tensor {
        shape: vec![2],
        axes: vec!["channel".into()],
        values: vec![1.0, 2.0],
    };
    let mut p = policy("channel", 2, 0.1);
    p.group_size = 0;
    assert!(encode(&valid, &p).is_err());
    p.group_size = 2;
    p.candidate_bits = vec![32];
    assert!(encode(&valid, &p).is_err());
    for values in [vec![f32::NAN, 1.0], vec![f32::INFINITY, 1.0], vec![1.0]] {
        assert!(encode(
            &Tensor {
                values,
                ..valid.clone()
            },
            &policy("channel", 2, 0.1)
        )
        .is_err());
    }
    assert!(encode(
        &Tensor {
            shape: vec![usize::MAX, 2],
            axes: vec!["a".into(), "b".into()],
            values: vec![]
        },
        &p
    )
    .is_err());
}

#[test]
fn groups_cross_roaring_container_boundaries() {
    let tensor = Tensor {
        shape: vec![65540],
        axes: vec!["channel".into()],
        values: (0..65540).map(|i| (i % 4) as f32).collect(),
    };
    let tile = encode(&tensor, &policy("channel", 4096, 0.0)).unwrap();
    assert_eq!(decode(&tile).unwrap(), tensor);
}

#[test]
fn one_tile_can_mix_low_precision_and_exact_groups() {
    let tensor = Tensor {
        shape: vec![512],
        axes: vec!["channel".into()],
        values: (0..512)
            .map(|i| if i < 256 { 0.0 } else { (i as f32).sin() })
            .collect(),
    };
    let tile = encode(&tensor, &policy("channel", 256, 0.0)).unwrap();
    let report = tile.report().unwrap();
    assert_eq!(report.groups_by_bits.get(&2), Some(&1));
    assert_eq!(report.groups_by_bits.get(&32), Some(&1));
    assert_eq!(decode(&tile).unwrap(), tensor);
}
