use std::path::Path;

use serde::Deserialize;
use shifou::{Address, Cache, Error, Policy, Result, Tensor};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PutRequest {
    address: Address,
    tensor: Tensor,
    policy: Policy,
}

fn json<T: serde::de::DeserializeOwned>(path: &str) -> Result<T> {
    Ok(serde_json::from_reader(std::fs::File::open(path)?)?)
}

fn main() {
    if let Err(error) = run() {
        eprintln!("shifou: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    match args.as_slice() {
        [command, directory] if command == "demo" => demo(Path::new(directory)),
        [command, directory, input] if command == "put" => {
            let request: PutRequest = json(input)?;
            let mut cache = Cache::open(directory)?;
            let report = cache.put(&request.address, &request.tensor, &request.policy)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
            Ok(())
        }
        [command, directory, input] if command == "get" => {
            let address: Address = json(input)?;
            let cache = Cache::open(directory)?;
            println!("{}", serde_json::to_string_pretty(&cache.get(&address)?)?);
            Ok(())
        }
        [command, directory, input] if command == "remove" => {
            let address: Address = json(input)?;
            let mut cache = Cache::open(directory)?;
            println!("{}", cache.remove(&address)?);
            Ok(())
        }
        _ => Err(Error::Invalid("usage: shifou demo CACHE | put CACHE request.json | get CACHE address.json | remove CACHE address.json".into())),
    }
}

fn demo(path: &Path) -> Result<()> {
    let mut cases = Vec::new();
    for (model, shape, axes) in [
        (
            "synthetic-a",
            vec![64, 2, 64],
            vec!["token", "head", "channel"],
        ),
        (
            "synthetic-b",
            vec![3, 96, 32],
            vec!["head", "token", "channel"],
        ),
    ] {
        let count: usize = shape.iter().product();
        let values: Vec<f32> = (0..count)
            .map(|i| {
                if i % 251 == 0 {
                    30.0
                } else {
                    ((i * 17 % 101) as f32 - 50.0) / 50.0
                }
            })
            .collect();
        for (slot, policy) in [("key", Policy::keys(0.08)), ("value", Policy::values(0.08))] {
            cases.push((
                Address {
                    namespace: "demo".into(),
                    model_fingerprint: model.into(),
                    prefix_fingerprint: "synthetic-prefix-v1".into(),
                    layer: 0,
                    slot: slot.into(),
                },
                Tensor {
                    shape: shape.clone(),
                    axes: axes.iter().map(|x| x.to_string()).collect(),
                    values: values.clone(),
                },
                policy,
            ));
        }
    }
    {
        let mut cache = Cache::open(path)?;
        for (address, tensor, policy) in &cases {
            cache.put(address, tensor, policy)?;
        }
    }
    let cache = Cache::open(path)?;
    println!("Synthetic tensors; f32 source baseline. No model-quality or GPU benchmark.");
    println!("model        slot    f32_bytes payload_bytes record_bytes max_error  groups_by_bits");
    for (address, original, policy) in &cases {
        let restored = cache
            .get(address)?
            .ok_or_else(|| Error::Corrupt("demo record missing".into()))?;
        let max_error = original
            .values
            .iter()
            .zip(&restored.tensor.values)
            .map(|(&a, &b)| (a as f64 - b as f64).abs())
            .fold(0.0, f64::max);
        if max_error > policy.max_abs_error || restored.tensor.shape != original.shape {
            return Err(Error::Corrupt("demo reconstruction check failed".into()));
        }
        let report = restored.report;
        println!(
            "{:<12} {:<7} {:>9} {:>13} {:>12} {:>9.5}  {:?}",
            address.model_fingerprint,
            address.slot,
            report.codec.raw_f32_bytes,
            report.codec.payload_bytes,
            report.total_roaring_bytes,
            max_error,
            report.codec.groups_by_bits
        );
    }
    println!("Reopened and verified all four records. record_bytes includes encoded metadata, but excludes database/WAL allocation.");
    Ok(())
}
