use std::{path::Path, rc::Rc, sync::Arc};

use anyhow::{ensure, Result};
use attention_rs::InputMetadata;
use candle_core::{DType, Device, Tensor};
use half::bf16;
use parking_lot::RwLock;
use rayon::prelude::*;
use shifou::{Address, PackedBuffer, PackedSnapshot};
use xinfer::{
    models::{
        layers::{distributed::Comm, VarBuilderX},
        qwen3::Qwen3ForCausalLM,
    },
    utils::{config::Config, downloader::ModelPaths, progress::ProgressReporter},
};

pub type NativeCache = Vec<(Tensor, Tensor)>;

/// Validated BF16 pages decoded once for repeated upload from host RAM.
pub struct PreparedBf16 {
    format: String,
    blocks: usize,
    tiles: Vec<Vec<bf16>>,
}

impl PreparedBf16 {
    #[allow(dead_code)]
    pub fn resident_bytes(&self) -> usize {
        self.tiles
            .iter()
            .map(|tile| tile.capacity() * std::mem::size_of::<bf16>())
            .sum()
    }
    /// Prepare exact BF16 pages on a CPU fetch worker before the GPU model is loaded.
    #[allow(dead_code)]
    pub fn from_packed_offline(
        config: &Config,
        attention: &[(Address, PackedSnapshot)],
        capacity: usize,
    ) -> Result<Self> {
        let format = packed_format("auto");
        let (slots, heads, dim) = check_packed_bf16_geometry(config, &format, attention, capacity)?;
        let tiles = attention
            .par_iter()
            .map(|(_, snapshot)| decode_bf16_tile(&snapshot.buffers[0], slots, heads, dim))
            .collect();
        Ok(Self {
            format,
            blocks: slots / BLOCK,
            tiles,
        })
    }
}

pub struct Model {
    pub device: Device,
    pub config: Config,
    pub inner: Qwen3ForCausalLM,
    pub mode: String,
}

const BLOCK: usize = 16;

fn model_format(mode: &str) -> String {
    format!("xinfer/17499e450a174e25be333f88c654ff6743fd4465/attention-c0f19f2/{mode}/bf16/block16/token-head-channel/v1")
}

fn packed_format(mode: &str) -> String {
    format!("{}/packed-bf16/v1", model_format(mode))
}

fn check_packed_bf16_geometry(
    config: &Config,
    format: &str,
    attention: &[(Address, PackedSnapshot)],
    capacity: usize,
) -> Result<(usize, usize, usize)> {
    ensure!(
        config.num_hidden_layers > 0 && attention.len() == config.num_hidden_layers * 2,
        "incomplete attention layer set"
    );
    let heads = config.num_key_value_heads;
    let dim = config
        .head_dim
        .ok_or_else(|| anyhow::anyhow!("missing BF16 head_dim"))?;
    ensure!(heads > 0 && dim > 0, "invalid BF16 attention geometry");
    let first = &attention[0].0;
    let prefix = attention[0]
        .1
        .buffers
        .first()
        .and_then(|b| b.shape.first())
        .copied()
        .unwrap_or(0);
    ensure!(
        prefix > 0 && prefix <= capacity,
        "invalid BF16 restore boundary"
    );
    let slots = capacity.div_ceil(BLOCK) * BLOCK;
    for (index, (address, snapshot)) in attention.iter().enumerate() {
        let expected_slot = if index.is_multiple_of(2) {
            "key"
        } else {
            "value"
        };
        ensure!(
            address.namespace == first.namespace
                && address.model_fingerprint == first.model_fingerprint
                && address.prefix_fingerprint == first.prefix_fingerprint
                && address.layer == (index / 2) as u32
                && address.slot == expected_slot,
            "attention address order or identity mismatch"
        );
        snapshot.validate()?;
        ensure!(
            snapshot.format == format && snapshot.buffers.len() == 1,
            "BF16 format mismatch"
        );
        let buffer = &snapshot.buffers[0];
        ensure!(
            buffer.name == "kv"
                && buffer.dtype == "bf16-le"
                && buffer.shape == [prefix, heads, dim],
            "BF16 buffer geometry mismatch"
        );
    }
    Ok((slots, heads, dim))
}

fn decode_bf16_tile(buffer: &PackedBuffer, slots: usize, heads: usize, dim: usize) -> Vec<bf16> {
    let mut words = Vec::with_capacity(slots * heads * dim);
    words.extend(
        buffer
            .bytes
            .chunks_exact(2)
            .map(|pair| bf16::from_bits(u16::from_le_bytes([pair[0], pair[1]]))),
    );
    words.resize(slots * heads * dim, bf16::from_bits(0));
    words
}

impl Model {
    pub fn load(directory: &Path, mode: &str) -> Result<Self> {
        let config: Config =
            serde_json::from_slice(&std::fs::read(directory.join("config.json"))?)?;
        ensure!(
            config
                .architectures
                .as_ref()
                .is_some_and(|a| a.iter().any(|x| x == "Qwen3ForCausalLM")),
            "this experiment adapter supports the Qwen3 architecture"
        );
        ensure!(
            config.head_dim == Some(128),
            "pinned native prefill kernels require head_dim=128"
        );
        let device = xinfer::utils::new_device(0)?;
        let paths = ModelPaths {
            tokenizer_filename: directory.join("tokenizer.json"),
            tokenizer_config_filename: directory.join("tokenizer_config.json"),
            config_filename: directory.join("config.json"),
            generation_config_filename: directory.join("generation_config.json"),
            filenames: vec![directory.join("model.safetensors")],
            auxiliary_filenames: vec![],
            chat_template_filename: None,
        };
        let vb = VarBuilderX::new(&paths, false, DType::BF16, &device)?;
        let inner = Qwen3ForCausalLM::new(
            &vb,
            Rc::new(Comm::default()),
            &config,
            DType::BF16,
            false,
            &device,
            Arc::new(RwLock::new(Box::new(ProgressReporter::new(0)))),
        )?;
        ensure!(
            matches!(mode, "auto" | "turbo4" | "turbo3"),
            "unsupported cache mode"
        );
        Ok(Self {
            mode: mode.into(),
            device,
            config,
            inner,
        })
    }

    pub fn empty_cache(&self, capacity: usize) -> Result<NativeCache> {
        let shape = if self.mode != "auto" {
            (capacity.div_ceil(BLOCK), 1, 1, 1)
        } else {
            (
                capacity.div_ceil(BLOCK),
                BLOCK,
                self.config.num_key_value_heads,
                self.config.head_dim.unwrap(),
            )
        };
        if self.mode != "auto" {
            self.init_tq(capacity)?;
        }
        (0..self.config.num_hidden_layers)
            .map(|_| {
                Ok((
                    Tensor::zeros(shape, DType::BF16, &self.device)?,
                    Tensor::zeros(shape, DType::BF16, &self.device)?,
                ))
            })
            .collect()
    }

    pub fn duplicate(&self, cache: &NativeCache) -> Result<NativeCache> {
        cache
            .iter()
            .map(|(k, v)| Ok((k.copy()?, v.copy()?)))
            .collect()
    }

    pub fn forward(&self, tokens: &[u32], start: usize, cache: &NativeCache) -> Result<Vec<f32>> {
        ensure!(!tokens.is_empty(), "empty forward");
        let prefill = start == 0;
        ensure!(
            prefill || tokens.len() == 1,
            "resume uses single-token decoding"
        );
        self.forward_with_prefill(tokens, start, cache, prefill)
    }

    /// Append a batch of new prompt tokens after a restored BF16 prefix.
    pub fn forward_chunk(
        &self,
        tokens: &[u32],
        start: usize,
        cache: &NativeCache,
    ) -> Result<Vec<f32>> {
        ensure!(
            self.mode == "auto",
            "chunked append is validated only for BF16 cache mode"
        );
        ensure!(
            start > 0 && !tokens.is_empty(),
            "chunked append needs a prefix and new tokens"
        );
        self.forward_with_prefill(tokens, start, cache, true)
    }

    fn forward_with_prefill(
        &self,
        tokens: &[u32],
        start: usize,
        cache: &NativeCache,
        prefill: bool,
    ) -> Result<Vec<f32>> {
        let _prefill_guard = xinfer::models::layers::linear::set_linear_is_prefill(prefill);
        let end = start + tokens.len();
        let capacity = cache[0].0.dim(0)? * BLOCK;
        ensure!(end <= capacity, "cache capacity exceeded");
        let positions: Vec<i64> = (start..end).map(|x| x as i64).collect();
        let block_ids: Vec<u32> = (0..end.div_ceil(BLOCK) as u32).collect();
        let metadata = InputMetadata {
            is_prefill: prefill,
            is_mla: false,
            sequence_ids: Some(vec![0]),
            mamba_slot_mapping: None,
            slot_mapping: Tensor::from_vec(positions.clone(), tokens.len(), &self.device)?,
            block_tables: Some(Tensor::from_vec(
                block_ids.clone(),
                (1, block_ids.len()),
                &self.device,
            )?),
            block_tables_host: Some(vec![block_ids]),
            context_lens_host: Some(vec![end as u32]),
            context_lens: Some(Tensor::from_vec(vec![end as u32], 1, &self.device)?),
            cu_seqlens_q: if prefill {
                Some(Tensor::from_vec(
                    vec![0u32, tokens.len() as u32],
                    2,
                    &self.device,
                )?)
            } else {
                None
            },
            cu_seqlens_k: if prefill {
                Some(Tensor::from_vec(vec![0u32, end as u32], 2, &self.device)?)
            } else {
                None
            },
            max_seqlen_q: if prefill { tokens.len() } else { 0 },
            max_seqlen_k: if prefill { end } else { 0 },
            max_context_len: end,
            seqlens: if prefill {
                Some(vec![tokens.len() as u32])
            } else {
                None
            },
            flashinfer_metadata: None,
            is_mtp_verify: false,
        };
        let logits = self.inner.forward(
            &Tensor::from_vec(tokens.to_vec(), tokens.len(), &self.device)?,
            &Tensor::from_vec(positions, tokens.len(), &self.device)?,
            Some(cache),
            &metadata,
            false,
        )?;
        Ok(logits.flatten_all()?.to_vec1::<f32>()?)
    }

    pub fn export(&self, cache: &NativeCache, tokens: usize) -> Result<Vec<shifou::Tensor>> {
        let heads = self.config.num_key_value_heads;
        let dim = self.config.head_dim.unwrap();
        let mut tiles = Vec::new();
        for (key, value) in cache {
            for tensor in [key, value] {
                let slots = tensor.dim(0)? * BLOCK;
                let values = tensor
                    .reshape((slots, heads, dim))?
                    .narrow(0, 0, tokens)?
                    .contiguous()?
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                tiles.push(shifou::Tensor {
                    shape: vec![tokens, heads, dim],
                    axes: vec!["token".into(), "head".into(), "channel".into()],
                    values,
                });
            }
        }
        Ok(tiles)
    }

    /// Replace one prefix tile while sharing the untouched native layer buffers.
    /// Decode writes only slots at and after the prefix boundary.
    pub fn replace_tile(
        &self,
        cache: &NativeCache,
        index: usize,
        tile: &shifou::Tensor,
        capacity: usize,
    ) -> Result<NativeCache> {
        ensure!(self.mode == "auto", "replacement requires the BF16 cache");
        ensure!(index < cache.len() * 2, "tile index out of range");
        tile.validate()?;
        ensure!(
            tile.shape.len() == 3
                && tile.axes == ["token", "head", "channel"]
                && tile.shape[0] <= capacity
                && tile.shape[1] == self.config.num_key_value_heads
                && tile.shape[2] == self.config.head_dim.unwrap(),
            "tile geometry does not match the model"
        );
        let slots = capacity.div_ceil(BLOCK) * BLOCK;
        let mut values = tile.values.clone();
        values.resize(slots * tile.shape[1] * tile.shape[2], 0.0);
        let replacement = Tensor::from_vec(
            values,
            (slots / BLOCK, BLOCK, tile.shape[1], tile.shape[2]),
            &self.device,
        )?
        .to_dtype(DType::BF16)?;
        let mut result = cache.clone();
        if index.is_multiple_of(2) {
            result[index / 2].0 = replacement;
        } else {
            result[index / 2].1 = replacement;
        }
        Ok(result)
    }

    pub fn restore(&self, tiles: &[shifou::Tensor], capacity: usize) -> Result<NativeCache> {
        ensure!(
            tiles.len() == self.config.num_hidden_layers * 2,
            "incomplete layer set"
        );
        let mut native = Vec::new();
        for tile in tiles {
            tile.validate()?;
            ensure!(
                tile.shape.len() == 3 && tile.axes == ["token", "head", "channel"],
                "unsupported tile layout"
            );
            ensure!(
                tile.shape[0] <= capacity
                    && tile.shape[1] == self.config.num_key_value_heads
                    && tile.shape[2] == self.config.head_dim.unwrap(),
                "tile geometry does not match the model"
            );
            let slots = capacity.div_ceil(BLOCK) * BLOCK;
            let mut values = tile.values.clone();
            values.resize(slots * tile.shape[1] * tile.shape[2], 0.0);
            native.push(
                Tensor::from_vec(
                    values,
                    (slots / BLOCK, BLOCK, tile.shape[1], tile.shape[2]),
                    &self.device,
                )?
                .to_dtype(DType::BF16)?,
            );
        }
        let cache = native
            .chunks_exact(2)
            .map(|x| (x[0].clone(), x[1].clone()))
            .collect();
        self.device.synchronize()?;
        Ok(cache)
    }

    pub fn packed_bf16_format(&self) -> String {
        packed_format(&self.mode)
    }

    /// Export all attention layers in K, V order without expanding BF16 to F32.
    pub fn export_packed_bf16(
        &self,
        cache: &NativeCache,
        tokens: usize,
    ) -> Result<Vec<PackedSnapshot>> {
        ensure!(self.mode == "auto", "raw BF16 export needs the BF16 cache");
        ensure!(
            cache.len() == self.config.num_hidden_layers,
            "incomplete native cache"
        );
        let heads = self.config.num_key_value_heads;
        let dim = self.config.head_dim.unwrap();
        let format = self.packed_bf16_format();
        let mut snapshots = Vec::with_capacity(cache.len() * 2);
        for (key, value) in cache {
            for tensor in [key, value] {
                let slots = tensor.dim(0)? * BLOCK;
                ensure!(
                    tokens > 0 && tokens <= slots,
                    "invalid BF16 export boundary"
                );
                let words = tensor
                    .reshape((slots, heads, dim))?
                    .narrow(0, 0, tokens)?
                    .contiguous()?
                    .flatten_all()?
                    .to_vec1::<bf16>()?;
                let bytes = words
                    .iter()
                    .flat_map(|word| word.to_bits().to_le_bytes())
                    .collect();
                let snapshot = PackedSnapshot {
                    format: format.clone(),
                    buffers: vec![PackedBuffer {
                        name: "kv".into(),
                        dtype: "bf16-le".into(),
                        shape: vec![tokens, heads, dim],
                        bytes,
                    }],
                };
                snapshot.validate()?;
                snapshots.push(snapshot);
            }
        }
        Ok(snapshots)
    }

    fn check_packed_bf16(
        &self,
        attention: &[(Address, PackedSnapshot)],
        capacity: usize,
    ) -> Result<(usize, usize, usize)> {
        ensure!(self.mode == "auto", "raw BF16 restore needs the BF16 cache");
        check_packed_bf16_geometry(
            &self.config,
            &self.packed_bf16_format(),
            attention,
            capacity,
        )
    }

    /// Exact BF16 payload bytes after padding to engine block size.
    #[allow(dead_code)]
    pub fn prepared_bf16_bytes(
        &self,
        attention: &[(Address, PackedSnapshot)],
        capacity: usize,
    ) -> Result<usize> {
        let (slots, heads, dim) = self.check_packed_bf16(attention, capacity)?;
        Ok(slots * heads * dim * std::mem::size_of::<bf16>() * attention.len())
    }

    /// Validate and decode a stored prefix once for repeated restoration from
    /// host RAM. The padded tiles keep continuation slots distinct per fork.
    pub fn prepare_packed_bf16(
        &self,
        attention: &[(Address, PackedSnapshot)],
        capacity: usize,
    ) -> Result<PreparedBf16> {
        let (slots, heads, dim) = self.check_packed_bf16(attention, capacity)?;
        let tiles = attention
            .iter()
            .map(|(_, snapshot)| decode_bf16_tile(&snapshot.buffers[0], slots, heads, dim))
            .collect();
        Ok(PreparedBf16 {
            format: self.packed_bf16_format(),
            blocks: slots / BLOCK,
            tiles,
        })
    }

    /// Decode independent host tiles on CPU workers for a prepared-host tier.
    #[allow(dead_code)]
    pub fn prepare_packed_bf16_parallel(
        &self,
        attention: &[(Address, PackedSnapshot)],
        capacity: usize,
    ) -> Result<PreparedBf16> {
        let (slots, heads, dim) = self.check_packed_bf16(attention, capacity)?;
        let tiles = attention
            .par_iter()
            .map(|(_, snapshot)| decode_bf16_tile(&snapshot.buffers[0], slots, heads, dim))
            .collect();
        Ok(PreparedBf16 {
            format: self.packed_bf16_format(),
            blocks: slots / BLOCK,
            tiles,
        })
    }

    /// Upload decoded BF16 pages without reconstructing them from byte arrays.
    pub fn restore_prepared_bf16(&self, prepared: &PreparedBf16) -> Result<NativeCache> {
        ensure!(self.mode == "auto", "prepared BF16 needs the BF16 cache");
        ensure!(
            prepared.format == self.packed_bf16_format()
                && prepared.tiles.len() == self.config.num_hidden_layers * 2,
            "prepared BF16 model or layer mismatch"
        );
        let heads = self.config.num_key_value_heads;
        let dim = self.config.head_dim.unwrap();
        let mut native = Vec::with_capacity(prepared.tiles.len());
        for words in &prepared.tiles {
            ensure!(
                words.len() == prepared.blocks * BLOCK * heads * dim,
                "prepared BF16 tile geometry mismatch"
            );
            native.push(Tensor::from_slice(
                words,
                (prepared.blocks, BLOCK, heads, dim),
                &self.device,
            )?);
        }
        self.device.synchronize()?;
        Ok(native
            .chunks_exact(2)
            .map(|pair| (pair[0].clone(), pair[1].clone()))
            .collect())
    }

    /// Validate and restore a stored BF16 prefix in one call.
    pub fn restore_packed_bf16(
        &self,
        attention: &[(Address, PackedSnapshot)],
        capacity: usize,
    ) -> Result<NativeCache> {
        let (slots, heads, dim) = self.check_packed_bf16(attention, capacity)?;
        let native = attention
            .iter()
            .map(|(_, snapshot)| {
                let words = decode_bf16_tile(&snapshot.buffers[0], slots, heads, dim);
                Tensor::from_vec(words, (slots / BLOCK, BLOCK, heads, dim), &self.device)
            })
            .collect::<candle_core::Result<Vec<_>>>()?;
        self.device.synchronize()?;
        Ok(native
            .chunks_exact(2)
            .map(|pair| (pair[0].clone(), pair[1].clone()))
            .collect())
    }

    /// Decode independent attention tiles on CPU workers, then upload them in
    /// the original layer order. The peer probe selects this path explicitly.
    #[allow(dead_code)]
    pub fn restore_packed_bf16_parallel(
        &self,
        attention: &[(Address, PackedSnapshot)],
        capacity: usize,
    ) -> Result<NativeCache> {
        let (slots, heads, dim) = self.check_packed_bf16(attention, capacity)?;
        let words = attention
            .par_iter()
            .map(|(_, snapshot)| decode_bf16_tile(&snapshot.buffers[0], slots, heads, dim))
            .collect::<Vec<_>>();
        let native = words
            .into_iter()
            .map(|words| Tensor::from_vec(words, (slots / BLOCK, BLOCK, heads, dim), &self.device))
            .collect::<candle_core::Result<Vec<_>>>()?;
        self.device.synchronize()?;
        Ok(native
            .chunks_exact(2)
            .map(|pair| (pair[0].clone(), pair[1].clone()))
            .collect())
    }
}

impl Model {
    pub fn format(&self) -> String {
        model_format(&self.mode)
    }

    fn tq_mode(&self) -> attention_rs::TurboquantMode {
        match self.mode.as_str() {
            "turbo4" => attention_rs::TurboquantMode::Turbo4,
            "turbo3" => attention_rs::TurboquantMode::Turbo3,
            _ => unreachable!("TQ method on ordinary cache"),
        }
    }

    fn init_tq(&self, capacity: usize) -> Result<()> {
        let pages = capacity.div_ceil(BLOCK);
        let h = self.config.num_key_value_heads;
        let d = self.config.head_dim.unwrap();
        ensure!(
            d.is_power_of_two(),
            "TQ rotation requires a power-of-two head dimension"
        );
        let kbytes = if self.mode == "turbo3" {
            (d * 3).div_ceil(8)
        } else {
            d / 2
        };
        let mut layers = Vec::new();
        for _ in 0..self.config.num_hidden_layers {
            layers.push(attention_rs::TurboquantLayerCache {
                k_absmax: Some(Tensor::zeros((pages, BLOCK, h), DType::F32, &self.device)?),
                k_quant: Some(Tensor::zeros(
                    (pages, BLOCK, h, kbytes),
                    DType::U8,
                    &self.device,
                )?),
                v_absmax: Tensor::zeros((pages, BLOCK, h), DType::F32, &self.device)?,
                v_quant: Tensor::zeros((pages, BLOCK, h, d / 2), DType::U8, &self.device)?,
            });
        }
        attention_rs::init_turboquant_cache(self.tq_mode(), layers, BLOCK);
        Ok(())
    }

    pub fn export_tq(&self, tokens: usize) -> Result<shifou::PackedSnapshot> {
        ensure!(self.mode != "auto", "not a TQ cache");
        let mut buffers = Vec::new();
        for layer in 0..self.config.num_hidden_layers {
            let result = attention_rs::with_turboquant_layer(layer, |tq, _| -> Result<()> {
                for (name, t) in [
                    ("k_absmax", tq.k_absmax.as_ref().unwrap()),
                    ("k_quant", tq.k_quant.as_ref().unwrap()),
                    ("v_absmax", &tq.v_absmax),
                    ("v_quant", &tq.v_quant),
                ] {
                    let mut shape = vec![t.dim(0)? * BLOCK];
                    shape.extend_from_slice(&t.dims()[2..]);
                    let slice = t.reshape(shape)?.narrow(0, 0, tokens)?.contiguous()?;
                    let (dtype, bytes) = match slice.dtype() {
                        DType::U8 => ("u8", slice.flatten_all()?.to_vec1::<u8>()?),
                        DType::F32 => (
                            "f32-le",
                            slice
                                .flatten_all()?
                                .to_vec1::<f32>()?
                                .iter()
                                .flat_map(|x| x.to_le_bytes())
                                .collect(),
                        ),
                        _ => anyhow::bail!("unsupported TQ buffer dtype"),
                    };
                    buffers.push(shifou::PackedBuffer {
                        name: format!("{layer}/{name}"),
                        dtype: dtype.into(),
                        shape: slice.dims().to_vec(),
                        bytes,
                    });
                }
                Ok(())
            })
            .ok_or_else(|| anyhow::anyhow!("missing TQ layer"))?;
            result?;
        }
        let snapshot = shifou::PackedSnapshot {
            format: self.format(),
            buffers,
        };
        snapshot.validate()?;
        Ok(snapshot)
    }

    pub fn restore_tq(
        &self,
        snapshot: &shifou::PackedSnapshot,
        tokens: usize,
        capacity: usize,
    ) -> Result<()> {
        snapshot.validate()?;
        ensure!(snapshot.format == self.format(), "TQ format mismatch");
        ensure!(
            snapshot.buffers.len() == self.config.num_hidden_layers * 4,
            "TQ layer count mismatch"
        );
        ensure!(tokens <= capacity, "TQ prefix exceeds capacity");
        let h = self.config.num_key_value_heads;
        let d = self.config.head_dim.unwrap();
        let kbytes = if self.mode == "turbo3" {
            (d * 3).div_ceil(8)
        } else {
            d / 2
        };
        let pages = capacity.div_ceil(BLOCK);
        let mut layers = Vec::new();
        for layer in 0..self.config.num_hidden_layers {
            let mut loaded = Vec::new();
            for (i, name, width, dtype) in [
                (0, "k_absmax", 1, "f32-le"),
                (1, "k_quant", kbytes, "u8"),
                (2, "v_absmax", 1, "f32-le"),
                (3, "v_quant", d / 2, "u8"),
            ] {
                let b = &snapshot.buffers[layer * 4 + i];
                let shape = if dtype == "u8" {
                    vec![tokens, h, width]
                } else {
                    vec![tokens, h]
                };
                ensure!(
                    b.name == format!("{layer}/{name}") && b.shape == shape && b.dtype == dtype,
                    "TQ buffer geometry/name/dtype mismatch"
                );
                let mut native_shape = vec![pages, BLOCK, h];
                if dtype == "u8" {
                    native_shape.push(width);
                }
                let count = pages * BLOCK * h * width;
                loaded.push(if dtype == "u8" {
                    let mut bytes = b.bytes.clone();
                    bytes.resize(count, 0);
                    Tensor::from_vec(bytes, native_shape, &self.device)?
                } else {
                    let mut values: Vec<f32> = b
                        .bytes
                        .chunks_exact(4)
                        .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
                        .collect();
                    ensure!(
                        values.iter().all(|x| x.is_finite() && *x >= 0.0),
                        "invalid TQ scale"
                    );
                    values.resize(count, 0.0);
                    Tensor::from_vec(values, native_shape, &self.device)?
                });
            }
            layers.push(attention_rs::TurboquantLayerCache {
                k_absmax: Some(loaded[0].clone()),
                k_quant: Some(loaded[1].clone()),
                v_absmax: loaded[2].clone(),
                v_quant: loaded[3].clone(),
            });
        }
        self.device.synchronize()?;
        attention_rs::init_turboquant_cache(self.tq_mode(), layers, BLOCK);
        Ok(())
    }
}
