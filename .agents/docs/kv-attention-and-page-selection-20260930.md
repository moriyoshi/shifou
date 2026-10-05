# KV attention and page selection experiments, 2026-09-30

## KV attention opportunities from set operations, 2026-09-30

This investigation concerns computation over sets, beyond persistence or dense
payload compression. It made no production changes. Shifou compact keys use two centroids per channel over the old prefix,
while its values use two centroids per token/head across channels.

### Candidate sets and selective KV fetch

For one attention head let B_l be token IDs in the query's selected LSH bucket
for table l. Then C = ( union_l B_l union Recent ) intersect Eligible and
Missing = C minus Resident are ordinary set queries. Eligibility must preserve
request identity and causal boundaries; residency is used to schedule fetches,
not to silently discard nonresident attention candidates. Existing operations
are OrdSet::union_all, and, and_not, and cardinality-only intersections.
Layer/head selections can also be unioned to coalesce block fetches.

[MagicPIG](https://arxiv.org/abs/2410.16179) supplies the LSH sampling precedent.
Its sampling estimator and inclusion probabilities matter: taking a bucket union
and applying ordinary subset softmax does not reproduce the paper's guarantees.
[Quest](https://arxiv.org/abs/2406.10774) selects KV pages using query-dependent
min/max scores; yesno could combine resulting page masks but does not replace
that numeric scoring step. [SparQ](https://arxiv.org/abs/2312.04985) is another
selective-fetch precedent. These are research leads, not implemented integrations.

A standalone release probe lives at
../yesno/.agents-workspace/tmp/kv-set-probe-20260930. It compares resident eager yesno,
prebuilt Expr with collect_set, RoaringBitmap MultiOps union, dense word masks,
and ordinary sorted/deduplicated posting vectors. All methods return the same
sorted selected-ID and missing-ID vectors, verified against the posting-vector
oracle before timing. The timed region includes result allocation and enumeration.
Index construction, hashing a real query, persistence, GPU transfer, and attention
are excluded. This establishes kernel cost only, with no accuracy claim.

Construction: N in {8192, 131072, 1048576}; 16 independently seeded synthetic
buckets at density 2^-b with b in {8,12}; append the last 128 token IDs as an
additional posting. Eligible excludes IDs divisible by 7. Resident excludes IDs
divisible by 4. Bucket membership uses SplitMix64 finalization of
(t XOR ((l+1)*123456789)) plus 0x9e3779b97f4a7c15, masked to b bits and compared
with zero. The exact generator is in the retained scratch source. Query planes
and buckets are reused across iterations. Five samples per method, rotating
method order; each sample averages 128 executions, or 32 at N=1048576.
Numbers below are the median of those five means, in microseconds, on this
shared aarch64 host. Dependencies resolved offline: yesno local checkout,
roaring 0.11.5, arrow-buffer 59.3.0. Warm, repeated selections are favorable to
cache locality and do not model changing real queries.

| N | Hash bits | Selected | yesno eager | yesno Expr | Roaring | Dense | Posting vectors |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8192 | 8 | 535 | 8.37 | 15.73 | 5.63 | 1.82 | 5.89 |
| 8192 | 12 | 132 | 5.42 | 14.61 | 2.17 | 1.31 | 1.32 |
| 131072 | 8 | 6946 | 36.13 | 38.75 | 33.43 | 9.86 | 55.98 |
| 131072 | 12 | 551 | 6.29 | 10.21 | 3.89 | 5.90 | 3.46 |
| 1048576 | 8 | 54164 | 357.36 | 334.24 | 327.14 | 164.92 | 701.99 |
| 1048576 | 12 | 3627 | 41.67 | 51.20 | 29.21 | 61.93 | 29.48 |

Yesno beats the full dense scan by about 1.5x in the longest sparse case,
but does not beat the Roaring reference in any measured case. Dense words win
at higher selected density, while posting vectors are competitive when very
few IDs survive. Expr setup is visible on small sets. There is an opportunity
for selective attention as a system, but this probe does not establish that
choosing yesno provides a speed advantage over an optimized alternative.
A real integration must include all layers and query heads, index maintenance,
actual memory traffic, and attention-output error; per-head microseconds multiply.

### Direct low-bit score computation

For current one-bit keys write k[t,j] = a[j] + d[j]*x[t,j], x in {0,1}.
For query q, let w[j] = q[j]*d[j]. Then

    score[t] = sum_j q[j]*a[j] + sum_j w[j]*x[t,j].

If w is quantized once per query as lambda times a signed b-bit integer,
with bit-plane sets Q_r and two's-complement weights alpha_r, and K_t is the
set of channels whose x[t,j] is one, the quantized score is

    score_hat[t] = base + lambda * sum_r alpha_r * |K_t intersect Q_r|.

This is exact for that quantized representation; quantizing w introduces
additional error. OrdSet::and_cardinality supplies the primitive, and
BitMatrix::counted_mul supplies a related integer count product. For D=128,
each intersection needs two dense u64 words. Tiny per-token OrdSets can cost
more in directories and dispatch than the arithmetic; packed rows must be
compared with a direct word-popcount baseline and the existing GPU kernel.
The current counted_mul implementation transposes its RHS on each call, which
must be counted or avoided with a reusable layout in a real integration.
Current compact code ordering also requires rearrangement for per-token rows.

For genuine +/-1 sign vectors of equal amplitude, q dot k = D - 2*|Q XOR K|.
That identity alone is not a valid replacement for current shifou centroids:
the centroids vary by channel. QJL's asymmetric sketch estimator is also not
plain Hamming distance; see https://arxiv.org/abs/2406.03482.

One-bit value accumulation admits the same construction after attention:
v[t,j] = a[t] + d[t]*x[t,j], w[t] = p[t]*d[t], and the channel sets index
tokens. Selected-token masks can restrict those sets before weighted sums.
The float softmax, scales, exact tail, and sparse exception corrections still
need their numeric paths. Prior scalar one-bit quality was poor, so this is a
kernel experiment to evaluate with accuracy, not a serving recommendation.

### Adaptive sparse refinements

For a stable bit-plane refinement codec, Requested intersect Available minus
Resident identifies refinement blocks to read. Requested intersect Outliers
identifies exact corrections to apply. These operations can reduce bytes read
when extra precision is genuinely sparse. Existing compact 1/2/4-bit candidates
use different quantizers; a 2-bit candidate cannot be upgraded to its 4-bit
candidate by appending bit planes. A nested codec would be required first.

## Binary candidate selection with real Qwen attention, 2026-09-30

The follow-up experiment used Qwen3-0.6B with the same pinned xinfer, Candle,
attention.rs, model, and WikiText-2 artifacts as shifou's earlier evaluation.
Research source is under ../yesno/.agents-workspace/tmp/kv-attention-probe-20260930;
production sources and dependency checkouts were not changed. The experiment
copies the pinned Qwen and attention adapters into that standalone crate and
adds a hook after native attention. The common codec is shifou compact4, not
native TurboQuant. Native TurboQuant kernel quality remains a separate question.

### Construction and controls

A 128-dimensional key is multiplied by fixed random signs generated with
SplitMix64 ( seed offset 42 ), followed by a Walsh-Hadamard transform. Its signs
form a 128-bit sketch; retain the original key norm separately. The same
transform is applied to a query. Candidate ranking uses

    key_norm * cos( pi * HammingDistance / 128 ).

Query norm is a positive common ranking factor and is omitted. This is a
heuristic angular sketch, not a reproduction of MagicPIG's sampling estimator.
Select ceil( context / denominator ) tokens for denominator 8, 4, or 2, then
union the first four and most recent 32 tokens. Break score ties by token ID.
There is no causal filtering gap: the isolated adapter has one request and
exports exactly the valid causal cache. Rescore candidates with the original
BF16 query and decoded compact4 KV, then apply subset softmax and weighted V.

Three equivalent candidate implementations are compared: per-token OrdSet
xor_cardinality; two packed u64 XOR/popcounts; and BitMatrix::counted_mul with
Hamming distance reconstructed as key_weight + query_weight - 2*intersection.
The matrix path includes query construction, RHS transposition, counts output,
selection, candidate-set construction and union with mandatory tokens.
Candidate IDs must agree exactly between all three paths for every replay query.

The main replay has prefix 8192 at corpus offset 4096, followed by one actual
corpus token, giving 8193 valid KV positions. Capture all 28 layers and 16 query
heads per layer, with two query heads per KV head: 448 queries, 1344 head-policy
rows. Sketches use original BF16 keys. KV uses compact4 keys grouped over tokens
and values grouped over channels, exact newest 32 tokens, decoded and rounded
back to BF16 to match Model::restore. CPU attention then operates on f32 copies
of those BF16 values. Thus packed decoding is outside the timed region.

The CPU full-BF16 attention output was compared against native CUDA attention
before using replay results. Maximum relative L2 disagreement was 0.002725.
Timing warms each operation, takes three samples of three executions each, and
uses their median. The table averages those per-query medians across all heads.
These are warm CPU measurements on the shared GB10 aarch64 host; method order
is fixed. They include allocation and returned outputs but exclude index setup,
encoding, device transfers, persistence, and model layers outside attention.
No comparison with an optimized CPU BLAS kernel or GPU attention was made.

### Replay results

Full scalar CPU attention averaged 449.04 us per query head.

| Requested fraction | Actual tokens | Mean attention mass | Mean relative L2 vs full4 | Pages touched | OrdSet pipeline us | BitMatrix pipeline us | Direct packed pipeline us |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1/8 | 12.78% | 82.62% | 0.2113 | 68.40% | 828.71 | 98.82 | 82.70 |
| 1/4 | 25.21% | 88.27% | 0.1423 | 87.70% | 917.27 | 171.66 | 149.63 |
| 1/2 | 50.11% | 94.17% | 0.0696 | 98.41% | 1129.17 | 347.46 | 320.04 |

The packed yesno path beats the scalar full-attention reference, but remains
slower than the direct two-word implementation. Per-token OrdSets are the wrong
representation for these short dense sketches. Selecting 1/8 retains only 1.99%
of attention mass in the worst head; 237 of 448 heads retain less than 90%.
An unconstrained exact-score top-k with the same candidate count retains 96.59%
mean mass, versus 82.62% for sketches. There is substantial ranking error beyond
the candidate budget itself. The exact-score control is an upper bound and is
not offered as a way to avoid reading all keys.

Page fractions count distinct 16-token pages per query head. They are not
measured I/O. GQA unions and shifou's channel-grouped key layout can increase
physical overfetch further. The 12.78% token selection is therefore not an 87%
reduction in bytes transferred. For D=128, ideal packed4 K+V code access would
be 128 bytes per selected old token, plus group parameters and exact-tail bytes;
our timed kernel actually reads decoded f32 values. A packed sketch plus norm
uses 20 bytes per KV token; the matrix path also retains a 4-byte row weight.
This side index is an additional cost, not free compression.

The replay compact payload totals 252,524,608 bytes. Building all three index
representations together took 2.81 seconds; encoding and decoding compact4 took
10.43 seconds. Those combined setup numbers are not a benchmark of a standalone
matrix index builder. Persistence and incremental index maintenance were not
measured.

### Continuation quality

Eight teacher-forced tokens were scored at each of two disjoint contexts:
prefix 2048 at offset 4096, and prefix 8192 at offset 16384. Compact4 encodes only
the prefix; continuation KV remains exact BF16. Every decode layer receives the
substituted sparse attention output, so effects propagate through later layers
and continuation queries. Index keys for the prefix come from original BF16;
new keys are appended from the current continuation. Model parameters are fixed
before evaluating either slice; no selection budget was tuned on their logits.

The hook runs native attention to update the cache, copies data to CPU, and
replaces its output. It is a quality instrument with redundant computation and
full-cache transfers, not a serving implementation or throughput measurement.
The 2K continuation uses actual BitMatrix selection and checks IDs against the
direct packed implementation on every query. Its full result JSON exactly
matches the earlier direct-packed continuation. The 8K continuation uses the
equivalent packed selector. Replay validated matrix equivalence on all 8K
reference queries; matrix-backed 8K continuation itself was not rerun.

| Attention policy | KL from BF16, 2K | KL from BF16, 8K |
| --- | ---: | ---: |
| Native compact4, full attention | 0.010699 | 0.032882 |
| CPU compact4, full attention control | 0.010361 | 0.032631 |
| Half + mandatory | 0.027415 | 0.041098 |
| Quarter + mandatory | 0.036898 | 0.034356 |
| Eighth + mandatory | 0.039752 | 0.086758 |

The full CPU replacement has KL 0.000558 and 0.000994 from native compact4 on
the 2K and 8K slices, respectively. This measures the observed numerical-path
confound in those controls; it does not make sparse policy error additive.
Quality varies nonmonotonically with budget on these tiny samples, and top-token
agreement alone conceals distribution shifts. Sixteen scored tokens across two
contexts do not establish general model quality or a safe serving policy.

### Decision and reproduction

BitMatrix supplies useful packed sketch arithmetic. This experiment establishes
no yesno-specific advantage over direct word operations and no end-to-end
inference speedup. The next useful design should choose pages coherently and
allow per-head candidate budgets or a full-attention fallback calibrated on
separate data. The weak heads and scattered-page fetches need to be addressed
before optimizing another cardinality kernel.

Build and run from the shifou root, using new output filenames on every run:

```sh
CUDA_COMPUTE_CAP=121 cargo build --release --offline \
  --manifest-path ../yesno/.agents-workspace/tmp/kv-attention-probe-20260930/Cargo.toml \
  --target-dir .agents-workspace/tmp/target
.agents-workspace/tmp/target/release/kv-attention-probe \
  .agents-workspace/tmp/models/Qwen3-0.6B \
  .agents-workspace/tmp/corpora/wikitext-2-test.txt \
  ../yesno/.agents-workspace/tmp/kv-attention-probe-20260930/replay-new.json 8192 4096
```

Append `quality` to run eight-token continuation evaluation instead of replay;
use 2048 4096 or 8192 16384 for the recorded quality cases. The retained final
source uses matrix selection for continuation. It reserves eight matrix rows
for appending keys and asserts equality with packed selection per query.

The durable numeric summary is
[kv-attention-experiment-20260930.json](./kv-attention-experiment-20260930.json).
Full per-head replay rows and build/run logs remain alongside the research
source. The isolated release build and formatting check passed. Correctness
checks included native attention comparison, candidate equality, and equality
of matrix-backed and packed continuation result JSON. No production Rust code
was changed by this experiment.

## Bitmap layout and candidate primitives for KV sketches, 2026-09-30

The separate research crate at `../yesno/.agents-workspace/tmp/kv-layout-probe-20260930`
reused the pinned Qwen3-0.6B/xinfer adapter and the 8,192-token prefix at corpus
offset 4,096. It captured one subsequent query across all 28 layers and 16
query heads per layer: 448 real queries, grouped into 224 pairs that share a KV
head. This experiment operates on BF16 Q and K for ranking and attention-mass
measurement; it does not measure model-output KL or a GPU attention kernel.

### Which sketch layout pays?

Each key sketch is 128 bits, exactly two `u64` words. Four layouts produced
identical score arrays for every query:

- Two packed words per token, with separate f32 norms ( 20 bytes/token ).
- `#[repr(C)]` entries containing two words and one f32 norm ( 24 bytes/token
  after alignment ).
- `BitMatrix` packed rows read through public `row_words`, with separate norms.
- `BitMatrix::counted_mul`, reconstructing Hamming distance from intersection
  counts and row weights. This also stores four bytes per token for row weight.

The query sketch was prepared before each timed scan. Each scan wrote all token
scores into a reused output vector. Seven warm samples of seven repetitions
were timed per query; the table averages the per-query medians on the shared
GB10 aarch64 host. The two paired methods score both query heads in one scan
and divide elapsed time by two. Timing excludes index construction, query
projection, top-k selection, attention, and data transfer.

| Layout / score path | Mean us per query | Approximate index bytes/token |
| --- | ---: | ---: |
| Separate packed words and norms | 6.18 | 20 |
| Interleaved padded entry | 6.05 | 24 |
| Existing `BitMatrix::row_words` | 7.43 | 20 |
| `BitMatrix::counted_mul` | 20.85 | 24 |
| Two query heads, direct packed scan | 5.37 | 20 |
| Two query heads, batched `counted_mul` | 16.79 | 24 |

The interleaved entry saves about 0.13 us, or 2%, over the separate arrays in
this run while spending 20% more index bytes. Prior runs varied enough that this
small advantage is not established. The row geometry itself has no padding at
128 columns. Paired direct scanning gives the clearest local gain: both queries
reuse the same two key words and norm. Batching also helps `counted_mul`, but
its allocation, generic counts and per-row work remain roughly three times the
direct paired path. `row_words` already exposes the packed representation; it
is 1.25 us slower than reading the two-word array in this measurement.

### Page-aligned bitmap summaries

For each 16-token KV page, AND and OR its 128-bit sketches. The two masks take
32 bytes/page, about two bytes per token. For a query bit of one where the page
OR is zero, every key in the page mismatches. A query zero where the page AND
is one also mismatches every key. Counting those positions is a safe lower
bound on every key's Hamming distance; the probe asserted the bound against
the actual minimum on every full page and query.

On this model the bound averaged 32.57 mismatches while the true nearest key
per page averaged 58.57. At the illustrative Hamming cutoff 50, the bound can
rule out 23.92% of pages; exact page minima show that 87.86% lie beyond 50.
This bound would therefore miss most possible page skips. It only bounds raw
Hamming distance: the current candidate rank also uses key norm, so a serving
prune would need a valid bound on that full score.

With a budget of about one eighth of the tokens, ranking individual tokens by
the existing sketch score retains 83.24% of exact BF16 attention mass on
average but touches 350.88 of 513 pages. Selecting about one eighth of pages
by their highest sketch score touches 67.13 pages and retains 78.52% mass.
Ranking pages by the sum of exponentiated sketch scores retains 78.84% mass at
67.11 pages. An oracle that ranks pages by exact attention mass retains 91.62%
at 65.41 pages, so the page constraint alone does not explain the loss. All
policies include the first page and newest three pages. Page count is an access
proxy; actual reads and full-model quality were not measured here.

### Primitive decision

`OrdSet::union_all`, `and` and `and_not` already form the useful page-mask
algebra: combine selected and mandatory pages, intersect with eligible pages,
then subtract resident pages to schedule reads. No new yesno set operator is
needed for that path. The shifou/xinfer adapter still needs a way to fetch only
the selected pages and an accurate way to rank them.

If an actual consumer uses `BitMatrix` sketches, two possible generic APIs
are worth a dedicated implementation benchmark:

1. A checked `BitMatrix::from_row_words(rows, cols, words)` constructor would
   accept an already packed token-major sketch array. It must reject the wrong
   length and nonzero padding bits, and initialize or invalidate the cached
   population count correctly. The current research adapter calls `set` once
   per one bit; no isolated constructor-time saving was measured here.
2. A row-wise Hamming or intersection-count operation accepting multiple packed
   query rows could write counts directly without transposing a right-hand
   matrix. Its target is the direct paired scan near 5.37 us per query, not the
   current 16.79 us of paired `counted_mul`. Preserve arbitrary column widths
   and zero-tail invariants; key norms and ranking remain model-side concerns.

Neither is yet justified as a new public API: the existing `row_words` and
`OrdSet` operations can support a first real selective-attention consumer, and
that consumer should determine whether the generic interface saves enough to
carry a semver promise. Page AND/OR reduction is also easy outside yesno from
packed rows; the measured bound is too loose to prioritize a crate API for it.

The durable [numeric summary](./kv-layout-experiment-20260930.json) records
the construction and aggregate measurements. Full per-head rows, build logs
and exact-equality checks remain with the scratch crate. The isolated release
build and `cargo fmt --check` passed. No production Rust code was changed.

## Page-centroid KV selection and chunk access, 2026-09-30

The isolated probe at `../yesno/.agents-workspace/tmp/kv-page-probe-20260930` reused
Qwen3-0.6B ( revision `c1899de289a04d12100db370d81485cdf75e47ca` ),
the pinned xinfer/Candle adapter, and WikiText-2 test text. One actual query
was captured after a 2,048-token prefix at corpus offset 4,096 and after a
separate 8,192-token prefix at offset 16,384. The 8K slice contains 448
layer/query-head observations, or 224 pairs sharing a KV head. Pages have 16
tokens. Every policy also retains the first and newest three pages. Static
attention mass uses exact BF16 QK softmax as its oracle.

The page score is query dot product with the mean key of the page. A physically
stored BF16 centroid takes 256 bytes/page, or 16 bytes/token. The static CPU
ranking probe reads `u16` BF16 words and converts them to f32; the model-quality
hook rounds an f32 centroid while scoring, giving BF16 decisions without
measuring storage access. Ranking timing uses five warm samples of five
rankings per query, reports the median per query and averages across heads.
It includes top-page selection but excludes construction, persistence,
transfer, and attention. The `Summary` struct retains arrays for other
policies, so timing does not isolate the index's cache footprint.

| 8K policy | Budget | Pages/head of 513 | BF16 attention mass | Rank us/head |
| --- | ---: | ---: | ---: | ---: |
| BF16 centroid | 1/8 | 65.86 | 93.08% | 20.00 |
| f32 centroid | 1/8 | 65.86 | 93.08% | 22.63 |
| Max-token sketch | 1/8 | 66.75 | 84.99% | 15.96 |
| Exact page-mass oracle | 1/8 | about 66 | 94.25% | not timed |
| BF16 centroid | 1/4 | 129.61 | 96.10% | 20.30 |
| Exact page-mass oracle | 1/4 | about 130 | 96.96% | not timed |

At 2K, the f32 centroid retained 88.02% mass at the one-eighth budget,
versus 89.31% for the page oracle and 81.07% for max-token sketch. The
coordinate min/max and centroid-plus-radius upper bounds were asserted
against every exact page maximum; neither ranked as well as the centroid.
At 8K and one eighth, min/max retained 91.33% mass at 30.28 us/head,
the radius bound 84.39%, and a 128-bit centroid sketch 83.81% at 2.20 us/head.
The BF16 centroid halves index bytes relative to f32 with nearly identical
selection. Its 20.00 versus 22.63 us/head in this shared-structure probe is
not evidence of an independent BF16 speed advantage. The initial timing path
mistakenly calculated an unused f32 centroid and query sketch for BF16 and
other policies; it was fixed before these timing figures were recorded.
SIMD scoring was not evaluated.

Eight teacher-forced continuation tokens were scored at each prefix. The
native full BF16 output is the reference. `native_compact4` uses shifou
compact4 for the complete prefix and full attention. Page policies use exact
attention over selected decoded compact4 KV, BF16 centroids from original
prefix keys, and exact continuation KV. The CPU hook substitutes its output
after native attention. These are model-quality measurements, not serving
throughput or native TurboQuant measurements.

| Policy | Mean KL from BF16, 2K | Mean KL from BF16, 8K |
| --- | ---: | ---: |
| Full compact4 | 0.010699 | 0.032882 |
| BF16 centroid pages, 1/8 | 0.037857 | 0.110116 |
| BF16 centroid pages, 1/4 | 0.023047 | 0.063336 |
| Earlier token sketch, 1/8 | 0.039752 | 0.086758 |
| Earlier token sketch, 1/4 | 0.036898 | 0.034356 |

On the 8K slice, page centroids retain more static attention mass than the
token sketch but give worse continuation KL at both budgets. Static mass is
not a sufficient quality target here. The experiment does not isolate which
omitted tokens cause the difference. Eight scored logits per prefix do not
establish a robust error rate or safe serving policy. The full CPU compact4
replacement control in the preceding experiment had KL 0.000994 from native
compact4 at 8K, which bounds the observed numerical-path difference in that
control, not the sparse policy's error.

### GQA unions and read granularity

The held-out 8K probe retained selected page IDs for both centroid budgets.
For each KV head, it unions the requests of its two query heads. A hypothetical
physical fetch unit containing `g` consecutive pages is touched when any of
its pages is selected; the last unit is capped at 513 pages. This is a logical
access estimate, not measured disk I/O or latency. A layer-wide page record
across eight KV heads would touch the union over all 16 query heads.

| Budget | Pages/query head | Pages/KV head after union | One-page units | Four-page units | Sixteen-page units | Layer-wide page union |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1/8 | 65.86 | 85.24 | 16.62% | 30.39% | 50.99% | 208.29 pages ( 40.60% ) |
| 1/4 | 129.61 | 162.21 | 31.62% | 51.09% | 72.82% | 334.71 pages ( 65.25% ) |

This favours page chunks scoped by KV head, with the two GQA requests
coalesced before fetch. Wider physical units quickly erase sparse access.
Fractions exclude parameters, exact tail, centroid index, metadata, cache
residency, storage amplification, and index planning. They cannot be read
as speedup or bytes transferred by current shifou.

At four bits, one full 16-token K page for **one** KV head and 128 channels is
1,024 bytes; V is another 1,024 bytes. The earlier 8,192-byte calculation
included all eight KV heads and was incorrectly attributed to one. Four
consecutive K/V page pairs for one KV head fill one 8,192-byte yesno bitmap
chunk. One bitmap chunk per head-specific page would pad each 2,048-byte
payload by a factor of four. Conversely, putting one page across all KV heads
in a chunk couples different heads' requests. The four-page layout matches
the table's four-page fetch unit and can be constructed with public
`BitmapContainer::from_words` and `OrdSet::from_chunks`.
The existing public `Snapshot::key_stream` builds one visible-prefix plan;
its `ChunkStream::seek` skips unselected prefixes without decoding their
payloads. `Snapshot::key_stream_prefix_range` can bound a contiguous run.
`BitmapContainer::try_words` or `copy_words_into` exposes selected dense
chunks. **No new yesno selected-chunk primitive is needed for a first
prototype.** Planning all prefixes still costs metadata work and needs
measurement with a real consumer.

Current shifou `get_packed` calls `Snapshot::load`, reconstructs every payload
byte, checks one SHA-256 over the full payload, then splits buffers. Its K
codes are grouped across the entire prefix by head/channel, so token pages
are not contiguous today. Selective reads require a new shifou page-major
format, partial reconstruction, and page-level integrity verification.
Merely calling `seek` on today's format does not provide page reads.
The follow-up measured that read path below; it remains a research adapter,
not shifou's production format.

The [numeric summary](./kv-page-experiment-20260930.json) records construction,
aggregate metrics, and source result files. The isolated release build and
formatting check passed, and the probe asserted score bounds and selected-mask
counts. No production Rust code was changed.

## Real compact4 page chunks in a reopened yesno DB, 2026-09-30

The same isolated probe added a `page-store` mode, using Qwen3-0.6B's actual
layer-0 K/V prefix at 8,192 tokens and WikiText-2 offset 16,384. It encoded
both tensors with shifou compact4, preserving the first 8,160 tokens' code
nibbles and treating the newest 32 exact tokens as resident outside this DB
payload. Compact4 K codes are channel-major across the old prefix; V codes
are token/head-major. The probe transposed those code nibbles, without
requantizing, into four consecutive 16-token K/V page pairs per KV head per
yesno bitmap chunk. The last chunk for each head pads two absent pages with
zeros. This is the head-scoped four-page fetch unit above, not a one-page
chunk. Eight heads produce 1,024 chunks and 8,388,608 bitmap bytes, compared
with 8,355,840 unpadded source code bytes. The bits are 51.18% one. The
portable Roaring representation is 8,396,820 bytes, 0.49% above the original
code bytes. A 32-byte SHA-256 digest per chunk would add another 32 KiB before
manifest overhead; the benchmark retains those expected digests in memory.

The page IDs came from the real BF16-centroid selection on that same query.
The two query heads per KV head were unioned, then each requested page was
rounded up to its four-page chunk. Layer 0 needs 291 of 1,024 chunks at the
one-eighth budget ( 28.42% of code bytes ), or 493 chunks at one quarter
( 48.14% ). Pages in the exact tail are not stored here. These layer-0
fractions are slightly below the all-layer estimates of 30.39% and 51.09%.

The probe wrote one set with `BitmapContainer::from_words` and
`OrdSet::from_chunks` into a one-shard yesno DB, committed, checkpointed,
closed and reopened it. Seven warm, interleaved rounds timed each read arm;
the table gives medians. Each call resolves its own snapshot chunks,
allocates and copies its output bytes. The selected arms use one
`Snapshot::key_stream`, `seek` to each ascending wanted prefix, and
`try_words` to copy the bitmap bytes. Every verified arm compares its output
with a digest computed from the pre-persistence code. The full ordinal arm
models shifou's `Snapshot::load` plus bit-by-bit `set_to_bytes` and whole
SHA-256, omitting its metadata/sentinel handling.

| Read arm, layer-0 code payload | Median warm ms | Bytes copied |
| --- | ---: | ---: |
| Full ordinal reconstruction + whole SHA-256 | 81.53 | 8.00 MiB |
| Full dense chunk copy, no hash | 0.94 | 8.00 MiB |
| Full dense chunk copy + whole SHA-256 | 15.48 | 8.00 MiB |
| Full dense chunk copy + per-chunk SHA-256 | 15.59 | 8.00 MiB |
| Selected one eighth, no hash | 0.35 | 2.27 MiB |
| Selected one eighth, per-chunk SHA-256 | 4.49 | 2.27 MiB |
| Selected one quarter, no hash | 0.48 | 3.85 MiB |
| Selected one quarter, per-chunk SHA-256 | 7.55 | 3.85 MiB |

Comparing matching per-chunk integrity arms, selection is 3.47 times faster
at one eighth and 2.06 times at one quarter for this warm layer-0 cache read.
The much larger difference from ordinal reconstruction combines two changes:
dense bitmap access removes bit-by-bit materialization, and selection copies
and hashes fewer bytes. SHA-256 dominates the verified dense paths here;
without hashes, the one-eighth selective copy is only about 2.7 times faster
than full dense copy. No cold-cache, GPU, attention, parameter/tail read,
centroid-index read, or end-to-end model latency was measured. Stored hash
lookup and authenticated manifest updates also remain unmeasured. The earlier
eight-logit quality result still makes neither sparse budget a safe default.

One write-side sample took 201.73 ms for compact4 encoding the K/V layer,
42.38 ms to transpose into bitmap chunks, and 2.78 ms to serialize a portable
Roaring view. Commit plus checkpoint varied from about 226 to 871 ms across
otherwise identical runs, so that combined number is not a stable write
latency estimate. The production codec could emit page-major codes directly
and avoid the measured transpose pass; that design has not been implemented.
The [storage result](./kv-page-experiment-20260930.json) points to the full
samples and isolated source. No production code was changed.

## Adaptive page ranking and physical fetch granularity, 2026-10-05

The isolated `kv-page-probe` was extended with three research modes. They use
the same pinned Qwen3-0.6B BF16 query/key captures as the earlier page study:
one query after 2,048 WikiText-2 tokens at offset 4,096 and another after
8,192 tokens at offset 16,384. Each capture has 28 layers and 16 query heads,
or 448 query/head cases, and 16-token KV pages. The last page contains the
new query token, giving 129 or 513 candidate pages. These modes are screening
tests over two contexts, not a trained or validated serving policy.

### Conservative softmax-mass certificates

`pages-certified` tests two previously validated upper bounds on the maximum
query/key score in a page: coordinate-wise key minima/maxima and centroid plus
the maximum Euclidean radius. It bounds the omitted partition mass by the
number of tokens in each omitted page times the exponential of that page's
upper score. The selector includes the first and newest three pages, reads
pages in descending upper-bound order, incorporates the selected pages' exact
scores, and stops when the conservative omitted-mass ratio is at most 1%, 5%,
or 10%. The fixture calculates every exact score afterward to check the
certificate; there were no violations. A suffix-sum accumulator avoids
catastrophic cancellation when very large page bounds are removed.
The fixture computes all exact scores for its oracle and uses their maximum
as a common exponential scale; that scale cancels from the selection ratio.
The selection loop consumes a page's exact mass only when it selects that page.

| Prefix | Bound | Allowed omission | Mean pages selected | Heads selecting every page |
| ---: | --- | ---: | ---: | ---: |
| 2K | Coordinate | 10% | 128.87 / 129 | 434 / 448 |
| 2K | Radius | 10% | 129 / 129 | 448 / 448 |
| 8K | Coordinate | 10% | 510.36 / 513 | 372 / 448 |
| 8K | Radius | 10% | 512.71 / 513 | 437 / 448 |

The 1% and 5% settings select still more pages. These bounds are safe on the
captured keys but too loose for useful page skipping. A certificate based on
these summaries should not drive the first selective-attention implementation.

### Estimated page mass with diagonal variance

`pages-variance` compares a centroid score with a page-mass estimate
`log(page_token_count) + q dot centroid / sqrt(D) + Var(q dot K)/(2D)`.
The variance term uses only per-coordinate key variances, so it ignores
cross-coordinate covariance and provides no bound. It increases a page's
summary by 128 f32 values. Across 28 layers, eight KV heads, and 513 pages,
that extra variance plane alone is 58,834,944 bytes ( 56.11 MiB ), before
centroids or indexing overhead. The selector includes the first and newest three
pages, then takes pages in descending estimated mass until the estimated
cumulative fraction reaches 90%, 95%, or 99%. Exact BF16 softmax mass is the
oracle after selection.

| Prefix | Policy | Target | Mean selected pages | Mean true mass | Lowest true mass | Heads below 90% true mass |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 2K | Centroid | 99% | 72.19 / 129 | 98.74% | 64.74% | 7 / 448 |
| 2K | Variance | 99% | 77.20 / 129 | 99.23% | 92.96% | 0 / 448 |
| 8K | Centroid | 99% | 190.90 / 513 | 98.69% | 83.70% | 4 / 448 |
| 8K | Variance | 95% | 104.03 / 513 | 96.32% | 55.10% | 22 / 448 |
| 8K | Variance | 99% | 210.90 / 513 | 98.95% | 59.08% | 3 / 448 |

The variance term improves average mass, but the estimated target is not a
quality guarantee. One 8K head retained only 59.08% of its true attention
mass at the 99% setting. Calibrating a full-attention fallback from the 2K
context did not reliably identify all weak 8K heads: a cutoff of 98% true
mass on the 2K trace flagged 14 heads, but three 8K heads still fell below
95% true mass. This was a cross-context screening check, not a calibrated
production policy.

`page-quality-variance` used the same adaptive rule during eight actual
Qwen continuation tokens. As in the earlier page-quality probe, the page
index is built from BF16 prefix keys, the KV values are restored from shifou
compact4, and a CPU attention hook replaces the native attention output. The
hook also computes native attention to maintain the cache, so these runs
measure continuation quality, not speed. The table compares against the
full compact4 continuation in the same run.

| Prefix | Policy | Mean pages per query | Mean KL from full compact4 | Mean KL from BF16 |
| ---: | --- | ---: | ---: | ---: |
| 2K | Full compact4 | 129 | 0 | 0.01070 |
| 2K | Variance 95% | 48.96 | 0.00928 | 0.02501 |
| 2K | Variance 99% | 79.89 | 0.00153 | 0.01238 |
| 8K | Full compact4 | 513 | 0 | 0.03288 |
| 8K | Variance 95% | 121.89 | 0.00572 | 0.05056 |
| 8K | Variance 99% | 228.63 | 0.00268 | 0.03631 |

The earlier fixed quarter-page centroid selector used about 130 pages and
had 0.00988 KL from full compact4 at 8K, so the adaptive variance 95% arm
reduced that error while selecting slightly fewer pages. All three 8K
compact4 policies agreed on the top choice for these eight scored tokens;
that small sample does not establish safe generation quality.

### GQA union and storage layout

The `pages-variance` output retained selected page IDs for the 8K adaptive
variance cases. The captured query adds page 512 after the 8,192-token
prefix; the persisted-prefix byte counts below exclude that new page.
Qwen3-0.6B has two query heads per KV head, eight KV heads
per layer, and its current BF16 page store holds all eight KV heads together
in each 16-token K or V tile page. The physical page request is therefore
the union across 16 query heads of a layer. A head-scoped layout would union
only the two query heads sharing that KV head. The table counts exact unions
for the captured query, before any residency subtraction or transfer timing.

| Estimated target | Pages per query head | Pages per KV-head pair | Physical bytes, current layer pages | Physical bytes, head pages |
| ---: | ---: | ---: | ---: | ---: |
| 95% | 104.03 / 513 | 144.45 / 513 | 654.00 MiB ( 73.0% of full ) | 251.04 MiB ( 28.0% ) |
| 99% | 210.90 / 513 | 266.97 / 513 | 835.88 MiB ( 93.3% of full ) | 465.45 MiB ( 51.9% ) |

At this model geometry, one 16-token K+V page for one KV head is exactly
8 KiB: `16 * 128 * 2 bytes * 2 tensors`. It could occupy one yesno bitmap
chunk without padding. This makes head-scoped storage a concrete next layout
experiment. The byte figures assume one read per distinct selected page and
do not include the extra keys, manifests, seeks, hashes, GPU scatter writes,
or serving concurrency that a head-scoped implementation would need. The
current whole-layer layout erases most of the apparent 99% selector saving;
its one-query 6.7% byte reduction does not justify a quality tradeoff by
itself.

Scratch source is in
`../yesno/.agents-workspace/tmp/kv-page-probe-20260930/src/{page,page_quality,main}.rs`.
The result files there start with `result-pages-certified-`,
`result-pages-variance-`, and `result-page-quality-variance-`; the 8K
`-ids.json` file records page masks for the physical-union calculation. The
release probe built and `cargo fmt --all --check` passed. No production
yesno, shifou, or xinfer source was changed for these experiments.

### Persisted head-page layout with real BF16 KV, 2026-10-05

The scratch `head-layout` mode tested the storage implication above against
the same 8,192-token Qwen3-0.6B prefix. It exported all 28 layers' real BF16
K/V tensors and published two representations to one fresh yesno DB. The
layer-wide copy uses 56 keys, one per K or V tile, and four 8 KiB bitmap
chunks per 16-token page. The head-scoped copy uses 224 keys, one per layer
and KV head, and one 8 KiB chunk per 16-token K+V head page. Both represent
896 MiB of exact prefix state. The paired DB occupies about 1.8 GiB of
allocated disk. Every chunk is a bitmap container; the timed readers seek
only requested prefixes, copy bitmap words and check a SHA-256 digest per
8 KiB chunk. An untimed reconstruction of all 56 tiles from the head-scoped
records matched the original source tile hashes exactly.

The 95% and 99% masks are the variance-policy IDs above, unioned across
query heads at each layout's physical granularity. Page 512, created by the
subsequent query token, is excluded from this stored prefix. Six alternating
rounds ran on CPU 16 with a warm OS page cache; the table uses the median of
rounds 1-5, leaving round 0 as warmup. Each result includes key-stream
setup, seeks, copying into an 8 KiB temporary, and hashing, but no GPU writes
or model decoding.

| Requested prefix | Layer-wide bytes | Layer-wide read | Head-scoped bytes | Head-scoped read | Read ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full | 896 MiB | 1,820 ms | 896 MiB | 1,817 ms | 1.00x |
| Variance 95% | 654 MiB | 1,348 ms | 251 MiB | 518 ms | 2.60x |
| Variance 99% | 835.9 MiB | 1,695 ms | 465.4 MiB | 951 ms | 1.78x |

The read gain follows fewer verified bytes; the extra 168 head keys did not
erase it. Individual rounds had large outliers: the maximum full head read
was 6,368 ms and the maximum 99% layer read was 4,072 ms. Five warm samples
per arm establish a useful median comparison, not a tail-latency claim.

The GPU path is a separate constraint. Xinfer's existing attention cache is
`[block, token, head, channel]`. A single head is a strided view, while the
pinned Candle `slice_set` operation requires contiguous source and destination
tensors. Its `slice_assign` alternative creates a full returned tensor.
Therefore the 8 KiB head record cannot yet be written efficiently into a
partly resident xinfer cache with the public in-place API. A small GPU scatter
primitive or a compatible head-major cache layout is required before the
storage-read gain can become a serving result. No selective GPU import or
inference was timed in this follow-up.

The scratch implementation and machine-readable results are
`../yesno/.agents-workspace/tmp/kv-page-probe-20260930/src/head_layout.rs`
and `result-head-layout-8192-16384-20261005.json`. It built in release mode
and `cargo fmt --all --check` passed. No production source changed.

### Head-page GPU scatter, 2026-10-05

A scratch CUDA scatter kernel now imports the persisted 8 KiB head records
into xinfer's existing `[block, token, head, channel]` BF16 cache. It stages
real Qwen3-0.6B KV bytes read through yesno, packs `(page, head)` metadata,
and writes the selected head slices in place. The first implementation staged
and launched by head group; a second stages all selected records for a layer
and launches once per layer (28 launches for this 28-layer model). This uses
Candle's exposed CUDA storage and cudarc/NVRTC in the scratch probe. It is
not a production xinfer API or a serving integration.

Two paired release-mode runs reused the 8,192-token, 896 MiB BF16 fixture
above and the captured variance-policy masks. They were pinned to CPU 16.
Each timed yesno read checks SHA-256 per 8 KiB chunk. GPU time includes host
BF16 conversion, staging tensors, device copies, scatter writes, and a device
synchronization. The table shows the two runs as ranges, in their measured
order. Cache allocation, original model prefill, CUDA compilation, GPU
readback for verification, and model continuation are outside these times.

| Requested prefix | Layout | Read bytes | Yesno read | GPU import | Read + import |
| --- | --- | ---: | ---: | ---: | ---: |
| Full | Head, layer-batched | 896 MiB | 2,057-2,067 ms | 342-345 ms | 2,399-2,412 ms |
| Variance 95% | Current layer pages | 654 MiB | 1,537-1,546 ms | 116-117 ms | 1,653-1,663 ms |
| Variance 95% | Head pages, per-group scatter | 251.04 MiB | 574-580 ms | 637-642 ms | 1,216-1,217 ms |
| Variance 95% | Head pages, layer-batched scatter | 251.04 MiB | 576-580 ms | 157-163 ms | 734-743 ms |
| Variance 99% | Current layer pages | 835.88 MiB | 1,932-1,937 ms | 148-151 ms | 2,083-2,085 ms |
| Variance 99% | Head pages, per-group scatter | 465.45 MiB | 1,051-1,077 ms | 995-1,082 ms | 2,045-2,159 ms |
| Variance 99% | Head pages, layer-batched scatter | 465.45 MiB | 1,069-1,073 ms | 212-219 ms | 1,281-1,292 ms |

For this warm-cache fixture, the batched head path reduced read plus import
by about 2.24-2.25x at the 95% target and 1.61-1.63x at 99% against the
current layer-page path. Batching also cut head import time about 4-5x
relative to the per-group scatter. Every selected GPU head slice was compared
bit for bit against the source cache. Full head restoration gave identical
next-token logits (`max_abs_logit_diff = 0.0`) in both runs. These are
observed fixture timings, not tail-latency or multi-request throughput
measurements.

A partial restored cache cannot yet be passed to xinfer's dense attention
path: unselected pages remain empty and the kernel would attend to them. A
selected-page attention kernel and a sound, encapsulated scatter API in
xinfer are needed to turn this restoration result into end-to-end sparse
inference. The earlier eight-token continuation study estimates selection
quality, but this GPU experiment measures restoration only. It also assumes
no selected page is already resident, so a live cache's incremental transfer
cost would depend on its residency pattern.

The scratch implementation is at
`../yesno/.agents-workspace/tmp/kv-page-probe-20260930/src/head_layout.rs`;
the machine-readable paired runs are
`result-head-gpu-batched-8192-16384-20261005.json` and
`result-head-gpu-batched-8192-16384-20261005-r2.json` in the same scratch
directory. The probe passed `cargo fmt --all --check`. No production source
changed.

The pinned Qwen decode route passes one `InputMetadata` instance through
all layers. Its `block_tables` and `context_lens` are per sequence, and the
attention call passes those unchanged to `flash_attn_with_kvcache_advanced`.
They cannot describe the measured masks, which differ by layer and query
head. A common block-table union would discard much of the head-layout
benefit. A practical next kernel experiment should consume selected page IDs
per layer and query head directly while preserving RoPE's original token
positions, then compare its output and latency against full attention.
