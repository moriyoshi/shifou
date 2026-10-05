# Calibrated affine KV precision, 2026-09-29

## Construction

Use the pinned Qwen3-0.6B BF16 adapter and WikiText-2 test corpus in
xinfer-evaluation.md. The shifou Cache::estimate and
Cache::estimate_encoded methods share the same record preparation code as
Cache::put. Their size includes the portable Roaring payload and metadata
for the exact address, without writing a trial database.

Train on token offset 0, prefix length 128, and eight continuation tokens.
For each of 28 layers and K/V slots, change only that tile to a candidate
decoded under maximum f32 absolute error 0, 0.05, 0.20 or 0.50. Keep every
other prefix tile BF16. Measure mean KL(reference || candidate) from the
next-token distributions. Group size is 64 with the usual 16-token exact
tail and 2% clipping candidate.

At held-out token offset 4096, evaluate prefixes of length 128 and 512,
each with 32 teacher-forced scored tokens. The current prefix determines
each option's exact encoded size; its continuation and labels are not used
to select precision. Set the budget to the sum of complete record sizes
under uniform error 0.20. Begin with the cheapest option per tile and buy
the largest calibrated KL reduction per additional byte until no further
affordable improvement is predicted. This greedy rule is deterministic but
has no global-optimality or additive-quality guarantee.

Persist all selected tiles, close and reopen the database, compare f32
reconstructions bit for bit, then compare all 32 logit vectors and 16
greedy tokens against the pre-persistence selected cache. Every control
passed. The held-out uniform comparison runs in the same process with
the same model, prefix, tokenizer, and continuation.

## Results

The full sensitivity matrix, selected policies and measured rows are in
adaptive-results.json. For 128 tokens, adaptive used 16,254,281 bytes versus
16,254,968 uniform; perplexity was 38.997 versus 39.098, while KL increased
from 0.01442 to 0.01629 and top-1 agreement fell from 100% to 90.63%.
For 512 tokens, adaptive used 41,520,261 versus 41,572,621 bytes; perplexity
was 64.552 versus 64.732, KL rose from 0.01015 to 0.01088, and agreement
fell from 93.75% to 90.63%. Across 64 scored tokens, perplexity was 50.173
versus 50.308, but greedy matches were 21/32 versus 30/32. These small
held-out samples do not establish a robust quality improvement.

The proxy sum of isolated KL scores improved in both cases (about 0.0495
versus 0.0591), whereas actual joint KL worsened. Independent tile effects
did not add reliably. A future selector should evaluate joint policy changes
on separate calibration prefixes and impose KL or ranking guardrails. It must
keep held-out continuation logits out of policy selection.

## Joint-calibrated follow-up

The second experiment begins with the isolated-score policy, then compares
its actual whole-cache KL on the same eight-token calibration continuation
with the uniform policy. It accepts the better one. Up to 40 affordable
single or paired tile changes are shortlisted by isolated scores in each
of two rounds; a change is accepted only when whole-cache calibration KL
improves. Exactly 82 joint policy trials were scored per held-out prefix.
The same prefix data determine exact candidate sizes, and held-out suffix
logits remain unseen while selecting.

Calibration KL improved from 0.00897 under uniform error 0.20 to 0.00490 for
the 128-token held-out budget and 0.00542 for the 512-token budget. Both
selected records fit below their uniform budgets and survived persistence
with unchanged logits. Held-out KL still worsened: 0.01784 versus 0.01442
at 128 tokens, and 0.01033 versus 0.01015 at 512. Perplexity worsened from
39.098 to 39.335 at 128 and improved from 64.732 to 63.629 at 512. Top-1
agreement fell from 100% to 90.63% and from 93.75% to 90.63%, respectively.
The full selected policies and measurements are in adaptive-joint-results.json.

This follow-up rules out the simple explanation that summing isolated scores
was the only problem. Searching 82 policies against the same eight tokens
likely overfits the calibration context. Multiple disjoint calibration
prefixes plus a separate validation set are a better next experiment than
more search on this one continuation. No held-out result was used to choose
or revise a policy in these experiments.

## Cost and limits

Sensitivity calibration took 25.06 seconds; preparing candidate encodings
and exact costs took 15.15 seconds for the 128-token prefix and 67.70
seconds for 512. Persistence and read timings are in adaptive-results.json.
These are offline research costs, not serving latency estimates. The two
held-out prefixes overlap and both come from the same corpus as calibration.
The codec's f32 absolute error is not a guarantee for logits, KL, ranking,
or BF16 rounding.

The layerwise selection applies to the generic affine codec. The pinned
xinfer TurboQuant mode is process-global and does not support independent
TurboQuant modes per layer. No study code was added to yesno-core.
