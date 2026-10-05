# Persisting a 56-tile cache: batching, checkpoints, atomicity, 2026-09-29

## Question

`Cache::put_packed` commits, waits for visibility and **checkpoints** once per
tile, and a Qwen3-0.6B cache is 56 tiles. The recorded `compact-onebit` run
measured `persist_ms` of 1375-1495 at a 128-token prefix and 1699-2038 at 512.

An inference was offered first, and it is recorded because it was wrong in an
instructive way. Fitting `time = fixed + k * bytes` to two points, using the real
payload growth ( 7.0 -> 17.83 MB rather than the 4x token ratio ), puts a fixed
term near 1.2 s. That is **consistent with** 56 fixed barriers and does not
establish it: the same timer covers staging, a per-tile collision snapshot, a
SHA-256 over each payload and the Roaring encode, all per-tile and none an fsync.
Two observations cannot separate two affine terms.

## Real-model result

The measurement that matters, because it is the actual path with the pinned
Qwen3-0.6B model and WikiText-2 corpus. `persist_ms` for the same policies,
before and after `onebit::persist` was changed to one `put_packed_many`:

```text
  prefix 128     before      after
    k2v1         1375.3 ms   267.3 ms
    adaptive_mid 1379.5      276.6
    adaptive_u2  1433.7      248.8
    sparse_onebit 1494.9     339.9

  prefix 512
    k2v1         1698.5      435.4
    adaptive_mid 1805.5      423.3
    adaptive_u2  1774.1      490.7
    sparse_onebit 2037.8     733.1
```

Between 2.8x and 5.8x, in the same direction in all eight policy-prefix
combinations. **The two sides are different runs on a shared machine**, so
individual ratios are approximate; eight independent agreements are what make the
direction and the rough magnitude credible, not any single pair.

Correctness is the run's own assertion, not a separate check.
`onebit::persist` requires `stored_bytes` to equal the pre-computed estimate, and
requires the reopened `logits` and greedy tokens to equal the pre-persistence
ones exactly. The run exits non-zero if either differs. It exited zero.

## Synthetic decomposition, and what it does not support

A four-arm instrument, since removed to `.agents-workspace/tmp/write-strategies`,
wrote 56 synthetic tiles at ~50% bit density into a fresh database per arm:

1. per-tile checkpoint -- what `put_packed` did
2. no **explicit** checkpoint -- the same loop with the call removed
3. one final checkpoint -- arm 2 plus a single checkpoint
4. one batch -- `put_packed_many`

Arm 1 minus arm 2 bounds what the explicit checkpoints cost. Arm 3 minus arm 4
bounds what the per-tile commits, waits and collision snapshots cost.

```text
  56 tiles, 6.68 MB logical        56 tiles, 17.00 MB logical
    1  per-tile checkpoint  1.545 s   2.402 s
    2  no explicit ckpt     0.540     0.961
    3  one final ckpt       0.523     1.002
    4  one batch            0.331     0.735
```

**One run per arm, on a host at load average 31 from an unrelated process.** The
same public-API arms repeated five times spanned 1.44-2.16 s and 0.32-0.83 s at
the smaller size -- a 1.7x spread within one arm. So these numbers support the
**ordering** and nothing finer. In particular they do not support a percentage
split, a per-checkpoint millisecond figure, or the claim that the remaining
per-tile cost is fixed with respect to payload size; an earlier draft asserted
all three and none of them follows from one run each.

Three further limits on arm 2, which an earlier draft ignored:

* **It is not "no checkpoints".** yesno's `CheckpointPolicy` fires on dirty
  bytes, WAL bytes and elapsed time, so the engine may have checkpointed during
  that arm on its own. Arm 1 minus arm 2 is therefore the cost of the *explicit*
  per-tile calls above whatever the policy did, not the cost of checkpointing.
* **The arms are sequential**, each hundreds of milliseconds long, on a machine
  whose load varied. They are not contemporaneous and a within-run comparison is
  not privileged over a between-run one here.
* Arm 3 came out below arm 2 at the smaller size and above it at the larger.
  That inversion is the noise floor showing itself, and it is the reason "a final
  checkpoint is free" is not claimed.

## Atomicity, which is not a timing result and does not depend on the above

56 commits are 56 durable points, so an interrupted write leaves a cache that is
**partly** persisted -- and nothing can detect it, because every tile is
individually well formed and carries its own manifest. A reader finding 30 of 56
sees 30 valid tiles, not a torn write.

* `per_tile_writes_can_leave_a_partial_cache` pins that as the old behaviour.
* `a_rejected_batch_writes_nothing_at_all` pins the opposite for
  `put_packed_many`: refused at the 40th tile, none of the 39 already prepared is
  written.

This is the argument for batching that survives every caveat above. The speed is
recoverable by dropping checkpoints alone; all-or-nothing is not.

## Duplicate addresses in one batch

Refused, not merged and not last-wins. `store_set` emits a `DeleteKey` before its
chunks, so a repeated key would erase the earlier entry while `put_packed_many`
still returned a report for both -- telling the caller 56 tiles were stored when
55 were. Measured before the check existed: `Ok` with two reports, and only the
second tile in the database.

The same preflight map closes a hole batching would otherwise have opened.
`put_packed` catches two distinct addresses landing on one 63-bit key, because
the first write is visible when the second is checked; inside a batch neither is
stored yet, so both would pass and one would silently overwrite the other. That
branch cannot be reached from a test -- a first attempt searched 200 000
candidate addresses for a key collision, found none, and passed green having
checked nothing -- so the decision is extracted into `repeated_key` and asserted
on its inputs instead.

## Recommendation

1. `put_packed_many` for the whole cache. Now wired into `onebit::persist`, with
   the real-model numbers above.
2. If a per-tile path is kept, stop checkpointing in it. The synthetic arms say
   that is the larger part of its cost; the exact share is not established here.
3. Re-measure on a quiet host before any synthetic figure is quoted.

Dense-path work is deliberately not attempted. The direct bitmap APIs are public
in yesno-core -- `BitmapContainer::from_words`, `OrdSet::from_chunks`,
`copy_words_into` -- so it stays shifou-side whenever it is taken up.

## A free change taken first

`bytes_to_set` emitted strictly ascending ordinals and passed them to
`OrdSet::from_iter_unsorted`, which collects, sorts, dedups and then delegates to
`from_sorted_slice`. It was sorting sorted data.
`the_ordinals_are_already_sorted_and_unique` asserts the precondition rather than
inferring it from a round trip, because a round trip cannot tell "sorted" from
"sorted by someone else": `from_iter_unsorted` returns the same answer either
way, so a contents test would pass whether or not the substitution was sound.

## Reproduction

Real model: build `experiments/xinfer` in release and run mode `compact-onebit`
with the pinned model and corpus, into a **fresh** output directory, exactly as
`compact-onebit.md` describes. Synthetic: `cargo run --release` in
`.agents-workspace/tmp/write-strategies`, which measures the two public-API arms;
the four-arm decomposition needed crate-internal access and its construction is
described above.
