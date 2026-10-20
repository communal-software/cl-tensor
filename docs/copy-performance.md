# Bulk tensor copying performance

Bulk copying reduces complete Qwen3 0.6B CPU inference time in all three measured pairs. F16 prefill takes about one third less time; Q8_0 prefill takes about one fifth less time. These are measurements on one M1 under background load, not isolated hardware limits; see [Limitations](#limitations).

## Results

128-token prompt, serial execution, one native thread. Times are medians across three fresh processes per variant. Positive paired reductions mean less time.[^paired]

| Weights | Phase | Generic copy (ms) | Bulk copy (ms) | Median paired reduction |
|---|---|---:|---:|---:|
| F16 | prefill | 3,598.79 | 2,392.05 | 33.5% |
| F16 | decode | 151.48 | 137.53 | 8.7% |
| F16 | generate | 4,658.67 | 2,965.61 | 36.7% |
| Q8_0 | prefill | 5,613.49 | 4,488.33 | 20.0% |
| Q8_0 | decode | 183.26 | 171.91 | 6.3% |
| Q8_0 | generate | 6,813.12 | 5,182.44 | 24.3% |

The first and third F16 pairs agree: prefill reductions are 33.5% and 33.1%; generation reductions are 35.9% and 36.7%. Q8_0 prefill reductions range from 20.0% to 20.7%, and generation from 23.1% to 24.5%. All six cases improve in every pair; no samples are discarded.

## Copy path

Compatible independent simple vectors copy their shared contiguous inner runs with Common Lisp `replace`. Whole contiguous tensors use one run; padded, broadcast or reversed outer dimensions use multiple runs. The path supports all built-in vector-backed numeric storage types and preserves F16/BF16 bit patterns.

Shared storage retains scalar traversal order. Noncontiguous inner layouts, foreign storage and opaque storage retain their existing copying hooks. See [copying and a padded-view example](constructors.md#access-and-copy).

## Method and reproduction

The [exact harnesses, runtime source, logs, environment observations and hashes](benchmark-runs/2026-10-20-bulk-copy/) preserve the measurement. Before/after order alternates across three pairs. Both variants load the same runtime patch; the generic variant disables only the internal bulk-copy helper by replacing it with a function returning `nil`. This baseline includes one additional helper call per copy relative to the original runtime.

Each timing uses one warmup and the median of three calls, with full collection before each call. Model loading, pinned model SHA checks, tokenizer work and expected-token preparation are untimed. Prefill has no KV cache. Decode overwrites one position after a prepared prefix. Generation resets its session and includes prefill plus four output tokens; every measured generation returns `(9625 374 785 6722)` and finishes at the requested length.

The archived runner writes 36 validated timing rows. On the measured checkout, copy `copy-inference-bench.lisp` to `/tmp/`, then run the archived `run-copy-inference.py`. Both files retain the measured machine's checkout paths. The metadata identifies runtime revisions plus the exact modified `access.lisp`; the revision alone does not identify the measured tensor source.

## Validation

The full SBCL tensor suite passes 136,532 checks; the full local inference suite passes 39,739 checks. CCL 1.13 passes 302 copy checks, covering vector layouts, encoded bits, aliases and foreign storage. Validation logs and exact test launchers accompany the timing evidence.

## Limitations

Two unrelated CPU-bound jobs remain active. Additional compiler jobs interfere with the second pair's F16 baseline, where prefill takes 5.92 s rather than approximately 3.6 s. That pair reports 53–63% reductions and remains in the raw results. The median paired summary and the first/third pair comparison avoid presenting this outlier as the expected gain.

This is one Apple M1 with 8 GiB RAM, macOS 15.7.5 and SBCL 2.6.8. Memory and process snapshots cover complete processes, including loading and preparation; they do not isolate each timed call. The comparison covers Qwen at 128 tokens with automatic dispatch, rather than other models, prompt lengths, GPUs, executors or native-default threading.

[^paired]: Each pair reports `100 × (1 − bulk/generic)`. The table uses the median of three paired reductions. Its value can differ from the ratio of the independent timing medians. Full per-pair reductions remain in `summary.json`.
