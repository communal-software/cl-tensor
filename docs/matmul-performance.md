# Matmul performance

Interleaved 256-product batches measure 3.59–3.75× faster with portable Lisp and 4.27–4.67× faster with native GEMM on this Apple M1. Tiny single matrices pay extra arithmetic-validation time; the contiguous batch control stays close to baseline.[^comparison]

## Layouts

Inputs contain 0.5 and 0.25, and every result equals `K / 8`. Output holes contain −7 and remain unchanged. Arrays are initialized outside timings.

| Case | Output shape | Output strides | K |
|---|---|---|---|
| Matrix 3×2 | `(3 2)` | `(2 3)` | 4 |
| Matrix 9×8 | `(9 8)` | `(8 9)` | 8 |
| Interleaved matrices | `(256 3 2)` | `(8 2 3)` | 4 |
| Interleaved batches | `(256 2 2)` | `(4 3 2)` | 4 |
| Contiguous control | `(256 3 2)` | `(6 2 1)` | 4 |

The interleaved-matrix case exercises direct destinations that fail the sorted-stride matrix proof. The interleaved-batch case exercises matrices that pass that proof individually but fail the combined batch proof.

## Complete-call timings

Native GEMM uses a threshold of one, so these small products exercise native dispatch. Values are medians across three fresh processes, in microseconds per complete `matmul!` call.[^timing]

| Native case | Dtype | Baseline µs | Direct µs | Speedup |
|---|---|---:|---:|---:|
| Matrix 3×2 | f64 | 3.60 | 3.85 | 0.93× |
| Matrix 9×8 | f64 | 8.59 | 7.26 | 1.18× |
| Interleaved matrices | f32 | 554.41 | 129.69 | 4.27× |
| Interleaved matrices | f64 | 549.23 | 128.38 | 4.28× |
| Interleaved batches | f32 | 482.87 | 103.40 | 4.67× |
| Interleaved batches | f64 | 480.70 | 103.51 | 4.64× |
| Contiguous control | f64 | 23.23 | 23.08 | 1.01× |

| Portable Lisp case | Dtype | Baseline µs | Direct µs | Speedup |
|---|---|---:|---:|---:|
| Interleaved matrices | f32 | 493.14 | 137.38 | 3.59× |
| Interleaved matrices | f64 | 495.58 | 137.10 | 3.61× |
| Interleaved batches | f32 | 411.91 | 111.41 | 3.70× |
| Interleaved batches | f64 | 410.32 | 109.29 | 3.75× |

SBCL reports native-path Lisp heap allocation of about 480 KB → 141 KB per interleaved-matrix batch and 428 KB → 106 KB per interleaved-batch call for f64. These totals include cl-tensor destination validation, which still uses a hash table for these layouts; they exclude native packing and stack workspace.

The proof-only check for shapes `(2 10 10)` and `(2 1000000 1000000)` measures about 1 µs per call and zero reported Lisp heap bytes in both cases. It validates metadata without allocating backing tensors. This checks workspace scaling for these fixtures, not a runtime bound for arbitrary layouts.

## Reproduce

```sh
sbcl --dynamic-space-size 4096 --non-interactive --load tests/matmul-bench.lisp
```

To compare source revisions, export their GEMM and matmul implementations and supply both paths:

```sh
git show cb1a96b:matmul.lisp > /tmp/matmul-baseline.lisp
git -C ../trivial-simd show 5cf7dcd:blas/level3.lisp > /tmp/gemm-baseline.lisp
CL_TENSOR_MATMUL_BASELINE=/tmp/matmul-baseline.lisp \
TRIVIAL_SIMD_GEMM_BASELINE=/tmp/gemm-baseline.lisp \
sbcl --dynamic-space-size 4096 --non-interactive --load tests/matmul-bench.lisp
```

The benchmark restores the current implementations after the baseline measurements. Retained samples: [run 1](benchmark-runs/2026-10-07-matmul-1.txt), [run 2](benchmark-runs/2026-10-07-matmul-2.txt), [run 3](benchmark-runs/2026-10-07-matmul-3.txt).

## Limitations

- Measurements describe ARM64 SBCL 2.6.8 on one Apple M1; they are not timing acceptance thresholds.
- General destination validation and conservative snapshots remain covered by the [elementwise workspace limitations](elementwise.md#limitations).
- Exact GEMM validation has bounded workspace but can take substantial search time on difficult layouts: trivial-simd #154.

[^comparison]: The baseline sources are cl-tensor `cb1a96b` and trivial-simd `5cf7dcd`. Both phases use the same installed native library; the changes are Lisp validation and dispatch. Portable and native backends run separately for f32 and f64. A speedup below one means a slower call.
[^timing]: Each case checks values and guards, warms up five calls, calibrates until a batch lasts at least 50 ms, and takes the median of three further batches. The table takes the median across three serial fresh-process runs. Compilation is excluded; outputs are reused. Heap allocation uses a separate sample of up to 100 calls after full GC.
