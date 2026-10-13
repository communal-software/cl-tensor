# Fusion and executor measurements

Fusion avoids intermediate expression tensors and reduces allocation in these workloads. Complete-call timing depends on layout and kernel setup. A three-worker executor reduces elapsed time for independent matmul rows and reduction groups on this machine.

## Recorded run

Apple M1, SBCL 2.6.8, native backend, f32, 2026-10-13. Multiply-add and softmax use 256×256 tensors; row sums use 2,048×256; matmul multiplies two 256×256 matrices. Reversed inputs reverse the inner axis; broadcast multiply-add uses a 256-element right operand. Inputs are preallocated; each measured operation allocates its result.[^measurement]

| Multiply-add layout | Composed µs | Fused µs | Composed bytes | Fused bytes |
|---|---:|---:|---:|---:|
| Contiguous | 64.3 | 74.7 | 524,320 | 305,210 |
| Reversed | 174.1 | 3,230.0 | 524,320 | 334,456 |
| Broadcast rows | 97.5 | 205.1 | 524,320 | 412,858 |

| Operation | Sequential µs | Three workers µs | Sequential bytes | Three workers bytes |
|---|---:|---:|---:|---:|
| Fused nested softmax | 993.6 | 424.5 | 557,006 | 537,338 |
| Matmul | 524.5 | 275.8 | 262,160 | 262,160 |
| Row sums | 91.2 | 80.3 | 6,446 | 6,312 |

Composed softmax takes 343.2 µs and allocates 793,034 bytes per call. It computes maximum, exponentials and sums through existing tensor operations. Fused softmax uses dependent kernel passes per row, with fewer tensor allocations and more setup work.

Contiguous fusion saves approximately 42% of multiply-add allocation but is slightly slower in this run. Reversed fusion saves allocation and pays for Lisp scalar packing. Compatible broadcast rows use upstream batches, which still set up elementwise kernels per row.

The [retained output](benchmark-runs/2026-10-13-fusion-execution-native.txt) includes first-call timings and warmed measurements.

## Reproduce

```sh
sbcl --non-interactive --load tests/fusion-execution-bench.lisp
```

The benchmark validates every case against a reference before measuring warmed calls. Its reusable worker-pool fixture is loaded from the test system; the library itself has no thread-library dependency. Pool creation and input construction are excluded.

## Limitations

- Full-result staging and complete-row packing remain tracked in [#13](https://github.com/communal-software/cl-tensor/issues/13).
- Strided packing and per-row/pass setup remain tracked in [#14](https://github.com/communal-software/cl-tensor/issues/14).
- Results describe these shapes, this backend and this machine. They do not establish a universal crossover or speed guarantee. Select executor thresholds using application workloads.

[^measurement]: Warm timing is the median of three ten-call averages after three warm-up calls. SBCL allocation is the byte-counter difference over ten calls after a full GC, including allocations in worker threads. The first-call column measures the first invocation of each benchmark case; compilation is excluded, and kernel initialization can be shared with earlier cases. Timer and allocation-counter granularity contribute variation.
