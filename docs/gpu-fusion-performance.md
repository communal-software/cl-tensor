# GPU fusion measurements

GPU fusion reduces complete-call time for multiply-add, squared row sums and nested stable softmax in this run. Allocating and destination calls include synchronous completion; destination calls include staged computation and copying.

## Recorded run

Apple M1, SBCL 2.6.8, Metal, f32, 256×256 inputs, 2026-10-18. Broadcast multiply-add uses a 256-element right operand; other multiply-add cases use scalar 2. Reversed inputs reverse the inner axis. All cases validate against independent scalar references before warmed timing.[^measurement]

| Operation/layout | Composed allocate µs | Fused allocate µs | Composed out µs | Fused out µs |
|---|---:|---:|---:|---:|
| Multiply-add, contiguous | 787.7 | 388.4 | 1,140.7 | 732.9 |
| Multiply-add, reversed | 781.8 | 385.9 | 1,138.0 | 736.9 |
| Multiply-add, broadcast | 791.1 | 393.6 | 1,148.4 | 742.5 |
| Squared row sums | 699.5 | 475.8 | 926.4 | 755.1 |
| Nested stable softmax | 1,720.0 | 959.1 | 2,061.5 | 1,317.0 |

The [retained output](benchmark-runs/2026-10-18-gpu-fusion-sbcl.txt) includes first-call timings and Lisp allocation counters. These counters exclude device memory; their granularity can report zero for small batches.

## Reproduce

```sh
sbcl --non-interactive --load tests/gpu-fusion-bench.lisp
```

Inputs and destinations are preallocated outside timing. Allocating measurements include result release; destination measurements reuse their destination. Composed operations use the public tensor interfaces with the same completion guarantees. Softmax computes maximum, subtraction, exponentials, sum and division; fusion uses dependent GPU passes.

## Limitations

- Full-shape reduction operands and conditional demand masks retain temporary GPU storage: [upstream #83](https://github.com/communal-software/cl-gpu/issues/83).
- Calls synchronize and destination forms stage full results: [#22](https://github.com/communal-software/cl-tensor/issues/22).
- GPU plan lookup takes a short shared lock; concurrent-call contention is tracked in [#31](https://github.com/communal-software/cl-tensor/issues/31).
- These measurements describe one shape and machine; they establish no universal speed guarantee or device-memory bound.

[^measurement]: Warm timing is the median of three ten-call averages after three warm-up calls. First-call allocating timings stop before validation and result release. First-call timing includes lazy kernel preparation, but pipelines can be shared with earlier cases; named allocating/out functions share plans. Lisp allocation is the SBCL byte-counter difference across ten additional complete calls. Transfers, reference validation, input construction and output allocation for destination forms are excluded. Timer and allocation-counter granularity contribute variation.
