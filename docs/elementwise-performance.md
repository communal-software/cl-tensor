# Elementwise measurements

Native strided elementwise calls traverse storage directly. These measurements include tensor broadcasting, validation and dispatch; inputs and outputs are preallocated.

## Recorded run

SBCL 2.6.8, Apple M1, native backend, `:f32` addition, 2026-10-06. Square views use side lengths 32 and 256. The comparison dispatcher is from commit `760d455`.[^measurement]

| Elements/layout | Comparison µs/call | N-D µs/call | Comparison bytes/call | N-D bytes/call |
|---|---:|---:|---:|---:|
| 1,024 contiguous | 1.48 | 1.88 | 1,228 | 1,677 |
| 65,536 contiguous | 10.51 | 10.42 | 1,212 | 1,654 |
| 65,536 transposed input | 276.57 | 89.81 | 507,926 | 1,720 |
| 65,536 stride-two input | 255.81 | 88.23 | 507,926 | 1,654 |
| 65,536 reversed input | 251.28 | 87.80 | 507,926 | 1,671 |
| 65,536 broadcast column | 204.35 | 16.42 | 192,860 | 1,720 |

N-D metadata contributes a fixed setup cost to short calls. Ordinary stride traversal allocates no element buffers; alias snapshots and irregular output validation have separate [workspace limitations](elementwise.md#limitations).

## Reproduce

```sh
sbcl --non-interactive --load tests/elementwise-bench.lisp
git show 760d455:elementwise.lisp > /tmp/cl-tensor-elementwise-comparison.lisp
CT_ELEMENTWISE_BASELINE=/tmp/cl-tensor-elementwise-comparison.lisp \
  sbcl --non-interactive --load tests/elementwise-bench.lisp
```

`CT_ELEMENTWISE_BASELINE` loads a comparison implementation into the benchmark process. Use a fresh process for each run. The script reports contiguous, transposed, stride-two, reversed and broadcast inputs for both sizes. Allocation reporting uses SBCL's byte counter.

## Limitations

Small-call layout setup remains a tuning opportunity: trivial-simd #152.

[^measurement]: Timings are wall-clock averages after 20 warm-up calls: 20,000 iterations at 1,024 elements and 4,000 at 65,536. A full GC precedes each SBCL measurement; bytes/call excludes data construction. Timing and allocation counter granularity cause small variation between cases. Results describe this machine and workload, rather than a cross-platform speed guarantee.
