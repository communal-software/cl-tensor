# Testing

The suite uses FiveAM.

```sh
sbcl --non-interactive --eval '(require :asdf)' --eval '(asdf:test-system :cl-tensor)'
```

`asdf:test-system` loads `cl-tensor/tests` and signals an error if any check fails. From a REPL, `(cl-tensor/tests:run-tests)` returns true when every check passes.

| File | Covers |
|---|---|
| `tests/tensor.lisp` | Dtype inference, construction, view validation, indexing |
| `tests/constructors.lisp` | Filled tensors, nested data, sequences, identity matrices, access, copying and foreign storage |
| `tests/elementwise.lisp` | Arithmetic, exp/sin/cos, SiLU and tanh-GELU against scalar and composed references, masks, conversion, dtype errors, broadcast/strided views, aliases across axes, and one N-D dispatch per operation |
| `tests/shape.lisp` | View sharing, reshape copying, slice bounds, joins/splits, gather shapes/dtypes/bounds, strided and foreign aliases, and seeded coordinate references |
| `tests/reductions.lisp` | All reducers, dtypes, axes, widening, empty groups, aliases, foreign storage and seeded enumeration |
| `tests/normalization.lisp` | Softmax, weighted RMSNorm, axis sets, widened sums, empty inputs, arbitrary layouts, aliases and seeded scalar references |
| `tests/matmul.lisp` | Encoded/f32 products in either order, vector/matrix/batch shapes, broadcast/reversed/padded layouts, aliases, validation, direct foreign weights without packing, interleaved output guards, storage sharing, dispatch counts and seeded reference products |
| `tests/fusion.lisp` | Inline/named kernels, both precisions, nested reductions, axis sets, broadcasting/strides, foreign/custom aliases, native lazy selection and dispatch, selected-branch errors, failure staging, buffer bounds, tensor allocation counts and seeded coordinate references |
| `tests/execution.lisp` | Reversed/uneven ranges, reusable worker pool, backend/FMA bindings, completion after errors, concurrent/cold kernels, encoded/foreign/custom storage, aliases and selector ownership |
| `tests/extensions.lisp` | External CLOS descriptors, operation tables, opaque blocks, packed gather, mixed matmul, conversion, bounded fallback buffers, safe direct selectors, validation, hidden aliases and failure atomicity |
| `tests/random.lisp` | Seeded random shapes, strides, broadcasting and output uniqueness checked against independent Lisp calculations[^seed] |

## Backends and examples

The default run uses the selected trivial-simd backend. To check the portable Lisp backend:[^backend]

```sh
sbcl --non-interactive --eval '(require :asdf)' \
  --eval '(asdf:load-system :cl-tensor/tests)' \
  --eval '(let ((trivial-simd::*backend* :lisp)) (asdf:test-system :cl-tensor))'
```

Run the constructor, broadcasting, activation, normalization, shape, embedding lookup, reduction and matmul examples:

```sh
sbcl --non-interactive --load examples/tensors.lisp
sbcl --non-interactive --load examples/fusion.lisp
sbcl --non-interactive --load examples/executor.lisp
```

Run the external dtype/storage example:

```lisp
(load "examples/extensions.lisp")
(cl-tensor/extension-example:run-example)
```

Check normal and forced ASDF test loading for recursive-operation warnings in a fresh process:

```sh
sbcl --non-interactive --load tests/asdf-loading.lisp
```

See [extensions](extensions.md) for its public protocol, [dispatch measurements](extensions-performance.md) for built-in regression controls, and [staging measurements](extensions-staging-performance.md) for direct execution and custom-storage packing.

Measure elementwise time and, on SBCL, allocation:

```sh
sbcl --non-interactive --load tests/elementwise-bench.lisp
```

See [elementwise measurements](elementwise-performance.md) for layouts, recorded results and baseline comparison.

Compare complete normalization calls using typed Lisp and existing kernels:

```sh
sbcl --non-interactive --load tests/normalization-bench.lisp
```

See [normalization measurements](normalization-performance.md) for validation, path selection and retained results.

Measure interleaved matrix and batch destinations, complete-call time and Lisp allocation:

```sh
sbcl --dynamic-space-size 4096 --non-interactive --load tests/matmul-bench.lisp
```

See [matmul measurements](matmul-performance.md) for layouts, baseline comparison and retained results.

Measure custom-storage fallback and opaque executor staging:

```sh
sbcl --non-interactive --load tests/extensions-bench.lisp
```

Measure composed/fused expressions and sequential/three-worker operations:

```sh
sbcl --non-interactive --load tests/fusion-execution-bench.lisp
```

See [fusion and executor measurements](fusion-performance.md) for complete-call timings, allocation and retained output. Worker-pool fixtures use Bordeaux Threads in the test system; the core system has no thread-library dependency.

## Device tests

GPU storage tests are optional; the core suite needs no cl-gpu dependency.

```sh
sbcl --non-interactive --eval '(require :asdf)' \
  --eval '(asdf:test-system "cl-tensor/gpu")'
```

The mock-device suite checks CPU allocation, specialized bulk transfers, signed 64-bit storage, reversed/broadcast/foreign uploads, buffer windows, ownership and error cleanup. See [device tensors](device.md).

## Validation

SBCL 2.6.8 and ECL 26.5.5 execute the full suite and the tensor, extension, fusion and executor examples. Recorded run, 2026-10-17:

| Implementation/backend | Passing checks |
|---|---:|
| SBCL default/native | 136,268 |
| SBCL Lisp | 136,268 |
| ECL | 137,360 |
| Experimental ARM64 CCL (2026-10-07 run) | 31,358; direct selectors, fusion and executors are untested, subject to the [CCL limitation](#limitations) |

Normal and forced ASDF loading pass in a fresh SBCL process. The forced rerun passes 136,249 checks with the random state advanced from the initial run.[^seed]

The extension checks cover public-interface-only external packages, opaque copying and writes, conversion destination selection, bounded elementwise buffers, malformed direct selectors, hidden aliases, validation before execution, and unchanged destinations on executor or late input-read failures. SBCL also injects a second-chunk kernel failure.[^injection] See [dispatch measurements](extensions-performance.md) and [staging measurements](extensions-staging-performance.md) for timing and allocation.

Elementwise math checks cover exp/sin/cos and SiLU/tanh-GELU in both precisions,
including scalar and composed references, broadcast/reversed layouts, foreign
aliases, custom storage and one upstream N-D dispatch per operation.

Gather checks cover scalar and batched token indices, negative and out-of-range indices, all built-in dtypes, raw encoded-float bits, signed and zero strides, foreign aliases, packed copies, hidden custom-storage aliases and unchanged destinations on gather failures.

## Limitations

- The experimental ARM64 CCL `1.13 (v1.13-459-g690ff7ea)` build can crash or end compiled normalization reference checks prematurely while reporting success. Its green summary does not establish complete normalization coverage: [#9](https://github.com/communal-software/cl-tensor/issues/9). SBCL native/Lisp and ECL execute the full checks.

[^seed]: SBCL runs use a fixed seed, so failures reproduce. Other Lisps seed from the clock.
[^backend]: The backend binding is an upstream test hook, not a cl-tensor public API. Use `:native` only when the native library is available; ARM64 does not provide the `:sbcl` SIMD backend.

[^injection]: ECL compiled calls bypass function-cell replacement, so the injected kernel-failure check runs on SBCL. The input-read and executor-failure checks run on both implementations.
