# Testing

The suite uses FiveAM.

```sh
sbcl --non-interactive --eval '(require :asdf)' --eval '(asdf:test-system :cl-tensor)'
```

`asdf:test-system` loads `cl-tensor/tests` and signals an error if any check fails. From a REPL, `(cl-tensor/tests:run-tests)` returns true when every check passes.

## Continuous integration

GitHub Actions runs the CPU FiveAM suite. GPU tests and benchmarks are separate local checks.

| Trigger | Platforms | Lisps | Backends |
|---|---|---|---|
| Push to `trunk`, pull request targeting `trunk` | Linux x86-64 | SBCL | Native, Lisp |
| Manual run, any tag push | Linux x86-64, Linux ARM64, macOS ARM64 | SBCL, ECL | Native, Lisp |

Changes confined to `docs/**`, Markdown files or `LICENSE` skip ordinary CI. Mixed code/documentation changes run CI. Tags and manual runs always select the full twelve-job matrix. A newer run cancels an older run for the same pull request or ref.[^ci-filter]

Choose **Actions → CI → Run workflow** to run the full matrix on a selected branch or tag.

CI builds the pinned `trivial-simd` shared library before loading tests and verifies the requested backend and source directories. Its logs include the dependency commit, Lisp version and selected backend. Dependencies are installed without a cache.

Linux SBCL jobs use checksummed SBCL 2.6.8 binaries. ECL and macOS SBCL come from the platform package manager. Quicklisp retries interrupted downloads up to three attempts.

### Reproduce a CI test

With Quicklisp dependencies installed and a built `trivial-simd` checkout:

```sh
export CL_TENSOR_DIR="$PWD"
export TRIVIAL_SIMD_DIR="$HOME/git/trivial-simd"
export CI_QUICKLISP_DIR="$HOME/quicklisp"
export CI_LISP=sbcl
export TRIVIAL_SIMD_BACKEND=native
bash .github/scripts/lisp.sh .github/scripts/test.lisp
```

Set `CI_LISP=ecl` or `TRIVIAL_SIMD_BACKEND=lisp` for the other combinations. Both backends use a built native library because some tests exercise native operations explicitly.

### Update the dependency pin

Change the full `trivial-simd` commit SHA in `.github/workflows/ci.yml`. Build that revision and run both backends locally, then request a full CI run before merging the update.

## CPU coverage

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

GPU tests are optional; the core suite needs no cl-gpu dependency.

```sh
sbcl --non-interactive --eval '(require :asdf)' \
  --eval '(asdf:test-system "cl-tensor/gpu")'
sbcl --non-interactive --eval '(require :asdf)' \
  --eval '(asdf:test-system "cl-tensor/gpu/metal/tests")'
```

| Suite | Coverage |
|---|---|
| Mock device | Native-f64 capability gates, supported operation families, empty-call rejection, failure staging, all storage dtypes, bulk/raw transfers, placement, lifetime, validation, elementwise/scalar/conversion dispatch, matmul batch grouping/window offsets and failure cleanup |
| Gather protocol | Dtype/index combinations, explicit placement, status-only downloads, empty indices, synchronous return and failure cleanup |
| Gather Metal | Independent coordinate references, raw half bits, bounds/extreme indices, windows/strides, aliases and destination padding |
| Fusion protocol | Inline/named dispatch, placement/dtype errors, cache reuse, concurrent first use, redefinition and compilation/launch/completion cleanup |
| Fusion Metal | Independent scalar arithmetic/reduction references, lazy selection, nested groups, windows/strides, aliases, padding guards, empty/scalar outputs and staged failures |
| OpenCL f64 | CPU-double and independent scalar references, direct rounding, raw copies, strided windows, aliases, normalization and empty/failure validation |
| Metal | F64 storage/conversion rejection; CPU and independent references for elementwise, masks, integer arithmetic/conversion, half-format bits, reductions, normalization and matmul; layouts, windows, aliases, errors and scalar results |

Matmul checks include vector products, broadcast batches, aliases across batches, irregular destinations with guard values, and zero-contraction products. Computation and device-copy checks forbid operand/result downloads; checked operations permit the documented single-word error status. Integers and encoded storage compare exactly; f32 values use relative/absolute tolerance `1e-4`.

Run native-f64 conformance on a selected capable OpenCL device:

```lisp
(asdf:load-system "cl-gpu/opencl")
(asdf:test-system "cl-tensor/gpu/opencl/tests")
(load "examples/device-f64.lisp")
(cl-tensor/device-f64-example:run-example)
```

The suite uses `cl-gpu:*device*` when bound, otherwise the default OpenCL device, and fails if native f64 is unavailable. An explicitly selected PoCL CPU device is supported. Arithmetic fixtures compare against CPU doubles with `abs(error) ≤ 2e-13 × max(1, abs(reference))`; integers, encoded formats, rounding boundaries and raw copies compare exactly. Computation forbids operand/result downloads and permits only documented error-status reads. See [native f64](device-f64.md) for the support matrix.

Run `examples/gpu-gather.lisp` for batched embedding lookup with explicit transfers and cleanup.

Run `examples/gpu-fusion.lisp` for inline/named expressions, nested reductions and in-place destinations. Measure synchronous allocating/out fusion against composed GPU operations with `sbcl --non-interactive --load tests/gpu-fusion-bench.lisp`; see [GPU fusion measurements](gpu-fusion-performance.md).

Run the Metal suite with MPS disabled to check the DSL kernels:

```lisp
(asdf:load-system "cl-tensor/gpu/metal/tests")
(let ((cl-gpu/metal:*use-mps* nil))
  (asdf:test-system "cl-tensor/gpu/metal/tests"))
```

Run `examples/device.lisp` for upload, reduction, softmax, weighted RMSNorm, matmul, vector dot, broadcast arithmetic, masks, conversion, download and explicit cleanup. See [device tensors](device.md) for setup and lifetime rules.

## Validation

SBCL 2.6.8 and ECL 26.5.5 execute the full suite and the tensor, extension, fusion and executor examples. Recorded run, 2026-10-19:

| Implementation/backend | Passing checks |
|---|---:|
| SBCL default/native | 136,268 |
| SBCL Lisp after GPU loading | 136,268 |
| ECL default/native | 134,989 |
| ECL Lisp after GPU loading | 136,181 |
| Experimental ARM64 CCL (2026-10-07 run) | 31,358; direct selectors, fusion and executors are untested, subject to the [CCL limitation](#limitations) |

The SBCL and ECL GPU protocol suites each pass 2,014 checks; real Metal execution passes 10,201 checks on both Lisps with MPS enabled and again with MPS disabled. These include gather, elementwise and fusion dispatch, scalar operands, all conversion pairs, every raw half encoding, integer wrapping, lazy nested reductions, concurrent fusion plans and staged failures. The CPU Lisp-backend suite also passes after the GPU extension loads. The embedding example passes on both Lisps.

Native f64 conformance passes 465 checks on Ubuntu 24.04 ARM64 with SBCL 2.2.9 and PoCL’s explicitly selected CPU device (`cpu--0x000`). The f64 example returns the expected double-precision addition and sum.

Normal and forced ASDF loading pass in a fresh SBCL process. Repeated runs advance the random state and may execute a different number of generated checks.[^seed]

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

[^ci-filter]: GitHub does not evaluate path filters for tag pushes. Docs-only pull requests do not produce CI checks, so this path-filtered workflow should not be configured as a required branch-protection check.
