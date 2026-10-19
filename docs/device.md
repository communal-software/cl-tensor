# Device tensors

Load `cl-tensor/gpu` and a cl-gpu backend to store tensors on a GPU. Transfers are explicit. Tensor dtypes describe values: device placement does not change `:f32` into a different dtype.

```lisp
(asdf:load-system "cl-tensor/gpu")
(asdf:load-system "cl-gpu/metal")

(cl-tensor/gpu:with-device-tensor
    (input (cl-tensor/gpu:to-device
            (cl-tensor:from-data '((1 2 3) (4 5 6)))))
  (cl-tensor/gpu:with-device-tensor
      (result (cl-tensor:sum input :axis 1))
    (cl-tensor:tref (cl-tensor/gpu:from-device result) 1)))
;; => 15.0
```

## Allocate and transfer

Functions below belong to `cl-tensor/gpu`.

| Call | Result |
|---|---|
| `(make-device-tensor shape &key (dtype :f32) device)` | Independent zeroed device tensor |
| `(to-device cpu-tensor &key device)` | Independent contiguous device copy |
| `(from-device device-tensor)` | Independent contiguous CPU copy |
| `(free-device-tensor tensor)` | Release the tensor's GPU array; return NIL |
| `(with-device-tensor (name form) ...)` | Bind a device tensor and release it on exit |

Storage supports [native-capable `:f64`](device-f64.md), `:f32`, `:f16`, `:bf16` and signed/unsigned 8-, 16-, 32- and 64-bit integers (`:s8` through `:s64`, `:u8` through `:u64`). Omitted `:device` uses cl-gpu's current device. Ordinary `make-tensor`, `zeros` and other constructors allocate CPU storage, including after the GPU system loads.

Transfers preserve shape and dtype. Uploads use specialized vectors; noncontiguous CPU inputs pack their logical values first. Noncontiguous downloads gather on the device before one bulk download. Neither transfer performs dtype conversion. Half-format transfers preserve raw 16-bit encodings, including NaN payloads and signed zero.

`tensor-storage` returns a `cl-gpu:gpu-array`. GPU storage has no scalar read/write access: download before calling `tref` or assigning elements.

## Operations

| Operation | Device behavior |
|---|---|
| `sum`, `mean`, `prod`, `minimum`, `maximum` | F32 or native f64 input and result |
| `argmin`, `argmax` | F32 or native f64 input; signed 64-bit result |
| `softmax`, `rmsnorm` | F32 or native f64; matching scalar or device tensor RMSNorm weights |
| `take`, `take!` | Same-device integer indices; retain supported input dtype (excluding f64), device and raw bits |
| `matmul` | F32 vectors, matrices and broadcast batches; device-resident results |
| Arithmetic, bounds, `negate`, `abs` | F32, native f64 and integer inputs/results; integer arithmetic wraps |
| `sqrt`, `reciprocal`, exp/log, trigonometric functions and activations | F32 or native f64 input/result |
| `compare`, `select` | U8 comparison masks; selection accepts any nonzero mask byte |
| `astype`, `dequantize` | Explicit device conversion; independent device result |
| `fuse`, `fuse!`, `define-kernel` | F32 expressions and nested reductions; s64 top-level arg results |
| Matching `!` forms | Write a matching destination on the same device |
| `copy-tensor`, copying `reshape` | Independent device copy, retaining dtype |
| Metadata views | Share storage; offsets and signed/zero input strides work |

[Reduction](reductions.md) axis, `keepdims`, empty-group and logical index rules apply. [Normalization](normalization.md) axis and weight-broadcasting rules apply. An empty axis vector selects no axes; NIL selects all axes.

Reductions and normalization accept f32 and native f64 inputs; matrix products require f32. F64 sum, mean and normalization accept `:accumulate :f64`; f32 inputs reject widened accumulation. F16 and BF16 remain storage-only: convert to f32 or native f64 before arithmetic. Floating-point arithmetic uses upstream GPU kernels; results need not be bit-identical to CPU results.

All tensor operands and a supplied destination must occupy the same device. Host tensor weights, CPU destinations and tensors on another device signal errors. Scalar operands and weights are accepted. Use explicit transfers to cross the CPU/device boundary.

Calls wait for completion before returning. Computation reads inputs before destination writes, including aliased weights and shifted input/output views. Validation and computation errors leave a destination unchanged; a failure during the final copy may partially update it.[^staging]

## Elementwise operations and conversion

[Elementwise](elementwise.md) broadcasting, scalar coercion, mask and dtype rules apply. Numeric tensor operands have identical dtypes. Scalar-only destination calls use the destination as numeric context; `select` with two scalar branches requires `:dtype`.

```lisp
(cl-tensor/gpu:with-device-tensor
    (input (cl-tensor/gpu:to-device (cl-tensor:from-data '(1 2 3))))
  (cl-tensor/gpu:with-device-tensor (mask (cl-tensor:compare :gt input 1))
    (cl-tensor/gpu:with-device-tensor (chosen (cl-tensor:select mask input 0))
      (cl-tensor/gpu:with-device-tensor (encoded (cl-tensor:astype chosen :bf16))
        (cl-tensor/gpu:from-device encoded)))))
;; BF16 CPU tensor; values are raw encoded bits
```

`astype` preserves shape, accepts all four [rounding modes](conversion.md), clamps integer narrowing and float-to-integer overflow, and copies raw bits for matching dtypes. F16/BF16 conversion pairs require real floating formats; use f32 as an intermediate for integer conversion. `dequantize` uses the same conversion path, defaulting to f32. F64 requires [native capability](device-f64.md); complex device storage is unsupported.

Integer division truncates toward zero; signed minimum divided by -1 wraps. Zero integer divisors, NaN-to-integer conversion and invalid tensor clamp bounds signal `cl-gpu:gpu-error`. Destination forms preserve the destination on these computation failures.[^status]

## Expression fusion

[Expression fusion](fusion.md#device-execution) uses the same inline and named interfaces on f32 device tensors. Arithmetic regions and dependent reduction passes use reusable GPU plans. Numeric selection is lazy, and all reducers share the supplied axis set.

Run `sbcl --non-interactive --load examples/gpu-fusion.lisp`. See [GPU fusion measurements](gpu-fusion-performance.md) for complete-call timing and temporary-storage limits.

## Lifetime

Metadata views keep their GPU array alive. Garbage collection releases owning arrays; use `with-device-tensor` for predictable cleanup of allocations and results.

Releasing a tensor whose storage owns a buffer invalidates **every tensor sharing that storage**, including slices and reshapes. Release the allocation once; do not independently manage each metadata view. Repeated release is harmless. Operations on released storage signal errors.

A tensor constructed over `cl-gpu:make-gpu-array-view` borrows a buffer window. Releasing that tensor does not free the root buffer; its owner controls the lifetime. Window offsets and tensor offsets both contribute to addressing.

## Matrix products

`matmul` and `matmul!` use the [matrix multiplication](matmul.md#shapes) shape and broadcasting rules. Both inputs and a supplied destination are f32 tensors on one device. Vector dot products return rank-zero device tensors; downloading is explicit.

```lisp
(cl-tensor/gpu:with-device-tensor
    (a (cl-tensor/gpu:to-device (cl-tensor:from-data '((1 2 3) (4 5 6)))))
  (cl-tensor/gpu:with-device-tensor
      (b (cl-tensor/gpu:to-device (cl-tensor:from-data '(1 2 3))))
    (cl-tensor/gpu:with-device-tensor (result (cl-tensor:matmul a b))
      (cl-tensor/gpu:from-device result))))
;; f32 CPU tensor, shape (2), values (14 32)
```

Inputs accept transposed, padded, signed and zero strides, including GPU-array windows. Destinations accept any unique layout. Aliased inputs are consumed before destination writes. Empty outputs still validate shapes, placement and dtype; zero contraction dimensions produce zeros, including an empty vector dot product.

Products use f32 GEMM, including vector products; compatible batch axes use constant-stride batched GEMM. Metal selects MPS for eligible layouts and DSL kernels otherwise. Dimensions are limited to 1,073,741,823. Results follow upstream arithmetic and may differ from CPU results in their final digits.

Run [the device example](../examples/device.lisp) with:

```sh
sbcl --non-interactive --load examples/device.lisp
```

See [testing](testing.md#device-tests) for mock-device and Metal suites.

## Gather

`take` and `take!` gather slices with the [shape API](shape.md#gather). Input and indices, plus the destination for `take!`, reside on the same device. Upload CPU indices explicitly. All eight integer index dtypes are accepted, including scalar and batched indices. Negative indices count from the end.

Gather accepts the storage dtypes above except f64; see [f64 limitations](device-f64.md#limitations). It retains the input dtype and device, including raw f16/bf16 encodings. Signed/zero strides, GPU-array windows and unique irregular destinations are supported. Bounds failures signal `cl-gpu:gpu-error` with the axis and its length. Nonempty indices are checked even when another dimension makes the result empty.

Only a single u32 error status is downloaded for nonempty indices; operand and result values stay on-device. Empty indices need no launch or status read. Calls wait for completion. `take!` consumes aliased input/index values before destination writes and preserves the destination on validation or gather-computation errors. Final-copy failures may leave partial output.[^gather]

Run [the embedding example](../examples/gpu-gather.lisp) for explicit uploads, batched token lookup, allocating and destination forms, downloads and cleanup.

## Limitations

- F64 requires native capability and excludes matrix products, gather and fusion: [f64 limitations](device-f64.md#limitations). Complex device storage is unsupported: [#28](https://github.com/communal-software/cl-tensor/issues/28).
- Joins lack GPU dispatch: [#29](https://github.com/communal-software/cl-tensor/issues/29). Unsupported calls signal errors without implicit downloads.
- Calls synchronize, destination forms retain full-result scratch, allocating gather retains an extra result buffer, and copies sharing a root buffer conservatively stage even disjoint ranges: [#22](https://github.com/communal-software/cl-tensor/issues/22).
- Irregular destination uniqueness checks may require O(output elements) host metadata workspace: [#5](https://github.com/communal-software/cl-tensor/issues/5).
- Upstream kernels use one threadgroup per reduction group and upload layout metadata per call: [cl-gpu #74](https://github.com/communal-software/cl-gpu/issues/74), [cl-gpu #49](https://github.com/communal-software/cl-gpu/issues/49).
- Generic elementwise indexing decodes every axis per element: [cl-gpu #76](https://github.com/communal-software/cl-gpu/issues/76).

[^staging]: Destination forms compute into independent GPU storage, wait for successful completion, then copy into the destination and wait again. Tensor values stay on the device throughout computation and copying; layout metadata is uploaded from the host. The CPU executor hook does not schedule GPU kernels.

[^status]: Checked integer division, float-to-integer conversion and tensor clamp bounds download a single u32 error flag. Operand and result values stay on the device. Scalar-only arithmetic and selection create a temporary rank-zero device operand for the upstream kernel context. Floating exceptional values follow Metal; CPU floating-point trap settings do not control GPU arithmetic.

[^gather]: The adapter calls public `cl-gpu/gather:gather!` and synchronizes after its final copy. Upstream owns normalized indices, metadata, status and result staging, releasing scratch on success and errors. `take!` uses that staging directly; allocating `take` also has a dispatch-owned result buffer that is released on failure. Workspace is O(index count + result size). The final copy has no rollback guarantee after submission, including completion errors.
