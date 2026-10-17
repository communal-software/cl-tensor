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

Storage supports `:f32` and `:s64`. Omitted `:device` uses cl-gpu's current device. Ordinary `make-tensor`, `zeros` and other constructors allocate CPU storage, including after the GPU system loads.

Transfers preserve shape and dtype. Uploads use specialized vectors; noncontiguous CPU inputs pack their logical values first. Noncontiguous downloads gather on the device before one bulk download. Neither transfer performs dtype conversion.

`tensor-storage` returns a `cl-gpu:gpu-array`. GPU storage has no scalar read/write access: download before calling `tref` or assigning elements.

## Operations

| Operation | Device behavior |
|---|---|
| `sum`, `mean`, `prod`, `minimum`, `maximum` | F32 input and result |
| `argmin`, `argmax` | F32 input; signed 64-bit result |
| `softmax`, `rmsnorm` | F32 input and result; scalar or device tensor RMSNorm weights |
| `matmul` | F32 vectors, matrices and broadcast batches; device-resident results |
| Matching `!` forms | Write a matching destination on the same device |
| `copy-tensor`, copying `reshape` | Independent device copy, retaining dtype |
| Metadata views | Share storage; offsets and signed/zero input strides work |

[Reduction](reductions.md) axis, `keepdims`, empty-group and logical index rules apply. [Normalization](normalization.md) axis and weight-broadcasting rules apply. An empty axis vector selects no axes; NIL selects all axes.

Computation accepts only f32. `:accumulate :f64` signals an error. S64 supports storage, copying and downloads of arg results, but no arithmetic. Floating-point arithmetic uses upstream GPU kernels; results need not be bit-identical to CPU results.

All tensor operands and a supplied destination must occupy the same device. Host tensor weights, CPU destinations and tensors on another device signal errors. Scalar weights are accepted. Use explicit transfers to cross the CPU/device boundary.

Calls wait for completion before returning. Computation reads inputs before destination writes, including aliased weights and shifted input/output views. Validation and computation errors leave a destination unchanged; a failure during the final copy may partially update it.[^staging]

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

## Limitations

- Elementwise operations, conversion, fusion, joins and gather lack GPU dispatch; extra storage dtypes are unsupported: [#23](https://github.com/communal-software/cl-tensor/issues/23). Unsupported calls signal errors without implicit downloads.
- Calls synchronize, destination forms retain full-result scratch, and copies sharing a root buffer conservatively stage even disjoint ranges: [#22](https://github.com/communal-software/cl-tensor/issues/22).
- Irregular destination uniqueness checks may require O(output elements) host metadata workspace: [#5](https://github.com/communal-software/cl-tensor/issues/5).
- Upstream kernels use one threadgroup per reduction group and upload layout metadata per call: [cl-gpu #74](https://github.com/communal-software/cl-gpu/issues/74), [cl-gpu #49](https://github.com/communal-software/cl-gpu/issues/49).

[^staging]: Destination forms compute into independent GPU storage, wait for successful completion, then copy into the destination and wait again. Tensor values stay on the device throughout computation and copying; layout metadata is uploaded from the host. The CPU executor hook does not schedule GPU kernels.
