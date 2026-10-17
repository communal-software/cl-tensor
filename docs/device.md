# Device tensors

Load `cl-tensor/gpu` and a cl-gpu backend to store tensors on a GPU. Transfers are explicit. Tensor dtypes describe values: device placement does not change `:f32` into a different dtype.

```lisp
(asdf:load-system "cl-tensor/gpu")
(asdf:load-system "cl-gpu/metal")

(cl-tensor/gpu:with-device-tensor
    (input (cl-tensor/gpu:to-device (cl-tensor:from-data '(1 2 3))))
  (cl-tensor:tref (cl-tensor/gpu:from-device input) 1))
;; => 2.0
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

## Views and copying

Metadata views share storage and accept signed and zero input strides. `copy-tensor` and copying `reshape` paths return independent device storage with the same dtype. GPU copies accept strided destinations and stage aliases on the device.

Calls complete synchronously. Explicit transfers cross the CPU/device boundary; ordinary copying does not upload or download tensors implicitly.

## Lifetime

Metadata views keep their GPU array alive. Garbage collection releases owning arrays; use `with-device-tensor` for predictable cleanup of allocations and results.

Releasing a tensor whose storage owns a buffer invalidates **every tensor sharing that storage**, including slices and reshapes. Release the allocation once; do not independently manage each metadata view. Repeated release is harmless. Operations on released storage signal errors.

A tensor constructed over `cl-gpu:make-gpu-array-view` borrows a buffer window. Releasing that tensor does not free the root buffer; its owner controls the lifetime. Window offsets and tensor offsets both contribute to addressing.

Run [the device example](../examples/device.lisp) with:

```sh
sbcl --non-interactive --load examples/device.lisp
```

See [testing](testing.md#device-tests) for the mock-device suite.

## Limitations

- Computational GPU dispatch, extra storage dtypes, joins and gather are unsupported: [#23](https://github.com/communal-software/cl-tensor/issues/23).
- Copies sharing a root buffer conservatively stage even disjoint ranges: [#22](https://github.com/communal-software/cl-tensor/issues/22).
- Irregular destination uniqueness checks may require O(output elements) host metadata workspace: [#5](https://github.com/communal-software/cl-tensor/issues/5).
