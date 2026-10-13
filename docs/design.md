# Design

A tensor is typed storage plus a strided window onto it. Views share storage; constructors, `copy-tensor`, `astype` and allocating operations return independent storage.

| Slot | Meaning |
|---|---|
| `storage` | Storage protocol object; built-ins are simple vectors and trivial-simd `vector-view`s |
| `dtype` | Element type keyword, see [dtypes and strides](dtypes-and-strides.md) |
| `shape` | Size of each axis |
| `strides` | Storage step per axis, in elements |
| `offset` | Storage index of the first element |

The element at indices `i0 … in` lives at `offset + Σ ik · stridek`.

```lisp
(let* ((storage (make-array 6 :element-type 'single-float))
       (a (ct:make-tensor-view storage '(2 3)))
       (b (ct:make-tensor-view storage '(3 2) :strides '(1 3))))
  (eq (ct:tensor-storage a) (ct:tensor-storage b)))
;; => T
```

## Tensor type

`tensor` is a structure with read-only slots.[^struct] Shape and strides are `(simple-array fixnum (*))`. `make-tensor` allocates zeroed contiguous storage; `make-tensor-view` wraps existing storage and validates it.

A view is valid when:

- the strides match the rank, the offset is non-negative and every size is non-negative;
- the dtype validates storage compatibility and any layout restrictions;
- every reachable storage index lies inside the storage. An empty view is always valid.

## Extension points

Dtypes are registered CLOS descriptors. Storage kinds implement logical length, type and optional scalar access. Numerical operations select a dtype method before built-in execution; accepted extension out-forms stage independent results and copy through the dtype protocol, or use an explicitly selected alias-safe, failure-atomic direct executor. See [dtype and storage extensions](extensions.md).

Autograd attaches through a wrapper around a tensor, or through a subtype defined with `(:include tensor)`. Code that accepts a `tensor` accepts both.

## Shape and reduction execution

Shape operations build validated views; reshape copies when its strides cannot represent the requested logical order. Concatenate, stack and take allocate independent storage. Gather copies selected slices through the dtype copy protocol; its out-form stages the result before writing the destination.

Reductions group retained and reduced axes into rows, packing when inner rows are not contiguous. Compatible real float batches execute natively; integer and complex batches use upstream typed reducers. Destination-writing reductions compute before scattering, so overlapping views read their original values.

See [reductions](reductions.md) and [shape operations](shape.md) for their interfaces.

## Normalization

Softmax and RMSNorm group selected axes into contiguous inner rows using the reduction layout helpers. Typed single- and double-float loops compute into independent storage; optional widened accumulation affects sums. RMSNorm applies broadcast weights to that storage before any destination write. Results restore the input axis order, and out-forms copy into validated strided destinations.

See [normalization](normalization.md) for formulas, axes and precision, and [measurements](normalization-performance.md) for execution comparisons.

## Matrix multiplication

Matmul broadcasts batch axes and promotes vectors into matrix axes. Matrix products wrap existing storage with independent signed row/column strides and combine compatible batch axes into constant-stride GEMM runs. Vector products use dot/GEMV and retain packing for irregular matrix layouts. Overlapping inputs are snapshotted before batch writes. Destinations proven unique by upstream receive direct writes; other valid matrix layouts scatter from reusable scratch.

See [matrix multiplication](matmul.md) for shapes, dtypes and output behavior.

## Operation execution

Elementwise operations compute the broadcast shape and validate tensor dtypes, then pass storage, offsets and broadcast strides to one trivial-simd N-D operation. Upstream validates output uniqueness and snapshots aliased inputs before execution. Compatible contiguous axes coalesce into bulk runs; native-supported operations traverse arbitrary layouts in C. Other cases traverse storage directly in Lisp. Inputs support arbitrary strides. Out-forms reject layouts in which two logical elements share a storage location.

Lisp storage uses array access. Foreign storage uses typed CFFI access and remains owned by the caller; its memory must stay valid for the entire operation.

See [limitations](limitations.md) for what is not built yet.

## Fusion

Expression macros lower f32/f64 tensor expressions into reusable trivial-simd kernels. Nested reducers share an axis set and operate on complete logical groups. Direct layouts use slices or row batches; other layouts pack bounded elementwise chunks or complete reduction rows. Destination-writing fusion stages an independent result before copying. See [fusion](fusion.md).


[^struct]: Accessors are `tensor-storage`, `tensor-dtype`, `tensor-shape`, `tensor-strides` and `tensor-offset`. The shape and stride vectors are not copied on access; do not modify them.
