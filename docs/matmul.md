# Matrix multiplication

`matmul` multiplies vectors, matrices and batches into an independent contiguous tensor. `(matmul! out a b)` writes the result into `out` and returns it. Both operands are tensors. Built-in kernels accept matching `:f32` or `:f64` dtypes, or one `:f16`/`:bf16` operand and one `:f32` operand. Mixed encoded/f32 products return `:f32`; `matmul!` requires an f32 destination. Use [astype](conversion.md) for other dtype combinations. [Extension methods](extensions.md) can accept mixed input dtypes and select the result dtype.

## Encoded weights

```lisp
(let* ((x (ct:from-data '((1 2 3))))
       (weights (ct:astype (ct:from-data '((1 0 1) (0 1 0))) :f16)))
  (ct:matmul x (ct:transpose weights)))
;; f32 result: ((4 2))
```

Encoded operands work on either side, with the same vector, batch, broadcasting and alias rules. A foreign `:u16` storage view carries `:f16` or `:bf16` as its tensor dtype. Contiguous weight rows are consumed directly by [trivial-simd's encoded kernels](https://github.com/communal-software/trivial-simd/blob/trunk/docs/encoded-matmul.md), without an f32 weight buffer. Irregular layouts pack encoded bits or f32 activations and scatter output; buffers are reused across batches. Inputs that overlap the destination are snapshotted before writes.[^encoded]


## Shapes

The final two axes are matrix axes; leading axes broadcast using the [elementwise broadcasting rules](elementwise.md#broadcasting-and-scalars). Contraction dimensions must match. Scalars and rank-zero tensors signal errors.

| Left shape | Right shape | Result shape |
|---|---|---|
| `(K)` | `(K)` | `()` — rank-zero tensor |
| `(M K)` | `(K)` | `(M)` |
| `(K)` | `(K N)` | `(N)` |
| `(M K)` | `(K N)` | `(M N)` |
| `(2 1 M K)` | `(1 3 K N)` | `(2 3 M N)` |
| `(K)` | `(2 K N)` | `(2 N)` |

A left vector behaves as a `(1 K)` matrix and a right vector as a `(K 1)` matrix; the inserted axes disappear from the result.

```lisp
(let ((a (ct:from-data '((1 2 3) (4 5 6))))
      (b (ct:from-data '(2 3 4))))
  (ct:matmul a b))
;; => #<TENSOR :F32 (2)> with values (20 47)

(ct:tref (ct:matmul (ct:from-data '(1 2 3)) (ct:from-data '(4 5 6))))
;; => 32f0
```

## Destinations and empty dimensions

The destination must match the result shape and dtype exactly. Padded, transposed and reversed destinations work; layouts mapping multiple logical elements to one location signal errors before writes. Overlapping inputs are read as if copied before any destination write, including aliases across batches and foreign views.[^output]

```lisp
(let ((out (ct:transpose (ct:zeros '(3 2)))))
  (ct:matmul! out (ct:ones '(2 4)) (ct:ones '(4 3))))
;; => OUT, shape (2 3), every value is 4f0
```

An empty output performs no writes but still validates operands, shapes and dtype. A zero contraction dimension produces typed zeros, including a zero for an empty vector dot product.

## Execution

The following execution details describe built-in storage and dtypes. Extension methods use independent result storage and protocol copying unless they select a safe direct executor; readable custom storage with a built-in dtype uses packed backend fallback.

Vectors use BLAS dot and matrix-vector routines; matrix products use stride-aware GEMM views over their backing storage. Transposed, reversed, padded and zero-stride input matrices work directly. Compatible batch axes combine into constant-stride runs, with one batched GEMM call per run; broadcast operands use a zero batch stride. A single product uses ordinary GEMM.[^backend]

Interleaved destination matrices write directly to their backing storage. Compatible interleaved batches use batched GEMM; upstream checks addressed elements with exact arithmetic and bounded validation workspace.[^validation]

```lisp
(let* ((data (ct:tensor-storage (ct:zeros '(17))))
       (out (ct:make-tensor-view data '(2 3 2) :strides '(8 2 3))))
  (ct:matmul! out (ct:ones '(2 3 4)) (ct:ones '(4 2))))
;; => OUT, every logical value is 4f0; unused storage stays zero
```

Matrix-vector products retain transpose flags and pack irregular matrices into reusable buffers. Zero-stride vectors materialize. Matrix strides outside GEMM kernel limits use reusable scratch and individual products.

See [performance limitations](limitations.md#performance) for workspace costs shared with other operations.

See [matmul measurements](matmul-performance.md) for interleaved destinations and complete-call benchmarks.

## Limitations

- General destination validation can use workspace proportional to element count, and input snapshots can include disjoint interleaved views. See [performance limitations](limitations.md#performance).

[^output]: Every potentially overlapping input is snapshotted before any batch writes. Ordinary validation precedes writes; computation errors may leave part of the destination updated.
[^validation]: Upstream checks matrix and constant-stride batch uniqueness and input/output overlap without enumerating addresses into a table. Difficult layouts can require substantial search time. cl-tensor separately validates the complete N-D destination before dispatch.
[^backend]: Floating-point order and exceptional values follow trivial-simd and the active Lisp floating-point environment; final bits may differ between backends. BLAS kernel dimensions, leading dimensions and vector increments are bounded by 1,073,741,823, and accessed storage offsets by 2⁶⁰−1. Irrelevant singleton-axis strides are ignored. GEMM uses signed row/column strides; input strides outside kernel limits pack into reusable scratch. Batch runs split at the upstream count limit. Native batching requires the upstream batch symbols; older native libraries fall back to per-product dispatch.

[^encoded]: Native kernels accumulate in f32; the Lisp fallback accumulates in f64 and rounds the output to f32. Arithmetic can differ in the final digits. Two encoded operands and combinations with f64 require explicit conversion. Stable pinned addresses avoid unnecessary snapshots for disjoint mixed Lisp/foreign storage in encoded matmul; other operations retain their documented alias limitations.
