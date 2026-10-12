# Dtypes and strides

Dtypes are strict. Built-in storage supports arbitrary strides; extensions validate their own layout restrictions.

## Dtypes

| Dtype | Element type | Bytes | Computable |
|---|---|---|---|
| `:f32` `:f64` | `single-float` `double-float` | 4, 8 | yes |
| `:c32` `:c64` | complex single/double | 8, 16 | yes |
| `:s8` `:s16` `:s32` `:s64` | `(signed-byte n)` | n/8 | yes |
| `:u8` `:u16` `:u32` `:u64` | `(unsigned-byte n)` | n/8 | yes |
| `:f16` `:bf16` | `(unsigned-byte 16)` | 2 | encoded storage; mixed f32 matmul |

- Built-in elementwise arithmetic requires matching dtypes. Conversion is explicit, with `astype`; extension methods declare the combinations they accept.
- Scalars and constructor values are coerced to the tensor dtype. Integer dtypes require in-range integers; floats and ratios signal an error. Real dtypes reject complex values, and complex dtypes accept real values with a zero imaginary component.
- Storage of `(unsigned-byte 16)` infers as `:u16`. Pass `:f16` or `:bf16` explicitly:

```lisp
(let ((bits (make-array 4 :element-type '(unsigned-byte 16))))
  (list (ct:tensor-dtype (ct:make-tensor-view bits '(4)))
        (ct:tensor-dtype (ct:make-tensor-view bits '(4) :dtype :bf16))))
;; => (:U16 :BF16)
```

Integer arithmetic wraps at the dtype width; division truncates toward zero. [Conversion](conversion.md) rounds and clamps instead. `:f16` and `:bf16` access preserves raw bits; use `astype` to encode or decode numeric values.

[Matrix multiplication](matmul.md) supports matching `:f32` or `:f64` operands, or one `:f16`/`:bf16` operand with one `:f32` operand and f32 output, through built-in kernels. Extension methods can accept mixed dtypes and select their result dtype. See [dtype and storage extensions](extensions.md).

## Strides

Strides count elements, not bytes, and may be negative or zero.

| Stride | Gives |
|---|---|
| positive | forward walk |
| negative | reversed axis (give a matching `:offset`) |
| zero | broadcast axis, every index reads the same element |

```lisp
(let ((v (ct:make-tensor-view (make-array 6 :element-type 'single-float) '(2 3)
                              :strides '(-3 1) :offset 3)))
  (list (ct:tensor-index v '(0 0)) (ct:tensor-index v '(1 0))))
;; => (3 0)
```

## Layout predicates

| Function | True when |
|---|---|
| `contiguous-p` | elements occupy consecutive storage in row-major order |
| `inner-contiguous-p` | the innermost axis has unit stride |

Axes of size 1 ignore their stride.[^fast]

[^fast]: Built-in elementwise operations pass storage, offsets and broadcast strides to one trivial-simd N-D call. Compatible contiguous axes coalesce into bulk runs; other layouts use native strided traversal or Lisp fallback. See [performance limitations](limitations.md#performance).
