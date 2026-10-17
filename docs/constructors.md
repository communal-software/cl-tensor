# Constructors and access

Constructors return independent, row-major contiguous tensors with dtype `:f32` unless specified. Shapes are lists or vectors of non-negative integers; `nil` is a scalar shape and a zero dimension makes the tensor empty.

[Custom dtypes](extensions.md) allocate through their storage method. Scalar value constructors require writable storage; opaque storage supplies allocation and copying without scalar access.

## Filled tensors and data

| Call | Result |
|---|---|
| `(zeros shape &key (dtype :f32))` | Zero-filled tensor; equivalent to `make-tensor` |
| `(ones shape &key (dtype :f32))` | One-filled tensor |
| `(full shape value &key (dtype :f32))` | Tensor filled with `value` |
| `(from-data data &key (dtype :f32))` | Copy of rectangular nested lists/vectors or a Lisp array |

`from-data` accepts numeric scalar input as rank zero, rejects ragged data, and does not infer dtype. Numeric leaves follow the [dtype coercion rules](dtypes-and-strides.md#dtypes). Empty nested lists retain their represented dimensions: `'(() ())` has shape `(2 0)`.[^arrays]

```lisp
(ct:from-data '((1 2) (3 4)) :dtype :s16)
;; => #<TENSOR :S16 (2 2)>
(ct:full '(2 3) 0.5 :dtype :f64)
;; => #<TENSOR :F64 (2 3)>
```

## Sequences and diagonals

| Call | Result |
|---|---|
| `(arange stop &key dtype)` | Values from 0 to exclusive `stop`, step 1 |
| `(arange start stop &key dtype)` | Values from `start` to exclusive `stop`, step 1 |
| `(arange start stop step &key dtype)` | Values with the given nonzero step |
| `(linspace start stop &key (num 50) (endpoint t) (dtype :f32))` | `num` evenly spaced real samples |
| `(eye rows &key (columns rows) (k 0) (dtype :f32))` | Ones on diagonal `column = row + k`, zeros elsewhere |

`arange` accepts real bounds and steps; mismatched direction returns an empty vector. Integer dtypes require every generated value to be an in-range integer. `linspace` accepts only `:f32` or `:f64`: zero samples return empty, one returns `start`, and `:endpoint nil` excludes `stop`.

```lisp
(ct:arange 5 0 -2 :dtype :s8) ; values: 5, 3, 1
(ct:linspace 0 1 :num 3)     ; values: 0.0, 0.5, 1.0
(ct:eye 2 :columns 3 :k 1)   ; rows: (0 1 0), (0 0 1)
```

## Access and copying

`(tref tensor &rest indices)` reads an element; `(setf (tref tensor indices...) value)` writes a coerced value and returns the supplied value. There must be one integer index per axis, starting at zero. Negative indices and out-of-bounds indices signal an error.

`(copy-tensor tensor)` preserves dtype, shape and logical values, with independent contiguous storage. It handles reversed, transposed and zero-stride views, including foreign storage. Copying preserves storage placement: device copies stay on the same GPU. See [device tensors](device.md) for explicit transfers and lifetime. Printing shows dtype and shape without reading elements.

```lisp
(let ((a (ct:zeros '(2 3))))
  (setf (ct:tref a 1 2) 7)
  (ct:tref (ct:copy-tensor a) 1 2))
;; => 7.0
```

For `:f16` and `:bf16`, `zeros` creates zero bits and `tref` reads/writes unsigned 16-bit patterns. Other value constructors reject these storage-only dtypes; use [conversion](conversion.md) to encode values.

[^arrays]: Multidimensional Lisp arrays preserve their dimensions and contain numeric leaves. Nested lists and vectors may contain further lists, vectors or numeric arrays. A zero-sized Lisp array preserves axes beyond the empty dimension; a nested list cannot represent unobserved axes.
