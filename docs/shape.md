# Shape operations

Shape operations create views sharing storage. `concatenate`, `stack` and `take` allocate independent contiguous results; `reshape` copies only when its copy policy requires it.

| Call | Behavior |
|---|---|
| `(reshape input shape &key (copy :auto))` | Preserve logical row-major element order |
| `(transpose input &key axes)` | Permute every axis; default reverses axis order |
| `(slice input &key selectors)` | Basic indexing and ranges as a view |
| `(squeeze input &key axis)` | Remove singleton axes; default removes all |
| `(expand-dims input &key axis)` | Insert one singleton axis; axis is required |
| `(concatenate tensors &key (axis 0))` | Join along an existing axis |
| `(stack tensors &key (axis 0))` | Join along a new axis |
| `(split input sections-or-cuts &key (axis 0))` | Return a list of views along an existing axis |
| `(take input indices &key (axis 0))` | Gather slices into a new tensor |
| `(take! out input indices &key (axis 0))` | Gather slices and return the supplied destination |

Negative axes count from the end. For `stack` and `expand-dims`, axes refer to the resulting rank. All operations preserve dtype, including raw `:f16` and `:bf16` storage.

[Storage extensions](extensions.md) validate each resulting view and may reject unsupported layouts. Copying paths use the dtype copy method, allowing opaque storage when its layout supports the operation.

## Reshape and transpose

A reshape shape contains non-negative dimensions and at most one inferred `-1`. Its element count must match the input. Inference with a zero known product is ambiguous and signals an error.

| Copy policy | Result |
|---|---|
| `:auto` | Share storage when strides allow; otherwise copy |
| `:never` | Share storage or signal an error |
| `:always` | Allocate independent storage |

Compatible padded, stepped, reversed and zero-stride layouts can reshape without a copy.[^strides] Transpose requires a permutation of all axes.

```lisp
(let* ((a (ct:reshape (ct:arange 6) '(2 3)))
       (b (ct:transpose a)))
  (ct:reshape b '(6)))
;; values: 0, 3, 1, 4, 2, 5; independent storage
```

## Gather

`take` replaces the selected axis with the shape of `indices`. Input rank must be positive. Indices are tensors of any rank with a signed or unsigned integer dtype (`:s8` through `:s64`, or `:u8` through `:u64`). Scalar indices remove the selected axis; repeated indices repeat slices. Negative indices count from the end. Values outside the axis bounds signal an error, including when another dimension is empty. An empty index tensor selects no slices.

| Input shape | Index shape | Axis | Result shape |
|---|---|---|---|
| `(vocabulary features)` | `(tokens)` | `0` | `(tokens features)` |
| `(vocabulary features)` | `(batch tokens)` | `0` | `(batch tokens features)` |
| `(rows columns)` | `()` | `1` | `(rows)` |

```lisp
(let* ((embeddings (ct:from-data '((1 2) (3 4) (5 6))))
       (ids (ct:from-data '((2 0) (1 -1)) :dtype :s64)))
  (ct:take embeddings ids))
;; shape: (2 2 2); values: 5, 6, 1, 2, 3, 4, 5, 6
```

Input and index tensors support padded, reversed and zero-stride views, including foreign storage. Gathering preserves the input dtype and copies raw `:f16`/`:bf16` bits. Custom storage uses its dtype's view validation and copy methods; packed formats may reject slices that break their block layout.

`take!` requires the exact result shape and input dtype, nonoverlapping logical destination elements, and dtype copy support. It validates every index and gathers into independent storage before copying to `out`, preserving input/index aliases and leaving the destination unchanged on gathering failures.[^gather-copy]

## Slicing

`:selectors` is a list or vector with one selector per supplied axis. Omitted trailing axes are retained.

| Selector | Meaning |
|---|---|
| `:all` | Retain the complete axis |
| Integer | Select one element and drop the axis; negative indices count from the end |
| `(start stop)` | Range with exclusive stop and step 1 |
| `(start stop step)` | Range with a nonzero signed step |

Bounds are clipped to the axis. Negative bounds count from the end. `nil` endpoints select the beginning/end appropriate to the step direction. For reverse ranges, an omitted stop differs from an explicit `-1`.

```lisp
(let ((a (ct:reshape (ct:arange 12) '(3 4))))
  (ct:slice a :selectors '(1 (nil nil -1))))
;; shape: (4); values: 7, 6, 5, 4; shared storage
```

Integer indices outside the axis and excess selectors signal errors. Selecting every axis by integer produces a rank-zero tensor; use `tref` to read its scalar.

## Singleton axes and joining

`squeeze :axis` accepts an integer or sequence; every selected dimension must be one. An empty vector removes none. `expand-dims` accepts one axis.

Concatenate requires matching dtypes and all dimensions outside the join axis. Stack requires identical shapes and dtypes. Both accept a nonempty sequence of tensors, including strided views.

Split accepts a positive section count dividing the axis evenly, or ordered cut points between zero and the axis size. Repeated cuts create empty views. Uneven splits use explicit cuts.

```lisp
(mapcar #'ct:tensor-shape (ct:split (ct:arange 7) '(2 5)))
;; => (#(2) #(3) #(2))
```

## Limitations

- Slicing supports basic integer and range selectors. Use `take` for axis-based integer gathers. Boolean masks, general advanced indexing, ellipsis and insertion selectors are outside this API.
- Gather out-forms use full-result scratch: [#11](https://github.com/communal-software/cl-tensor/issues/11).

[^strides]: Reshape groups compatible stride chunks, ignoring singleton-axis strides. A new dimension cannot cross a discontinuity between chunks. Views retain their storage owner; callers keep foreign memory valid for the view's lifetime.

[^gather-copy]: A custom dtype's final destination copy retains its own failure contract. Result staging and normalized-index storage require O(result size + index count) workspace; no upstream gather kernel is required.
