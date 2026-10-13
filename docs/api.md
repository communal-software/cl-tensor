# API

Package `cl-tensor`, nickname it locally (`ct` below). Use qualified names: `abs`, `sqrt`, `min`, `max`, `concatenate`, `log`, `tanh`, `exp`, `sin` and `cos` shadow Common Lisp symbols.

| Topic | Reference |
|---|---|
| Filled tensors, sequences, nested data, element access and copying | [Constructors and access](constructors.md) |
| Broadcasting arithmetic, activations, comparisons and selection | [Elementwise operations](elementwise.md) |
| Inline/named expression kernels and nested reductions | [Expression fusion](fusion.md) |
| Dynamic executor hook for independent output groups | [External executors](execution.md) |
| Axis reductions and logical arg indices | [Reductions](reductions.md) |
| Softmax and weighted RMSNorm | [Normalization](normalization.md) |
| Vector, matrix and batched multiplication | [Matrix multiplication](matmul.md) |
| Views, reshape, slicing, gathering and joining | [Shape operations](shape.md) |
| Explicit dtype changes, half-float encoding and dequantization | [Conversion](conversion.md) |
| External dtypes, storage and operation methods | [Extensions](extensions.md) |

## Construction

| Function | Returns |
|---|---|
| `(make-tensor shape &key (dtype :f32))` | Zeroed contiguous tensor |
| `(make-tensor-view storage shape &key dtype strides offset)` | Tensor sharing `storage`; signals an error if the view is invalid |
| `(row-major-strides shape)` | Row-major strides for `shape` |

`shape` is a list or vector of non-negative integers. In `make-tensor-view`, `dtype` defaults to the dtype inferred from `storage`.

```lisp
(ct:make-tensor '(2 3) :dtype :f64)
;; => #<TENSOR :F64 (2 3)>
```

## Inspection

| Function | Returns |
|---|---|
| `tensor-storage` `tensor-dtype` `tensor-shape` `tensor-strides` `tensor-offset` | The slots |
| `tensor-rank` `tensor-size` | Number of axes, number of elements |
| `(tensor-index tensor indices)` | Storage index of the element at the index list; signals an error when out of bounds |
| `contiguous-p` `inner-contiguous-p` | Layout predicates, see [dtypes and strides](dtypes-and-strides.md) |
| `tensorp` | Type predicate |

## Dtypes

| Function | Returns |
|---|---|
| `*dtypes*` | Registered descriptors; enumerate without modifying |
| `register-dtype`, `find-dtype` | Register and resolve dtype descriptors |
| `(storage-dtype storage)` | Dtype inferred from storage |
| `(dtype-element-type dtype)` `(dtype-bytes dtype)` | Logical Lisp element type, bytes per logical element |
| `(dtype-storage-only-p dtype)` | True for built-in `:f16`/`:bf16` and storage-only extensions |
| `(dtype-block-size dtype)`, `(dtype-block-bytes dtype)` | Elements and stored bytes per block |
