# Reductions

Reduce all axes or selected axes into an independent tensor. A matching `!` form writes into a supplied output: `(sum! out input :axis 1)` returns `out`.

| Function | Result | Dtypes |
|---|---|---|
| `sum`, `prod` | Sum or product | All computational dtypes |
| `mean` | Arithmetic mean | All computational dtypes |
| `minimum`, `maximum` | Smallest or largest value | Real and integer |
| `argmin`, `argmax` | Logical index of the first extreme | Real and integer; result `:s64` |

For f32 device inputs, the [GPU extension](device.md) keeps results on-device, including signed 64-bit arg indices. Transfers are explicit and widened accumulation is unsupported.

Elementwise bounds use [`min` and `max`](elementwise.md).

The [external executor](execution.md) schedules independent retained-axis groups. Each group's reduction stays intact. [Expression fusion](fusion.md) combines elementwise work and nested reducers without intermediate expression tensors.

The dtype rules below describe built-in implementations. [Extension methods](extensions.md) retain the reduction shape rules and select their accepted dtypes and result dtype.

## Axes and shape

All reducers accept `:axis` and `:keepdims`. `:axis nil` reduces every axis; an integer selects one axis, and a list or vector selects several. Negative axes count from the end. Duplicate and out-of-range axes signal errors. An empty vector selects no axes.

Results are tensors, including rank-zero whole-tensor results. `:keepdims t` retains reduced axes at size one.

```lisp
(let ((a (ct:from-data '((1 2 3) (4 5 6)))))
  (list (ct:tensor-shape (ct:sum a))
        (ct:tensor-shape (ct:sum a :axis -1 :keepdims t))))
;; => (#() #(2 1))
```

Multi-axis arg reductions flatten the selected axes in their original axis order, regardless of the order supplied. Indices describe logical row-major positions within each reduction group, including reversed and broadcast views. Ties select the first logical occurrence.

## Dtypes and empty groups

Numeric results retain the input dtype, except integer `mean`, which converts to `:f64` before summation. Integer sums and products wrap at the dtype width. Storage-only half floats require [`astype`](conversion.md) before computation.

`sum` and `mean` accept `:accumulate :f64`: `:f32` becomes `:f64`, and `:c32` becomes `:c64`. Double precision remains double. Integer inputs reject this option.

```lisp
(ct:tref (ct:sum (ct:from-data '(1f8 1f0 -1f8)) :accumulate :f64))
;; => 1d0
```

Empty groups produce typed zero for `sum` and typed one for `prod`. `mean` and extrema signal errors if an output element needs an empty group. An empty output performs no writes, but still validates axes, dtype and destination.

## Output and storage

The output must match the result shape and dtype exactly. Strided destinations work; layouts mapping multiple output elements to one location signal errors. Inputs are read before any destination write, including overlapping Lisp and foreign views.[^scratch]

Floating-point order and exceptional values follow trivial-simd and the active Lisp floating-point environment. Last bits may differ between backends.

## Limitations

- Irregular reductions can require O(input elements) packing, and out-forms use O(output elements) scratch: [#7](https://github.com/communal-software/cl-tensor/issues/7).
- Native integer and complex row batching remains upstream work: trivial-simd #148.

[^scratch]: Whole contiguous reductions use upstream vector reducers. Compatible layouts use contiguous inner-row batches; other layouts pack logical rows. Out-forms compute into temporary storage and scatter only after reduction succeeds.
