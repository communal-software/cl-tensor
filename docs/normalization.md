# Normalization

Built-in `softmax` and `rmsnorm` kernels normalize groups of `:f32` or `:f64` values without changing tensor shape or dtype. Allocating forms return independent contiguous storage; destination-first `!` forms return the supplied output.

| Call | Group computation |
|---|---|
| `(softmax input &key (axis -1) accumulate)` | `exp(x - maximum) / sum(exp(x - maximum))` |
| `(rmsnorm input &key (axis -1) weights (epsilon 1e-6) accumulate)` | `x / sqrt(mean(x²) + epsilon)`, multiplied by weights |

Out-forms take the same keywords: `(softmax! out input ...)` and `(rmsnorm! out input ...)`.

```lisp
(ct:softmax (ct:from-data '((1000 1001 1002) (0 0 0))))
;; rows approximately (0.09003057 0.24472848 0.66524094)
;;                    (0.33333334 0.33333334 0.33333334)

(ct:rmsnorm (ct:from-data '(3 4))
            :weights (ct:from-data '(1 2)) :epsilon 0)
;; approximately (0.84852815 2.2627418)
```

[Extension methods](extensions.md) retain the normalization shape and axis rules and declare their accepted dtypes and result dtype.

The [GPU extension](device.md) supports f32 softmax and RMSNorm with scalar or same-device tensor weights. Results stay on-device; widened accumulation is unsupported.

## Axes

Omitting `:axis` selects the last axis. Explicit `:axis nil` selects every axis. An integer selects one axis; a list or vector selects several. Negative axes count from the end. Duplicate and out-of-range axes signal errors.

Selected axes form one group in original axis order, regardless of the order supplied. Other axes identify independent groups. The result keeps every axis and its original size; there is no `:keepdims` keyword.

```lisp
(ct:softmax logits :axis -1)      ; each last-axis row
(ct:softmax logits :axis nil)     ; one whole-tensor group
(ct:rmsnorm x :axis '(0 2))       ; one group for each axis-1 position
```

An empty axis vector `#()` selects no axes, so each element forms a singleton group. Its softmax is one; its RMSNorm is `x / sqrt(x² + epsilon)`, with weights applied. Rank-zero tensors require explicit `:axis nil` or `:axis #()`.

## Weights and precision

RMSNorm weights default to one. Scalars coerce to the input dtype. Tensor weights must have the same dtype and broadcast to the input shape without expanding it. A last-axis vector supplies shared feature weights; `(batch 1 features)` supplies batch-specific weights for `(batch rows features)` input.

Epsilon is a nonnegative real coerced to the input dtype. Zero epsilon with an all-zero group divides by zero and follows the active floating-point environment.

`:accumulate nil` uses the input precision. `:accumulate :f64` widens sums for f32 input; squares and exponentials are still computed in f32, and the mean, reciprocal scale and output return to input precision.[^precision] F64 remains f64. Other accumulation values signal errors.

```lisp
(ct:rmsnorm x :weights weights :epsilon 1e-5 :accumulate :f64)
```

## Output and errors

Output shape and dtype must exactly match the input. Reversed, transposed and padded destinations work; destinations mapping multiple elements to one storage location signal errors. Inputs and weights are read before destination writes, including exact in-place calls, cross-group overlaps and shared foreign-memory views.[^execution]

Empty tensors return empty results without evaluating groups, while still validating axes, dtype, weights, epsilon, accumulation and destination. Argument validation errors leave the destination unchanged.

Maximum subtraction avoids exponential overflow for ordinary finite softmax inputs. Extreme differences, square overflow, underflow, NaNs, infinities and traps follow the active backend and Lisp environment. There is no special nonfinite-input policy or cross-backend bit-identity guarantee.

## Device precision

Device normalization accepts f32 and [native-capable f64](device-f64.md). F64 operands and accumulators retain double precision. F64 softmax and RMSNorm accept `:accumulate :f64`; device f32 inputs reject widening. Results stay on the device.

## Limitations

- Normalization uses full-result scratch, and portable f64 weight/output stages allocate per element: [#8](https://github.com/communal-software/cl-tensor/issues/8).
- Irregular groups use the reduction packing helper: [#7](https://github.com/communal-software/cl-tensor/issues/7).

[^precision]: F32 widened softmax sums the stored single-float exponentials in double precision, then rounds the reciprocal to single precision. Widened RMSNorm sums single-float squares in double precision, then rounds the mean before adding epsilon. Widening sums does not prevent single-float square overflow.
[^execution]: Reduction helpers reuse compatible inner rows and pack irregular layouts. Typed loops handle Lisp vectors and foreign views directly. Normalized values occupy independent storage; weights are applied there, then a same-dtype N-D conversion copies into the destination. See [execution measurements](normalization-performance.md).
