# Elementwise operations

Operations broadcast trailing axes and return independent contiguous tensors. A matching `!` form writes into a supplied output and returns it, for example `(add a b)` and `(add! out a b)`.

The dtype rules below describe built-in implementations. [Extension methods](extensions.md) select accepted dtype combinations and result dtypes before built-in execution.

## Operations

| Allocating call | Result | Dtypes |
|---|---|---|
| `(add a b)`, `(subtract a b)`, `(multiply a b)`, `(divide a b)` | Arithmetic | Real, complex, integer |
| `(negate input)`, `(abs input)` | Negation, absolute value | Real, complex, integer |
| `(sqrt input)`, `(reciprocal input)` | Square root, reciprocal | Real, complex floating-point |
| `(log input)`, `(tanh input)`, `(sigmoid input)` | Natural logarithm, hyperbolic tangent, logistic sigmoid | `:f32`, `:f64` |
| `(exp input)`, `(sin input)`, `(cos input)` | Exponential, sine and cosine in radians | `:f32`, `:f64` |
| `(silu input)`, `(gelu input)` | SiLU and tanh-approximate GELU | `:f32`, `:f64` |
| `(min a b)`, `(max a b)`, `(clamp input lower upper)` | Bounds | Real, integer |
| `(compare operator a b)` | `:u8` mask containing 0 or 1 | Equality for all computational dtypes; ordering for real/integer |
| `(select mask on-true on-false &key dtype)` | Element selected by mask | Real, complex, integer |

Out-forms prepend `out` to these arguments; `select!` has no `:dtype` keyword. Comparison operators are `:eq`, `:ne`, `:lt`, `:le`, `:gt`, `:ge`. Selection requires a `:u8` tensor mask: zero is false and any nonzero byte is true.

Complex `abs` returns `:f32` for `:c32` input or `:f64` for `:c64` input. Other numeric results retain the input dtype. Use package-qualified names for `abs`, `sqrt`, `min`, `max`, `log`, `tanh`, `exp`, `sin` and `cos`, which shadow Common Lisp symbols.

## Real-float math and activations

`log` computes the natural logarithm, and `exp` computes the exponential.
`sin` and `cos` take angles in radians. These operations, `tanh`, `sigmoid`,
`silu` and `gelu` retain the input dtype and shape. Their `!` forms follow the
same output and alias rules as other elementwise operations.

SiLU computes `x * sigmoid(x)` using stable sigmoid. `gelu` uses the standard
tanh approximation.[^gelu] Built-in activation kernels evaluate each expression
in one traversal without intermediate tensors.

```lisp
(ct:sigmoid (ct:from-data '(-1 0 1)))
;; approximately (0.26894143 0.5 0.7310586)
(ct:silu (ct:from-data '(-1 0 1)))
;; approximately (-0.26894143 0.0 0.7310586)
(ct:gelu (ct:from-data '(-1 0 1)))
;; approximately (-0.158808 0.0 0.841192)
(ct:cos (ct:from-data '(0d0) :dtype :f64))
;; values: (1d0)
```

These operations use scalar system math, without a fixed ULP bound or
cross-backend bit identity. Positive finite inputs have real logarithms.
Nonpositive logarithm inputs, exceptional values, underflow and floating-point
traps follow the active backend and Lisp environment. Sigmoid and SiLU use a
sign-dependent sigmoid formula that avoids exponential overflow.[^activation]
Exponential overflow and GELU's cubic intermediate follow the same backend policy.

## Broadcasting and scalars

Align shapes from the right. Each pair of dimensions must match or one must be 1; a missing axis behaves as 1. Broadcasting 0 with 1 produces 0.

```lisp
(let ((a (ct:from-data '((1) (2))))
      (b (ct:from-data '(10 20 30))))
  (ct:add a b))
;; => #<TENSOR :F32 (2 3)> with rows (11 21 31), (12 22 32)
```

Lisp scalars are accepted on either side and coerced to the numeric tensor dtype. Numeric tensors must have identical dtypes; convert explicitly with [astype](conversion.md). Allocating arithmetic needs a numeric tensor to determine dtype. For selection with two scalar choices, supply `:dtype`:

```lisp
(ct:select (ct:compare :gt (ct:arange 4) 1) 10 0 :dtype :s16)
;; values: 0, 0, 10, 10
```

Out-forms use the destination dtype when all numeric operands are scalars.[^scalar-out] Half-float storage must be converted before arithmetic. Integer arithmetic wraps at the dtype width, and division truncates toward zero; zero integer divisors signal `division-by-zero`.

## Output and aliases

Output shape and dtype must exactly match the result. Reversed, transposed and padded outputs work; outputs that map several logical elements to one storage location signal an error before writes.

Inputs are read as if copied before overlapping output writes, including aliases across rows and distinct foreign views of shared memory. Exact elementwise in-place mappings need no snapshot. Ordinary validation happens before writes, even for empty outputs; arithmetic errors may leave part of the output updated.[^execution]

## Device tensors

The optional GPU extension executes these APIs on f32 and integer device tensors. Real-float math requires f32; half formats remain storage-only. Broadcasting, scalar context and mask rules are the same. Calls complete synchronously, and destination forms stage results before writing. See [device tensors](device.md#elementwise-operations-and-conversion) for supported types, errors and transfer behavior.

## Limitations

- General output validation uses O(element count) workspace when the sorted-stride uniqueness proof fails: [#5](https://github.com/communal-software/cl-tensor/issues/5), trivial-simd #150.
- Mixed Lisp/foreign storage conservatively snapshots inputs, including disjoint memory: [#6](https://github.com/communal-software/cl-tensor/issues/6), trivial-simd #151.

[^scalar-out]: Scalars have shape `nil`, so scalar-only arithmetic requires a rank-zero output. `select!` also broadcasts its mask. Scalar-only `compare!` uses its `:u8` output as numeric context, so operands must be in-range integers.
[^execution]: Built-in backend-storage execution uses one upstream N-D call. Native-supported operations traverse signed and broadcast strides in C, using bulk kernels where available and scalar system math for activations. Complex operations, remaining conversion pairs and fallback configurations traverse storage directly in Lisp. See [workspace limitations](#limitations) for snapshot and layout-validation costs. Exceptional floating-point behavior follows trivial-simd and the active Lisp floating-point environment.

[^activation]: Sigmoid computes `z = exp(-abs(x))`, then `1/(1+z)` for nonnegative inputs or `z/(1+z)` otherwise. Underflow remains backend-dependent. See [examples](../examples/tensors.lisp) for activations and RoPE angles, and [normalization](normalization.md) for softmax and RMSNorm.
[^gelu]: GELU computes `0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x³)))`. Constants and intermediates use the input precision. This is the tanh approximation, with no exact-erf variant or approximation selector.
