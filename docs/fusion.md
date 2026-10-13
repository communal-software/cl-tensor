# Expression fusion

`fuse` evaluates an elementwise expression without intermediate expression tensors. `fuse!` stages the result and copies it into a supplied destination. Both accept f32/f64 tensors, broadcast operands and preserve arbitrary signed strides.

```lisp
(let ((a (ct:from-data '(1 2 3)))
      (b (ct:from-data '(2 3 4))))
  (ct:fuse (+ (* a b) 1)))
;; values: 3, 7, 13
```

## Interfaces

| Interface | Result |
|---|---|
| `(fuse expression &key axis keepdims)` | Independent contiguous tensor |
| `(fuse! out expression &key axis keepdims)` | Fills and returns `out` |
| `(define-kernel name (arguments...) expression)` | Defines `name` and `name!` with the same keyword options |

```lisp
(ct:define-kernel multiply-add (a b c) (+ (* a b) c))
(multiply-add a b 1)
(multiply-add! out a b 1)
```

Leaves are variable names or real constants. Inline variable leaves are captured once, in first-occurrence order. Bind function-call results outside the expression. Named definitions require every variable leaf to appear in the argument list; unused arguments do not affect broadcasting.

At least one referenced operand is a tensor. Tensor operands share one f32/f64 dtype; scalar variables are coerced to that precision. Computation uses the input precision. Kernels compile with their containing code and are reused between calls.[^loading]

## Expressions

| Forms | Meaning |
|---|---|
| `+`, `-`, `*`, `/` | Left-folded arithmetic; unary `-` and `/` negate and reciprocate |
| `abs`, `sqrt`, `min`, `max` | Magnitude, root and bounds |
| `exp`, `sin`, `cos`, `log`, `tanh`, `ct:sigmoid` | Real-float math |
| `(trivial-simd:fma a b c)` | Explicit fused multiply-add |
| `=`, `/=`, `<`, `<=`, `>`, `>=` | Two-operand comparisons inside selection |
| `(ct:select comparison a b)` | Numeric selection with lazy branches |
| `ct:sum`, `ct:prod`, `ct:minimum`, `ct:maximum`, `ct:argmin`, `ct:argmax` | Top-level or nested reductions |

The corresponding tensor arithmetic/math names also work, for example `ct:add`, `ct:multiply` and `ct:exp`. Unsupported operators and invalid expression kinds signal errors during macro expansion. Masks are internal to selection; fusion returns numeric tensors.

`(+ (* a b) c)` uses separate multiplication and addition rounding. FMA requires the explicit form above. Floating-point traps and exceptional values follow the active trivial-simd backend; results need not be bit-identical across backends.

## Reduction groups

Every reducer in an expression uses the same `:axis` set over the broadcast shape. NIL reduces all axes; negative axes count from the end; `#()` selects no axes. Retained axes identify independent groups.

```lisp
(ct:fuse (ct:sum (* a b)) :axis -1 :keepdims t)

(ct:define-kernel stable-softmax (x)
  (/ (exp (- x (ct:maximum x)))
     (ct:sum (exp (- x (ct:maximum x))))))
(stable-softmax logits :axis -1)
```

A top-level reduction removes selected axes, or retains them at size one with `:keepdims t`. Other expressions retain the broadcast shape. `:keepdims t` requires a top-level reduction. Nested reductions broadcast their scalar result within each complete group, including when the input requires packing.

Top-level arg reducers return `:s64` logical indices within the selected axes, enumerated in ascending axis order. Ties retain the first occurrence. Nested arg indices convert to the computation precision.

Empty sum groups return zero; product groups return one. Top-level extrema and arg reducers reject empty groups when an output is required. Empty vector results and identity reductions skip nested evaluation.

## Destinations and storage

Out-forms validate shape, dtype, uniqueness and copy support before computing. Inputs are read before any destination copy, including shifted and hidden storage aliases. Computation or input-read errors leave the destination unchanged; an error in a custom destination's copy method follows that method's guarantees.

Built-in arrays and foreign views use direct contiguous slices or compatible row batches. Other layouts pack input values. Readable custom storage with a built-in f32/f64 dtype works through scalar access; custom destinations require copy support. Fusion bypasses custom operation selectors.

Run [the example](../examples/fusion.lisp) with `sbcl --non-interactive --load examples/fusion.lisp`.

## Limitations

- Integer/complex computation, custom dtypes, mask outputs, mean, widened accumulation, per-reducer axis sets and arbitrary Lisp calls are unsupported.
- Expressions containing selection execute through the upstream Lisp backend to preserve lazy branches. Native branch evaluation is tracked in [trivial-simd #81](https://github.com/communal-software/trivial-simd/issues/81).
- Out-forms stage complete results. Packed reduction operands and scalar-variable operands retain one complete row per worker; irregular nested vector results also use a row buffer. Plain elementwise packing is bounded to 4,096 elements per operand. Bounded multi-pass execution and transactional output are tracked in [#13](https://github.com/communal-software/cl-tensor/issues/13).
- Strided packing and per-row kernel setup can outweigh saved expression tensors. Direct N-D expression execution is tracked in [#14](https://github.com/communal-software/cl-tensor/issues/14).
- Upstream `define-kernel` is experimental. Nested reductions require multiple passes and may recompute expressions; see [upstream limitations](https://github.com/communal-software/trivial-simd/blob/trunk/docs/kernel-passes.md#limitations).

[^loading]: Native programs initialize lazily on first use. Inline macros also work through `eval`; this does not introduce per-call compilation inside a compiled call site. Named allocating and out-functions share a kernel closure. Retained functions remain callable after redefinition.
