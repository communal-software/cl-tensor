# Native f64 device tensors

F64 tensors retain double precision on devices with native f64 computation. Load `cl-tensor/gpu` and select an OpenCL device with `cl_khr_fp64`. Transfers are explicit; operations keep their results on that device.

## Support

| Capability | Native-f64 OpenCL | Metal or incapable device |
|---|---|---|
| Allocation, explicit transfers, inferred dtype and metadata views | F64 | Rejected |
| Copy, copying reshape, noncontiguous download | F64 | Rejected |
| Arithmetic, bounds, math functions and activations | F64 | Rejected |
| Comparisons and selection | U8 masks; f64 selected values | Rejected |
| Conversion with f32, f16/bf16 or integers | Direct conversion | Rejected |
| Sum, mean, product, extrema | F64 | Rejected |
| Arg extrema | S64 result | Rejected |
| Softmax and weighted RMSNorm | F64 | Rejected |
| Matrix products, gather and fused expressions | Unsupported for f64 | Unsupported for f64 |

An explicitly selected OpenCL CPU device, such as PoCL, is a valid execution device. Nothing selects a different device automatically. Ordinary CPU constructors retain their usual behavior.

## Select a device

```lisp
(asdf:load-system "cl-tensor/gpu")
(asdf:load-system "cl-gpu/opencl")
(cl-gpu:device-native-type-p device 'cl-gpu:f64)
```

The capability query is part of cl-gpu. Allocation and upload require a true result even for empty tensors. Constructing a tensor over an existing f64 GPU-array window also checks capability. Software emulation does not qualify as native support.[^capability]

```lisp
(load "examples/device-f64.lisp")
(cl-tensor/device-f64-example:run-example device)
;; => (:ADD #(16777218.0d0 2.0d0) :SUM 16777218.0d0)
```

Omit the argument to use the default OpenCL device. The example checks capability before uploading and releases its device tensors on exit. Its sum distinguishes f64 from f32.

## Precision and conversion

Operands have matching numeric dtypes. Scalars, activation constants, intermediates, reduction accumulators, normalization weights and epsilon use double precision. Value results retain f64; comparisons produce u8 and arg reductions produce s64.

`:accumulate nil` uses f64 for f64 inputs. Sum, mean, softmax and RMSNorm also accept `:accumulate :f64` with f64 inputs. F32 device inputs reject widened accumulation.

Conversions use one direct rounding step. Floating narrowing supports `:nearest-even`, `:truncate`, `:floor` and `:ceiling`, including f64 to f32. Float-to-integer conversion rounds, then clamps; NaN signals an error. Integer-to-f64 conversion rounds to nearest-even. Same-dtype copies preserve signed zeros and NaN payloads.[^conversion]

Finite arithmetic agrees numerically with CPU-double references within operation-appropriate tolerances; it is not bitwise reproducible across backends. Exceptional values follow backend arithmetic. Comparisons with NaN are false except `:ne`; selection retains the selected value. Extrema use strict comparisons and earlier logical indices on ties, so NaN reduction results depend on traversal order. Softmax with an infinite maximum may produce NaNs. RMSNorm does not rescale squares to prevent overflow.

## Layouts and failures

[Device tensor](device.md) placement, lifetime, signed/zero input strides, GPU-array windows, aliases and destination rules apply. Noncontiguous downloads gather through a device copy before explicit bulk download. Unsupported families reject f64 even when the result would be empty.

Destination forms consume inputs before writing the destination. Validation and computation failures preserve it. A failure during the final copy can partially update it. Computation downloads no operand or result values; checked operations may read their documented u32 status.

Run the native-f64 conformance suite with a capable device selected:

```lisp
(asdf:test-system "cl-tensor/gpu/opencl/tests")
```

The suite fails explicitly when native f64 is unavailable. See [testing](testing.md#device-tests) for regression suites and validation results.

## Limitations

- F64 matrix products, gather and fused expressions are excluded: [#32](https://github.com/communal-software/cl-tensor/issues/32). Matrix prerequisites remain [cl-gpu #71](https://github.com/communal-software/cl-gpu/issues/71), with backend parity tracked in [cl-gpu #86](https://github.com/communal-software/cl-gpu/issues/86).
- Metal f64 tensors are rejected, including storage-only allocation and transfers. Optional software emulation is separate: [cl-gpu #56](https://github.com/communal-software/cl-gpu/issues/56).
- Synchronous calls and full-result destination staging retain the [device staging limitation](device.md#limitations).

[^capability]: Missing native support signals `cl-gpu:gpu-unsupported-capability`, a `gpu-error`, before device result/scratch allocation, upload packing, metadata uploads or launches. Its readers identify the device, operation and f64 element type. Underlying cl-gpu storage can exist on incapable devices; the tensor adapter rejects wrapping it.
[^conversion]: Cross-format NaNs become quiet NaNs with their sign and fitting leading payload bits. Floating narrowing includes directed subnormal and overflow rounding. Encoded-half/integer pairs remain unsupported; use an explicit real floating intermediate. CPU conversion has its own [rounding contract](conversion.md).
