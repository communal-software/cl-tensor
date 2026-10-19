# Dtype conversion

`(astype tensor dtype &key (rounding :nearest-even))` returns an independent contiguous tensor with the same shape and converted values. Built-in tensor operations reject mixed numeric dtypes; use this function to align them.

`(dequantize tensor &key (dtype :f32))` invokes an extension dequantization method or falls back to `astype`. Both functions dispatch through [dtype extension methods](extensions.md); custom conversions specify their accepted dtype combinations.

## Numeric conversion

| Conversion | Behavior |
|---|---|
| Integer to integer | Clamp to the destination range |
| Integer to float, float precision change | Round to the destination precision |
| Float to integer | Apply `:rounding`, then clamp |
| Real/integer to complex | Convert real part; imaginary part is zero |
| Complex precision change | Convert both components |
| Complex to real/integer | Require zero imaginary part, then convert |

`:rounding` accepts `:nearest-even`, `:truncate`, `:floor` or `:ceiling`. NaN-to-integer conversion signals an error; infinities clamp to the corresponding integer bound. Same-dtype conversion still copies.[^rounding]

```lisp
(ct:astype (ct:from-data '(0.5 1.5 2.5)) :s8)
;; values: 0, 2, 2
(ct:astype (ct:from-data '(0.5 1.5 2.5)) :s8 :rounding :truncate)
;; values: 0, 1, 2
```

## Half-float storage

`:f16` and `:bf16` hold encoded bits in unsigned 16-bit storage. Conversion accepts either encoding paired with `:f32`, `:f64`, or the other encoding. Conversions between encoded storage and integers or complex dtypes signal an error; use a real floating-point intermediate when needed.

```lisp
(let* ((values (ct:from-data '(1 2 3)))
       (encoded (ct:astype values :bf16)))
  (ct:tref (ct:astype encoded :f32) 1))
;; => 2.0
```

`tref`, `copy-tensor` and same-encoding `astype` preserve raw bits. Encoded narrowing follows the requested rounding mode; widening to real floats is exact.[^encoding]

## Device tensors

`astype` and `dequantize` retain device placement and return independent contiguous storage. Device storage supports f32, native-capable f64, f16/bf16 and all integer widths. Encoded pairs require floating formats; complex targets are unsupported. F64 conversions preserve double precision and support direct floating narrowing with all four rounding modes. See [native f64](device-f64.md#precision-and-conversion) for the device precision contract. Transfers preserve the dtype and raw half bits. See [device conversion](device.md#elementwise-operations-and-conversion) for completion and failure rules.

[^rounding]: Rounding modes affect float-to-integer conversion and encoded narrowing; other conversions ignore the mode after validating its name. Values follow trivial-simd's conversion contract. Conversion passes the full signed-stride layout to one upstream N-D call; native-supported float/encoded pairs execute directly in C and other pairs use Lisp.
[^encoding]: Matching encodings copy bit patterns, including NaN payloads. Different encodings use upstream conversion rules for signed zero, subnormals, infinities and NaN payloads.
