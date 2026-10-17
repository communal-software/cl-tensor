# cl-tensor

> **Work in progress.** This project is under development; expect missing features and breaking changes.

NumPy-like N-dimensional tensors for Common Lisp, built on
[trivial-simd](https://github.com/communal-software/trivial-simd) and `trivial-simd/blas`.

```lisp
(ql:quickload :cl-tensor)
(cl-tensor:make-tensor '(2 3))
;; => #<TENSOR :F32 (2 3)>
```

Construct, index and copy strided tensors; reshape and slice views; broadcast arithmetic, comparisons and selection; multiply matrices and batches; reduce and normalize axes; convert dtypes explicitly.

## Docs

- [Design](docs/design.md)
- [Dtypes and strides](docs/dtypes-and-strides.md)
- [Dtype and storage extensions](docs/extensions.md)
- [Device tensors](docs/device.md)
- [API](docs/api.md)
- [Constructors and access](docs/constructors.md)
- [Elementwise operations](docs/elementwise.md)
- [Expression fusion](docs/fusion.md)
- [External executors](docs/execution.md)
- [Reductions](docs/reductions.md)
- [Normalization](docs/normalization.md)
- [Matrix multiplication](docs/matmul.md)
- [Shape operations](docs/shape.md)
- [Conversion](docs/conversion.md)
- [Testing](docs/testing.md)

Work is tracked at <https://github.com/communal-software/cl-tensor/issues>.

## License

```text
cl-tensor

Copyright (C) 2026 George Watson

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
```
