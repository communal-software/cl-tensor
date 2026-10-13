# Limitations

Registered dtype/storage extensions, constructors, access, copying, broadcasting operations, expression fusion, reductions, softmax, RMSNorm, shape manipulation, real-float matrix multiplication, dtype conversion and external executor hooks are implemented. The following areas remain open.

| Area | Ticket |
|---|---|
| Inference architectures and production weight formats in `cl-inference` | cl-inference #1 |
| Autograd | [#3](https://github.com/communal-software/cl-tensor/issues/3) |
| Continuous integration | [#4](https://github.com/communal-software/cl-tensor/issues/4) |
| Complete normalization validation on experimental ARM64 CCL | [#9](https://github.com/communal-software/cl-tensor/issues/9) |

## Performance

| Limitation | Ticket |
|---|---|
| Fusion out-forms stage full results; irregular reduction inputs and scalar variables retain complete rows per worker | [#13](https://github.com/communal-software/cl-tensor/issues/13) |
| Fusion uses Lisp packing for irregular strides and per-row setup for compatible broadcast/dependent-pass layouts | [#14](https://github.com/communal-software/cl-tensor/issues/14) |
| Fused selection uses the Lisp backend to preserve lazy branches | [trivial-simd #81](https://github.com/communal-software/trivial-simd/issues/81) |
| Gather out-forms retain full-result scratch | [#11](https://github.com/communal-software/cl-tensor/issues/11) |
| Staged extension execution retains full-result scratch; non-elementwise accessible-storage fallback packs complete inputs | [#10](https://github.com/communal-software/cl-tensor/issues/10) |
| Short elementwise calls pay fixed N-D layout setup overhead | trivial-simd #152 |
| Exact GEMM layout proofs can require substantial search time for difficult layouts | trivial-simd #154 |
| Irregular reductions pack input layouts; reduction out-forms use output scratch | [#7](https://github.com/communal-software/cl-tensor/issues/7) |
| Normalization uses full-result scratch; portable f64 weight/output stages allocate per element, and restoring axis order can require another tensor | [#8](https://github.com/communal-software/cl-tensor/issues/8) |
| Native integer/complex row reduction batching | trivial-simd #148 |
| General output-layout validation uses O(element count) workspace when a sorted-stride uniqueness proof fails | [#5](https://github.com/communal-software/cl-tensor/issues/5), trivial-simd #150 |
| Mixed Lisp/foreign storage conservatively snapshots inputs, including disjoint memory | [#6](https://github.com/communal-software/cl-tensor/issues/6), trivial-simd #151 |
