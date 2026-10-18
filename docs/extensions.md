# Dtype and storage extensions

External packages register dtype descriptors and implement CLOS methods for storage and operations. Tensor APIs use keyword dtype names. Packed formats keep their layout and kernels in the extension.

## Register a dtype

```lisp
(defclass packed8 (ct:dtype) ())
(ct:register-dtype
 (make-instance 'packed8 :name :my-packed8 :element-type 'single-float
                        :block-size 32 :block-bytes 34 :storage-only t))
```

| Descriptor initarg | Meaning |
|---|---|
| `:name` | Unique keyword, read with `dtype-name` |
| `:element-type` | Logical scalar Lisp type; built-in encoded floats retain their raw-bit type |
| `:block-size` | Positive logical element count per block; default 1 |
| `:block-bytes` | Positive stored bytes per block |
| `:zero` | Default scalar zero for ordinary allocation; default 0 |
| `:storage-only` | Disables ordinary computational fallback; default NIL |
| `:operations` | Optional property list from operation keywords to selector functions |

`find-dtype` returns the registered descriptor. `dtype-block-size` and `dtype-block-bytes` expose its block geometry. `dtype-bytes` returns bytes per logical element, including fractions such as `17/16`; it is not an address calculation for packed storage.

Registration rejects duplicate names. Registering the identical object again succeeds. Registered metadata and `*dtypes*` are read-only by convention; use registration to extend the registry. Register during initialization, before concurrent tensor operations.[^registry]

## Implement storage

| Generic method | Contract |
|---|---|
| `storage-length (storage)` | Logical element capacity |
| `storage-element-type (storage)` | Logical scalar Lisp type |
| `storage-dtype (storage)` | Inferred dtype keyword; override for an encoded storage kind |
| `storage-readable-p (storage)` | Whether logical scalar reads are available; default NIL |
| `storage-writable-p (storage)` | Whether logical scalar writes are available; default NIL |
| `storage-ref (storage index)` | Read one logical scalar |
| `(setf storage-ref) (value storage index)` | Write one logical scalar |

Simple vectors and trivial-simd vector views implement these methods. Missing scalar access signals `unsupported-storage-access`. Packed storage may omit both access methods and rely on operation and copy methods instead.

Offsets and strides count logical elements. Core validates dimensions, stride rank, offsets and reachable bounds; then it calls `validate-storage-view`. Every metadata view goes through this validation. Extensions can reject partial blocks, transposes, broadcasts or other layouts they cannot represent.

## Implement dtype methods

These methods receive a registered descriptor as their first argument.

| Generic method | Contract |
|---|---|
| `allocate-storage (dtype shape)` | Independent zeroed storage for the requested shape |
| `coerce-scalar (dtype value)` | Validate/coerce a scalar constructor or assignment value |
| `validate-storage-view (dtype storage shape strides offset)` | Validate storage compatibility and extension layout restrictions; signal on failure |
| `copy-storage-supported-p (dtype out)` | Whether destination copying is supported |
| `copy-storage! (dtype out input)` | Copy equal-shape, equal-dtype tensors; return `out` |

Defaults allocate ordinary one-element-block arrays, enforce dtype/storage element-type compatibility, and copy through logical scalar access. Packed dtypes supply allocation and copying explicitly. Scalar constructors require writable storage; storage-only descriptors reject ordinary filled constructors. `zeros` uses the allocation method.

`copy-tensor`, copying reshape paths and concatenation use the copy protocol. A packed extension can support these without scalar access, provided its view validator accepts the required layouts. Copy methods receive independent source/destination storage from core; callers of the public copy method must handle aliases themselves or supply disjoint storage.

## Select an operation

Specialize `resolve-operation (dtype operation inputs options)`. Return the result dtype keyword, an executor function and an optional direct selector as multiple values. Return NIL to decline. Selection validates accepted dtype combinations and extension-specific options, without writing storage.

```lisp
(defmethod ct:resolve-operation ((dtype packed8) operation inputs options)
  (when (and (eq operation :matmul)
             (eq (ct:tensor-dtype (first inputs)) :my-packed8)
             (eq (ct:tensor-dtype (second inputs)) :f32))
    (values :f32 #'packed-matmul!)))
```

The executor has signature `(out inputs options)`. It fills all logical result elements, reads inputs without modifying them, and respects their validated layouts. Its return value is ignored. By default, the core allocates an independent result and checks a supplied destination's shape, dtype, uniqueness and copy support before invoking the executor. Destination-writing calls copy the completed result into the caller's destination, preserving aliased inputs.[^staging]

Core computes broadcasting, matmul and reduction shapes. Reduction and normalization axes in hook options are normalized, sorted lists of nonnegative axis numbers; an empty list means no axes. Gather uses one normalized nonnegative integer axis. Extensions choose the result dtype, including mixed matmul such as packed weights × `:f32` activations → `:f32`.

| Operation keys | Inputs and options |
|---|---|
| Elementwise public names as keywords | Operands in public argument order; `:dtype`, `:operator` |
| `:sum`, `:mean`, `:prod`, `:minimum`, `:maximum`, `:argmin`, `:argmax` | One tensor; `:axis`, `:keepdims`, `:accumulate` |
| `:softmax`, `:rmsnorm` | Input, then optional RMSNorm weights; `:axis`, `:accumulate`, `:epsilon` |
| `:take` | Input and integer index tensors; `:axis`; result dtype matches input |
| `:matmul`, `:dot` | Left and right tensors; no options |
| `:convert` | Input tensor; requested `:dtype`, `:rounding` |
| `:dequantize` | Input tensor; requested `:dtype` |

Dtype selection visits distinct input descriptors in operand order, then the requested destination dtype for conversion or an explicit dtype option. The first accepting method wins. Built-in methods decline unless their operation table supplies a selector. An operation-table selector receives `(dtype operation inputs options)` and follows the same multiple-value contract.

Vector–vector matmul tries `:dot`, then `:matmul`. Conversion hooks must return the requested dtype. `(dequantize tensor &key (dtype :f32))` tries `:dequantize`, then ordinary `astype` dispatch. Both return independent storage.

## Storage-specific execution

Storage hooks select placement independently of dtype. The optional [device extension](device.md) implements these hooks for GPU arrays while keeping ordinary dtype names and CPU allocation.

| Generic method | Contract |
|---|---|
| `resolve-storage-operation (storage operation out inputs options)` | Return result dtype, executor and optional direct selector, or NIL to decline |
| `allocate-storage-like (dtype storage shape)` | Independent zeroed storage matching the prototype's placement; default uses `allocate-storage` |
| `storage-copy-supported-p (storage dtype)` | Whether the storage accepts same-dtype copies; default uses `storage-writable-p` |
| `copy-storage-between! (dtype destination-storage source-storage out input)` | Copy tensors through storage-specific methods; default uses scalar access |
| `release-storage (storage)` | Release temporary storage; default does nothing |

Storage selection visits distinct input storage objects in operand order, then destination storage. An accepting selector uses its storage as the result allocation prototype. If all storage selectors decline, dtype selection follows the order above. The executor and direct-selector contracts are identical to dtype selection. A storage selector validates placement combinations before returning and may reject unsupported operations explicitly.

Default `copy-storage-supported-p` and `copy-storage!` methods delegate to the storage copy hooks. Existing dtype-specialized overrides retain control. `copy-tensor` and copying reshape paths allocate through `allocate-storage-like`; ordinary constructors use `allocate-storage`.

Core calls `release-storage` for staged operation results after copying, and for failed operation/copy allocations. Successful allocating calls transfer ownership to the caller. A specialized allocation method cleans up its own partial allocations if it signals before returning.

## Safe direct execution

Return an optional third function to inspect a supplied destination after core checks its shape, dtype, uniqueness and copy support. This direct selector has signature `(out inputs options)`, performs no writes, and returns a direct executor or NIL to use staging. Core rejects non-function selectors and non-function executor results. Allocating calls use the ordinary executor.

```lisp
(values :my-packed8 #'staged-negate! #'select-direct-negate)
```

The direct executor has signature `(out inputs options)` and fills the supplied destination. It guarantees the same result as reading all inputs before any destination writes, including hidden storage aliases. Validation and computation errors leave the destination unchanged. An executor that stages internally follows the final-copy failure contract when committing its completed result.[^staging] Core trusts this declaration; it does not inspect opaque layouts or add snapshots. Its return value is ignored, and the public out-form returns `out`.[^direct]

The [opaque example](../examples/extensions.lisp) accepts separate buffers and exact aliases for negation, checks every code for overflow before writing, and declines shifted aliases. A declined direct selector uses the ordinary executor with full-result staging. Existing two-value selectors also use staging.

## Fallback and example

When all selectors decline, built-in dtypes use existing kernels. Readable custom storage uses ordinary-array backend fallback. Elementwise operations pack broadcasted logical inputs in reusable buffers of at most 4,096 elements per tensor operand. Destination writes retain an independent full result and copy it only after all chunks succeed. Gather retains its storage-specific slice-copy fallback. Other operation families pack complete inputs. Numerical operations on custom dtypes without accepting methods signal `unsupported-operation`; computation does not implicitly dequantize.

Ordinary arrays retain built-in dtype inference even after extensions register scalar types. Encoded/custom storage overrides `storage-dtype` or supplies `:dtype` explicitly.

[The runnable example](../examples/extensions.lisp) defines an opaque format with two signed-byte codes and one float scale per block. Its matmul demonstration decodes before computing; negation operates on codes directly. Production extensions supply their own kernels without changing core.

```lisp
(load "examples/extensions.lisp")
(cl-tensor/extension-example:run-example)
;; => :F32 vector (17 39), and independent decoded weights
```

See [dispatch measurements](extensions-performance.md) for built-in storage and [staging measurements](extensions-staging-performance.md) for custom storage and direct execution.

## Limitations

- Staged executors and elementwise fallback retain O(output elements) scratch. Reduction, normalization, matmul and conversion fallback pack complete custom-storage inputs: [#10](https://github.com/communal-software/cl-tensor/issues/10).
- Production weight formats and model integration belong to the inference project: cl-inference #1.

[^registry]: Descriptor replacement requires a fresh Lisp image. Reloading methods on an existing descriptor class is supported; duplicate registration does not replace its metadata. Concurrent registry mutation is outside the protocol.
[^staging]: Executor failures leave a supplied destination unchanged. A failing final copy method can partially update it; extensions validate their copy prerequisites before writing.
[^direct]: The extension owns any workspace and transaction mechanism. Opting into direct execution does not establish a bounded-memory guarantee. Validate every computation condition that can fail before writing, or provide rollback. Internally staged GPU gather permits partial writes after final-copy submission; its completion wait reports errors before returning.
