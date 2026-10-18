# External executors

Bind `ct:*executor*` to a synchronous range executor to schedule CPU matmul output rows and ordinary or fused CPU reduction groups. NIL runs sequentially. Core creates no threads and owns no worker pool. Device operations use their GPU backend and do not invoke this hook.

```lisp
(let ((ct:*executor* executor))
  (ct:matmul weights input)
  (ct:sum activations :axis -1))
```

## Contract

The executor is a function `(count callback)`. It calls `(callback start end)` for disjoint half-open ranges covering `[0,count)` exactly once. Bounds must be integers satisfying `0 <= start <= end <= count`; empty ranges are allowed. Core checks each range's bounds. Coverage and disjointness are executor obligations.

The executor selects partition sizes, worker count and scheduling thresholds. It may execute serially. It waits for every started callback before returning, and waits for the remaining workers before propagating a worker error. Return values are ignored.

Zero groups skip scheduling; one group runs serially. Validation and zero-width identity cases can complete without calling the executor.

```lisp
(defun range-executor (count callback)
  (loop for start from 0 below count by 64
        do (funcall callback start (min count (+ start 64)))))
```

[The runnable example](../examples/executor.lisp) uses a serial range executor. The tests use a reusable three-worker pool and verify worker completion after failures.

## Operation groups

| Operation | One group |
|---|---|
| Matrix/batched matmul | One output row, in batch then row order |
| Matrix-vector or vector-matrix product | One output element, in batch then element order |
| Axis reduction | One retained-axis coordinate |
| Fused expression with reductions | One retained-axis coordinate shared by all nested reducers |

Contractions remain intact. Dot products and whole-tensor reductions have one group and remain sequential. Parallel scheduling does not change the reduction-axis enumeration or arg tie rule, although backend batching can affect floating-point results.

Inputs and destinations are validated before scheduling. Matmul snapshots overlapping inputs before any worker writes. Each callback owns its mutable scratch and adapters; reductions and fusion stage results before destination copies.

Callbacks inherit the operation's backend/FMA bindings and run with `*executor*` bound to NIL, preventing recursive scheduling. Encoded matmul establishes storage address contexts in workers. Foreign memory remains caller-owned and must stay valid until the executor finishes.[^bindings]

## Pool adapters

A pool accepting `(worker-id worker-count)` jobs adapts without a cl-tensor dependency on that pool:

```lisp
(defun pool-executor (pool run-pool)
  (lambda (count callback)
    (funcall run-pool pool
             (lambda (id workers)
               (funcall callback
                        (floor (* count id) workers)
                        (floor (* count (1+ id)) workers))))))
```

For cl-qwen's existing internal pool, supply `#'cl-qwen::run-pool`. It waits for workers and propagates errors; the adapter captures the pool. The caller creates and closes the pool.

Custom operation selectors control their own execution. Core does not automatically partition opaque/custom kernels. A selector may choose to use the dynamic executor under its own contract.

## Errors

Reduction and fusion computation failures leave supplied destinations unchanged. Matmul may leave a partially written destination, matching its direct execution behavior. Once an error reaches the caller, no worker may still access operation buffers.

## Limitations

- The executor must guarantee complete, disjoint coverage and completion; core does not retain an O(group count) coverage map.
- Single contractions remain sequential: [#18](https://github.com/communal-software/cl-tensor/issues/18).
- Per-call executor overrides are unsupported: [#20](https://github.com/communal-software/cl-tensor/issues/20).
- Automatic elementwise/normalization scheduling and built-in pool management are outside this interface.
- More workers can increase overhead and scratch allocation. Choose thresholds using [complete-call measurements](fusion-performance.md).

[^bindings]: Core propagates trivial-simd backend and FMA settings. Executors remain responsible for any additional application bindings and worker floating-point environment. Concurrent mutation of inputs, tensor metadata or dtype registration is outside the operation contract.
