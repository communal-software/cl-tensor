(defpackage #:cl-tensor
  (:use #:cl)
  (:shadow #:abs #:sqrt #:min #:max #:concatenate #:log #:tanh #:exp #:sin #:cos)
  (:export #:dtype #:dtype-name #:register-dtype #:find-dtype
           #:dtype-block-size #:dtype-block-bytes #:dtype-zero
           #:storage-length #:storage-element-type #:storage-readable-p #:storage-writable-p
           #:storage-ref #:allocate-storage #:coerce-scalar #:validate-storage-view
           #:copy-storage! #:copy-storage-supported-p #:resolve-operation
           #:resolve-storage-operation #:allocate-storage-like #:storage-copy-supported-p
           #:copy-storage-between! #:release-storage
           #:unsupported-operation #:unsupported-operation-name #:unsupported-operation-dtypes
           #:unsupported-storage-access #:unsupported-access-storage #:unsupported-access-kind
           #:dequantize #:*dtypes*
           #:dtype-element-type
           #:dtype-bytes
           #:dtype-storage-only-p
           #:storage-dtype
           #:tensor
           #:tensorp
           #:tensor-storage
           #:tensor-dtype
           #:tensor-shape
           #:tensor-strides
           #:tensor-offset
           #:tensor-rank
           #:tensor-size
           #:tensor-index
           #:row-major-strides
           #:make-tensor
           #:make-tensor-view
           #:contiguous-p
           #:inner-contiguous-p
           #:zeros #:ones #:full #:from-data #:arange #:linspace #:eye
           #:tref #:copy-tensor #:astype
           #:reshape #:transpose #:slice #:squeeze #:expand-dims #:concatenate #:stack #:split
           #:take #:take!
           #:matmul #:matmul! #:*executor*
           #:fuse #:fuse! #:define-kernel
           #:softmax #:softmax! #:rmsnorm #:rmsnorm!
           #:sum #:sum! #:mean #:mean! #:minimum #:minimum! #:maximum #:maximum!
           #:argmin #:argmin! #:argmax #:argmax! #:prod #:prod!
           #:add #:add! #:subtract #:subtract! #:multiply #:multiply! #:divide #:divide!
           #:log #:log! #:tanh #:tanh! #:sigmoid #:sigmoid!
           #:exp #:exp! #:sin #:sin! #:cos #:cos! #:silu #:silu! #:gelu #:gelu!
           #:negate #:negate! #:abs #:abs! #:sqrt #:sqrt! #:reciprocal #:reciprocal!
           #:min #:min! #:max #:max! #:clamp #:clamp!
           #:compare #:compare! #:select #:select!))
