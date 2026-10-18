(asdf:defsystem "cl-tensor"
  :description "NumPy-like N-dimensional tensors for Common Lisp, built on trivial-simd"
  :author "George Watson"
  :license "GPL-3.0-or-later"
  :version "0.0.0"
  :depends-on ("trivial-simd" "trivial-simd/blas" "cffi")
  :serial t
  :components ((:file "package") (:file "dtype") (:file "tensor")
               (:file "access") (:file "dispatch") (:file "constructors") (:file "elementwise")
               (:file "shape") (:file "execution") (:file "reductions") (:file "fusion")
               (:file "normalization") (:file "matmul"))
  :in-order-to ((asdf:test-op (asdf:test-op "cl-tensor/tests"))))

(asdf:defsystem "cl-tensor/tests"
  :depends-on ("cl-tensor" "fiveam" "bordeaux-threads")
  :serial t
  :components ((:file "tests/package") (:file "tests/tensor")
               (:file "tests/constructors") (:file "tests/elementwise") (:file "tests/random")
               (:file "tests/shape") (:file "tests/reductions") (:file "tests/normalization")
               (:file "tests/matmul") (:file "tests/fusion") (:file "tests/execution")
               (:file "examples/extensions") (:file "tests/extensions"))
  :perform (asdf:test-op (op component)
             (declare (ignore op component))
             (unless (uiop:symbol-call :cl-tensor/tests :run-tests)
               (error "cl-tensor tests failed"))))

(asdf:defsystem "cl-tensor/gpu"
  :description "Explicit GPU tensor storage and tensor operations"
  :depends-on ("cl-tensor" "cl-gpu/reductions" "cl-gpu/blas" "cl-gpu/elementwise")
  :serial t
  :components ((:file "gpu/package")
               (:file "gpu/storage") (:file "gpu/elementwise") (:file "gpu/operations"))
  :in-order-to ((asdf:test-op (asdf:test-op "cl-tensor/gpu/tests"))))

(asdf:defsystem "cl-tensor/gpu/tests"
  :depends-on ("cl-tensor/gpu" "cl-tensor/tests")
  :serial t
  :components ((:file "tests/gpu") (:file "tests/gpu-operations") (:file "tests/gpu-elementwise"))
  :perform (asdf:test-op (op component)
             (declare (ignore op component))
             (unless (uiop:symbol-call :cl-tensor/gpu/tests :run-tests)
               (error "cl-tensor GPU protocol tests failed"))))

(asdf:defsystem "cl-tensor/gpu/metal/tests"
  :depends-on ("cl-tensor/gpu/tests" "cl-gpu/metal")
  :serial t
  :components ((:file "tests/gpu-metal") (:file "tests/gpu-elementwise-metal"))
  :perform (asdf:test-op (op component)
             (declare (ignore op component))
             (unless (uiop:symbol-call :cl-tensor/gpu/tests :run-metal-tests)
               (error "cl-tensor Metal tests failed"))))
