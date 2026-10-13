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
