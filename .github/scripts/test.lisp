(require :asdf)

(load (merge-pathnames "setup.lisp"
                       (uiop:ensure-directory-pathname (uiop:getenv "CI_QUICKLISP_DIR"))))
(asdf:initialize-source-registry
 `(:source-registry
   (:directory ,(uiop:getenv "CL_TENSOR_DIR"))
   (:directory ,(uiop:getenv "TRIVIAL_SIMD_DIR"))
   :inherit-configuration))

(asdf:load-system "cl-tensor/tests")

(unless trivial-simd::*native-available-p*
  (error "CI requires the built trivial-simd native library on both backends"))

(dolist (entry '(("cl-tensor" . "CL_TENSOR_DIR") ("trivial-simd" . "TRIVIAL_SIMD_DIR")))
  (unless (equal (truename (asdf:system-source-directory (car entry)))
                 (truename (uiop:ensure-directory-pathname (uiop:getenv (cdr entry)))))
    (error "Unexpected source directory for ~A" (car entry))))

(let ((expected (uiop:getenv "TRIVIAL_SIMD_BACKEND")))
  (unless (and expected (string-equal expected (trivial-simd:backend)))
    (error "Expected backend ~S, got ~S" expected (trivial-simd:backend))))

(format t "~&~A ~A; backend ~A; trivial-simd ~A at ~A~%"
        (lisp-implementation-type) (lisp-implementation-version)
        (trivial-simd:backend) (asdf:component-version (asdf:find-system "trivial-simd"))
        (asdf:system-source-directory "trivial-simd"))
(asdf:test-system "cl-tensor")
(uiop:quit 0)
