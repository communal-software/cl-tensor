(require :asdf)
(asdf:load-system :cl-tensor/tests)

(defpackage #:cl-tensor/fusion-bench
  (:use #:cl)
  (:local-nicknames (#:ct #:cl-tensor)))
(in-package #:cl-tensor/fusion-bench)

(ct:define-kernel multiply-add (a b c) (+ (* a b) c))
(ct:define-kernel stable-softmax (x)
  (/ (exp (- x (ct:maximum x)))
     (ct:sum (exp (- x (ct:maximum x))))))

(defun elapsed-microseconds (start)
  (* 1d6 (/ (- (get-internal-real-time) start) internal-time-units-per-second)))

(defun assert-close (a b)
  (assert (equalp (ct:tensor-shape a) (ct:tensor-shape b)))
  (loop for x in (cl-tensor/tests::tensor-values a) for y in (cl-tensor/tests::tensor-values b)
        do (assert (<= (abs (- x y)) (* 1d-5 (max 1d0 (abs y)))))))

(defun measure (name function reference)
  (let* ((start (get-internal-real-time)) (first (funcall function))
         (cold (elapsed-microseconds start)))
    (assert-close first (funcall reference))
    (dotimes (i 3) (funcall function))
    (let ((times
            (loop repeat 3 collect
                  (let ((start (get-internal-real-time)))
                    (dotimes (i 10) (funcall function))
                    (/ (elapsed-microseconds start) 10))))
          (bytes
            #+sbcl (progn
                     (sb-ext:gc :full t)
                     (let ((before (sb-ext:get-bytes-consed)))
                       (dotimes (i 10) (funcall function))
                       (/ (- (sb-ext:get-bytes-consed) before) 10d0)))
            #-sbcl nil))
      (format t "~&~A,~,3F,~,3F,~A~%" name cold (second (sort times #'<)) bytes))))

(format t "~&backend: ~S~%case,first-call-us,warm-median-us,bytes/call~%" trivial-simd::*backend*)
(let ((a (ct:full '(256 256) 0.25)) (b (ct:full '(256 256) 0.5)) (c 1))
  (dolist (layout '(:contiguous :reversed :broadcast))
    (let* ((a (if (eq layout :reversed) (ct:slice a :selectors '(:all (nil nil -1))) a))
           (b (if (eq layout :broadcast) (ct:full '(256) 0.5) b))
           (composed (lambda () (ct:add (ct:multiply a b) c)))
           (fused (lambda () (multiply-add a b c))))
      (measure (format nil "~A-composed" layout) composed composed)
      (measure (format nil "~A-fused" layout) fused composed))))

(multiple-value-bind (executor close) (cl-tensor/tests::make-threaded-test-executor 3)
  (unwind-protect
       (let* ((a (ct:full '(256 256) 0.25)) (weights (ct:full '(256 256) 0.5))
              (composed (lambda ()
                          (let ((exp (ct:exp (ct:subtract a (ct:maximum a :axis -1 :keepdims t)))))
                            (ct:divide exp (ct:sum exp :axis -1 :keepdims t)))))
              (fused (lambda () (stable-softmax a :axis -1))))
         (measure "softmax-composed" composed composed)
         (measure "softmax-fused" fused composed)
         (measure "softmax-fused-pool3" (lambda () (let ((ct:*executor* executor)) (funcall fused))) composed)
         (let ((serial (lambda () (ct:matmul a weights))))
           (measure "matmul-serial" serial serial)
           (measure "matmul-pool3" (lambda () (let ((ct:*executor* executor)) (funcall serial))) serial))
         (let* ((input (ct:full '(2048 256) 0.25)) (serial (lambda () (ct:sum input :axis -1))))
           (measure "reduction-serial" serial serial)
           (measure "reduction-pool3" (lambda () (let ((ct:*executor* executor)) (funcall serial))) serial)))
    (funcall close)))
