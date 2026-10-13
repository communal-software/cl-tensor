(require :asdf)
(asdf:load-system :cl-tensor)

(defpackage #:cl-tensor/fusion-example
  (:use #:cl)
  (:local-nicknames (#:ct #:cl-tensor)))
(in-package #:cl-tensor/fusion-example)

(ct:define-kernel multiply-add (a b c) (+ (* a b) c))
(ct:define-kernel stable-softmax (x)
  (/ (exp (- x (ct:maximum x)))
     (ct:sum (exp (- x (ct:maximum x))))))

(let* ((a (ct:from-data '((1 2 3) (4 5 6))))
       (b (ct:from-data '(2 3 4)))
       (out (ct:zeros '(2 3))))
  (assert (= 5 (ct:tref (ct:fuse (+ (* a b) 3)) 0 0)))
  (assert (eq out (multiply-add! out a b 3)))
  (assert (= 15 (ct:tref out 0 2)))
  (assert (= 20 (ct:tref (ct:fuse (ct:sum (* a b)) :axis -1) 0)))
  (let ((probabilities (stable-softmax a :axis -1)))
    (dotimes (row 2)
      (assert (< (abs (- 1 (ct:tref (ct:sum probabilities :axis -1) row))) 1e-6)))))

(format t "~&Fusion examples passed.~%")
