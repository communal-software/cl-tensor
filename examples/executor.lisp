(require :asdf)
(asdf:load-system :cl-tensor)

(defpackage #:cl-tensor/executor-example
  (:use #:cl)
  (:local-nicknames (#:ct #:cl-tensor)))
(in-package #:cl-tensor/executor-example)

(defun range-executor (count callback)
  (loop for start from 0 below count by 2
        do (funcall callback start (min count (+ start 2)))))

(let ((ct:*executor* #'range-executor)
      (a (ct:from-data '((1 2) (3 4) (5 6))))
      (b (ct:eye 2)))
  (assert (= 6 (ct:tref (ct:matmul a b) 2 1)))
  (assert (= 11 (ct:tref (ct:sum a :axis -1) 2)))
  (assert (= 61 (ct:tref (ct:fuse (ct:sum (* a a)) :axis -1) 2))))

(format t "~&Executor examples passed.~%")
