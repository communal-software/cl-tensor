(require :asdf)
(asdf:load-system "cl-tensor/gpu")
(asdf:load-system "cl-gpu/metal")

(defpackage #:cl-tensor/device-example
  (:use #:cl)
  (:local-nicknames (#:ct #:cl-tensor) (#:tg #:cl-tensor/gpu))
  (:export #:run-example))

(in-package #:cl-tensor/device-example)

(defun run-example ()
  (tg:with-device-tensor (input (tg:to-device (ct:from-data '((1 2 3) (4 5 6)))))
    (tg:with-device-tensor (weights (tg:to-device (ct:from-data '(1 2 3))))
      (tg:with-device-tensor (sum (ct:sum input :axis 1))
        (tg:with-device-tensor (softmax (ct:softmax input))
          (tg:with-device-tensor (normalized (ct:rmsnorm input :weights weights))
            (list :sum (ct:tensor-storage (tg:from-device sum))
                  :softmax (ct:tensor-storage (tg:from-device softmax))
                  :rmsnorm (ct:tensor-storage (tg:from-device normalized)))))))))

(format t "~S~%" (run-example))
