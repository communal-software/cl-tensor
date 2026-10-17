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
    (tg:with-device-tensor (copy (ct:copy-tensor (ct:transpose input)))
      (ct:tensor-storage (tg:from-device copy)))))

(format t "~S~%" (run-example))
