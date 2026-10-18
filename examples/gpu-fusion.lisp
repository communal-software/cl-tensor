(require :asdf)
(asdf:load-system "cl-tensor/gpu")
(asdf:load-system "cl-gpu/metal")

(defpackage #:cl-tensor/gpu-fusion-example
  (:use #:cl)
  (:local-nicknames (#:ct #:cl-tensor) (#:tg #:cl-tensor/gpu))
  (:export #:run-example))
(in-package #:cl-tensor/gpu-fusion-example)

(ct:define-kernel affine (a b c) (+ (* a b) c))
(ct:define-kernel stable-softmax (x)
  (/ (exp (- x (ct:maximum x)))
     (ct:sum (exp (- x (ct:maximum x))))))

(defun run-example ()
  (tg:with-device-tensor (input (tg:to-device (ct:from-data '((1 2 3) (4 5 6)))))
    (tg:with-device-tensor (weights (tg:to-device (ct:from-data '(1 2 3))))
      (tg:with-device-tensor (result (affine input weights 1))
        (tg:with-device-tensor (sums (ct:fuse (ct:sum (* input input)) :axis -1))
          (tg:with-device-tensor (probabilities (stable-softmax input :axis -1))
            (affine! result result 2 0)
            (list :affine (ct:tensor-storage (tg:from-device result))
                  :squared-row-sums (ct:tensor-storage (tg:from-device sums))
                  :softmax (ct:tensor-storage (tg:from-device probabilities)))))))))

(format t "~S~%" (run-example))
