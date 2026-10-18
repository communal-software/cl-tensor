(require :asdf)
(asdf:load-system "cl-tensor/gpu")
(asdf:load-system "cl-gpu/metal")

(defpackage #:cl-tensor/gpu-gather-example
  (:use #:cl)
  (:local-nicknames (#:ct #:cl-tensor) (#:tg #:cl-tensor/gpu))
  (:export #:run-example))

(in-package #:cl-tensor/gpu-gather-example)

(defun run-example ()
  (tg:with-device-tensor (embeddings (tg:to-device (ct:from-data '((1 2) (3 4) (5 6)))))
    (tg:with-device-tensor (ids (tg:to-device (ct:from-data '((2 0) (1 -1)) :dtype :s64)))
      (tg:with-device-tensor (batch (ct:take embeddings ids))
        (tg:with-device-tensor (out (tg:make-device-tensor '(2 2 2)))
          (ct:take! out embeddings ids)
          (values (tg:from-device batch) (tg:from-device out)))))))

(multiple-value-bind (batch out) (run-example)
  (assert (equalp (ct:tensor-storage batch) #(5f0 6f0 1f0 2f0 3f0 4f0 5f0 6f0)))
  (assert (equalp (ct:tensor-storage batch) (ct:tensor-storage out)))
  (format t "Embedding batch ~S: ~S~%" (ct:tensor-shape batch) (ct:tensor-storage batch)))
