(defpackage #:cl-tensor/gpu
  (:use #:cl)
  (:local-nicknames (#:ct #:cl-tensor) (#:gpu #:cl-gpu) (#:red #:cl-gpu/reductions)
                    (#:blas #:cl-gpu/blas) (#:ew #:cl-gpu/elementwise))
  (:export #:make-device-tensor #:to-device #:from-device
           #:free-device-tensor #:with-device-tensor))
