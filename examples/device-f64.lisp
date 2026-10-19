(require :asdf)
(asdf:load-system "cl-tensor/gpu")
(asdf:load-system "cl-gpu/opencl")

(defpackage #:cl-tensor/device-f64-example
  (:use #:cl)
  (:local-nicknames (#:ct #:cl-tensor) (#:tg #:cl-tensor/gpu) (#:gpu #:cl-gpu))
  (:export #:run-example))
(in-package #:cl-tensor/device-f64-example)

(defun run-example (&optional (device (cl-gpu/opencl:default-device)))
  (unless (gpu:device-native-type-p device 'gpu:f64)
    (error "Select an OpenCL device with native f64 support"))
  (tg:with-device-tensor (input (tg:to-device (ct:from-data '(16777217d0 1d0) :dtype :f64)
                                            :device device))
    (tg:with-device-tensor (shifted (ct:add input 1.0000000000000002d0))
      (tg:with-device-tensor (sum (ct:sum input :accumulate :f64))
        (list :add (ct:tensor-storage (tg:from-device shifted))
              :sum (ct:tref (tg:from-device sum)))))))
