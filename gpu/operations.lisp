(in-package #:cl-tensor/gpu)

(defmethod ct:resolve-storage-operation ((storage gpu:gpu-array) operation out inputs options)
  (unless (member operation '(:sum :mean :prod :minimum :maximum :argmin :argmax :softmax :rmsnorm))
    (error 'ct:unsupported-operation :operation operation
           :dtypes (mapcar #'ct:tensor-dtype (remove-if-not #'ct:tensorp inputs))))
  (when (getf options :accumulate) (error "GPU operations do not support widened accumulation"))
  (let ((device (gpu:gpu-array-device storage)))
    (dolist (tensor (append (remove-if-not #'ct:tensorp inputs) (when out (list out))))
      (unless (and (typep (ct:tensor-storage tensor) 'gpu:gpu-array)
                   (eq (tensor-device tensor) device))
        (error "GPU operations require tensors on one device; transfer tensors explicitly"))
      (tensor-view tensor))
    (dolist (input (remove-if-not #'ct:tensorp inputs))
      (unless (eq (ct:tensor-dtype input) :f32) (error "GPU computation requires :F32")))
    (when (and (eq operation :rmsnorm) (second inputs) (not (ct:tensorp (second inputs))))
      (unless (realp (second inputs)) (error "RMSNorm weights must be real"))
      (coerce (second inputs) 'single-float))
    ;; TODO: synchronous full-result staging; add measured safe direct execution (#22).
    (values (if (member operation '(:argmin :argmax)) :s64 :f32)
            (lambda (result operands parameters)
              (let* ((input (tensor-view (first operands)))
                     (target (tensor-view result))
                     (axes (coerce (getf parameters :axis) 'vector)))
                (case operation
                  (:softmax (red:softmax! target input :axis axes))
                  (:rmsnorm
                   (red:rmsnorm! target input :axis axes :epsilon (getf parameters :epsilon)
                                 :weights (if (ct:tensorp (second operands))
                                              (tensor-view (second operands)) (second operands))))
                  (otherwise
                   (funcall (ecase operation
                              (:sum #'red:sum!) (:mean #'red:mean!) (:prod #'red:prod!)
                              (:minimum #'red:minimum!) (:maximum #'red:maximum!)
                              (:argmin #'red:argmin!) (:argmax #'red:argmax!))
                            target input :axis axes :keepdims (getf parameters :keepdims))))
                (gpu:synchronize device))))))
