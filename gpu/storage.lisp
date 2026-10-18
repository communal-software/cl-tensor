(in-package #:cl-tensor/gpu)

(defparameter *device-elements*
  '((:f32 . gpu:f32) (:f16 . gpu:f16) (:bf16 . gpu:bf16)
    (:s8 . gpu:i8) (:u8 . gpu:u8) (:s16 . gpu:i16) (:u16 . gpu:u16)
    (:s32 . gpu:i32) (:u32 . gpu:u32) (:s64 . gpu:i64) (:u64 . gpu:u64)))

(defun device-element (dtype)
  (or (cdr (assoc dtype *device-elements*))
      (error "Unsupported device storage dtype ~S" dtype)))

(defun array-dtype (array)
  (or (car (rassoc (gpu:scalar-type-name (gpu:gpu-array-element-type array)) *device-elements*))
      (error "Unsupported device storage element type")))

(defun tensor-view (tensor)
  (gpu:make-strided-view (ct:tensor-storage tensor) (ct:tensor-shape tensor)
                         :strides (ct:tensor-strides tensor) :offset (ct:tensor-offset tensor)))

(defun root-array (array)
  (if (gpu:gpu-array-view-p array) (gpu:gpu-array-view-owner array) array))

(defun tensor-device (tensor)
  (gpu:gpu-array-device (ct:tensor-storage tensor)))

(defmethod ct:storage-length ((storage gpu:gpu-array)) (gpu:gpu-array-length storage))
(defmethod ct:storage-dtype ((storage gpu:gpu-array)) (array-dtype storage))
(defmethod ct:storage-element-type ((storage gpu:gpu-array))
  (upgraded-array-element-type (ct:dtype-element-type (array-dtype storage))))

(defmethod ct:validate-storage-view :after ((dtype ct:dtype) (storage gpu:gpu-array) shape strides offset)
  (unless (eq (ct:dtype-name dtype) (array-dtype storage))
    (error "Device storage dtype does not match tensor dtype"))
  (gpu:make-strided-view storage shape :strides strides :offset offset))

(defmethod ct:allocate-storage-like ((dtype ct:dtype) (storage gpu:gpu-array) shape)
  (gpu:make-gpu-array (reduce #'* shape) :element-type (device-element (ct:dtype-name dtype))
                     :device (gpu:gpu-array-device storage)))

(defmethod ct:storage-copy-supported-p ((storage gpu:gpu-array) dtype)
  (eq (array-dtype storage) (ct:dtype-name dtype)))

(defmethod ct:release-storage ((storage gpu:gpu-array)) (gpu:free storage))

(defun make-device-tensor (shape &key (dtype :f32) device)
  "Allocate an independent zeroed tensor of a supported device dtype."
  (unless (every (lambda (n) (typep n '(integer 0 #.most-positive-fixnum))) shape)
    (error "Shape must contain nonnegative dimensions"))
  (let ((storage (gpu:make-gpu-array (reduce #'* shape) :element-type (device-element dtype)
                                   :device device))
        (success nil))
    (unwind-protect
         (let ((tensor (ct:make-tensor-view storage shape :dtype dtype)))
           (gpu:synchronize (gpu:gpu-array-device storage))
           (setf success t)
           tensor)
      (unless success (gpu:free storage)))))

(defun free-device-tensor (tensor)
  "Release TENSOR's GPU array, invalidating all tensors sharing that owning array."
  (check-type tensor ct:tensor)
  (unless (typep (ct:tensor-storage tensor) 'gpu:gpu-array)
    (error "Expected a device tensor"))
  (gpu:free (ct:tensor-storage tensor)))

(defmacro with-device-tensor ((name form) &body body)
  `(let ((,name ,form))
     (unwind-protect (progn ,@body) (free-device-tensor ,name))))

(defun copy-device-values! (out input)
  (ew:convert! (tensor-view out) (tensor-view input))
  (gpu:synchronize (tensor-device input))
  out)

(defmethod ct:copy-storage-between! ((dtype ct:dtype) (destination gpu:gpu-array)
                                    (source gpu:gpu-array) out input)
  (unless (and (eq (ct:tensor-dtype out) (ct:tensor-dtype input))
               (eq (ct:dtype-name dtype) (ct:tensor-dtype input))
               (equalp (ct:tensor-shape out) (ct:tensor-shape input))
               (eq (tensor-device out) (tensor-device input)))
    (error "Device copies require matching shape, dtype and device"))
  (tensor-view out)
  (tensor-view input)
  (unless (cl-tensor::unique-output-p out) (error "Output elements overlap"))
  ;; TODO: shared-root copies stage disjoint ranges; prove disjointness before staging (#22).
  (if (eq (root-array destination) (root-array source))
      (with-device-tensor (scratch (make-device-tensor (ct:tensor-shape input)
                                                      :dtype (ct:tensor-dtype input)
                                                      :device (tensor-device input)))
        (copy-device-values! scratch input)
        (copy-device-values! out scratch))
      (copy-device-values! out input)))

(defmethod ct:copy-storage-between! ((dtype ct:dtype) (destination gpu:gpu-array)
                                    (source t) out input)
  (declare (ignore out input))
  (error "Use TO-DEVICE to upload a CPU tensor"))

(defmethod ct:copy-storage-between! ((dtype ct:dtype) (destination t)
                                    (source gpu:gpu-array) out input)
  (declare (ignore out input))
  (error "Use FROM-DEVICE to download a device tensor"))

(defun to-device (tensor &key device)
  "Upload a CPU tensor into independent contiguous device storage."
  (check-type tensor ct:tensor)
  (let* ((storage (ct:tensor-storage tensor)) (dtype (ct:tensor-dtype tensor))
         (element (device-element dtype)) (n (ct:tensor-size tensor)))
    (unless (ct:storage-readable-p storage) (error "TO-DEVICE requires readable CPU storage"))
    (let* ((values (if (and (typep storage '(simple-array * (*))) (ct:contiguous-p tensor))
                       (if (and (zerop (ct:tensor-offset tensor)) (= n (length storage))) storage
                           (subseq storage (ct:tensor-offset tensor) (+ (ct:tensor-offset tensor) n)))
                       (let ((values (make-array n :element-type (ct:dtype-element-type dtype))))
                         (dotimes (i n values)
                           (let ((q i) (offset (ct:tensor-offset tensor)))
                             (loop for axis downfrom (1- (ct:tensor-rank tensor)) to 0
                                   do (multiple-value-bind (next coordinate) (floor q (aref (ct:tensor-shape tensor) axis))
                                        (incf offset (* coordinate (aref (ct:tensor-strides tensor) axis)))
                                        (setf q next)))
                             (setf (aref values i) (ct:storage-ref storage offset)))))))
           (array (gpu:to-device values :element-type element :device device
                                 :raw (member dtype '(:f16 :bf16)))) (success nil))
      (unwind-protect
           (let ((result (ct:make-tensor-view array (ct:tensor-shape tensor) :dtype dtype)))
             (gpu:synchronize (gpu:gpu-array-device array))
             (setf success t)
             result)
        (unless success (gpu:free array))))))

(defun from-device (tensor)
  "Download logical values into an independent contiguous CPU tensor."
  (check-type tensor ct:tensor)
  (unless (typep (ct:tensor-storage tensor) 'gpu:gpu-array)
    (error "Expected a device tensor"))
  (tensor-view tensor)
  (if (ct:contiguous-p tensor)
      (let* ((window (gpu:make-gpu-array-view (ct:tensor-storage tensor) (ct:tensor-size tensor)
                                               :offset (ct:tensor-offset tensor)))
             (values (gpu:from-device window :raw (member (ct:tensor-dtype tensor) '(:f16 :bf16)))))
        (ct:make-tensor-view values (ct:tensor-shape tensor) :dtype (ct:tensor-dtype tensor)))
      (with-device-tensor (packed (ct:copy-tensor tensor)) (from-device packed))))
