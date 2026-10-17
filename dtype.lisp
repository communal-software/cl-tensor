(in-package #:cl-tensor)

(defclass dtype ()
  ((name :initarg :name :reader dtype-name)
   (element-type :initarg :element-type :reader dtype-info-element-type)
   (block-size :initarg :block-size :initform 1 :reader dtype-info-block-size)
   (block-bytes :initarg :block-bytes :reader dtype-info-block-bytes)
   (zero :initarg :zero :initform 0 :reader dtype-info-zero)
   (storage-only :initarg :storage-only :initform nil :reader dtype-info-storage-only)
   (operations :initarg :operations :initform nil :reader dtype-operations)))

(defvar *dtypes* nil)
(defvar *dtype-registry* (make-hash-table :test #'eq))
(defvar *builtin-dtypes* nil)

(defun register-dtype (descriptor)
  (check-type descriptor dtype)
  (let ((name (dtype-name descriptor))) (check-type name keyword))
  (unless (and (typep (dtype-info-block-size descriptor) '(integer 1 *))
               (typep (dtype-info-block-bytes descriptor) '(integer 1 *)))
    (error "Dtype block dimensions must be positive integers"))
  (let ((existing (gethash (dtype-name descriptor) *dtype-registry*)))
    (when (and existing (not (eq existing descriptor)))
      (error "Dtype ~S is already registered" (dtype-name descriptor)))
    (unless existing
      (setf (gethash (dtype-name descriptor) *dtype-registry*) descriptor
            *dtypes* (append *dtypes* (list descriptor)))))
  descriptor)

(defun find-dtype (name)
  (or (gethash name *dtype-registry*)
      (error "Unknown dtype ~S" name)))

(defun dtype-element-type (name)
  (dtype-info-element-type (find-dtype name)))

(defun dtype-block-size (name)
  (dtype-info-block-size (find-dtype name)))

(defun dtype-block-bytes (name)
  (dtype-info-block-bytes (find-dtype name)))

(defun dtype-bytes (name)
  (/ (dtype-block-bytes name) (dtype-block-size name)))

(defun dtype-storage-only-p (name)
  (dtype-info-storage-only (find-dtype name)))

(defun dtype-zero (name)
  (dtype-info-zero (find-dtype name)))

(dolist (spec '((:f32 single-float 4 0f0) (:f64 double-float 8 0d0)
                (:c32 (complex single-float) 8 #C(0f0 0f0))
                (:c64 (complex double-float) 16 #C(0d0 0d0))
                (:s8 (signed-byte 8) 1 0) (:u8 (unsigned-byte 8) 1 0)
                (:s16 (signed-byte 16) 2 0) (:u16 (unsigned-byte 16) 2 0)
                (:s32 (signed-byte 32) 4 0) (:u32 (unsigned-byte 32) 4 0)
                (:s64 (signed-byte 64) 8 0) (:u64 (unsigned-byte 64) 8 0)
                (:f16 (unsigned-byte 16) 2 0 t) (:bf16 (unsigned-byte 16) 2 0 t)))
  (destructuring-bind (name type bytes zero &optional storage-only) spec
    (let ((descriptor (or (gethash name *dtype-registry*)
                          (register-dtype (make-instance 'dtype :name name :element-type type
                                                        :block-bytes bytes :zero zero
                                                        :storage-only storage-only)))))
      (pushnew descriptor *builtin-dtypes*))))

(define-condition unsupported-operation (error)
  ((operation :initarg :operation :reader unsupported-operation-name)
   (dtypes :initarg :dtypes :reader unsupported-operation-dtypes))
  (:report (lambda (condition stream)
             (format stream "No implementation of ~S for dtypes ~S"
                     (unsupported-operation-name condition)
                     (unsupported-operation-dtypes condition)))))

(define-condition unsupported-storage-access (error)
  ((storage :initarg :storage :reader unsupported-access-storage)
   (access :initarg :access :reader unsupported-access-kind))
  (:report (lambda (condition stream)
             (format stream "Storage ~S does not support ~S"
                     (unsupported-access-storage condition) (unsupported-access-kind condition)))))

(defgeneric storage-element-type (storage))
(defgeneric storage-length (storage))
(defgeneric storage-dtype (storage))
(defgeneric storage-readable-p (storage))
(defgeneric storage-writable-p (storage))
(defgeneric storage-ref (storage index))
(defgeneric (setf storage-ref) (value storage index))
(defgeneric allocate-storage (dtype shape))
(defgeneric coerce-scalar (dtype value))
(defgeneric validate-storage-view (dtype storage shape strides offset))
(defgeneric copy-storage! (dtype out input))
(defgeneric copy-storage-supported-p (dtype out))
(defgeneric resolve-operation (dtype operation inputs options))
(defgeneric resolve-storage-operation (storage operation out inputs options))
(defgeneric allocate-storage-like (dtype storage shape))
(defgeneric storage-copy-supported-p (storage dtype))
(defgeneric copy-storage-between! (dtype destination-storage source-storage out input))
(defgeneric release-storage (storage))

(defmethod resolve-storage-operation ((storage t) operation out inputs options)
  (declare (ignore operation out inputs options))
  nil)

(defmethod allocate-storage-like ((descriptor dtype) (storage t) shape)
  (allocate-storage descriptor shape))

(defmethod storage-copy-supported-p ((storage t) descriptor)
  (declare (ignore descriptor))
  (storage-writable-p storage))

(defmethod release-storage ((storage t)) nil)

(defmethod resolve-operation ((descriptor dtype) operation inputs options)
  (let ((selector (getf (dtype-operations descriptor) operation)))
    (when selector (funcall selector descriptor operation inputs options))))

(defmethod storage-element-type ((storage trivial-simd:vector-view))
  (upgraded-array-element-type (dtype-element-type (trivial-simd:vector-view-type storage))))

(defmethod storage-element-type ((storage vector))
  (unless (typep storage '(simple-array * (*)))
    (error "Storage must be a simple vector"))
  (array-element-type storage))

(defmethod storage-length ((storage trivial-simd:vector-view))
  (trivial-simd:vector-view-length storage))

(defmethod storage-length ((storage vector))
  (unless (typep storage '(simple-array * (*)))
    (error "Storage must be a simple vector"))
  (length storage))

(defmethod storage-readable-p ((storage t)) nil)
(defmethod storage-writable-p ((storage t)) nil)
(defmethod storage-readable-p ((storage vector)) (typep storage '(simple-array * (*))))
(defmethod storage-writable-p ((storage vector)) (typep storage '(simple-array * (*))))
(defmethod storage-readable-p ((storage trivial-simd:vector-view)) t)
(defmethod storage-writable-p ((storage trivial-simd:vector-view)) t)

(defun dtype-matches-storage-p (name storage)
  (equal (upgraded-array-element-type (dtype-element-type name))
         (storage-element-type storage)))

(defmethod storage-dtype ((storage t))
  (let ((match (find-if (lambda (descriptor)
                          (and (not (dtype-info-storage-only descriptor))
                               (dtype-matches-storage-p (dtype-name descriptor) storage)))
                        *builtin-dtypes*)))
    (if match (dtype-name match)
        (error "No dtype for storage of element type ~S" (storage-element-type storage)))))

(defmethod allocate-storage ((descriptor dtype) shape)
  (unless (= 1 (dtype-info-block-size descriptor))
    (error 'unsupported-storage-access :storage descriptor :access :allocate))
  (make-array (reduce #'* shape) :element-type (dtype-info-element-type descriptor)
                               :initial-element (dtype-info-zero descriptor)))

(defmethod validate-storage-view ((descriptor dtype) storage shape strides offset)
  (declare (ignore shape strides offset))
  (unless (dtype-matches-storage-p (dtype-name descriptor) storage)
    (error "Storage of element type ~S cannot hold dtype ~S"
           (storage-element-type storage) (dtype-name descriptor))))
