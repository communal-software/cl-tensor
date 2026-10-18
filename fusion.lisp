(in-package #:cl-tensor)

(defstruct fusion-program function keys parameters body reducers outer device-plan)

(defgeneric execute-storage-fusion (storage program operands out axis keepdims))

(defmethod execute-storage-fusion (storage program operands out axis keepdims)
  (declare (ignore storage program operands out axis keepdims))
  (values nil nil))

(defun execute-fusion (program operands out axis keepdims)
  (dolist (tensor (if out (append (remove-if-not #'tensorp operands) (list out))
                     (remove-if-not #'tensorp operands)))
    (multiple-value-bind (result handled)
        (execute-storage-fusion (tensor-storage tensor) program operands out axis keepdims)
      (when handled (return-from execute-fusion result))))
  (run-fusion (fusion-program-function program) (fusion-program-keys program) operands
              (fusion-program-reducers program) (fusion-program-outer program) out axis keepdims))

(eval-when (:compile-toplevel :load-toplevel :execute)
  (defun fusion-operator (operator)
    (case operator
      ((cl:+ add) 'cl:+) ((cl:- subtract negate) 'cl:-)
      ((cl:* multiply) 'cl:*) ((cl:/ divide reciprocal) 'cl:/)
      ((cl:abs abs) 'cl:abs) ((cl:sqrt sqrt) 'cl:sqrt)
      ((cl:min min) 'cl:min) ((cl:max max) 'cl:max)
      ((cl:exp exp) 'cl:exp) ((cl:sin sin) 'cl:sin) ((cl:cos cos) 'cl:cos)
      ((cl:log log) 'cl:log) ((cl:tanh tanh) 'cl:tanh)
      ((sigmoid trivial-simd:sigmoid) 'trivial-simd:sigmoid)
      (trivial-simd:fma 'trivial-simd:fma)
      ((select trivial-simd:select) 'trivial-simd:select)
      ((sum trivial-simd:sum) 'trivial-simd:sum)
      ((prod trivial-simd:prod) 'trivial-simd:prod)
      ((minimum trivial-simd:minimum) 'trivial-simd:minimum)
      ((maximum trivial-simd:maximum) 'trivial-simd:maximum)
      ((argmin trivial-simd:argmin) 'trivial-simd:argmin)
      ((argmax trivial-simd:argmax) 'trivial-simd:argmax)
      ((cl:= cl:/= cl:< cl:<= cl:> cl:>=) operator)
      (t (error "Unsupported fusion operator ~S" operator))))

  (defun fusion-reducer (operator)
    (case operator
      (trivial-simd:sum :sum) (trivial-simd:prod :prod)
      (trivial-simd:minimum :minimum) (trivial-simd:maximum :maximum)
      (trivial-simd:argmin :argmin) (trivial-simd:argmax :argmax)))

  (defun parse-fusion (expression)
    (let ((leaves nil) (parameters nil) (reducers nil))
      (labels ((walk (form)
                 (cond
                   ((realp form) (values form :numeric))
                   ((and (symbolp form) form (not (eq form t)) (not (keywordp form)))
                    (let ((entry (assoc form leaves)))
                      (unless entry
                        (setf entry (cons form (gensym "INPUT")))
                        (push entry leaves)
                        (push (cdr entry) parameters))
                      (values (cdr entry) :numeric)))
                   ((consp form)
                    (let* ((operator (fusion-operator (first form)))
                           (reducer (fusion-reducer operator))
                           (arguments (rest form))
                           (count (length arguments))
                           (comparison (member operator '(cl:= cl:/= cl:< cl:<= cl:> cl:>=)))
                           (selection (eq operator 'trivial-simd:select))
                           (arity (cond ((or selection (eq operator 'trivial-simd:fma)) 3)
                                        (comparison 2)
                                        ((or reducer
                                             (member operator '(cl:abs cl:sqrt cl:exp cl:sin cl:cos
                                                                cl:log cl:tanh trivial-simd:sigmoid))) 1))))
                      (unless (if arity (= count arity) (plusp count))
                        (error "Invalid fusion arity in ~S" form))
                      (when reducer (pushnew reducer reducers))
                      (let ((lowered
                              (loop for argument in arguments for index from 0
                                    collect (multiple-value-bind (body kind) (walk argument)
                                              (unless (eq kind (if (and selection (zerop index)) :mask :numeric))
                                                (error "Invalid fusion operand in ~S" form))
                                              body))))
                        (values (cons operator lowered) (if comparison :mask :numeric)))))
                   (t (error "Invalid fusion expression ~S" form)))))
        (multiple-value-bind (body kind) (walk expression)
          (unless (eq kind :numeric) (error "Fusion requires a numeric result"))
          (values body (mapcar #'car (nreverse leaves)) (nreverse parameters)
                  reducers (and (consp body) (fusion-reducer (first body))))))))

  (defun fusion-keys (parameters)
    (mapcar (lambda (parameter)
              (list (intern (format nil "~A-START" parameter) :keyword)
                    (intern (format nil "~A-ROW-STRIDE" parameter) :keyword)))
            parameters))

  (defun fusion-program-form (body parameters reducers outer)
    (let ((kernel (gensym "FUSED-KERNEL")))
      `(progn
         (trivial-simd:define-kernel ,kernel ,parameters ,body)
         (make-fusion-program :function (symbol-function ',kernel)
                              :keys ',(fusion-keys parameters) :parameters ',parameters
                              :body ',body :reducers ',reducers :outer ',outer))))

  (defun fusion-call-form (expression out axis keepdims)
    (multiple-value-bind (body leaves parameters reducers outer) (parse-fusion expression)
      `(execute-fusion
        (load-time-value ,(fusion-program-form body parameters reducers outer))
        (list ,@leaves) ,out ,axis ,keepdims))))

(defmacro fuse (expression &key axis keepdims)
  (fusion-call-form expression nil axis keepdims))

(defmacro fuse! (out expression &key axis keepdims)
  (let ((target (gensym "OUT")))
    `(let ((,target ,out))
       (check-type ,target tensor)
       ,(fusion-call-form expression target axis keepdims))))

(defmacro define-kernel (name arguments expression)
  (unless (and (symbolp name) name (not (eq name t)) (not (keywordp name)) (listp arguments)
               (every (lambda (argument)
                        (and (symbolp argument) argument (not (eq argument t)) (not (keywordp argument))
                             (not (member argument lambda-list-keywords))))
                      arguments)
               (= (length arguments) (length (remove-duplicates arguments))))
    (error "Invalid tensor kernel definition"))
  (multiple-value-bind (body leaves parameters reducers outer) (parse-fusion expression)
    (unless (every (lambda (leaf) (member leaf arguments)) leaves)
      (error "Tensor kernel expression references an undeclared argument"))
    (let ((program (gensym "PROGRAM")) (out (gensym "OUT"))
          (axis (gensym "AXIS")) (keepdims (gensym "KEEPDIMS"))
          (out-name (intern (format nil "~A!" name) (or (symbol-package name) *package*)))
          (unused (set-difference arguments leaves)))
      `(let ((,program ,(fusion-program-form body parameters reducers outer)))
         (defun ,name (,@arguments &key ((:axis ,axis)) ((:keepdims ,keepdims)))
           ,@(when unused `((declare (ignore ,@unused))))
           (execute-fusion ,program (list ,@leaves) nil ,axis ,keepdims))
         (defun ,out-name (,out ,@arguments &key ((:axis ,axis)) ((:keepdims ,keepdims)))
           ,@(when unused `((declare (ignore ,@unused))))
           (check-type ,out tensor)
           (execute-fusion ,program (list ,@leaves) ,out ,axis ,keepdims))))))

(defun fusion-slice (function parameters arrays starts destination destination-start count outer)
  (let ((options (loop for (start-key) in parameters for start in starts append (list start-key start))))
    (if outer
        (apply function (append arrays (list :end count) options))
        (apply function destination
               (append arrays (list :end count :destination-start destination-start) options)))))

(defun fusion-buffers (inputs capacity dtype direct)
  (loop for input in inputs for use-direct in direct
        collect (unless use-direct
                  (make-array capacity :element-type (dtype-element-type dtype)
                                       :initial-element (if (tensorp input) (dtype-zero dtype) input)))))

(defun fusion-elementwise (function parameters inputs result)
  (let* ((shape (tensor-shape result)) (size (tensor-size result))
         (capacity (cl:min size +elementwise-pack-size+))
         (strides (row-major-strides shape))
         (direct (every (lambda (input)
                          (or (not (tensorp input))
                              (and (backend-storage-p (tensor-storage input))
                                   (equalp (broadcast-strides input shape) strides))))
                        inputs)))
    (when (zerop size) (return-from fusion-elementwise result))
    (when (plusp (length shape))
      (let* ((axis (1- (length shape))) (width (aref shape axis)))
        (when (and (not direct) (<= width +elementwise-pack-size+)
                   (every (lambda (input)
                            (or (not (tensorp input))
                                (and (backend-storage-p (tensor-storage input))
                                     (or (= width 1) (= 1 (aref (broadcast-strides input shape) axis))))))
                          inputs))
          (let ((*executor* nil))
            (fusion-grouped function parameters inputs shape (list axis)
                            (loop for kept below axis collect kept) width (/ size width) result nil))
          (return-from fusion-elementwise result))))
    (let* ((buffers (fusion-buffers inputs capacity (tensor-dtype result)
                                   (mapcar (lambda (input) (and direct (tensorp input))) inputs)))
           (arrays (loop for input in inputs for buffer in buffers
                         collect (if (and direct (tensorp input)) (tensor-storage input) buffer)))
           (start 0) (chunk 0))
      (flet ((execute (count)
               (fusion-slice function parameters arrays
                             (mapcar (lambda (input)
                                       (if (and direct (tensorp input)) (+ (tensor-offset input) start) 0))
                                     inputs)
                             (tensor-storage result) start count nil)
               (incf start count)))
        (if direct
            (loop while (< start size) do (execute (cl:min capacity (- size start))))
            ;; TODO: scalar strided packing; use upstream N-D expression execution (#14).
            (let* ((tensors (remove-if-not #'tensorp inputs))
                   (packed (loop for input in inputs for buffer in buffers when (tensorp input) collect buffer)))
              (call-with-offsets
               shape (mapcar (lambda (input) (broadcast-strides input shape)) tensors)
               (mapcar #'tensor-offset tensors)
               (lambda (offsets)
                 (loop for input in tensors for buffer in packed for offset in offsets
                       do (setf (aref buffer chunk) (storage-ref (tensor-storage input) offset)))
                 (incf chunk)
                 (when (= chunk capacity) (execute chunk) (setf chunk 0))) t)
              (when (plusp chunk) (execute chunk))))))
    result))

(defun fusion-layout-step (shape strides)
  (let ((step 0) (count 1))
    (loop for axis downfrom (1- (length shape)) to 0
          for dimension = (aref shape axis)
          when (> dimension 1)
            do (if (= count 1) (setf step (aref strides axis))
                   (unless (= (aref strides axis) (* count step))
                     (return-from fusion-layout-step nil)))
               (setf count (* count dimension)))
    step))

(defun fusion-batch (function parameters inputs arrays group-shape group-layouts start end width result outer)
  (let ((options
          (loop for (start-key stride-key) in parameters for input in inputs for strides in group-layouts
                append (list start-key
                             (if (tensorp input)
                                 (linear-layout-offset start group-shape strides (tensor-offset input)) 0)
                             stride-key
                             (if (tensorp input) (fusion-layout-step group-shape strides) 0)))))
    (if outer
        (apply function (append arrays (list :rows (- end start) :row-length width
                                            :destination (tensor-storage result) :destination-start start) options))
        (apply function (tensor-storage result)
               (append arrays (list :rows (- end start) :row-length width
                                    :destination-start (* start width) :destination-row-stride width) options)))))

(defun fusion-grouped (function parameters inputs shape axes kept width rows result outer)
  (let* ((group-shape (index-vector (mapcar (lambda (axis) (aref shape axis)) kept)))
         (row-shape (index-vector (mapcar (lambda (axis) (aref shape axis)) axes)))
         (layouts (mapcar (lambda (input) (when (tensorp input) (broadcast-strides input shape))) inputs))
         (group-layouts (mapcar (lambda (strides)
                                 (when strides (index-vector (mapcar (lambda (axis) (aref strides axis)) kept))))
                               layouts))
         (row-layouts (mapcar (lambda (strides)
                               (when strides (index-vector (mapcar (lambda (axis) (aref strides axis)) axes))))
                             layouts))
         (result-group-strides
           (unless outer (index-vector (mapcar (lambda (axis) (aref (tensor-strides result) axis)) kept))))
         (result-row-strides
           (unless outer (index-vector (mapcar (lambda (axis) (aref (tensor-strides result) axis)) axes))))
         (direct (loop for input in inputs for strides in row-layouts
                       collect (and (tensorp input) (backend-storage-p (tensor-storage input))
                                    (or (= width 1) (eql 1 (fusion-layout-step row-shape strides))))))
         (batch (and (loop for input in inputs for use-direct in direct for strides in group-layouts
                           always (or (not (tensorp input))
                                      (and use-direct (fusion-layout-step group-shape strides))))
                     (or outer (and (or (= width 1) (eql 1 (fusion-layout-step row-shape result-row-strides)))
                                    (or (= rows 1) (eql width (fusion-layout-step group-shape result-group-strides))))))))
    (call-with-operation-ranges
     rows
     (lambda (start end)
       ;; TODO: full-row packing; use strided multi-pass execution for wide groups (#13).
       (let* ((buffers (fusion-buffers inputs width (tensor-dtype (find-if #'tensorp inputs)) direct))
              (arrays (loop for input in inputs for buffer in buffers for use-direct in direct
                            collect (if use-direct (tensor-storage input) buffer)))
              (output (unless batch
                        (if outer (tensor-storage result)
                            (make-array width :element-type (dtype-element-type (tensor-dtype result)))))))
         (if batch
             (fusion-batch function parameters inputs arrays group-shape group-layouts start end width result outer)
             (loop for group from start below end
               do (let ((starts
                          (loop for input in inputs for buffer in buffers for use-direct in direct
                                for group-strides in group-layouts for inner-strides in row-layouts
                                collect
                                (if (not (tensorp input)) 0
                                    (let ((base (linear-layout-offset group group-shape group-strides
                                                                      (tensor-offset input))))
                                      (if use-direct base
                                          (let ((position 0))
                                            (call-with-offsets
                                             row-shape (list inner-strides) (list base)
                                             (lambda (offset)
                                               (setf (aref buffer position) (storage-ref (tensor-storage input) offset))
                                               (incf position)))
                                            0)))))))
                    (if outer
                        (setf (aref output group)
                              (fusion-slice function parameters arrays starts nil 0 width outer))
                        (progn
                          (fusion-slice function parameters arrays starts output 0 width nil)
                          (let ((position 0)
                                (base (linear-layout-offset group group-shape result-group-strides 0)))
                            (call-with-offsets
                             row-shape (list result-row-strides) (list base)
                             (lambda (offset)
                               (setf (aref (tensor-storage result) offset) (aref output position))
                               (incf position))))))))))))))

(defun run-fusion (function parameters operands reducers outer out axis keepdims)
  (when out (check-type out tensor))
  (when (and keepdims (not outer)) (error "KEEPDIMS requires a top-level reduction"))
  (let* ((context (or (find-if #'tensorp operands) (error "Fusion requires a tensor operand")))
         (dtype (tensor-dtype context))
         (shape (broadcast-shape operands))
         (rank (length shape)))
    (unless (member dtype '(:f32 :f64)) (error "Fusion requires f32/f64 tensors"))
    (let* ((inputs (mapcar (lambda (input)
                             (if (tensorp input)
                                 (progn
                                   (unless (eq dtype (tensor-dtype input)) (error "Fusion tensor dtypes must match"))
                                   (unless (storage-readable-p (tensor-storage input))
                                     (error 'unsupported-storage-access :storage (tensor-storage input) :access :read))
                                   input)
                                 (coerce-value input dtype)))
                           operands))
           (axes (if (null axis) (loop for axis below rank collect axis) (normalize-axes axis rank)))
           (kept (loop for axis below rank unless (member axis axes) collect axis))
           (width (reduce #'* axes :key (lambda (axis) (aref shape axis)) :initial-value 1))
           (rows (reduce #'* kept :key (lambda (axis) (aref shape axis)) :initial-value 1))
           (result-shape (if outer
                             (if keepdims
                                 (index-vector (loop for dimension across shape for axis from 0
                                                     collect (if (member axis axes) 1 dimension)))
                                 (index-vector (mapcar (lambda (axis) (aref shape axis)) kept)))
                             shape))
           (result-dtype (if (member outer '(:argmin :argmax)) :s64 dtype)))
      (when out (check-copy-destination out result-shape result-dtype))
      (when (and outer (plusp rows) (zerop width) (member outer '(:minimum :maximum :argmin :argmax)))
        (error "Empty fusion groups have no ~S" outer))
      ;; TODO: full-result staging for out-forms; use transactional destinations (#13).
      (let ((result (make-tensor result-shape :dtype result-dtype)))
        (cond ((and outer (zerop width))
               (when (eq outer :prod) (fill (tensor-storage result) (coerce-value 1 dtype))))
              ((zerop (tensor-size result)))
              (reducers (fusion-grouped function parameters inputs shape axes kept width rows result outer))
              (t (fusion-elementwise function parameters inputs result)))
        (if out (copy-storage! (find-dtype result-dtype) out result) result)))))
