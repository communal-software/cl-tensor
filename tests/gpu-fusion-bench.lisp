(require :asdf)
(asdf:load-system "cl-tensor/gpu/metal/tests")

(defpackage #:cl-tensor/gpu-fusion-bench
  (:use #:cl)
  (:local-nicknames (#:ct #:cl-tensor) (#:tg #:cl-tensor/gpu)))
(in-package #:cl-tensor/gpu-fusion-bench)

(ct:define-kernel affine (a b) (+ (* a b) 1))
(ct:define-kernel squared-sum (a) (ct:sum (* a a)))
(ct:define-kernel stable-softmax (a)
  (/ (exp (- a (ct:maximum a))) (ct:sum (exp (- a (ct:maximum a))))))

(defun composed-affine (a b &optional out)
  (tg:with-device-tensor (product (ct:multiply a b))
    (if out (ct:add! out product 1) (ct:add product 1))))

(defun composed-sum (a &optional out)
  (tg:with-device-tensor (squares (ct:multiply a a))
    (if out (ct:sum! out squares :axis -1) (ct:sum squares :axis -1))))

(defun composed-softmax (a &optional out)
  (tg:with-device-tensor (maximum (ct:maximum a :axis -1 :keepdims t))
    (tg:with-device-tensor (shifted (ct:subtract a maximum))
      (tg:with-device-tensor (numerator (ct:exp shifted))
        (tg:with-device-tensor (denominator (ct:sum numerator :axis -1 :keepdims t))
          (if out (ct:divide! out numerator denominator) (ct:divide numerator denominator)))))))

(defun elapsed-us (function count)
  (let ((start (get-internal-real-time)))
    (dotimes (i count) (funcall function))
    (* 1d6 (/ (- (get-internal-real-time) start) internal-time-units-per-second count))))

(defun measure (name function first)
  (dotimes (i 3) (funcall function))
  (let ((times (loop repeat 3 collect (elapsed-us function 10)))
        (bytes #+sbcl (let ((before (sb-ext:get-bytes-consed)))
                       (dotimes (i 10) (funcall function))
                       (/ (- (sb-ext:get-bytes-consed) before) 10d0))
               #-sbcl nil))
    (format t "~A,~,3F,~,3F,~A~%" name first (second (sort times #'<)) bytes)))

(defun benchmark (name shape reference allocating into)
  (tg:with-device-tensor (out (tg:make-device-tensor shape))
    (flet ((allocate () (tg:with-device-tensor (result (funcall allocating))
                         result))
           (write-out () (funcall into out)))
      (let* ((start (get-internal-real-time))
             (result (funcall allocating))
             (first (* 1d6 (/ (- (get-internal-real-time) start) internal-time-units-per-second))))
        (unwind-protect
             (assert (cl-tensor/gpu/tests::same-values-p reference (tg:from-device result)))
          (tg:free-device-tensor result))
        (measure (concatenate 'string name "/allocate") #'allocate first))
      (let ((first (elapsed-us #'write-out 1)))
        (assert (cl-tensor/gpu/tests::same-values-p reference (tg:from-device out)))
        (measure (concatenate 'string name "/out") #'write-out first)))))


(defun run-benchmarks ()
  (let* ((rows 256) (columns 256)
         (cpu (ct:from-data (loop for i below rows collect
                             (loop for j below columns collect (+ 0.2f0 (* 0.01f0 (mod (+ i j) 31)))))))
         (weights (ct:from-data (loop for j below columns collect (+ 0.3f0 (* 0.01f0 j))))))
    (format t "~&256x256 f32; median of three ten-call batches; complete synchronous calls~%")
    (format t "case,first-call-us,warm-median-us,Lisp-bytes/call~%")
    (tg:with-device-tensor (a (tg:to-device cpu))
      (tg:with-device-tensor (b (tg:to-device weights))
        (dolist (layout '(:contiguous :reversed :broadcast))
          (let* ((input (if (eq layout :reversed)
                            (ct:make-tensor-view (ct:tensor-storage a) '(256 256) :strides '(256 -1) :offset 255) a))
                 (right (if (eq layout :broadcast) b 2))
                 (expected (ct:from-data
                            (loop for i below rows collect
                              (loop for j below columns
                                    for x = (ct:tref cpu i (if (eq layout :reversed) (- 255 j) j))
                                    collect (+ (* x (if (eq layout :broadcast) (ct:tref weights j) 2)) 1))))))
            (benchmark (format nil "affine/~(~A~)/composed" layout) '(256 256) expected
                       (lambda () (composed-affine input right)) (lambda (out) (composed-affine input right out)))
            (benchmark (format nil "affine/~(~A~)/fused" layout) '(256 256) expected
                       (lambda () (affine input right)) (lambda (out) (affine! out input right)))))
        (let ((expected (ct:from-data (loop for i below rows collect
                                      (loop for j below columns for x = (ct:tref cpu i j) sum (* x x))))))
          (benchmark "row-squares/composed" '(256) expected
                     (lambda () (composed-sum a)) (lambda (out) (composed-sum a out)))
          (benchmark "row-squares/fused" '(256) expected
                     (lambda () (squared-sum a :axis -1)) (lambda (out) (squared-sum! out a :axis -1))))
        (let ((expected (ct:from-data
                         (loop for i below rows collect
                           (let* ((row (loop for j below columns collect (float (ct:tref cpu i j) 1d0)))
                                  (maximum (reduce #'max row))
                                  (values (mapcar (lambda (x) (exp (- x maximum))) row))
                                  (sum (reduce #'+ values)))
                             (mapcar (lambda (x) (/ x sum)) values))))))
          (benchmark "softmax/composed" '(256 256) expected
                     (lambda () (composed-softmax a)) (lambda (out) (composed-softmax a out)))
          (benchmark "softmax/fused" '(256 256) expected
                     (lambda () (stable-softmax a :axis -1)) (lambda (out) (stable-softmax! out a :axis -1))))))))

(run-benchmarks)
