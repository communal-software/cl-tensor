(require :asdf)
(asdf:load-system :cl-tensor)

(let* ((x (cl-tensor:from-data '((1 2 3))))
       (weights (cl-tensor:astype (cl-tensor:from-data '((1 0 1) (0 1 0))) :f16))
       (out (cl-tensor:matmul x (cl-tensor:transpose weights))))
  (assert (eq (cl-tensor:tensor-dtype out) :f32))
  (assert (= 4f0 (cl-tensor:tref out 0 0)))
  (assert (= 2f0 (cl-tensor:tref out 0 1))))
