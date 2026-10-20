(require :asdf)

(load (uiop:getenv "CI_QUICKLISP_BOOTSTRAP"))

(defun download-with-retries (function &rest arguments)
  (loop for attempt from 1 to 3
        do (handler-case (return (apply function arguments))
             (stream-error (condition)
               (when (= attempt 3) (error condition))
               (format *error-output* "~&Retrying interrupted download: ~A~%" condition)
               (sleep attempt)))))

(let ((fetch (symbol-function (find-symbol "FETCH" :qlqs-http))))
  (setf (symbol-function (find-symbol "FETCH" :qlqs-http))
        (lambda (&rest arguments) (apply #'download-with-retries fetch arguments))))

(uiop:symbol-call :quicklisp-quickstart :install
                 :path (uiop:ensure-directory-pathname (uiop:getenv "CI_QUICKLISP_DIR")))

(let ((fetch (cdr (assoc "http" ql-http::*fetch-scheme-functions* :test #'equal))))
  (setf (cdr (assoc "http" ql-http::*fetch-scheme-functions* :test #'equal))
        (lambda (&rest arguments) (apply #'download-with-retries fetch arguments))))

(uiop:symbol-call :ql :quickload '("cffi" "trivial-garbage" "fiveam" "bordeaux-threads")
                 :prompt nil)
(uiop:quit 0)
