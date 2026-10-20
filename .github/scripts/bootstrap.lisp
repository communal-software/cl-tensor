(require :asdf)

(load (uiop:getenv "CI_QUICKLISP_BOOTSTRAP"))
(uiop:symbol-call :quicklisp-quickstart :install
                 :path (uiop:ensure-directory-pathname (uiop:getenv "CI_QUICKLISP_DIR")))
(uiop:symbol-call :ql :quickload '("cffi" "trivial-garbage" "fiveam" "bordeaux-threads")
                 :prompt nil)
(uiop:quit 0)
