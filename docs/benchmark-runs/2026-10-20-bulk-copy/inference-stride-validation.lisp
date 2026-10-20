(require :asdf)
(load "/Users/george/quicklisp/setup.lisp")
(asdf:initialize-source-registry '(:source-registry (:directory "/Users/george/git/cl-inference/") (:directory "/Users/george/git/trivial-simd/") :inherit-configuration))
(asdf:test-system :cl-inference/local)
