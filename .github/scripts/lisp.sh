#!/usr/bin/env bash
set -euo pipefail
export CI_LISP_SCRIPT="$1"

case "$CI_LISP" in
  sbcl) exec sbcl --dynamic-space-size 4096 --noinform --no-userinit --no-sysinit --script "$1" ;;
  ecl) exec ecl -norc -eval '(handler-bind ((error (lambda (condition) (format *error-output* "~&~A~%" condition) (ext:quit 1)))) (load (ext:getenv "CI_LISP_SCRIPT")))' ;;
  *) echo "Unsupported CI_LISP: $CI_LISP" >&2; exit 1 ;;
esac
