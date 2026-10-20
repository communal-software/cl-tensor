#!/usr/bin/env bash
set -euo pipefail

case "$RUNNER_ARCH" in
  X64)
    arch=x86-64
    checksum=afe02b0f3c3404cc8182307fea26710a80b0580b0737551012b197eb42fca8e4
    ;;
  ARM64)
    arch=arm64
    checksum=62126e9f5e2af002a9a33c69fb5fe92f7d56a0c69b454b1ef2538396b4bd898f
    ;;
  *) echo "Unsupported Linux architecture: $RUNNER_ARCH" >&2; exit 1 ;;
esac

archive="$RUNNER_TEMP/sbcl.tar.bz2"
curl --fail --location --retry 3 --retry-all-errors \
  "https://github.com/roswell/sbcl_bin/releases/download/2.6.8/sbcl-2.6.8-$arch-linux-binary.tar.bz2" \
  --output "$archive"
echo "$checksum  $archive" | sha256sum --check
mkdir -p "$RUNNER_TEMP/sbcl-source"
tar -xjf "$archive" -C "$RUNNER_TEMP/sbcl-source" --strip-components=1
(cd "$RUNNER_TEMP/sbcl-source" && INSTALL_ROOT="$RUNNER_TEMP/sbcl" sh install.sh)
echo "$RUNNER_TEMP/sbcl/bin" >> "$GITHUB_PATH"
echo "SBCL_HOME=$RUNNER_TEMP/sbcl/lib/sbcl" >> "$GITHUB_ENV"
