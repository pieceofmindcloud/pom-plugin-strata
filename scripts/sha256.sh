#!/usr/bin/env bash
# The SHA-256 of a file as hex, on Linux (sha256sum), macOS (shasum) and Git Bash on Windows.
set -euo pipefail
if command -v sha256sum >/dev/null 2>&1; then
  sha256sum "$1" | awk '{print $1}'
else
  shasum -a 256 "$1" | awk '{print $1}'
fi
