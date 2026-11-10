#!/usr/bin/env bash
# Builds the plugin library for one platform, on that platform: Linux x86_64
# (CUDA and HIP), macOS aarch64 (Metal, Apple Silicon) or Windows x86_64 (CUDA).
# The Strata runtime is not part of the library: see scripts/build-runtime.sh
# and scripts/package.sh.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
platform=""
output="${root}/dist-release"
profile="release"

usage() {
  printf '%s\n' \
    'Usage: scripts/build.sh --platform <linux-x86_64|macos-aarch64|windows-x86_64> [options]' \
    '  --output <dir>   artifact directory (default: dist-release)' \
    '  --debug          build the debug profile' \
    'Environment: STRATA_RUNTIME_URL, STRATA_RUNTIME_SHA256, STRATA_RUNTIME_SIZE pin the' \
    '  runtime release asset the library downloads (scripts/package.sh sets them).'
}

die() {
  printf 'build: %s\n' "$1" >&2
  exit 2
}

while (($# > 0)); do
  case "$1" in
    --platform) (($# >= 2)) || die '--platform requires a value'; platform="$2"; shift 2 ;;
    --output) (($# >= 2)) || die '--output requires a value'; output="$2"; shift 2 ;;
    --debug) profile="debug"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
done

case "$platform" in
  linux-x86_64) target="x86_64-unknown-linux-gnu"; library="libpom_plugin_strata.so"; ext="so" ;;
  macos-aarch64) target="aarch64-apple-darwin"; library="libpom_plugin_strata.dylib"; ext="dylib" ;;
  windows-x86_64) target="x86_64-pc-windows-msvc"; library="pom_plugin_strata.dll"; ext="dll" ;;
  *) die "unsupported platform: $platform" ;;
esac
cargo_args=(build --locked --target "$target")
[[ "$profile" == debug ]] || cargo_args+=(--release)
artifact="${root}/target/${target}/${profile}/${library}"

"$root/scripts/build-ui.sh"
(
  cd "$root"
  cargo "${cargo_args[@]}" >&2
)
[[ -f "$artifact" ]] || die "artifact not found: $artifact"

mkdir -p "$output"
destination="${output}/pom-plugin-strata-${platform}.${ext}"
cp "$artifact" "$destination"
sha256="$("$root/scripts/sha256.sh" "$destination")"
size="$(wc -c < "$destination" | tr -d '[:space:]')"
printf 'artifact=%s\nsha256=%s\nsize=%s\nplatform=%s\ntarget=%s\n' \
  "$destination" "$sha256" "$size" "$platform" "$target"
