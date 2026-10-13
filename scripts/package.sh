#!/usr/bin/env bash
# Builds an installable release: the Strata runtime archive (a release asset
# the node downloads and verifies), the plugin library pinned to that archive,
# and the per-platform manifest of POM's "Add from GitHub" flow.
#
# Upload every file it prints to the GitHub release v<version> of this
# repository: the library, pom-plugin-linux-x86_64.json and the runtime
# archives (CUDA, and HIP for AMD cards).
#
# STRATA_RUNTIME_ARCHIVE / STRATA_RUNTIME_HIP_ARCHIVE: runtime archives built
# beforehand (scripts/build-runtime.sh --archive [--hip], as the release
# workflow builds them on their own runners); without them both are built here.
# STRATA_PACKAGE_HIP=0 leaves the HIP runtime out.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
version=""
output="${root}/dist-release"
repository="${STRATA_PLUGIN_REPOSITORY:-pieceofmindcloud/pom-plugin-strata}"

die() {
  printf 'package: %s\n' "$1" >&2
  exit 2
}

while (($# > 0)); do
  case "$1" in
    --platform) [[ "$2" == linux-x86_64 ]] || die 'only linux-x86_64 is supported'; shift 2 ;;
    --version) version="$2"; shift 2 ;;
    --output) output="$2"; shift 2 ;;
    --repository) repository="$2"; shift 2 ;;
    -h|--help) printf 'Usage: scripts/package.sh --version <semver> [--repository owner/name]\n'; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
done
[[ "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+([-+][0-9A-Za-z.-]+)?$ ]] || die 'version must be a semantic version without the v prefix'
command -v jq >/dev/null 2>&1 || die 'jq is required'
[[ "$(jq -er '.plugin_code' "$root/ui/manifest.json")" == "$(jq -er '.plugin_code' "$root/release/manifest.json")" ]] \
  || die 'release manifest plugin code does not match the UI manifest'

# archive=, sha256 and size of a runtime archive: the one given, or built now
runtime_pin() {
  local given="$1"; shift
  if [[ -n "$given" ]]; then
    [[ -f "$given" ]] || die "no runtime archive at $given"
    printf 'runtime_asset=%s\nruntime_sha256=%s\nruntime_size=%s\n' "$given" \
      "$(sha256sum "$given" | awk '{print $1}')" "$(wc -c < "$given" | tr -d '[:space:]')"
  else
    RUNTIME_VERSION="$version" "$root/scripts/build-runtime.sh" --archive --version "$version" "$@"
  fi
}
field() { printf '%s\n' "$1" | sed -n "s/^$2=//p"; }

runtime_output="$(runtime_pin "${STRATA_RUNTIME_ARCHIVE:-}")"
runtime_asset="$(field "$runtime_output" runtime_asset)"
export STRATA_RUNTIME_SHA256="$(field "$runtime_output" runtime_sha256)"
export STRATA_RUNTIME_SIZE="$(field "$runtime_output" runtime_size)"
export STRATA_RUNTIME_URL="https://github.com/${repository}/releases/download/v${version}/$(basename "$runtime_asset")"
hip_asset=""
if [[ "${STRATA_PACKAGE_HIP:-1}" != 0 ]]; then
  hip_output="$(runtime_pin "${STRATA_RUNTIME_HIP_ARCHIVE:-}" --hip)"
  hip_asset="$(field "$hip_output" runtime_asset)"
  export STRATA_RUNTIME_HIP_SHA256="$(field "$hip_output" runtime_sha256)"
  export STRATA_RUNTIME_HIP_SIZE="$(field "$hip_output" runtime_size)"
  export STRATA_RUNTIME_HIP_URL="https://github.com/${repository}/releases/download/v${version}/$(basename "$hip_asset")"
fi

build_output="$("$root/scripts/build.sh" --platform linux-x86_64 --output "$output")"
artifact="$(printf '%s\n' "$build_output" | sed -n 's/^artifact=//p')"
sha256="$(printf '%s\n' "$build_output" | sed -n 's/^sha256=//p')"
size="$(printf '%s\n' "$build_output" | sed -n 's/^size=//p')"
public_manifest="${output}/pom-plugin-linux-x86_64.json"
jq -n \
  --slurpfile contract "$root/release/manifest.json" \
  --arg release_id "v${version}" \
  --arg version "$version" \
  --arg asset "$(basename "$artifact")" \
  --arg sha256 "$sha256" \
  --argjson size "$size" \
  '($contract[0]) + {release_id: $release_id, version: $version, os: "linux", arch: "x86_64",
     asset: $asset, sha256: $sha256, size: $size}' > "$public_manifest"

# Metadata for the license server upload (scripts/publish-to-license-server.sh),
# the same shape the harness plugin writes.
metadata="${output}/pom-plugin-strata-linux-x86_64.metadata.json"
jq -n \
  --arg plugin_code "$(jq -er '.plugin_code' "$root/release/manifest.json")" \
  --arg version "$version" \
  --arg artifact "$(basename "$artifact")" \
  --arg sha256 "$sha256" \
  --argjson size "$size" \
  --argjson plugin_abi "$(jq -er '.plugin_abi' "$root/release/manifest.json")" \
  '{plugin_code: $plugin_code, version: $version, platform: "linux-x86_64", target: "x86_64-unknown-linux-gnu",
    plugin_abi: $plugin_abi, sha256: $sha256, size: $size, artifact: $artifact}' > "$metadata"

printf 'artifact=%s\npublic_manifest=%s\nmetadata=%s\nruntime_asset=%s\nruntime_url=%s\nhip_runtime_asset=%s\nsha256=%s\nsize=%s\nversion=%s\n' \
  "$artifact" "$public_manifest" "$metadata" "$runtime_asset" "$STRATA_RUNTIME_URL" "$hip_asset" "$sha256" "$size" "$version"
