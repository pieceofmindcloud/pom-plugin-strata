#!/usr/bin/env bash
# Builds an installable release: the Strata runtime archive (a release asset
# the node downloads and verifies), the plugin library pinned to that archive,
# and the per-platform manifest of POM's "Add from GitHub" flow.
#
# Upload every file it prints to the GitHub release v<version> of this
# repository: the library, pom-plugin-linux-x86_64.json and the runtime archive.
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

runtime_output="$(RUNTIME_VERSION="$version" "$root/scripts/build-runtime.sh" --archive --version "$version")"
runtime_asset="$(printf '%s\n' "$runtime_output" | sed -n 's/^runtime_asset=//p')"
export STRATA_RUNTIME_SHA256="$(printf '%s\n' "$runtime_output" | sed -n 's/^runtime_sha256=//p')"
export STRATA_RUNTIME_SIZE="$(printf '%s\n' "$runtime_output" | sed -n 's/^runtime_size=//p')"
export STRATA_RUNTIME_URL="https://github.com/${repository}/releases/download/v${version}/$(basename "$runtime_asset")"

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

printf 'artifact=%s\npublic_manifest=%s\nruntime_asset=%s\nruntime_url=%s\nsha256=%s\nsize=%s\nversion=%s\n' \
  "$artifact" "$public_manifest" "$runtime_asset" "$STRATA_RUNTIME_URL" "$sha256" "$size" "$version"
