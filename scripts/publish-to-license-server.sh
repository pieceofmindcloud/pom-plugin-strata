#!/usr/bin/env bash
# Registers a version already released on GitHub (by the Publish release workflow) with the license server, so the
# plugin manifest the POM fetches from `/v1/plugins/strata/manifest` picks it up.
#
# It reuses the upload contract the POM and the logs plugin already use (`POST /internal/releases/upload`, multipart):
# the library the release published is downloaded back, its sha256 is checked against the public manifest published
# with it (pom-plugin-linux-x86_64.json), and only then is it uploaded. The runtime archives are not uploaded: the node
# downloads them from the release, at the URLs the manifest names.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
version=""
min_host_version="0.1.0"
channel="stable"
feature_set="strata.inference"
reason="release strata plugin"
repository="${GITHUB_REPOSITORY:-pieceofmindcloud/pom-plugin-strata}"
dry_run=false

usage() {
  printf '%s\n' \
    'Usage: POM_RELEASE_TOKEN=... POM_RELEASE_API=... ./scripts/publish-to-license-server.sh --version v0.2.0 [options]' \
    '  --version <tag>         GitHub release tag to publish (required)' \
    '  --min-host <version>    minimum host version (default: 0.1.0)' \
    '  --channel <channel>     release channel (default: stable)' \
    '  --feature-set <a,b>     feature_set metadata (default: strata.inference)' \
    '  --reason <text>         release reason' \
    '  --dry-run               download and verify, without uploading'
}

die() {
  printf 'publish-to-license-server: %s\n' "$1" >&2
  exit 2
}

while (($# > 0)); do
  case "$1" in
    --version) (($# >= 2)) || die '--version requires a value'; version="$2"; shift 2 ;;
    --min-host|--min-host-version) (($# >= 2)) || die '--min-host requires a value'; min_host_version="$2"; shift 2 ;;
    --channel) (($# >= 2)) || die '--channel requires a value'; channel="$2"; shift 2 ;;
    --feature-set) (($# >= 2)) || die '--feature-set requires a value'; feature_set="$2"; shift 2 ;;
    --reason) (($# >= 2)) || die '--reason requires a value'; reason="$2"; shift 2 ;;
    --dry-run) dry_run=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
done

[[ -n "$version" ]] || die '--version is required'
command -v gh >/dev/null 2>&1 || die 'the GitHub CLI (gh) is required'
command -v jq >/dev/null 2>&1 || die 'jq is required'

POM_RELEASE_API="${POM_RELEASE_API:-https://admin-license.pieceofmind.cloud}"
if [[ "$dry_run" != true ]]; then
  [[ -n "${POM_RELEASE_TOKEN:-}" ]] || die 'POM_RELEASE_TOKEN is required'
fi

# The names package.sh gives the release assets (see its header and the public manifest it writes).
platform="linux-x86_64"
os="linux"
arch="x86_64"
download_dir="$(mktemp -d "${TMPDIR:-/tmp}/pom-plugin-strata-publish.XXXXXX")"
trap 'rm -rf "$download_dir"' EXIT

gh release download "$version" --repo "$repository" --dir "$download_dir" --clobber \
  --pattern "pom-plugin-strata-${platform}.so" --pattern "pom-plugin-${platform}.json"

artifact="${download_dir}/pom-plugin-strata-${platform}.so"
manifest="${download_dir}/pom-plugin-${platform}.json"
[[ -f "$artifact" ]] || die "release ${version} has no library asset: pom-plugin-strata-${platform}.so"
[[ -f "$manifest" ]] || die "release ${version} has no public manifest: pom-plugin-${platform}.json"

expected_sha256="$(jq -r '.sha256' "$manifest")"
actual_sha256="$(sha256sum "$artifact" | awk '{print $1}')"
[[ "$expected_sha256" == "$actual_sha256" ]] \
  || die "sha256 mismatch for ${platform}: release manifest says ${expected_sha256}, downloaded library hashes to ${actual_sha256}"

artifact_size="$(wc -c < "$artifact" | tr -d '[:space:]')"
release_id="strata-${version#v}-${os}-${arch}"

if [[ "$dry_run" == true ]]; then
  printf 'release_id=%s\nstatus=dry-run\nsha256=%s\nsize=%s\nartifact=%s\n' \
    "$release_id" "$actual_sha256" "$artifact_size" "$artifact"
  exit 0
fi

upload_url="${POM_RELEASE_API%/}"
[[ "$upload_url" == */internal/releases/upload ]] || upload_url="${upload_url}/internal/releases/upload"
curl --fail --silent --show-error \
  -H "Authorization: Bearer ${POM_RELEASE_TOKEN}" \
  -F "plugin=strata" \
  -F "version=${version#v}" \
  -F "os=${os}" \
  -F "arch=${arch}" \
  -F "plugin_abi=1" \
  -F "min_host_version=${min_host_version}" \
  -F "channel=${channel}" \
  -F "feature_set=${feature_set}" \
  -F "reason=${reason}" \
  -F "artifact=@${artifact}" \
  "$upload_url" >/dev/null
printf 'release_id=%s\nstatus=published\nsha256=%s\nsize=%s\n' "$release_id" "$actual_sha256" "$artifact_size"
