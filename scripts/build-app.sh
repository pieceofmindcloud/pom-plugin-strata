#!/usr/bin/env bash
# Adds Strata's app to a built runtime: the web shell and its host server
# (apps/host of the Strata source), and the portable Node.js it runs on.
#
#   scripts/build-app.sh <Strata source> <runtime directory>
#
# Runs on the build host (curl, tar, xz), after the runtime's container build.
# The Node.js release is the one strata.lock pins, checked against the
# release's SHASUMS256.txt. The web shell is built with that Node, with
# relative asset paths so it works under the POM's plugin proxy prefix. The
# agent (the DeepSeek Harness) is not added here: the plugin installs it from
# npm on the node (src/app.rs), at the version written to app/APP.json.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
# shellcheck disable=SC1091
source "$root/strata.lock"
src="${1:?usage: scripts/build-app.sh <Strata source> <runtime directory>}"
out="${2:?usage: scripts/build-app.sh <Strata source> <runtime directory>}"
host="$src/apps/host"
[[ -f "$host/package.json" && -f "$host/server/index.mjs" ]] \
  || { echo "build-app: $src has no apps/host: this Strata has no app, the runtime stays without it" >&2; exit 0; }
[[ -d "$out" ]] || { echo "build-app: no runtime at $out" >&2; exit 2; }

echo "== app: Node.js ${NODE_VERSION}"
cache="$root/build/cache/node"
dist="node-v${NODE_VERSION}-linux-x64"
archive="$cache/${dist}.tar.xz"
mkdir -p "$cache"
[[ -f "$archive" ]] || curl -fsSL -o "$archive.part" "https://nodejs.org/dist/v${NODE_VERSION}/${dist}.tar.xz"
[[ -f "$archive" ]] || mv "$archive.part" "$archive"
expected="$(curl -fsSL "https://nodejs.org/dist/v${NODE_VERSION}/SHASUMS256.txt" | awk -v name="${dist}.tar.xz" '$2 == name {print $1}')"
actual="$(sha256sum "$archive" | awk '{print $1}')"
[[ -n "$expected" && "$actual" == "$expected" ]] \
  || { rm -f "$archive"; echo "build-app: the Node.js archive does not match SHASUMS256.txt" >&2; exit 1; }
rm -rf "$cache/$dist"
tar -xJf "$archive" -C "$cache"
node="$cache/$dist/bin/node"
npm_cli="$cache/$dist/lib/node_modules/npm/bin/npm-cli.js"

echo "== app: web shell"
work="$root/build/cache/app-build"
rm -rf "$work"
mkdir -p "$work"
cp -a "$host/." "$work/"
rm -rf "$work/node_modules" "$work/dist"
(
  cd "$work"
  export PATH="$cache/$dist/bin:$PATH" npm_config_cache="$root/build/cache/npm" npm_config_update_notifier=false
  "$node" "$npm_cli" ci --no-audit --no-fund --loglevel=error
  # relative asset paths: the shell is served under the POM's proxy prefix (src/proxy.rs)
  "$node" node_modules/vite/bin/vite.js build --base ./ --sourcemap false --logLevel warn
)

echo "== app: into the runtime"
rm -rf "$out/app" "$out/node"
mkdir -p "$out/app/public" "$out/node/bin" "$out/node/lib/node_modules"
cp -a "$work/server" "$work/dist" "$work/package.json" "$out/app/"
cp -a "$work/public/examples" "$out/app/public/"
[[ ! -f "$src/LICENSE" ]] || cp "$src/LICENSE" "$out/app/LICENSE"
find "$out/app/server" -name '*.test.mjs' -delete
install -m 0755 "$node" "$out/node/bin/node"
cp -a "$cache/$dist/lib/node_modules/npm" "$out/node/lib/node_modules/"
cp "$cache/$dist/LICENSE" "$out/node/LICENSE"
printf '{\n  "node_version": "%s",\n  "dsh_version": "%s"\n}\n' "$NODE_VERSION" "$DSH_VERSION" > "$out/app/APP.json"
rm -rf "$work"
echo "== app ready: $(du -sh "$out/app" | cut -f1) app, $(du -sh "$out/node" | cut -f1) Node.js"
