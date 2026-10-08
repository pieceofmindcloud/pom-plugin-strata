//! Where the Strata runtime (engine, server, pack tools, Python packages)
//! lives on this node.
//!
//! The plugin library never embeds it: the engine alone is hundreds of
//! megabytes. Three sources, first match wins:
//!
//! 1. `STRATA_RUNTIME_DIR` in the environment of the POM node - an explicit
//!    directory, for an operator who built it by hand.
//! 2. `build/runtime` of the source this library was compiled from, when it
//!    exists. That is the development flow: `scripts/build-runtime.sh` on the
//!    host, then the POM's local rebuild, which compiles the mounted source;
//!    the node reads the runtime straight from the mount.
//! 3. The release asset pinned at build time (`STRATA_RUNTIME_URL`, size and
//!    SHA-256), downloaded once into the plugin directory and verified before
//!    it is unpacked.

use sha2::{Digest, Sha256};
use std::io::Read;
use std::path::{Path, PathBuf};

include!(concat!(env!("OUT_DIR"), "/runtime_release.rs"));

/// What a usable runtime directory must contain.
pub fn is_runtime(dir: &Path) -> bool {
    dir.join("RUNTIME.json").is_file()
        && dir.join("engine/strata").is_file()
        && dir.join("pom_strata_server.py").is_file()
        && dir.join("strata/serve/server.py").is_file()
}

pub fn version_of(dir: &Path) -> Option<String> {
    let text = std::fs::read_to_string(dir.join("RUNTIME.json")).ok()?;
    let meta: serde_json::Value = serde_json::from_str(&text).ok()?;
    let commit = meta["strata_commit"].as_str().unwrap_or_default();
    let version = meta["runtime_version"].as_str().unwrap_or_default();
    Some(match (version.is_empty(), commit.len() >= 7) {
        (false, _) => version.to_owned(),
        (true, true) => format!("strata {}", &commit[..7]),
        _ => "local".to_owned(),
    })
}

/// A runtime that needs no download, when there is one.
pub fn local_runtime() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("STRATA_RUNTIME_DIR").map(PathBuf::from) {
        return is_runtime(&dir).then_some(dir);
    }
    let dev = PathBuf::from(DEV_RUNTIME_DIR);
    is_runtime(&dev).then_some(dev)
}

/// The directory a pinned release is unpacked into.
pub fn installed_dir(base: &Path) -> Option<PathBuf> {
    (!RUNTIME_SHA256.is_empty()).then(|| {
        base.join("runtime")
            .join(&RUNTIME_SHA256[..16])
            .join("runtime")
    })
}

pub fn has_release() -> bool {
    !RUNTIME_URL.is_empty()
}

/// Downloads, verifies and unpacks the pinned runtime. `progress` gets the
/// bytes read so far and the expected total.
pub fn install(base: &Path, progress: &dyn Fn(u64, u64)) -> Result<PathBuf, String> {
    let target = installed_dir(base).ok_or("this build pins no Strata runtime")?;
    if is_runtime(&target) {
        return Ok(target);
    }
    let holder = target.parent().ok_or("bad runtime directory")?;
    std::fs::create_dir_all(holder).map_err(|error| error.to_string())?;
    let archive = holder.join("runtime.tar.gz.part");
    let response = ureq::get(RUNTIME_URL)
        .call()
        .map_err(|error| format!("download failed: {error}"))?;
    let mut reader = response.into_reader().take(RUNTIME_SIZE + 1);
    let mut file = std::fs::File::create(&archive).map_err(|error| error.to_string())?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1 << 20];
    let mut read_total = 0u64;
    loop {
        let read = reader
            .read(&mut buffer)
            .map_err(|error| error.to_string())?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
        std::io::Write::write_all(&mut file, &buffer[..read]).map_err(|error| error.to_string())?;
        read_total += read as u64;
        progress(read_total, RUNTIME_SIZE);
    }
    drop(file);
    let digest = format!("{:x}", hasher.finalize());
    if read_total != RUNTIME_SIZE || digest != RUNTIME_SHA256 {
        let _ = std::fs::remove_file(&archive);
        return Err("the downloaded runtime does not match the pinned size and SHA-256".into());
    }
    let unpacked = holder.join("unpacking");
    let _ = std::fs::remove_dir_all(&unpacked);
    std::fs::create_dir_all(&unpacked).map_err(|error| error.to_string())?;
    let file = std::fs::File::open(&archive).map_err(|error| error.to_string())?;
    let mut tar = tar::Archive::new(flate2::read::GzDecoder::new(file));
    tar.set_preserve_permissions(true);
    for entry in tar.entries().map_err(|error| error.to_string())? {
        let mut entry = entry.map_err(|error| error.to_string())?;
        // `unpack_in` refuses paths that leave the directory.
        if !entry
            .unpack_in(&unpacked)
            .map_err(|error| error.to_string())?
        {
            return Err("the runtime archive has an unsafe path".into());
        }
    }
    let _ = std::fs::remove_file(&archive);
    let _ = std::fs::remove_dir_all(&target);
    std::fs::rename(unpacked.join("runtime"), &target).map_err(|error| error.to_string())?;
    let _ = std::fs::remove_dir_all(&unpacked);
    if is_runtime(&target) {
        Ok(target)
    } else {
        Err("the runtime archive is incomplete".into())
    }
}
