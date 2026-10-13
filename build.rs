use std::{
    env, fs,
    path::{Path, PathBuf},
};

fn content_type(name: &str) -> &'static str {
    match Path::new(name)
        .extension()
        .and_then(|extension| extension.to_str())
    {
        Some("js") => "text/javascript",
        Some("css") => "text/css",
        Some("md") => "text/markdown",
        Some("png") => "image/png",
        Some("json") => "application/json",
        _ => "application/octet-stream",
    }
}

/// Where the built UI is: the POM's local rebuild builds it in
/// `POM_PLUGIN_OUT_DIR` (`scripts/dev-ui.sh`, the source is read-only there),
/// any other build in the source's `ui/dist` (`scripts/build-ui.sh`).
fn ui_dist(root: &str) -> PathBuf {
    println!("cargo:rerun-if-env-changed=POM_PLUGIN_OUT_DIR");
    env::var("POM_PLUGIN_OUT_DIR")
        .ok()
        .filter(|value| !value.is_empty())
        .map(|out| Path::new(&out).join("ui-build/ui/dist"))
        .filter(|dist| dist.join("screens.js").is_file())
        .unwrap_or_else(|| Path::new(root).join("ui/dist"))
}

/// A release build without the built interface would install a plugin that
/// runs but shows no menu. Refuse it here, where the reason can be said.
fn require_built_ui(root: &str, dist: &Path) {
    println!(
        "cargo:rerun-if-changed={}",
        dist.join("screens.js").display()
    );
    let release = env::var("PROFILE").is_ok_and(|profile| profile == "release");
    let built = ["screens.js", "plugin.css"]
        .iter()
        .all(|file| dist.join(file).is_file());
    if release && !built {
        panic!(
            "the plugin interface is not built: run scripts/build-ui.sh in {root} \
             (it creates ui/dist, which git ignores) and build again"
        );
    }
}

/// The Strata runtimes this library runs, one per GPU backend: the release
/// assets pinned by the package script (`STRATA_RUNTIME_*` for CUDA,
/// `STRATA_RUNTIME_HIP_*` for HIP), and the development directories next to
/// this source.
fn runtime_release(root: &str) -> String {
    let mut out = String::new();
    for (prefix, constant, dev) in [
        ("STRATA_RUNTIME", "RUNTIME", "build/runtime"),
        ("STRATA_RUNTIME_HIP", "RUNTIME_HIP", "build/runtime-hip"),
    ] {
        for key in ["URL", "SHA256", "SIZE"] {
            println!("cargo:rerun-if-env-changed={prefix}_{key}");
        }
        let read = |key: &str| {
            env::var(format!("{prefix}_{key}"))
                .ok()
                .filter(|value| !value.is_empty())
        };
        let url = read("URL").unwrap_or_default();
        let sha = read("SHA256").unwrap_or_default();
        let size: u64 = read("SIZE")
            .map(|value| value.parse().expect("the runtime SIZE must be a number"))
            .unwrap_or(0);
        if !url.is_empty() {
            assert!(url.starts_with("https://"), "{prefix}_URL must be https");
            assert!(
                sha.len() == 64 && sha.bytes().all(|byte| byte.is_ascii_hexdigit()),
                "{prefix}_SHA256 must be a SHA-256 hex digest"
            );
            assert!(size > 0, "{prefix}_SIZE is required with the URL");
        }
        let dev = Path::new(root).join(dev);
        out.push_str(&format!(
            "pub static {constant}_URL: &str = {url:?};\n\
             pub static {constant}_SHA256: &str = {sha:?};\n\
             pub static {constant}_SIZE: u64 = {size};\n\
             pub static DEV_{constant}_DIR: &str = {:?};\n",
            dev.to_string_lossy()
        ));
    }
    out
}

fn main() {
    let root = env::var("CARGO_MANIFEST_DIR").expect("manifest directory");
    let dist = ui_dist(&root);
    require_built_ui(&root, &dist);
    let out = env::var("OUT_DIR").expect("output directory");
    fs::write(
        Path::new(&out).join("runtime_release.rs"),
        runtime_release(&root),
    )
    .expect("write runtime release metadata");

    let mut entries = Vec::new();
    for (directory, prefix, extensions) in [
        (dist.clone(), "ui", &["js", "css"][..]),
        (Path::new(&root).join("ui"), "ui", &["png"][..]),
        (dist.join("i18n"), "i18n", &["json"][..]),
        (Path::new(&root).join("docs"), "docs", &["md"][..]),
    ] {
        println!("cargo:rerun-if-changed={}", directory.display());
        let Ok(files) = fs::read_dir(&directory) else {
            continue;
        };
        for file in files.flatten() {
            let name = file.file_name().to_string_lossy().into_owned();
            let extension = Path::new(&name)
                .extension()
                .and_then(|value| value.to_str())
                .unwrap_or_default();
            if extensions.contains(&extension) {
                entries.push((
                    format!("{prefix}/{name}"),
                    file.path().to_string_lossy().into_owned(),
                ));
            }
        }
    }
    entries.sort();
    let mut generated = String::from("pub static UI_ASSETS: &[(&str, &str, &[u8])] = &[\n");
    for (name, path) in &entries {
        generated.push_str(&format!(
            "    ({name:?}, {:?}, include_bytes!({path:?})),\n",
            content_type(name)
        ));
    }
    generated.push_str("];\n");
    fs::write(Path::new(&out).join("ui_assets.rs"), generated).expect("write embedded asset index");
    println!("cargo:rerun-if-changed=ui/manifest.json");
    println!("cargo:rerun-if-changed=i18n");
    println!("cargo:rerun-if-changed=docs");
}
