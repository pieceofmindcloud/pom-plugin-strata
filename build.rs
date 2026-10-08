use std::{env, fs, path::Path};

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

/// A release build without the built interface would install a plugin that
/// runs but shows no menu. Refuse it here, where the reason can be said.
fn require_built_ui(root: &str) {
    println!("cargo:rerun-if-changed=ui/dist/screens.js");
    let release = env::var("PROFILE").is_ok_and(|profile| profile == "release");
    let built = ["ui/dist/screens.js", "ui/dist/plugin.css"]
        .iter()
        .all(|file| Path::new(root).join(file).is_file());
    if release && !built {
        panic!(
            "the plugin interface is not built: run scripts/build-ui.sh in {root} \
             (it creates ui/dist, which git ignores) and build again"
        );
    }
}

/// The Strata runtime this library runs: the release asset pinned by the
/// package script, and the development directory next to this source.
fn runtime_release(root: &str) -> String {
    for key in [
        "STRATA_RUNTIME_URL",
        "STRATA_RUNTIME_SHA256",
        "STRATA_RUNTIME_SIZE",
    ] {
        println!("cargo:rerun-if-env-changed={key}");
    }
    let read = |key: &str| env::var(key).ok().filter(|value| !value.is_empty());
    let url = read("STRATA_RUNTIME_URL").unwrap_or_default();
    let sha = read("STRATA_RUNTIME_SHA256").unwrap_or_default();
    let size: u64 = read("STRATA_RUNTIME_SIZE")
        .map(|value| value.parse().expect("STRATA_RUNTIME_SIZE must be a number"))
        .unwrap_or(0);
    if !url.is_empty() {
        assert!(
            url.starts_with("https://"),
            "STRATA_RUNTIME_URL must be https"
        );
        assert!(
            sha.len() == 64 && sha.bytes().all(|byte| byte.is_ascii_hexdigit()),
            "STRATA_RUNTIME_SHA256 must be a SHA-256 hex digest"
        );
        assert!(size > 0, "STRATA_RUNTIME_SIZE is required with the URL");
    }
    let dev = Path::new(root).join("build/runtime");
    format!(
        "pub static RUNTIME_URL: &str = {url:?};\n\
         pub static RUNTIME_SHA256: &str = {sha:?};\n\
         pub static RUNTIME_SIZE: u64 = {size};\n\
         pub static DEV_RUNTIME_DIR: &str = {:?};\n",
        dev.to_string_lossy()
    )
}

fn main() {
    let root = env::var("CARGO_MANIFEST_DIR").expect("manifest directory");
    require_built_ui(&root);
    let out = env::var("OUT_DIR").expect("output directory");
    fs::write(
        Path::new(&out).join("runtime_release.rs"),
        runtime_release(&root),
    )
    .expect("write runtime release metadata");

    let mut entries = Vec::new();
    for (directory, prefix, extensions) in [
        ("ui/dist", "ui", &["js", "css"][..]),
        ("ui", "ui", &["png"][..]),
        ("ui/dist/i18n", "i18n", &["json"][..]),
        ("docs", "docs", &["md"][..]),
    ] {
        println!("cargo:rerun-if-changed={directory}");
        let Ok(files) = fs::read_dir(Path::new(&root).join(directory)) else {
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
