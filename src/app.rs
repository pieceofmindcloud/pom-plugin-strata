//! Strata's app: the web shell with chats, tasks, workspaces and the agent
//! (Strata's `apps/host`), served on the plugin screen under `/app/`
//! (`proxy.rs`).
//!
//! The runtime carries the app and the portable Node.js it runs on
//! (`scripts/build-app.sh`: `app/`, `node/`). The agent, the DeepSeek Harness,
//! is installed from npm on this machine the first time, at the version the
//! runtime names (`app/APP.json`), into `<plugin dir>/app/harness/<version>`;
//! neither the plugin library nor the runtime carries it.
//!
//! The app never talks to Strata directly. Its model connection is the POM's
//! OpenAI endpoint with the key the POM hands over in `host.configure`, so
//! every request it makes goes through the POM like any other client's. It
//! starts only once that configuration has arrived.

use crate::engine::Group;
use serde_json::{json, Value};
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

const DSH_PACKAGE: &str = "@deepseek-ai/dsh";
/// The id of the connection this plugin writes into the app's model list.
const CONNECTION_ID: &str = "pom";
const POLL: Duration = Duration::from_secs(2);
/// How often the POM's model list is compared with the app's connection.
const MODELS_EVERY: Duration = Duration::from_secs(15);

/// What the POM hands over in `host.configure` that the app needs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Gateway {
    pub openai_base_url: String,
    pub api_key: String,
    pub workspace_root: Option<PathBuf>,
}

impl Gateway {
    /// `{"gateway": {"openai_base_url", "api_key"}, "workspace_root"}`; `None`
    /// when the POM sent no gateway (it has none ready yet).
    pub fn from_configure(request: &Value) -> Option<Self> {
        let gateway = &request["gateway"];
        let openai_base_url = gateway["openai_base_url"].as_str()?.trim_end_matches('/');
        let api_key = gateway["api_key"].as_str()?;
        if openai_base_url.is_empty() || api_key.is_empty() {
            return None;
        }
        Some(Self {
            openai_base_url: openai_base_url.to_owned(),
            api_key: api_key.to_owned(),
            workspace_root: request["workspace_root"]
                .as_str()
                .filter(|path| Path::new(path).is_absolute())
                .map(PathBuf::from),
        })
    }
}

#[derive(Debug, Clone)]
struct State {
    phase: &'static str,
    detail: Option<String>,
    port: Option<u16>,
    gateway: Option<Gateway>,
    harness: Option<String>,
}

/// Where the runtime is now (the engine's choice), read on every round.
pub type RuntimeDir = Arc<dyn Fn() -> Option<PathBuf> + Send + Sync>;

pub struct App {
    shared: Arc<Mutex<State>>,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl App {
    pub fn start(base: PathBuf, runtime: RuntimeDir) -> Self {
        let shared = Arc::new(Mutex::new(State {
            phase: "waiting",
            detail: Some("waiting for the POM's configuration".into()),
            port: None,
            gateway: None,
            harness: None,
        }));
        let stop = Arc::new(AtomicBool::new(false));
        let thread_shared = Arc::clone(&shared);
        let thread_stop = Arc::clone(&stop);
        // A long-lived thread: the children it starts get the parent-death
        // signal tied to it (see engine.rs).
        let thread = thread::Builder::new()
            .name("pom-plugin-strata-app".into())
            .spawn(move || Manager::new(base, runtime, thread_shared, thread_stop).run())
            .expect("start the app manager");
        Self {
            shared,
            stop,
            thread: Some(thread),
        }
    }

    fn state(&self) -> State {
        self.shared
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    /// A new configuration restarts the app with it.
    pub fn configure(&self, gateway: Gateway) {
        let mut state = self
            .shared
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.gateway.as_ref() != Some(&gateway) {
            state.gateway = Some(gateway);
        }
    }

    pub fn port(&self) -> Option<u16> {
        self.state().port
    }

    /// For the plugin screen; never the key.
    pub fn status_json(&self) -> Value {
        let state = self.state();
        json!({"state": state.phase, "detail": state.detail, "harness": state.harness})
    }
}

impl Drop for App {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

struct Running {
    group: Group,
    gateway: Gateway,
    runtime: PathBuf,
    port: u16,
    since: Instant,
    /// The models the app's "POM" connection lists, and when the POM was last asked.
    models: Vec<String>,
    checked: Instant,
}

struct Manager {
    base: PathBuf,
    runtime: RuntimeDir,
    shared: Arc<Mutex<State>>,
    stop: Arc<AtomicBool>,
    running: Option<Running>,
    /// No new attempt before this (after a failure).
    retry_at: Option<Instant>,
    failures: u32,
}

impl Manager {
    fn new(
        base: PathBuf,
        runtime: RuntimeDir,
        shared: Arc<Mutex<State>>,
        stop: Arc<AtomicBool>,
    ) -> Self {
        Self {
            base,
            runtime,
            shared,
            stop,
            running: None,
            retry_at: None,
            failures: 0,
        }
    }

    fn update(&self, change: impl FnOnce(&mut State)) {
        let mut state = self
            .shared
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        change(&mut state);
    }

    fn set(&self, phase: &'static str, detail: Option<String>) {
        self.update(|state| {
            state.phase = phase;
            state.detail = detail;
        });
    }

    fn log(&self) -> PathBuf {
        self.base.join("run/strata-app.log")
    }

    fn run(mut self) {
        while !self.stop.load(Ordering::Acquire) {
            self.round();
            thread::sleep(POLL);
        }
        self.halt();
    }

    fn halt(&mut self) {
        if let Some(running) = self.running.take() {
            running.group.stop(Duration::from_secs(5));
        }
        self.update(|state| state.port = None);
    }

    fn fail(&mut self, error: String) {
        self.halt();
        self.failures += 1;
        let wait = Duration::from_secs(5u64 << self.failures.min(6)).min(Duration::from_secs(300));
        self.retry_at = Some(Instant::now() + wait);
        let _ = append(&self.log(), &format!("[pom] the app stopped: {error}"));
        self.set("error", Some(error));
    }

    fn round(&mut self) {
        let gateway = self
            .shared
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .gateway
            .clone();
        let runtime = (self.runtime)().filter(|dir| has_app(dir));
        if let Some(running) = self.running.as_mut() {
            let exited = running.group.exited();
            let changed = gateway.as_ref() != Some(&running.gateway)
                || runtime.as_ref() != Some(&running.runtime);
            let (port, since) = (running.port, running.since);
            if let Some(status) = exited {
                self.fail(format!("the app ended ({status}); see run/strata-app.log"));
                return;
            }
            if changed {
                self.halt();
                self.set(
                    "starting",
                    Some("restarting with the new configuration".into()),
                );
                return;
            }
            let waiting = self
                .shared
                .lock()
                .map(|state| state.port.is_none())
                .unwrap_or(false);
            if waiting {
                if healthy(port) {
                    self.failures = 0;
                    self.update(|state| state.port = Some(port));
                    self.set("ready", None);
                } else if since.elapsed() > Duration::from_secs(120) {
                    self.fail("the app did not answer within 2 minutes".into());
                }
                return;
            }
            self.refresh_models();
            return;
        }
        if self.retry_at.is_some_and(|at| Instant::now() < at) {
            return;
        }
        let Some(gateway) = gateway else {
            self.set(
                "waiting",
                Some("waiting for the POM's configuration".into()),
            );
            return;
        };
        let Some(runtime) = runtime else {
            self.set("unavailable", Some("this Strata runtime has no app".into()));
            return;
        };
        match self.launch(gateway, runtime) {
            Ok(running) => {
                self.retry_at = None;
                self.running = Some(running);
            }
            Err(error) => self.fail(error),
        }
    }

    /// The POM serves the models it is deployed with, which change while the
    /// app runs (a model loads after the app started, another is deployed):
    /// the app's "POM" connection follows them through the app's own API, so
    /// its picker updates without a restart.
    fn refresh_models(&mut self) {
        let Some(running) = self.running.as_mut() else {
            return;
        };
        if running.checked.elapsed() < MODELS_EVERY {
            return;
        }
        running.checked = Instant::now();
        let Some(models) = gateway_models(&running.gateway) else {
            return;
        };
        // (an empty list is not saved: the app keeps a connection's last models)
        if models.is_empty() || models == running.models {
            return;
        }
        match post_connection(running.port, &running.gateway, &models) {
            Ok(()) => {
                running.models = models;
                let line = format!("[pom] the POM's models now: {}", running.models.join(", "));
                let _ = append(&self.log(), &line);
            }
            Err(error) => {
                let _ = append(
                    &self.log(),
                    &format!("[pom] updating the app's model list failed: {error}"),
                );
            }
        }
    }

    fn launch(&mut self, gateway: Gateway, runtime: PathBuf) -> Result<Running, String> {
        let node = runtime.join("node/bin/node");
        let meta: Value = std::fs::read_to_string(runtime.join("app/APP.json"))
            .ok()
            .and_then(|text| serde_json::from_str(&text).ok())
            .ok_or("the runtime's app/APP.json is missing")?;
        let version = meta["dsh_version"]
            .as_str()
            .filter(|version| valid_version(version))
            .ok_or("app/APP.json names no DeepSeek Harness version")?
            .to_owned();
        let home = self.base.join("app/home");
        let workspace = gateway
            .workspace_root
            .clone()
            .unwrap_or_else(|| self.base.join("app/workspace"));
        for dir in [&home, &workspace, &self.base.join("run")] {
            std::fs::create_dir_all(dir).map_err(|error| format!("{}: {error}", dir.display()))?;
        }
        let harness = self.harness(&node, &runtime, &version)?;
        self.update(|state| state.harness = Some(version.clone()));
        self.set("starting", Some("starting the app".into()));
        let models = gateway_models(&gateway).unwrap_or_default();
        write_connection(&home.join("data/model-providers.json"), &gateway, &models)?;
        keep_workspace(&home.join("data/workspaces.json"), &workspace)?;
        let port = free_port()?;
        let log = self.log();
        let _ = std::fs::write(&log, "");
        let mut command = Command::new(&node);
        command
            .arg(runtime.join("app/server/index.mjs"))
            .arg("--dist")
            .arg(runtime.join("app/dist"))
            .current_dir(&home)
            .env("HOME", &home)
            .env("STRATA_HOME", &home)
            .env("STRATA_WORKSPACE", &workspace)
            .env("STRATA_PORT", port.to_string())
            .env("STRATA_BIND", "127.0.0.1")
            .env("STRATA_HARNESS_DIR", &harness)
            // The POM runs the engine and owns the models: the app installs and downloads nothing.
            .env("STRATA_NO_BUNDLED_ENGINE", "1")
            .env("STRATA_NO_ACQUIRE", "1")
            .env("STRATA_REAL_ENGINE", "1")
            .env_remove("STRATA_SERVER_URL")
            .env("PATH", path_with(&runtime.join("node/bin")))
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let mut group =
            Group::spawn(&mut command).map_err(|error| format!("starting the app: {error}"))?;
        group.forward_lines(&log, "[strata app] ");
        let _ = append(
            &log,
            &format!(
                "[pom] app on 127.0.0.1:{port}, model connection: the POM's OpenAI endpoint ({} models)",
                models.len()
            ),
        );
        Ok(Running {
            group,
            gateway,
            runtime,
            port,
            since: Instant::now(),
            models,
            checked: Instant::now(),
        })
    }

    /// The Harness at `version`, installed from npm when it is not yet: the
    /// directory the app is given (`STRATA_HARNESS_DIR`).
    fn harness(&self, node: &Path, runtime: &Path, version: &str) -> Result<PathBuf, String> {
        let root = self.base.join("app/harness");
        let dir = root.join(version);
        let package = dir.join("node_modules").join(DSH_PACKAGE);
        if !(dir.join(".complete").is_file() && package.join("lib/bin.js").is_file()) {
            self.set(
                "installing",
                Some(format!(
                    "installing the agent ({DSH_PACKAGE}@{version}) from npm"
                )),
            );
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).map_err(|error| error.to_string())?;
            std::fs::write(
                dir.join("package.json"),
                serde_json::to_string_pretty(&json!({
                    "name": "pom-strata-harness",
                    "private": true,
                    "dependencies": {DSH_PACKAGE: version},
                }))
                .unwrap_or_default(),
            )
            .map_err(|error| error.to_string())?;
            let npm = runtime.join("node/lib/node_modules/npm/bin/npm-cli.js");
            let log = self.log();
            let _ = append(
                &log,
                &format!("[pom] installing {DSH_PACKAGE}@{version} from npm"),
            );
            let mut command = Command::new(node);
            command
                .arg(&npm)
                .args([
                    "install",
                    "--omit=dev",
                    "--ignore-scripts",
                    "--no-audit",
                    "--no-fund",
                    "--loglevel=error",
                ])
                .current_dir(&dir)
                .env("npm_config_cache", root.join(".npm-cache"))
                .env("npm_config_update_notifier", "false")
                .env("PATH", path_with(&runtime.join("node/bin")))
                .stdout(Stdio::piped())
                .stderr(Stdio::piped());
            let mut group = Group::spawn(&mut command).map_err(|error| format!("npm: {error}"))?;
            group.forward_lines(&log, "[npm] ");
            let started = Instant::now();
            let status = loop {
                if let Some(status) = group.exited() {
                    break status;
                }
                if self.stop.load(Ordering::Acquire)
                    || started.elapsed() > Duration::from_secs(1800)
                {
                    group.stop(Duration::from_secs(3));
                    return Err("the agent's install was stopped".into());
                }
                thread::sleep(Duration::from_millis(500));
            };
            if !status.success() || !package.join("lib/bin.js").is_file() {
                return Err(format!(
                    "installing {DSH_PACKAGE}@{version} from npm failed ({status}); see run/strata-app.log"
                ));
            }
            std::fs::write(dir.join(".complete"), version).map_err(|error| error.to_string())?;
            // only this version stays
            if let Ok(entries) = std::fs::read_dir(&root) {
                for entry in entries.flatten() {
                    let name = entry.file_name();
                    if name != version && valid_version(&name.to_string_lossy()) {
                        let _ = std::fs::remove_dir_all(entry.path());
                    }
                }
            }
        }
        // The app looks for the package's own node_modules; npm hoisted them
        // into the directory above, where Node resolves them anyway.
        let link = package.join("node_modules");
        if !link.exists() {
            #[cfg(unix)]
            std::os::unix::fs::symlink("../..", &link).map_err(|error| error.to_string())?;
        }
        Ok(package)
    }
}

/// A runtime that carries the app and its Node.js.
pub fn has_app(runtime: &Path) -> bool {
    runtime.join("app/server/index.mjs").is_file()
        && runtime.join("app/dist/index.html").is_file()
        && runtime.join("app/APP.json").is_file()
        && runtime.join("node/bin/node").is_file()
}

fn valid_version(version: &str) -> bool {
    !version.is_empty()
        && version.len() <= 64
        && version.as_bytes()[0].is_ascii_digit()
        && version
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'-' | b'+'))
}

fn path_with(dir: &Path) -> std::ffi::OsString {
    let mut paths = vec![dir.to_path_buf()];
    if let Some(current) = std::env::var_os("PATH") {
        paths.extend(std::env::split_paths(&current));
    }
    std::env::join_paths(paths).unwrap_or_default()
}

fn free_port() -> Result<u16, String> {
    let listener =
        std::net::TcpListener::bind(("127.0.0.1", 0)).map_err(|error| error.to_string())?;
    listener
        .local_addr()
        .map(|address| address.port())
        .map_err(|error| error.to_string())
}

fn healthy(port: u16) -> bool {
    ureq::get(&format!("http://127.0.0.1:{port}/api/health"))
        .timeout(Duration::from_secs(2))
        .call()
        .is_ok()
}

/// The models the POM's endpoint lists now (ids as the POM names them);
/// `None` when it did not answer.
fn gateway_models(gateway: &Gateway) -> Option<Vec<String>> {
    let answer: Value = ureq::get(&format!("{}/models", gateway.openai_base_url))
        .set("Authorization", &format!("Bearer {}", gateway.api_key))
        .timeout(Duration::from_secs(10))
        .call()
        .ok()?
        .into_json()
        .ok()?;
    Some(
        answer["data"]
            .as_array()?
            .iter()
            .filter_map(|model| model["id"].as_str())
            .filter(|id| !id.is_empty() && id.len() <= 256 && !id.contains(['\r', '\n']))
            .take(100)
            .map(str::to_owned)
            .collect(),
    )
}

/// The connection's model rows, as the app stores them.
fn model_rows(models: &[String]) -> Vec<Value> {
    models
        .iter()
        .map(|id| json!({"id": id, "name": id, "contextMax": null, "maxOutput": null, "vision": null, "reasoning": null}))
        .collect()
}

/// Saves the connection's new model list through the running app's API (the
/// app keeps the key it already has for an unchanged address).
fn post_connection(port: u16, gateway: &Gateway, models: &[String]) -> Result<(), String> {
    ureq::post(&format!("http://127.0.0.1:{port}/api/model-providers"))
        .timeout(Duration::from_secs(10))
        .send_json(json!({
            "id": CONNECTION_ID,
            "kind": "custom",
            "name": "POM",
            "baseUrl": gateway.openai_base_url,
            "protocol": "openai-chat",
            "models": model_rows(models),
        }))
        .map(|_| ())
        .map_err(|error| error.to_string())
}

/// Writes (or replaces) this plugin's connection in the app's model list,
/// keeping the connections the person added. The file holds the POM key, so
/// it is the owner's alone, as the app keeps it.
fn write_connection(file: &Path, gateway: &Gateway, models: &[String]) -> Result<(), String> {
    let mut connections: Vec<Value> = std::fs::read_to_string(file)
        .ok()
        .and_then(|text| serde_json::from_str(&text).ok())
        .unwrap_or_default();
    connections.retain(|connection| connection["id"] != CONNECTION_ID);
    connections.insert(
        0,
        json!({
            "id": CONNECTION_ID,
            "kind": "custom",
            "provider": null,
            "name": "POM",
            "baseUrl": gateway.openai_base_url,
            "protocol": "openai-chat",
            "apiKey": gateway.api_key,
            "models": model_rows(models),
        }),
    );
    if let Some(parent) = file.parent() {
        std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    let temp = file.with_extension("json.pom-tmp");
    let text = serde_json::to_string_pretty(&connections).map_err(|error| error.to_string())?;
    {
        let mut options = std::fs::OpenOptions::new();
        options.write(true).create(true).truncate(true);
        #[cfg(unix)]
        std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o600);
        let mut out = options.open(&temp).map_err(|error| error.to_string())?;
        out.write_all(text.as_bytes())
            .map_err(|error| error.to_string())?;
    }
    std::fs::rename(&temp, file).map_err(|error| error.to_string())
}

/// The app's first workspace (`root`, where chats work unless they choose
/// another folder) is the POM's common workspace, as the Harness plugin
/// registers it at every start. Without a saved list the app makes it itself
/// from `STRATA_WORKSPACE`; a saved one keeps the person's other workspaces.
fn keep_workspace(file: &Path, workspace: &Path) -> Result<(), String> {
    let Some(mut list) = std::fs::read_to_string(file)
        .ok()
        .and_then(|text| serde_json::from_str::<Vec<Value>>(&text).ok())
        .filter(|list| !list.is_empty())
    else {
        return Ok(());
    };
    let path = workspace.to_string_lossy();
    let name = workspace
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.to_string());
    let root = json!({"id": "root", "name": name, "path": path});
    match list.iter_mut().find(|entry| entry["id"] == "root") {
        Some(entry) if entry["path"] == root["path"] => return Ok(()),
        Some(entry) => *entry = root,
        None => list.insert(0, root),
    }
    let text = serde_json::to_string_pretty(&list).map_err(|error| error.to_string())?;
    std::fs::write(file, text).map_err(|error| format!("{}: {error}", file.display()))
}

fn append(log: &Path, line: &str) -> std::io::Result<()> {
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(log)?;
    writeln!(file, "{line}")
}

impl Group {
    /// Copies the child's output, line by line, to `log` and (prefixed) to
    /// this process's stderr, which the POM keeps as the node's log.
    pub(crate) fn forward_lines(&mut self, log: &Path, prefix: &'static str) {
        for stream in self.take_output() {
            let log = log.to_path_buf();
            let _ = thread::Builder::new()
                .name("pom-plugin-strata-app-output".into())
                .spawn(move || {
                    let mut file = std::fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(&log)
                        .ok();
                    for line in std::io::BufReader::new(stream)
                        .split(b'\n')
                        .map_while(Result::ok)
                    {
                        let line = String::from_utf8_lossy(&line);
                        let line = line.trim_end_matches('\r');
                        if let Some(file) = file.as_mut() {
                            let _ = writeln!(file, "{line}");
                        }
                        let _ = std::io::stderr()
                            .lock()
                            .write_all(format!("{prefix}{line}\n").as_bytes());
                    }
                });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_gateway_comes_from_host_configure() {
        let gateway = Gateway::from_configure(&json!({
            "gateway": {"openai_base_url": "http://127.0.0.1:8080/v1/", "api_key": "k"},
            "workspace_root": "/data/workspace",
        }))
        .unwrap();
        assert_eq!(gateway.openai_base_url, "http://127.0.0.1:8080/v1");
        assert_eq!(
            gateway.workspace_root,
            Some(PathBuf::from("/data/workspace"))
        );
        assert!(Gateway::from_configure(&json!({"gateway": {"openai_base_url": "x"}})).is_none());
        let relative = Gateway::from_configure(&json!({
            "gateway": {"openai_base_url": "http://a/v1", "api_key": "k"}, "workspace_root": "rel"
        }))
        .unwrap();
        assert_eq!(relative.workspace_root, None);
    }

    #[test]
    fn the_connection_replaces_its_own_and_keeps_the_others() {
        let dir = std::env::temp_dir().join(format!("pom-strata-app-{}", std::process::id()));
        let file = dir.join("data/model-providers.json");
        std::fs::create_dir_all(file.parent().unwrap()).unwrap();
        std::fs::write(
            &file,
            r#"[{"id":"mine","name":"Mine"},{"id":"pom","name":"old"}]"#,
        )
        .unwrap();
        let gateway = Gateway {
            openai_base_url: "http://127.0.0.1:8080/v1".into(),
            api_key: "secret".into(),
            workspace_root: None,
        };
        write_connection(&file, &gateway, &["model-a".into()]).unwrap();
        let saved: Vec<Value> =
            serde_json::from_str(&std::fs::read_to_string(&file).unwrap()).unwrap();
        assert_eq!(saved.len(), 2);
        assert_eq!(saved[0]["id"], "pom");
        assert_eq!(saved[0]["baseUrl"], "http://127.0.0.1:8080/v1");
        assert_eq!(saved[0]["models"][0]["id"], "model-a");
        assert_eq!(saved[1]["id"], "mine");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                std::fs::metadata(&file).unwrap().permissions().mode() & 0o777,
                0o600
            );
        }
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn the_first_workspace_follows_the_pom_workspace() {
        let dir = std::env::temp_dir().join(format!("pom-strata-ws-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("workspaces.json");
        keep_workspace(&file, Path::new("/srv/pom/workspace")).unwrap();
        assert!(!file.exists(), "no saved list: the app makes its own");
        std::fs::write(&file, r#"[{"id":"root","name":"old","path":"/old"},{"id":"w2","name":"mine","path":"/mine"}]"#).unwrap();
        keep_workspace(&file, Path::new("/srv/pom/workspace")).unwrap();
        let list: Vec<Value> =
            serde_json::from_str(&std::fs::read_to_string(&file).unwrap()).unwrap();
        assert_eq!(
            list[0],
            json!({"id": "root", "name": "workspace", "path": "/srv/pom/workspace"})
        );
        assert_eq!(list[1]["id"], "w2");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn only_plain_versions_name_a_folder() {
        assert!(valid_version("0.2.1-rc.5"));
        for bad in ["", "../x", "latest", "1/2", "1 2"] {
            assert!(!valid_version(bad), "{bad}");
        }
    }
}
