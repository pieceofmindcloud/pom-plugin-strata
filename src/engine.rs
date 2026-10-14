//! The Strata engine on this node: one model at a time, loaded when the POM
//! deploys it here and stopped when the POM takes it away.
//!
//! One manager thread owns every child process (the pack tool, then Strata's
//! server, which runs the engine). `query` never waits on it: commands go
//! through a channel and the state is read from a shared snapshot. Children
//! are started from that long-lived thread in their own process group, with
//! the parent-death signal set on Linux, so killing the plugin host takes the
//! engine down with it and never strands GPU memory.

use crate::gguf;
use crate::runtime;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{channel, Receiver, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

const PLUGIN_DIR: &str = "strata";
const POLL: Duration = Duration::from_secs(2);
/// How long Strata may take to stop before it is killed.
const STOP_GRACE: Duration = Duration::from_secs(45);
/// The context Strata's setup writes when nothing else is asked.
const DEFAULT_CONTEXT: u32 = 32768;
/// Below this the KV is small enough for the engine's default type.
const KV_TYPE_FROM: u32 = 8192;
/// From this context on, the KV lives in RAM and only a window stays in VRAM
/// (Strata's setup does the same).
const KV_STREAM_FROM: u32 = 65536;
const KV_RESIDENT: &str = "32768";

/// What the POM asks for when it deploys a model on this node.
#[derive(Debug, Clone, PartialEq)]
pub struct LoadRequest {
    pub model_path: PathBuf,
    /// The name the network knows the model by.
    pub model_label: String,
    pub context: u32,
    /// GPU indices as this node numbers them; empty = Strata's default (0).
    pub gpus: Vec<u32>,
    /// The node's AMD cards as HIP numbers them (`gpu_devices`): the HIP
    /// runtime runs on them when the node has no NVIDIA card.
    pub amd_gpus: Vec<u32>,
    /// Concurrent requests (Strata's `parallel` batch slots).
    pub sessions: u32,
    /// `int8`, `q4_0` or `k8v4`.
    pub kv: String,
    /// Engine settings the deployment chose (`crate::settings`).
    pub settings: serde_json::Map<String, Value>,
    /// A POM slice runs as one stage of a split across machines: the link
    /// the POM gives (`{}` for the first stage, which makes its own secret).
    pub link: Option<Link>,
    /// Where the bridge keeps the conversation sessions (`POM_STRATA_SESSIONS`):
    /// the node's KV snapshot folder, so the KV cache clear empties them.
    pub session_dir: Option<PathBuf>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Link {
    /// The earlier stage's secret, which a later stage presents to it.
    pub token: Option<String>,
    /// For a later stage: the loopback port where this node relays the link
    /// to the earlier stage's machine.
    pub prev_port: Option<u16>,
    /// For a later stage: how many frames the POM's relay keeps in flight at
    /// once (the deployment's `prefill_max_inflight_chunks`), so the engine
    /// asks for as many prompt pieces ahead.
    pub max_inflight: Option<u32>,
}

impl Link {
    fn from_json(value: &Value) -> Result<Option<Self>, String> {
        if value.is_null() {
            return Ok(None);
        }
        let token = match &value["token"] {
            Value::Null => None,
            token => Some(
                token
                    .as_str()
                    .filter(|token| {
                        (16..=256).contains(&token.len())
                            && token.bytes().all(|b| b.is_ascii_graphic())
                    })
                    .ok_or("link.token is malformed")?
                    .to_owned(),
            ),
        };
        let prev_port = match &value["prev_port"] {
            Value::Null => None,
            port => Some(
                port.as_u64()
                    .and_then(|port| u16::try_from(port).ok())
                    .filter(|port| *port > 0)
                    .ok_or("link.prev_port is not a port")?,
            ),
        };
        let max_inflight = match &value["max_inflight"] {
            Value::Null => None,
            count => Some(
                count
                    .as_u64()
                    .and_then(|count| u32::try_from(count).ok())
                    .filter(|count| *count > 0)
                    .ok_or("link.max_inflight is not a positive count")?,
            ),
        };
        Ok(Some(Self {
            token,
            prev_port,
            max_inflight,
        }))
    }
}

impl LoadRequest {
    /// The backend this load runs on: CUDA on the node's NVIDIA cards, HIP on
    /// its AMD ones when it has no NVIDIA card (a process runs one backend).
    pub fn backend(&self) -> runtime::Backend {
        if self.gpus.is_empty() && !self.amd_gpus.is_empty() {
            runtime::Backend::Hip
        } else {
            runtime::Backend::Cuda
        }
    }

    /// The cards of this load, numbered as its backend's runtime numbers them.
    pub fn backend_gpus(&self) -> &[u32] {
        match self.backend() {
            runtime::Backend::Cuda => &self.gpus,
            runtime::Backend::Hip => &self.amd_gpus,
        }
    }

    pub fn from_json(value: &Value) -> Result<Self, String> {
        let model_path = value["model_path"]
            .as_str()
            .filter(|path| !path.is_empty())
            .map(PathBuf::from)
            .ok_or("model_path is missing")?;
        if !model_path.is_absolute() {
            return Err("model_path must be absolute".into());
        }
        let model_label = value["model_label"]
            .as_str()
            .filter(|label| !label.trim().is_empty())
            .map(str::to_owned)
            .unwrap_or_else(|| {
                model_path
                    .file_stem()
                    .map(|stem| stem.to_string_lossy().into_owned())
                    .unwrap_or_else(|| "strata".into())
            });
        let context = value["n_ctx"]
            .as_u64()
            .filter(|context| *context >= 2048)
            .map(|context| context.min(1 << 20) as u32)
            .unwrap_or(DEFAULT_CONTEXT);
        let gpus = value["gpus"]
            .as_array()
            .map(|list| {
                list.iter()
                    .filter_map(Value::as_u64)
                    .filter(|gpu| *gpu < 64)
                    .map(|gpu| gpu as u32)
                    .collect()
            })
            .unwrap_or_default();
        let amd_gpus = value["gpu_devices"]
            .as_array()
            .map(|list| {
                list.iter()
                    .filter(|device| device["vendor"] == "amd")
                    .filter_map(|device| device["ordinal"].as_u64())
                    .filter(|ordinal| *ordinal < 64)
                    .map(|ordinal| ordinal as u32)
                    .collect()
            })
            .unwrap_or_default();
        let sessions = value["max_sessions"]
            .as_u64()
            .map(|sessions| sessions.clamp(1, 8) as u32)
            .unwrap_or(1);
        let kv = match value["kv"].as_str().unwrap_or("int8") {
            "q4_0" => "q4_0",
            "k8v4" => "k8v4",
            _ => "int8",
        }
        .to_owned();
        Ok(Self {
            model_path,
            model_label,
            context,
            gpus,
            amd_gpus,
            sessions,
            kv,
            settings: crate::settings::select(&value["settings"]),
            link: Link::from_json(&value["link"])?,
            session_dir: match value["session_dir"].as_str().filter(|dir| !dir.is_empty()) {
                Some(dir) => {
                    let dir = PathBuf::from(dir);
                    if !dir.is_absolute() {
                        return Err("session_dir must be absolute".into());
                    }
                    Some(dir)
                }
                None => None,
            },
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    /// No model.
    Idle,
    /// The runtime is being downloaded.
    Installing,
    /// The pack tool is preparing the model for Strata.
    Preparing,
    /// Strata is reading the weights.
    Loading,
    Ready,
    Stopping,
    Error,
}

impl Phase {
    pub fn as_str(self) -> &'static str {
        match self {
            Phase::Idle => "idle",
            Phase::Installing => "installing",
            Phase::Preparing => "preparing",
            Phase::Loading => "loading",
            Phase::Ready => "ready",
            Phase::Stopping => "stopping",
            Phase::Error => "error",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Snapshot {
    pub phase: Phase,
    pub detail: Option<String>,
    pub runtime: Option<PathBuf>,
    /// The HIP runtime (AMD cards), when there is one.
    pub runtime_hip: Option<PathBuf>,
    pub runtime_version: Option<String>,
    pub runtime_error: Option<String>,
    pub request: Option<LoadRequest>,
    /// The context Strata reports once loaded.
    pub context: Option<u32>,
    pub port: Option<u16>,
    /// A stage server's loopback port (an earlier stage of a split model).
    pub link_port: Option<u16>,
    /// The secret of this machine's earlier stage, from the load on.
    pub link_token: Option<String>,
    pub api_key: String,
    pub busy: bool,
    /// Requests the bridge is serving now (generations and session steps, queued or running).
    pub in_flight: u32,
    pub since: Instant,
    pub log: PathBuf,
}

impl Snapshot {
    /// For the POM (`engine.status`) and the plugin screen: never the key.
    pub fn status_json(&self) -> Value {
        json!({
            "state": self.phase.as_str(),
            "detail": self.detail,
            "model": self.request.as_ref().map(|request| request.model_label.clone()),
            "model_path": self.request.as_ref().map(|request| request.model_path.display().to_string()),
            "gpus": self.request.as_ref().map(|request| request.gpus.clone()),
            "n_ctx": self.context.or(self.request.as_ref().map(|request| request.context)),
            "busy": self.busy,
            "in_flight": self.in_flight,
            // An earlier stage of a split model runs the engine alone: no
            // server, so no Strata app to open; the last stage has it.
            "earlier_stage": self.port.is_none() && self.link_port.is_some(),
            "runtime": {
                "ready": self.runtime.is_some(),
                "version": self.runtime_version,
                "error": self.runtime_error,
            },
            "seconds_in_state": self.since.elapsed().as_secs(),
        })
    }

    /// How the POM node reaches the generation route, while the model is ready.
    pub fn upstream_json(&self) -> Value {
        match (self.phase, self.port) {
            (Phase::Ready, Some(port)) => {
                json!({"status": "ready", "port": port, "token": self.api_key})
            }
            (_, None) if self.link_port.is_some() => json!({
                "status": "error",
                "error": "this machine runs an earlier stage of the model; the last stage serves the requests",
            }),
            (Phase::Error, _) => json!({"status": "error", "error": self.detail}),
            _ => json!({"status": "starting"}),
        }
    }

    /// Where the POM node relays the stage link of a split model to, while
    /// this machine's earlier stage is ready (`engine.link`).
    /// The secret goes with every answer once a load gave this machine an
    /// earlier stage: the POM hands it only to the next stage's machine.
    pub fn link_json(&self) -> Value {
        let mut answer = match (self.phase, self.link_port) {
            (Phase::Ready, Some(port)) => json!({"status": "ready", "port": port}),
            (Phase::Error, _) => json!({"status": "error", "error": self.detail}),
            _ => json!({"status": "starting"}),
        };
        if let Some(token) = &self.link_token {
            answer["token"] = json!(token);
        }
        answer
    }

    /// The service the POM draws next to this machine.
    pub fn service(&self) -> Value {
        let (status, detail) = match (self.phase, &self.runtime) {
            (Phase::Installing, _) => ("installing", self.detail.clone()),
            (_, None) => (
                if self.runtime_error.is_some() {
                    "error"
                } else {
                    "waiting"
                },
                self.runtime_error
                    .clone()
                    .or(Some("the Strata runtime is not installed".into())),
            ),
            (Phase::Error, _) => ("error", self.detail.clone()),
            (Phase::Preparing | Phase::Loading, _) => (
                "starting",
                self.request
                    .as_ref()
                    .map(|request| request.model_label.clone()),
            ),
            (Phase::Ready, _) => (
                "ready",
                self.request
                    .as_ref()
                    .map(|request| request.model_label.clone()),
            ),
            _ => ("ready", Some("no model loaded".into())),
        };
        json!({
            "id": "strata",
            "kind": "inference",
            "name": "Strata",
            "status": status,
            "version": self.runtime_version,
            "detail": detail.map(|text| text.chars().take(190).collect::<String>()),
            "architectures": [gguf::STRATA_ARCHITECTURE],
            // A POM slice of a model runs as one stage of a split across machines.
            "stages": true,
            // The POM settings that apply to those stages: the nodes carry the hidden state between them.
            "pom_settings": ["prefill_max_inflight_chunks", "hidden_state_dtype"],
            "settings": crate::settings::schema(),
        })
    }
}

enum Command_ {
    Load(LoadRequest),
    Unload,
    Stop,
}

pub struct Engine {
    shared: Arc<Mutex<Snapshot>>,
    commands: Sender<Command_>,
    thread: Option<JoinHandle<()>>,
}

pub fn plugin_directory() -> Result<PathBuf, String> {
    let parent = std::env::var_os("POM_PLUGIN_DB")
        .map(PathBuf::from)
        .and_then(|database| database.parent().map(Path::to_path_buf))
        .filter(|parent| !parent.as_os_str().is_empty());
    let parent = match parent {
        Some(parent) => parent,
        None => std::env::current_dir().map_err(|error| error.to_string())?,
    };
    Ok(parent.join(PLUGIN_DIR))
}

fn random_key() -> String {
    let mut bytes = [0u8; 24];
    getrandom::getrandom(&mut bytes).expect("system randomness");
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

impl Engine {
    pub fn start(base: PathBuf) -> Self {
        let _ = std::fs::create_dir_all(base.join("run"));
        let shared = Arc::new(Mutex::new(Snapshot {
            phase: Phase::Idle,
            detail: None,
            runtime: None,
            runtime_hip: None,
            runtime_version: None,
            runtime_error: None,
            request: None,
            context: None,
            port: None,
            link_port: None,
            link_token: None,
            api_key: random_key(),
            busy: false,
            in_flight: 0,
            since: Instant::now(),
            log: base.join("run/strata-server.log"),
        }));
        let (commands, receiver) = channel();
        let thread_shared = Arc::clone(&shared);
        let thread = thread::Builder::new()
            .name("pom-plugin-strata-engine".into())
            .spawn(move || Manager::new(base, thread_shared, receiver).run())
            .expect("start the engine manager");
        Self {
            shared,
            commands,
            thread: Some(thread),
        }
    }

    pub fn snapshot(&self) -> Snapshot {
        self.shared
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    pub fn load(&self, request: LoadRequest) -> Result<(), String> {
        gguf::inspect(&request.model_path)?;
        self.commands
            .send(Command_::Load(request))
            .map_err(|_| "the engine manager stopped".to_owned())
    }

    pub fn unload(&self) {
        let _ = self.commands.send(Command_::Unload);
    }

    pub fn log_tail(&self, lines: usize) -> Vec<String> {
        let path = self.snapshot().log;
        let Ok(text) = std::fs::read(&path) else {
            return Vec::new();
        };
        let start = text.len().saturating_sub(256 * 1024);
        String::from_utf8_lossy(&text[start..])
            .lines()
            .rev()
            .take(lines)
            .map(str::to_owned)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect()
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        let _ = self.commands.send(Command_::Stop);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// A child process in its own group.
struct Group {
    child: Child,
}

impl Group {
    fn spawn(command: &mut Command) -> std::io::Result<Self> {
        #[cfg(unix)]
        unsafe {
            use std::os::unix::process::CommandExt;
            command.pre_exec(|| {
                libc::setsid();
                #[cfg(target_os = "linux")]
                libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGTERM);
                Ok(())
            });
        }
        // The plugin host talks to the POM over its own stdin/stdout: a child
        // must never inherit them.
        command.stdin(Stdio::null());
        Ok(Self {
            child: command.spawn()?,
        })
    }

    /// Copies the child's output, line by line, to the Strata log file and to
    /// this process's stderr. The POM reads the plugin host's stderr into the
    /// node's log, which is what reaches the network's admins as the process
    /// log - the same path llama.cpp's lines take.
    fn forward_output(&mut self, log: &Path) {
        let streams: Vec<Box<dyn std::io::Read + Send>> = [
            self.child
                .stdout
                .take()
                .map(|out| Box::new(out) as Box<dyn std::io::Read + Send>),
            self.child
                .stderr
                .take()
                .map(|err| Box::new(err) as Box<dyn std::io::Read + Send>),
        ]
        .into_iter()
        .flatten()
        .collect();
        for stream in streams {
            let log = log.to_path_buf();
            let _ = thread::Builder::new()
                .name("pom-plugin-strata-output".into())
                .spawn(move || copy_lines(stream, &log));
        }
    }

    fn exited(&mut self) -> Option<std::process::ExitStatus> {
        self.child.try_wait().ok().flatten()
    }

    fn signal(&self, _signal: i32) {
        #[cfg(unix)]
        unsafe {
            libc::kill(-(self.child.id() as i32), _signal);
        }
    }

    /// Asks the group to end, then forces it after `grace`.
    fn stop(mut self, grace: Duration) {
        #[cfg(unix)]
        self.signal(libc::SIGTERM);
        #[cfg(not(unix))]
        let _ = self.child.kill();
        let deadline = Instant::now() + grace;
        while Instant::now() < deadline {
            if self.exited().is_some() {
                // The engine may still be closing after the server ended.
                #[cfg(unix)]
                self.signal(libc::SIGTERM);
                thread::sleep(Duration::from_millis(500));
                #[cfg(unix)]
                self.signal(libc::SIGKILL);
                return;
            }
            thread::sleep(Duration::from_millis(200));
        }
        #[cfg(unix)]
        self.signal(libc::SIGKILL);
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

enum Work {
    None,
    Packing {
        group: Group,
        request: LoadRequest,
    },
    Serving {
        group: Group,
    },
    /// The engine alone, as the stage server of an earlier stage.
    Staging {
        group: Group,
    },
}

struct Manager {
    base: PathBuf,
    shared: Arc<Mutex<Snapshot>>,
    commands: Receiver<Command_>,
    work: Work,
    installer: Option<(runtime::Backend, Receiver<Result<PathBuf, String>>)>,
    /// A load asked while the runtime was not ready yet.
    waiting: Option<LoadRequest>,
}

impl Manager {
    fn new(base: PathBuf, shared: Arc<Mutex<Snapshot>>, commands: Receiver<Command_>) -> Self {
        Self {
            base,
            shared,
            commands,
            work: Work::None,
            installer: None,
            waiting: None,
        }
    }

    fn update(&self, change: impl FnOnce(&mut Snapshot)) {
        let mut snapshot = self
            .shared
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        change(&mut snapshot);
    }

    fn set_phase(&self, phase: Phase, detail: Option<String>) {
        self.update(|snapshot| {
            if snapshot.phase != phase {
                snapshot.since = Instant::now();
            }
            snapshot.phase = phase;
            snapshot.detail = detail;
        });
    }

    fn runtime(&self) -> Option<PathBuf> {
        self.shared
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .runtime
            .clone()
    }

    fn find_runtime(&mut self) {
        // the HIP runtime is looked for, never downloaded, here: a load on AMD cards installs it (load)
        let hip = runtime::local_runtime(runtime::Backend::Hip).or_else(|| {
            runtime::installed_dir(&self.base, runtime::Backend::Hip)
                .filter(|dir| runtime::is_runtime(dir))
        });
        self.update(|snapshot| snapshot.runtime_hip = hip);
        let cuda = runtime::local_runtime(runtime::Backend::Cuda).or_else(|| {
            runtime::installed_dir(&self.base, runtime::Backend::Cuda)
                .filter(|dir| runtime::is_runtime(dir))
        });
        if let Some(dir) = cuda {
            let version = runtime::version_of(&dir);
            self.update(|snapshot| {
                snapshot.runtime = Some(dir);
                snapshot.runtime_version = version;
            });
            return;
        }
        if !runtime::has_release(runtime::Backend::Cuda) {
            self.update(|snapshot| {
                snapshot.runtime_error = Some(
                    "no Strata runtime: this is a development build without build/runtime \
                     (run scripts/build-runtime.sh in the plugin source)"
                        .into(),
                )
            });
            return;
        }
        self.install(runtime::Backend::Cuda);
    }

    /// Downloads the pinned runtime of `backend` on a thread (poll_installer).
    fn install(&mut self, backend: runtime::Backend) {
        let (sender, receiver) = channel();
        let base = self.base.clone();
        let shared = Arc::clone(&self.shared);
        self.set_phase(
            Phase::Installing,
            Some(format!("downloading the Strata {} runtime", backend.name())),
        );
        let _ = thread::Builder::new()
            .name("pom-plugin-strata-install".into())
            .spawn(move || {
                let result = runtime::install(&base, backend, &|done, total| {
                    if let Ok(mut snapshot) = shared.lock() {
                        snapshot.detail = Some(format!(
                            "downloading the Strata {} runtime: {} of {} MB",
                            backend.name(),
                            done >> 20,
                            total >> 20
                        ));
                    }
                });
                let _ = sender.send(result);
            });
        self.installer = Some((backend, receiver));
    }

    fn run(mut self) {
        self.find_runtime();
        loop {
            match self.commands.recv_timeout(POLL) {
                Ok(Command_::Load(request)) => self.load(request),
                Ok(Command_::Unload) => self.unload(),
                Ok(Command_::Stop) | Err(RecvTimeoutError::Disconnected) => {
                    self.unload();
                    return;
                }
                Err(RecvTimeoutError::Timeout) => {}
            }
            self.poll_installer();
            self.poll_work();
        }
    }

    fn poll_installer(&mut self) {
        let Some((backend, receiver)) = &self.installer else {
            return;
        };
        let backend = *backend;
        let Ok(result) = receiver.try_recv() else {
            return;
        };
        self.installer = None;
        match result {
            Ok(dir) if backend == runtime::Backend::Hip => {
                self.update(|snapshot| snapshot.runtime_hip = Some(dir));
                self.set_phase(Phase::Idle, None);
                if let Some(request) = self.waiting.take() {
                    self.load(request);
                }
            }
            Ok(dir) => {
                let version = runtime::version_of(&dir);
                self.update(|snapshot| {
                    snapshot.runtime = Some(dir);
                    snapshot.runtime_version = version;
                    snapshot.runtime_error = None;
                });
                self.set_phase(Phase::Idle, None);
                if let Some(request) = self.waiting.take() {
                    self.load(request);
                }
            }
            Err(error) => {
                self.update(|snapshot| snapshot.runtime_error = Some(error.clone()));
                self.set_phase(Phase::Error, Some(error));
            }
        }
    }

    fn unload(&mut self) {
        self.waiting = None;
        match std::mem::replace(&mut self.work, Work::None) {
            Work::None => {}
            Work::Packing { group, .. } | Work::Serving { group, .. } | Work::Staging { group } => {
                self.set_phase(Phase::Stopping, None);
                group.stop(STOP_GRACE);
            }
        }
        self.update(|snapshot| {
            snapshot.request = None;
            snapshot.context = None;
            snapshot.port = None;
            snapshot.link_port = None;
            snapshot.link_token = None;
            snapshot.busy = false;
            snapshot.in_flight = 0;
        });
        if self.installer.is_none() {
            self.set_phase(Phase::Idle, None);
        }
    }

    /// The last line Strata wrote: why it stopped, most of the time.
    fn last_log_line(&self) -> Option<String> {
        let path = self.shared.lock().ok()?.log.clone();
        let text = std::fs::read(path).ok()?;
        let start = text.len().saturating_sub(8 * 1024);
        String::from_utf8_lossy(&text[start..])
            .lines()
            .rev()
            .map(str::trim)
            .find(|line| !line.is_empty() && !line.starts_with("[pom] starting"))
            .map(|line| line.chars().take(300).collect())
    }

    fn fail(&mut self, error: String) {
        self.append_log(&format!("[pom] {error}"));
        self.set_phase(Phase::Error, Some(error));
    }

    fn append_log(&self, line: &str) {
        use std::io::Write;
        let path = self.shared.lock().map(|snapshot| snapshot.log.clone());
        if let Ok(path) = path {
            if let Ok(mut file) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
            {
                let _ = writeln!(file, "{line}");
            }
        }
    }

    fn python_command(&self, runtime: &Path) -> Result<Command, String> {
        let python = std::env::var_os("STRATA_PYTHON").unwrap_or_else(|| "python3".into());
        let mut command = Command::new(python);
        command
            .env(
                "PYTHONPATH",
                format!(
                    "{}:{}",
                    runtime.join("pylib").display(),
                    runtime.join("strata").display()
                ),
            )
            .env("PYTHONUNBUFFERED", "1")
            .env("PYTHONDONTWRITEBYTECODE", "1")
            .env(
                "STRATA_GGUF_PY",
                runtime.join("strata/third_party/llama.cpp/gguf-py"),
            )
            .env("TMPDIR", self.base.join("run"))
            .current_dir(self.base.join("run"))
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        Ok(command)
    }

    fn load(&mut self, request: LoadRequest) {
        self.unload();
        self.update(|snapshot| snapshot.request = Some(request.clone()));
        if request.backend() == runtime::Backend::Hip {
            let hip = self
                .shared
                .lock()
                .ok()
                .and_then(|snapshot| snapshot.runtime_hip.clone());
            let Some(runtime) = hip else {
                if self.installer.is_some() {
                    self.waiting = Some(request);
                    return;
                }
                if runtime::has_release(runtime::Backend::Hip) {
                    self.waiting = Some(request);
                    self.install(runtime::Backend::Hip);
                    return;
                }
                return self.fail(
                    "this node's GPUs are AMD and there is no Strata HIP runtime \
                     (scripts/build-runtime.sh --hip in the plugin source)"
                        .into(),
                );
            };
            return self.load_with(runtime, request);
        }
        let Some(runtime) = self.runtime() else {
            if self.installer.is_some() {
                self.waiting = Some(request);
                return;
            }
            let error = self
                .shared
                .lock()
                .ok()
                .and_then(|snapshot| snapshot.runtime_error.clone())
                .unwrap_or_else(|| "the Strata runtime is not installed".into());
            return self.fail(error);
        };
        self.load_with(runtime, request)
    }

    /// The load once the runtime of its backend is here.
    fn load_with(&mut self, runtime: PathBuf, request: LoadRequest) {
        let model = match gguf::inspect(&request.model_path) {
            Ok(model) => model,
            Err(error) => return self.fail(error),
        };
        if let Some(stage) = model.stage {
            match &request.link {
                None => return self.fail("a POM slice needs the stage link the POM gives".into()),
                Some(link)
                    if !stage.first() && (link.prev_port.is_none() || link.token.is_none()) =>
                {
                    return self.fail(
                        "a later stage needs the relay to the earlier stage and its secret".into(),
                    )
                }
                // A stage before the last (the first or a middle one) serves the next: its own secret, which the
                // POM asks before the engine is up. A middle stage also presents the earlier one's (link.token).
                Some(_) if !stage.last() => {
                    let token = random_key();
                    self.update(|snapshot| snapshot.link_token = Some(token));
                }
                Some(_) => {}
            }
        }
        let pack = self.pack_dir(&request.model_path);
        if pack_ready(&pack) {
            return self.serve(runtime, request, &model, pack);
        }
        self.append_log(&format!(
            "[pom] preparing {} for Strata in {}",
            request.model_path.display(),
            pack.display()
        ));
        let mut command = match self.python_command(&runtime) {
            Ok(command) => command,
            Err(error) => return self.fail(error),
        };
        command
            .arg(runtime.join("strata/tools/iq_pack.py"))
            .arg("--gguf")
            .arg(&request.model_path)
            .arg("--out")
            .arg(&pack);
        if model.shards.len() > 2 || model.stage.is_some() {
            // Unsloth's files keep small projections as Q8_0; the engine reads them as BF16.
            // (A slice no longer says how many files its model had: the flag only converts what needs it.)
            command.arg("--compat-bf16");
        }
        match Group::spawn(&mut command) {
            Ok(mut group) => {
                group.forward_output(
                    &self
                        .shared
                        .lock()
                        .map(|snapshot| snapshot.log.clone())
                        .unwrap_or_default(),
                );
                self.set_phase(
                    Phase::Preparing,
                    Some("preparing the model for Strata".into()),
                );
                self.work = Work::Packing { group, request };
            }
            Err(error) => self.fail(format!("the pack tool did not start: {error}")),
        }
    }

    fn pack_dir(&self, model: &Path) -> PathBuf {
        use sha2::{Digest, Sha256};
        let stem = model
            .file_stem()
            .map(|stem| stem.to_string_lossy().to_lowercase())
            .unwrap_or_default();
        let stem: String = stem
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                    c
                } else {
                    '_'
                }
            })
            .take(80)
            .collect();
        let size = std::fs::metadata(model).map(|meta| meta.len()).unwrap_or(0);
        let digest = Sha256::digest(format!("{}\0{size}", model.display()).as_bytes());
        let tag: String = digest[..4]
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        self.base.join("packs").join(format!("{stem}-{tag}"))
    }

    fn mtp_dir(&self, model: &Path) -> Option<PathBuf> {
        let mut candidates = vec![self.base.join("mtp/rt")];
        for ancestor in model.ancestors().skip(1).take(3) {
            candidates.push(ancestor.join("mtp/rt"));
        }
        candidates
            .into_iter()
            .find(|dir| dir.join("experts.bin").is_file())
    }

    fn serve(
        &mut self,
        runtime: PathBuf,
        request: LoadRequest,
        model: &gguf::Model,
        pack: PathBuf,
    ) {
        let port = match free_port() {
            Ok(port) => port,
            Err(error) => return self.fail(error),
        };
        let mut args: Vec<String> = vec![
            "--pack".into(),
            pack.display().to_string(),
            "--native".into(),
            request.model_path.display().to_string(),
        ];
        if model.shards.len() <= 2 {
            if let Some(ple) = &model.ple_shard {
                args.push("--ple-gguf".into());
                args.push(ple.display().to_string());
            }
        }
        let name = request.model_path.to_string_lossy().to_lowercase();
        let profile = if name.contains("coder") {
            "expert-profile-coder.bin"
        } else {
            "expert-profile.bin"
        };
        args.extend(
            [
                "--expert-profile",
                &runtime
                    .join("strata/data")
                    .join(profile)
                    .display()
                    .to_string(),
                "--expert-cache",
                "auto",
                "--prefill",
                "auto",
            ]
            .map(str::to_owned),
        );
        let stage_server = model.stage.is_some_and(|stage| !stage.last());
        if stage_server {
            // The drafter belongs to the last stage, with the head.
        } else if let Some(mtp) = self.mtp_dir(&request.model_path) {
            args.extend(["--spec", "4", "--spec-min-p", "0.5", "--mtp"].map(str::to_owned));
            args.push(mtp.display().to_string());
        } else {
            // No drafter for this model (a slice without one): the native IQ engine still needs a verify window
            // (--spec T >= 2); the windows come from suffix matches in the context.
            args.extend(["--spec", "4"].map(str::to_owned));
        }
        if let Some(port) = request.link.as_ref().and_then(|link| link.prev_port) {
            args.push("--stage-prev".into());
            args.push(format!("127.0.0.1:{port}"));
        }
        args.push("--max-context".into());
        args.push(request.context.to_string());
        if request.context > KV_TYPE_FROM {
            args.push("--kv".into());
            args.push(request.kv.clone());
        }
        if request.context >= KV_STREAM_FROM {
            args.push("--kv-resident".into());
            args.push(KV_RESIDENT.into());
        }
        if stage_server {
            return self.serve_stage(runtime, request, args, port);
        }
        let api_key = self
            .shared
            .lock()
            .map(|snapshot| snapshot.api_key.clone())
            .unwrap_or_default();
        let mut config = json!({
            "exe": runtime.join("engine/strata").display().to_string(),
            "args": args,
            "cwd": self.base.join("run").display().to_string(),
            "tokenizer": pack.join("tokenizer").display().to_string(),
            "model_name": request.model_label,
            "log": self.base.join("run/strata-engine.log").display().to_string(),
            "lib_dirs": [],
            "port": port,
            "host": "127.0.0.1",
            "api_key": api_key,
            "open_browser": false,
        });
        // Several GPUs share the model (or this stage's layers): the bridge adds the layer split for a list.
        // The cards are numbered as the backend's runtime numbers them (HIP_VISIBLE_DEVICES for AMD).
        match request.backend_gpus() {
            [] => {}
            [first] => config["gpu"] = json!(first),
            many => config["gpu"] = json!(many),
        }
        config["gpus_asked"] = json!(!request.backend_gpus().is_empty());
        if request.backend() == runtime::Backend::Hip {
            config["backend"] = json!("hip");
            // the ROCm libraries the HIP runtime carries (a POM node has none)
            config["lib_dirs"] = json!([runtime.join("lib").display().to_string()]);
        }
        if request.sessions > 1 {
            config["parallel"] = json!(request.sessions);
        }
        let mut env = serde_json::Map::new();
        if let Some(token) = request.link.as_ref().and_then(|link| link.token.as_ref()) {
            env.insert("STRATA_STAGE_TOKEN".into(), json!(token));
        }
        // The POM's micro-pipelining: as many prompt pieces ahead as its relay keeps in flight.
        if let Some(count) = request.link.as_ref().and_then(|link| link.max_inflight) {
            env.insert("STRATA_STAGE_INFLIGHT".into(), json!(count.to_string()));
        }
        if !env.is_empty() {
            config["env"] = Value::Object(env);
        }
        // Applied by the bridge with Strata's own rules (serve/runconfig.py).
        if !request.settings.is_empty() {
            config["pom_settings"] = Value::Object(request.settings.clone());
        }
        let config_path = self.base.join("run/strata.json");
        if let Err(error) = write_private(
            &config_path,
            &serde_json::to_vec_pretty(&config).unwrap_or_default(),
        ) {
            return self.fail(format!("cannot write the Strata config: {error}"));
        }
        let mut command = match self.python_command(&runtime) {
            Ok(command) => command,
            Err(error) => return self.fail(error),
        };
        command
            .arg(runtime.join("pom_strata_server.py"))
            .arg("--engine")
            .arg("strata")
            .arg("--config")
            .arg(&config_path)
            .arg("--port")
            .arg(port.to_string());
        if let Some(dir) = &request.session_dir {
            command.env("POM_STRATA_SESSIONS", dir);
        }
        self.append_log(&format!(
            "[pom] starting Strata for {} on 127.0.0.1:{port}",
            request.model_label
        ));
        match Group::spawn(&mut command) {
            Ok(mut group) => {
                group.forward_output(
                    &self
                        .shared
                        .lock()
                        .map(|snapshot| snapshot.log.clone())
                        .unwrap_or_default(),
                );
                self.update(|snapshot| {
                    snapshot.port = Some(port);
                    snapshot.request = Some(request.clone());
                });
                self.set_phase(
                    Phase::Loading,
                    Some(format!("loading {}", request.model_label)),
                );
                self.work = Work::Serving { group };
            }
            Err(error) => self.fail(format!("Strata did not start: {error}")),
        }
    }

    /// An earlier stage of a split model: the engine alone, answering the
    /// later stage's machine (through the POM's relay) on a loopback port. No
    /// server, no head, no requests of its own.
    fn serve_stage(
        &mut self,
        runtime: PathBuf,
        request: LoadRequest,
        args: Vec<String>,
        port: u16,
    ) {
        let Some(token) = self
            .shared
            .lock()
            .ok()
            .and_then(|snapshot| snapshot.link_token.clone())
        else {
            return self.fail("the earlier stage has no link secret".into());
        };
        let mut command = Command::new(runtime.join("engine/strata"));
        command
            .arg("--serve")
            .arg("--stage-serve")
            .arg(format!("127.0.0.1:{port}"))
            .args(&args)
            .env("STRATA_STAGE_TOKEN", &token)
            .current_dir(self.base.join("run"))
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        // A middle stage: the earlier stage's secret, and how many prompt pieces it asks of it at once.
        if let Some(link) = request.link.as_ref() {
            if let Some(prev) = &link.token {
                command.env("STRATA_STAGE_PREV_TOKEN", prev);
            }
            if let Some(count) = link.max_inflight {
                command.env("STRATA_STAGE_INFLIGHT", count.to_string());
            }
        }
        // This node's GPUs for the stage: several share its layers (the engine places them, --layer-split auto).
        let gpus = request.backend_gpus();
        if !gpus.is_empty() {
            let list: Vec<String> = gpus.iter().map(u32::to_string).collect();
            if request.backend() == runtime::Backend::Hip {
                command.env("HIP_VISIBLE_DEVICES", list.join(","));
            } else {
                // numbered as nvidia-smi and the node number them, not fastest first
                command
                    .env("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
                    .env("CUDA_VISIBLE_DEVICES", list.join(","));
            }
            if gpus.len() > 1 {
                command.args(["--layer-split", "auto"]);
            }
        }
        if request.backend() == runtime::Backend::Hip {
            let lib = runtime.join("lib");
            let path = match std::env::var_os("LD_LIBRARY_PATH") {
                Some(old) if !old.is_empty() => {
                    format!("{}:{}", lib.display(), old.to_string_lossy())
                }
                _ => lib.display().to_string(),
            };
            command.env("LD_LIBRARY_PATH", path);
        }
        self.append_log(&format!(
            "[pom] starting Strata as an earlier stage of {} on 127.0.0.1:{port}",
            request.model_label
        ));
        match Group::spawn(&mut command) {
            Ok(mut group) => {
                group.forward_output(
                    &self
                        .shared
                        .lock()
                        .map(|snapshot| snapshot.log.clone())
                        .unwrap_or_default(),
                );
                self.update(|snapshot| {
                    snapshot.link_port = Some(port);
                    snapshot.request = Some(request.clone());
                });
                self.set_phase(
                    Phase::Loading,
                    Some(format!("loading {} (earlier stage)", request.model_label)),
                );
                self.work = Work::Staging { group };
            }
            Err(error) => self.fail(format!("Strata did not start: {error}")),
        }
    }

    /// Whether the stage server on `port` has said it listens.
    fn stage_listening(&self, port: u16) -> bool {
        let Ok(path) = self.shared.lock().map(|snapshot| snapshot.log.clone()) else {
            return false;
        };
        let Ok(text) = std::fs::read(path) else {
            return false;
        };
        let start = text.len().saturating_sub(256 * 1024);
        String::from_utf8_lossy(&text[start..]).contains(&format!("listening on 127.0.0.1:{port} "))
    }

    fn poll_work(&mut self) {
        match &mut self.work {
            Work::None => {}
            Work::Staging { group } => {
                if let Some(status) = group.exited() {
                    self.work = Work::None;
                    self.update(|snapshot| snapshot.link_port = None);
                    let reason = self
                        .last_log_line()
                        .map(|line| format!(": {line}"))
                        .unwrap_or_default();
                    return self.fail(format!("Strata stopped ({status}){reason}"));
                }
                let (phase, port) = match self.shared.lock() {
                    Ok(snapshot) => (snapshot.phase, snapshot.link_port),
                    Err(_) => return,
                };
                if phase != Phase::Ready && port.is_some_and(|port| self.stage_listening(port)) {
                    self.append_log("[pom] Strata's earlier stage is ready");
                    self.set_phase(Phase::Ready, None);
                }
            }
            Work::Packing { group, request } => {
                let Some(status) = group.exited() else {
                    return;
                };
                let request = request.clone();
                self.work = Work::None;
                let pack = self.pack_dir(&request.model_path);
                if !status.success() || !pack_ready(&pack) {
                    return self.fail(format!(
                        "preparing the model for Strata failed ({status}); see the Strata log"
                    ));
                }
                let (Some(runtime), Ok(model)) =
                    (self.runtime(), gguf::inspect(&request.model_path))
                else {
                    return self.fail("the model or the runtime went away".into());
                };
                self.serve(runtime, request, &model, pack);
            }
            Work::Serving { group, .. } => {
                if let Some(status) = group.exited() {
                    self.work = Work::None;
                    self.update(|snapshot| snapshot.port = None);
                    let reason = self
                        .last_log_line()
                        .map(|line| format!(": {line}"))
                        .unwrap_or_default();
                    return self.fail(format!("Strata stopped ({status}){reason}"));
                }
                let (port, key) = match self.shared.lock() {
                    Ok(snapshot) => (snapshot.port, snapshot.api_key.clone()),
                    Err(_) => return,
                };
                let Some(port) = port else { return };
                let status = ureq::get(&format!("http://127.0.0.1:{port}/pom/status"))
                    .set("Authorization", &format!("Bearer {key}"))
                    .timeout(Duration::from_secs(2))
                    .call()
                    .ok()
                    .and_then(|response| response.into_json::<Value>().ok());
                let Some(status) = status else { return };
                let loaded = status["loaded"].as_bool().unwrap_or(false);
                let context = status["max_context"].as_u64().map(|context| context as u32);
                let busy = status["busy"].as_bool().unwrap_or(false);
                let in_flight = status["in_flight"].as_u64().unwrap_or(0) as u32;
                self.update(|snapshot| {
                    snapshot.busy = busy;
                    snapshot.in_flight = in_flight;
                    if context.is_some_and(|context| context > 0) {
                        snapshot.context = context;
                    }
                });
                let phase = self
                    .shared
                    .lock()
                    .map(|snapshot| snapshot.phase)
                    .unwrap_or(Phase::Idle);
                if loaded && phase != Phase::Ready {
                    self.append_log("[pom] Strata is ready");
                    self.set_phase(Phase::Ready, None);
                }
            }
        }
    }
}

fn copy_lines(stream: Box<dyn std::io::Read + Send>, log: &Path) {
    use std::io::{BufRead, Write};
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(log)
        .ok();
    let reader = std::io::BufReader::new(stream);
    for line in reader.split(b'\n').map_while(Result::ok) {
        let line = String::from_utf8_lossy(&line);
        let line = line.trim_end_matches('\r');
        if let Some(file) = file.as_mut() {
            let _ = writeln!(file, "{line}");
        }
        // One write per line, so lines of the two streams never interleave.
        let _ = std::io::stderr()
            .lock()
            .write_all(format!("[strata] {}\n", line.trim_start_matches("[strata] ")).as_bytes());
    }
}

/// The pack tool writes `native_experts.txt` and the tokenizer last.
fn pack_ready(pack: &Path) -> bool {
    pack.join("native_experts.txt").is_file()
        && pack.join("tokenizer/vocab.json").is_file()
        && pack.join("tokenizer/chat_template.jinja").is_file()
}

fn free_port() -> Result<u16, String> {
    let listener =
        std::net::TcpListener::bind(("127.0.0.1", 0)).map_err(|error| error.to_string())?;
    listener
        .local_addr()
        .map(|address| address.port())
        .map_err(|error| error.to_string())
}

/// The config carries the API key: only this user reads it.
fn write_private(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    use std::io::Write;
    let mut options = std::fs::OpenOptions::new();
    options.create(true).write(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    options.open(path)?.write_all(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_load_request_is_read_with_safe_defaults() {
        let request = LoadRequest::from_json(&json!({
            "model_path": "/models/Q-IQ3_XXS-00001-of-00002.gguf",
            "model_label": "qwen3.8-flash-next",
            "n_ctx": 65536,
            "gpus": [0, 1, 99],
            "max_sessions": 40,
            "kv": "q4_0"
        }))
        .unwrap();
        assert_eq!(request.context, 65536);
        assert_eq!(request.gpus, vec![0, 1]);
        assert_eq!(request.sessions, 8);
        assert_eq!(request.kv, "q4_0");
        let plain =
            LoadRequest::from_json(&json!({"model_path": "/m/x.gguf", "kv": "f16"})).unwrap();
        assert_eq!(plain.model_label, "x");
        assert_eq!(plain.context, DEFAULT_CONTEXT);
        assert_eq!(plain.kv, "int8");
        assert!(LoadRequest::from_json(&json!({"model_path": "relative.gguf"})).is_err());
        assert!(LoadRequest::from_json(&json!({})).is_err());
    }

    #[test]
    fn the_upstream_is_only_given_while_ready() {
        let mut snapshot = Snapshot {
            phase: Phase::Loading,
            detail: None,
            runtime: None,
            runtime_hip: None,
            runtime_version: None,
            runtime_error: None,
            request: None,
            context: None,
            port: Some(1234),
            link_port: None,
            link_token: None,
            api_key: "k".into(),
            busy: false,
            in_flight: 0,
            since: Instant::now(),
            log: PathBuf::from("/dev/null"),
        };
        assert_eq!(snapshot.upstream_json()["status"], "starting");
        snapshot.phase = Phase::Ready;
        assert_eq!(snapshot.upstream_json()["port"], 1234);
        assert_eq!(snapshot.status_json()["earlier_stage"], false);
        assert!(snapshot.status_json().to_string().find("\"k\"").is_none());
        assert_eq!(snapshot.service()["kind"], "inference");
        assert_eq!(snapshot.service()["status"], "waiting");
        assert_eq!(snapshot.service()["stages"], true);
        assert_eq!(snapshot.service()["pom_settings"][1], "hidden_state_dtype");
        snapshot.port = None;
        snapshot.link_port = Some(4321);
        assert_eq!(snapshot.upstream_json()["status"], "error");
        // An earlier stage has no Strata app for the screen to open.
        assert_eq!(snapshot.status_json()["earlier_stage"], true);
        assert_eq!(snapshot.link_json()["port"], 4321);
        assert!(snapshot.link_json().get("token").is_none());
        snapshot.phase = Phase::Loading;
        snapshot.link_token = Some("s".repeat(32));
        assert_eq!(snapshot.link_json()["status"], "starting");
        assert_eq!(snapshot.link_json()["token"], "s".repeat(32));
    }

    #[test]
    fn a_node_with_only_amd_cards_runs_the_hip_runtime() {
        let amd = |ordinal: u32| json!({"vendor": "amd", "ordinal": ordinal, "index": ordinal + 1});
        let nvidia = json!({"vendor": "nvidia", "ordinal": 0, "index": 0});
        let both = LoadRequest::from_json(&json!({
            "model_path": "/m/x.gguf", "gpus": [0], "gpu_devices": [nvidia, amd(0)]
        }))
        .unwrap();
        assert_eq!(both.backend(), runtime::Backend::Cuda);
        assert_eq!(both.backend_gpus(), &[0]);
        let radeon = LoadRequest::from_json(&json!({
            "model_path": "/m/x.gguf", "gpus": [], "gpu_devices": [amd(0), amd(1)]
        }))
        .unwrap();
        assert_eq!(radeon.backend(), runtime::Backend::Hip);
        assert_eq!(radeon.backend_gpus(), &[0, 1]);
        let none = LoadRequest::from_json(&json!({"model_path": "/m/x.gguf"})).unwrap();
        assert_eq!(none.backend(), runtime::Backend::Cuda);
    }

    #[test]
    fn a_stage_link_is_read_and_checked() {
        let token = "t".repeat(32);
        let request = LoadRequest::from_json(&json!({
            "model_path": "/m/tail.gguf",
            "link": {"token": token, "prev_port": 4100, "max_inflight": 4}
        }))
        .unwrap();
        let link = request.link.unwrap();
        assert_eq!(link.prev_port, Some(4100));
        assert_eq!(link.max_inflight, Some(4));
        assert_eq!(link.token.as_deref(), Some(token.as_str()));
        let first = LoadRequest::from_json(&json!({"model_path": "/m/head.gguf", "link": {}}))
            .unwrap()
            .link
            .unwrap();
        assert_eq!((first.token, first.prev_port), (None, None));
        for bad in [
            json!({"token": "short"}),
            json!({"token": token, "prev_port": 0}),
            json!({"token": token, "prev_port": 4100, "max_inflight": 0}),
        ] {
            assert!(
                LoadRequest::from_json(&json!({"model_path": "/m/x.gguf", "link": bad})).is_err()
            );
        }
        assert!(LoadRequest::from_json(&json!({"model_path": "/m/x.gguf"}))
            .unwrap()
            .link
            .is_none());
    }
}
