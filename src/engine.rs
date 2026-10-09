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
    /// Concurrent requests (Strata's `parallel` batch slots).
    pub sessions: u32,
    /// `int8`, `q4_0` or `k8v4`.
    pub kv: String,
    /// Engine settings the deployment chose (`crate::settings`).
    pub settings: serde_json::Map<String, Value>,
}

impl LoadRequest {
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
            sessions,
            kv,
            settings: crate::settings::select(&value["settings"]),
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
    pub runtime_version: Option<String>,
    pub runtime_error: Option<String>,
    pub request: Option<LoadRequest>,
    /// The context Strata reports once loaded.
    pub context: Option<u32>,
    pub port: Option<u16>,
    pub api_key: String,
    pub busy: bool,
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
            (Phase::Error, _) => json!({"status": "error", "error": self.detail}),
            _ => json!({"status": "starting"}),
        }
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
            runtime_version: None,
            runtime_error: None,
            request: None,
            context: None,
            port: None,
            api_key: random_key(),
            busy: false,
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
        let start = text.len().saturating_sub(64 * 1024);
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
    Packing { group: Group, request: LoadRequest },
    Serving { group: Group },
}

struct Manager {
    base: PathBuf,
    shared: Arc<Mutex<Snapshot>>,
    commands: Receiver<Command_>,
    work: Work,
    installer: Option<Receiver<Result<PathBuf, String>>>,
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
        if let Some(dir) = runtime::local_runtime() {
            let version = runtime::version_of(&dir);
            self.update(|snapshot| {
                snapshot.runtime = Some(dir);
                snapshot.runtime_version = version;
            });
            return;
        }
        if let Some(dir) = runtime::installed_dir(&self.base).filter(|dir| runtime::is_runtime(dir))
        {
            let version = runtime::version_of(&dir);
            self.update(|snapshot| {
                snapshot.runtime = Some(dir);
                snapshot.runtime_version = version;
            });
            return;
        }
        if !runtime::has_release() {
            self.update(|snapshot| {
                snapshot.runtime_error = Some(
                    "no Strata runtime: this is a development build without build/runtime \
                     (run scripts/build-runtime.sh in the plugin source)"
                        .into(),
                )
            });
            return;
        }
        let (sender, receiver) = channel();
        let base = self.base.clone();
        let shared = Arc::clone(&self.shared);
        self.set_phase(
            Phase::Installing,
            Some("downloading the Strata runtime".into()),
        );
        let _ = thread::Builder::new()
            .name("pom-plugin-strata-install".into())
            .spawn(move || {
                let result = runtime::install(&base, &|done, total| {
                    if let Ok(mut snapshot) = shared.lock() {
                        snapshot.detail = Some(format!(
                            "downloading the Strata runtime: {} of {} MB",
                            done >> 20,
                            total >> 20
                        ));
                    }
                });
                let _ = sender.send(result);
            });
        self.installer = Some(receiver);
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
        let Some(receiver) = &self.installer else {
            return;
        };
        let Ok(result) = receiver.try_recv() else {
            return;
        };
        self.installer = None;
        match result {
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
            Work::Packing { group, .. } | Work::Serving { group, .. } => {
                self.set_phase(Phase::Stopping, None);
                group.stop(STOP_GRACE);
            }
        }
        self.update(|snapshot| {
            snapshot.request = None;
            snapshot.context = None;
            snapshot.port = None;
            snapshot.busy = false;
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
        let model = match gguf::inspect(&request.model_path) {
            Ok(model) => model,
            Err(error) => return self.fail(error),
        };
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
        if model.shards.len() > 2 {
            // Unsloth's files keep small projections as Q8_0; the engine reads them as BF16.
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
        if let Some(mtp) = self.mtp_dir(&request.model_path) {
            args.extend(["--spec", "4", "--spec-min-p", "0.5", "--mtp"].map(str::to_owned));
            args.push(mtp.display().to_string());
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
        match request.gpus.as_slice() {
            [] => {}
            [one] => config["gpu"] = json!(one),
            many => config["gpu"] = json!(many),
        }
        config["gpus_asked"] = json!(!request.gpus.is_empty());
        if request.sessions > 1 {
            config["parallel"] = json!(request.sessions);
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

    fn poll_work(&mut self) {
        match &mut self.work {
            Work::None => {}
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
                self.update(|snapshot| {
                    snapshot.busy = busy;
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
            runtime_version: None,
            runtime_error: None,
            request: None,
            context: None,
            port: Some(1234),
            api_key: "k".into(),
            busy: false,
            since: Instant::now(),
            log: PathBuf::from("/dev/null"),
        };
        assert_eq!(snapshot.upstream_json()["status"], "starting");
        snapshot.phase = Phase::Ready;
        assert_eq!(snapshot.upstream_json()["port"], 1234);
        assert!(snapshot.status_json().to_string().find("\"k\"").is_none());
        assert_eq!(snapshot.service()["kind"], "inference");
        assert_eq!(snapshot.service()["status"], "waiting");
    }
}
