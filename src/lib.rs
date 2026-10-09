//! POM plugin that runs a whole model on the Strata engine.
//!
//! A node with this plugin offers the `inference` service "Strata" to the
//! cluster. When the operator deploys a Qwen3.8-Flash-Next model
//! (architecture `qwen4exp`) on this machine with the Strata inference
//! service, the POM node does not load llama.cpp: it asks this plugin to load
//! the model (`engine.load`) on the GPUs the deployment chose, and the POM
//! gateway sends each prompt to Strata's generation route through
//! `engine.upstream` and the node's authenticated plugin route. Undeploying
//! asks `engine.unload`, which stops Strata and gives the GPU back.
//!
//! The library only supervises. Strata's engine, server and pack tools live in
//! a runtime directory built by `scripts/build-runtime.sh` (`runtime`).

pub mod engine;
pub mod gguf;
pub mod proxy;
pub mod runtime;
pub mod settings;

use engine::{Engine, LoadRequest};
use proxy::UiServer;
use serde_json::{json, Value};
use std::ffi::{c_char, c_void};
use std::sync::Arc;

const ABI_VERSION: u32 = 1;
static VERSION: &[u8] = b"0.1.0\0";
const UI_MANIFEST: &str = include_str!("../ui/manifest.json");

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ByteSlice {
    pub ptr: *const u8,
    pub len: usize,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ByteBuffer {
    pub ptr: *mut u8,
    pub len: usize,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct HostCallbacks {
    pub log: Option<unsafe extern "C" fn(message: ByteSlice)>,
}

pub type PluginHandle = *mut c_void;
pub type CreateFn = unsafe extern "C" fn(HostCallbacks, ByteSlice) -> PluginHandle;
pub type IngestFn = unsafe extern "C" fn(PluginHandle, ByteSlice) -> i32;
pub type QueryFn = unsafe extern "C" fn(PluginHandle, ByteSlice) -> ByteBuffer;
pub type FreeBufferFn = unsafe extern "C" fn(ByteBuffer);
pub type ShutdownFn = unsafe extern "C" fn(PluginHandle);

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PluginApiV1 {
    pub abi_version: u32,
    pub plugin_version: *const c_char,
    pub capabilities: u64,
    pub create: Option<CreateFn>,
    pub ingest: Option<IngestFn>,
    pub query: Option<QueryFn>,
    pub free_buffer: Option<FreeBufferFn>,
    pub shutdown: Option<ShutdownFn>,
}

unsafe impl Sync for PluginApiV1 {}

struct PluginState {
    /// An error when the plugin has no directory to work in.
    engine: Result<Arc<Engine>, String>,
    ui: Option<UiServer>,
}

include!(concat!(env!("OUT_DIR"), "/ui_assets.rs"));

unsafe fn input_bytes<'a>(input: ByteSlice) -> Result<&'a [u8], String> {
    if input.len == 0 {
        return Ok(&[]);
    }
    if input.ptr.is_null() {
        return Err("null byte slice".into());
    }
    Ok(std::slice::from_raw_parts(input.ptr, input.len))
}

fn ui_asset(path: &str) -> Option<&'static (&'static str, &'static str, &'static [u8])> {
    UI_ASSETS.iter().find(|(name, _, _)| *name == path)
}

fn ui_manifest() -> Result<Value, String> {
    let manifest: Value = serde_json::from_str(UI_MANIFEST).map_err(|error| error.to_string())?;
    let assets = manifest["assets"]
        .as_array()
        .ok_or("manifest has no assets")?;
    for asset in assets.iter().filter_map(Value::as_str) {
        if ui_asset(asset).is_none() {
            return Err(format!("asset is not embedded: {asset}"));
        }
    }
    Ok(manifest)
}

fn encode_base64(bytes: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut output = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let value = (u32::from(chunk[0]) << 16)
            | (u32::from(*chunk.get(1).unwrap_or(&0)) << 8)
            | u32::from(*chunk.get(2).unwrap_or(&0));
        for index in 0..4 {
            if index <= chunk.len() {
                output.push(TABLE[((value >> (18 - 6 * index)) & 63) as usize] as char);
            } else {
                output.push('=');
            }
        }
    }
    output
}

fn engine(state: &PluginState) -> Result<&Engine, String> {
    state.engine.as_deref().map_err(Clone::clone)
}

fn query_inner(state: &PluginState, request: &[u8]) -> Result<Value, String> {
    let request: Value = serde_json::from_slice(request).map_err(|error| error.to_string())?;
    match request["operation"].as_str().unwrap_or_default() {
        // Nothing of the node context is needed: the model path comes with
        // the deployment.
        "host.configure" | "host.event" => Ok(json!({"status": "ok"})),
        // The inference service this machine offers, and the architectures
        // it runs, for the dashboard and the deploy screen.
        "host.services" => match &state.engine {
            Ok(engine) => Ok(json!({"services": [engine.snapshot().service()]})),
            Err(_) => Ok(json!({"services": []})),
        },
        // The engine operations come from this node's POM process only: it
        // calls them through the local plugin host when a deployment puts a
        // model on this machine or takes it away. They are not `peer.*` nor
        // `rpc.*`, so no other machine and no browser reaches them.
        "engine.load" => {
            let load = LoadRequest::from_json(&request["params"])?;
            engine(state)?.load(load)?;
            Ok(json!({"status": "accepted"}))
        }
        "engine.unload" => {
            engine(state)?.unload();
            Ok(json!({"status": "accepted"}))
        }
        "engine.status" => Ok(engine(state)?.snapshot().status_json()),
        "engine.upstream" => Ok(engine(state)?.snapshot().upstream_json()),
        "engine.link" => Ok(engine(state)?.snapshot().link_json()),
        // Where the POM plugin proxy finds Strata's web app and monitor.
        "ui.upstream" => match &state.ui {
            Some(ui) => Ok(json!({"status": "ready", "port": ui.port, "token": ui.token})),
            None => Ok(json!({"status": "error", "error": "the interface server did not start"})),
        },
        // The plugin screen. The log tail is Strata's own progress lines
        // (load, speed, cache) and never a prompt; it is for the administrator.
        "rpc.status" => {
            let engine = engine(state)?;
            let mut status = engine.snapshot().status_json();
            if request["profile"] == "admin" {
                status["log"] = json!(engine.log_tail(500));
            }
            Ok(status)
        }
        "ui.manifest" => ui_manifest(),
        "ui.asset" => {
            let path = request["path"].as_str().ok_or("asset path is missing")?;
            let (_, content_type, bytes) = ui_asset(path).ok_or("unknown asset")?;
            Ok(json!({"content_type": content_type, "base64": encode_base64(bytes)}))
        }
        _ => Err("unknown operation".into()),
    }
}

fn buffer_from_bytes(bytes: Vec<u8>) -> ByteBuffer {
    let mut bytes = bytes.into_boxed_slice();
    let buffer = ByteBuffer {
        ptr: bytes.as_mut_ptr(),
        len: bytes.len(),
    };
    std::mem::forget(bytes);
    buffer
}

fn ui_token() -> String {
    let mut bytes = [0u8; 24];
    getrandom::getrandom(&mut bytes).expect("system randomness");
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

unsafe extern "C" fn create(_: HostCallbacks, config: ByteSlice) -> PluginHandle {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let config = input_bytes(config)?;
        if !config.is_empty() {
            serde_json::from_slice::<Value>(config).map_err(|error| error.to_string())?;
        }
        let engine = engine::plugin_directory().map(|base| Arc::new(Engine::start(base)));
        let ui = engine.as_ref().ok().and_then(|engine| {
            let engine = Arc::clone(engine);
            let upstream: proxy::Upstream = Arc::new(move || {
                let snapshot = engine.snapshot();
                snapshot.port.map(|port| (port, snapshot.api_key.clone()))
            });
            UiServer::start(upstream, ui_token()).ok()
        });
        let state = PluginState { engine, ui };
        Ok::<_, String>(Box::into_raw(Box::new(state)).cast::<c_void>())
    }));
    match result {
        Ok(Ok(handle)) => handle,
        _ => std::ptr::null_mut(),
    }
}

unsafe extern "C" fn ingest(handle: PluginHandle, event: ByteSlice) -> i32 {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if handle.is_null() {
            return false;
        }
        input_bytes(event).is_ok()
    }));
    matches!(result, Ok(true)).then_some(0).unwrap_or(-1)
}

unsafe extern "C" fn query(handle: PluginHandle, request: ByteSlice) -> ByteBuffer {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if handle.is_null() {
            return Err("plugin handle is null".to_owned());
        }
        query_inner(&*handle.cast::<PluginState>(), input_bytes(request)?)
            .and_then(|value| serde_json::to_vec(&value).map_err(|error| error.to_string()))
    }));
    match result {
        Ok(Ok(bytes)) => buffer_from_bytes(bytes),
        _ => ByteBuffer {
            ptr: std::ptr::null_mut(),
            len: 0,
        },
    }
}

unsafe extern "C" fn free_buffer(buffer: ByteBuffer) {
    if !buffer.ptr.is_null() {
        let slice = std::ptr::slice_from_raw_parts_mut(buffer.ptr, buffer.len);
        drop(Box::from_raw(slice));
    }
}

unsafe extern "C" fn shutdown(handle: PluginHandle) {
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if !handle.is_null() {
            // The interface server first, then Strata and its manager thread.
            let mut state = Box::from_raw(handle.cast::<PluginState>());
            state.ui = None;
            drop(state);
        }
    }));
}

static API: PluginApiV1 = PluginApiV1 {
    abi_version: ABI_VERSION,
    plugin_version: VERSION.as_ptr().cast(),
    capabilities: 0,
    create: Some(create),
    ingest: Some(ingest),
    query: Some(query),
    free_buffer: Some(free_buffer),
    shutdown: Some(shutdown),
};

/// Entry point the POM plugin host resolves as `pom_<plugin_code>_plugin_v1`.
///
/// # Safety
///
/// The returned table is static; the host must call its functions only as the
/// v1 plugin ABI specifies.
#[no_mangle]
pub unsafe extern "C" fn pom_strata_plugin_v1() -> *const PluginApiV1 {
    &API
}

#[cfg(test)]
mod tests {
    use super::*;

    fn offline_state() -> PluginState {
        PluginState {
            engine: Err("no plugin directory".into()),
            ui: None,
        }
    }

    fn query_json(state: &PluginState, request: Value) -> Result<Value, String> {
        query_inner(state, &serde_json::to_vec(&request).unwrap())
    }

    #[test]
    fn entry_exposes_the_generic_v1_boundary() {
        let api = unsafe { &*pom_strata_plugin_v1() };
        assert_eq!(api.abi_version, ABI_VERSION);
        assert!(api.create.is_some() && api.query.is_some() && api.shutdown.is_some());
    }

    #[test]
    fn manifest_resolves_all_built_assets() {
        if ui_asset("ui/screens.js").is_some() {
            assert!(ui_manifest().is_ok());
        } else {
            assert!(ui_manifest().is_err());
        }
    }

    #[test]
    fn services_are_listed_even_without_an_engine_and_unknown_operations_fail() {
        let state = offline_state();
        assert_eq!(
            query_json(&state, json!({"operation": "host.services"})).unwrap(),
            json!({"services": []})
        );
        assert!(query_json(&state, json!({"operation": "host.configure"})).is_ok());
        assert!(query_json(&state, json!({"operation": "nope"})).is_err());
        assert!(query_json(&state, json!({"operation": "peer.load"})).is_err());
        assert!(query_json(&state, json!({"operation": "rpc.load"})).is_err());
    }

    #[test]
    fn a_load_without_a_model_is_refused_before_the_engine() {
        let state = offline_state();
        let refused = query_json(&state, json!({"operation": "engine.load", "params": {}}));
        assert!(refused.unwrap_err().contains("model_path"));
    }
}
