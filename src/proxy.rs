//! Serves Strata's own web app and monitor inside the POM.
//!
//! The POM mounts a plugin's web server under
//! `/api/ui/plugins/strata/proxy/` (`ui.upstream`). This is that server, on
//! loopback. Every request must carry the token published in `ui.upstream`;
//! it is forwarded to Strata's server with Strata's API key added, so the
//! browser never learns the key. The browser's `Authorization`, cookies,
//! `Origin` and `Referer` never reach Strata.
//!
//! Strata's main page uses relative paths and works under the prefix as it is.
//! Its monitor page and scripts use a few absolute ones (`/web/...`,
//! `/v1/status`, `/load`): HTML pages get those quoted `/web/` paths rebased
//! and load `/_pom/base.js` first, which puts the prefix back on same-origin
//! `fetch`, `XMLHttpRequest` and `EventSource` calls that left without it.
//!
//! Strata's app (the web shell with chats, tasks and the agent, `src/app.rs`)
//! is served under `/app/`: its page names the prefix in `__STRATA_HOST__`
//! and loads `/_pom/app.js`, which moves every absolute same-origin address
//! it asks for (calls, pictures, frames, links) under `<prefix>/app`.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

const TOKEN_HEADER: &str = "x-pom-plugin-token";
const PREFIX_HEADER: &str = "x-forwarded-prefix";
const MAX_HEAD_BYTES: usize = 64 * 1024;
const BASE_SCRIPT_PATH: &str = "/_pom/base.js";

const BASE_SCRIPT: &str = r#"(function () {
  var prefix = "__PREFIX__";
  function rebase(url) {
    try {
      var parsed = new URL(url, window.location.href);
      if (parsed.origin !== window.location.origin) return url;
      if (parsed.pathname === prefix || parsed.pathname.indexOf(prefix + "/") === 0) return url;
      parsed.pathname = prefix + parsed.pathname;
      return parsed.href;
    } catch (error) {
      return url;
    }
  }
  var nativeFetch = window.fetch;
  window.fetch = function (input, init) {
    if (typeof input === "string" || input instanceof URL) {
      input = rebase(String(input));
    } else if (input instanceof Request) {
      var target = rebase(input.url);
      if (target !== input.url) input = new Request(target, input);
    }
    return nativeFetch.call(this, input, init);
  };
  var nativeOpen = XMLHttpRequest.prototype.open;
  XMLHttpRequest.prototype.open = function (method, url) {
    var args = Array.prototype.slice.call(arguments);
    args[1] = rebase(String(url));
    return nativeOpen.apply(this, args);
  };
  if (window.EventSource) {
    var NativeEventSource = window.EventSource;
    window.EventSource = function (url, options) {
      return new NativeEventSource(rebase(String(url)), options);
    };
    window.EventSource.prototype = NativeEventSource.prototype;
  }
})();
"#;

const APP_SCRIPT_PATH: &str = "/_pom/app.js";

/// The base script for the app's page, plus the addresses the page writes into
/// elements (`src`, `href`) and `window.open`: the app uses absolute paths
/// (`/api/files/raw?...`, `/brand/...`) that must stay under its prefix.
const APP_SCRIPT_EXTRA: &str = r#"(function () {
  var prefix = "__PREFIX__";
  function rebase(url) {
    try {
      var parsed = new URL(url, window.location.href);
      if (parsed.origin !== window.location.origin) return url;
      if (parsed.pathname === prefix || parsed.pathname.indexOf(prefix + "/") === 0) return url;
      parsed.pathname = prefix + parsed.pathname;
      return parsed.href;
    } catch (error) {
      return url;
    }
  }
  function absolute(value) {
    return typeof value === "string" && value.charAt(0) === "/" && value.charAt(1) !== "/";
  }
  function fix(element) {
    if (!element || element.nodeType !== 1) return;
    ["src", "href", "poster"].forEach(function (name) {
      var value = element.getAttribute && element.getAttribute(name);
      if (absolute(value)) {
        var target = rebase(value);
        if (target !== value) element.setAttribute(name, target);
      }
    });
  }
  new MutationObserver(function (records) {
    records.forEach(function (record) {
      if (record.type === "attributes") fix(record.target);
      else record.addedNodes.forEach(function (node) {
        fix(node);
        if (node.querySelectorAll) node.querySelectorAll("[src],[href],[poster]").forEach(fix);
      });
    });
  }).observe(document.documentElement, {subtree: true, childList: true, attributes: true, attributeFilter: ["src", "href", "poster"]});
  var nativeWindowOpen = window.open;
  window.open = function (url) {
    var args = Array.prototype.slice.call(arguments);
    if (typeof url === "string") args[0] = rebase(url);
    return nativeWindowOpen.apply(window, args);
  };
})();
"#;

/// Where Strata answers right now, read on every request (the port changes
/// when another model is loaded).
pub type Upstream = Arc<dyn Fn() -> Option<(u16, String)> + Send + Sync>;

/// Where Strata's app answers right now (`src/app.rs`), when it runs.
pub type AppUpstream = Arc<dyn Fn() -> Option<u16> + Send + Sync>;

pub struct UiServer {
    pub port: u16,
    pub token: String,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl UiServer {
    pub fn start(upstream: Upstream, app: AppUpstream, token: String) -> Result<Self, String> {
        let listener = TcpListener::bind(("127.0.0.1", 0)).map_err(|error| error.to_string())?;
        listener
            .set_nonblocking(true)
            .map_err(|error| error.to_string())?;
        let port = listener
            .local_addr()
            .map_err(|error| error.to_string())?
            .port();
        let stop = Arc::new(AtomicBool::new(false));
        let thread_stop = Arc::clone(&stop);
        let thread_token = token.clone();
        let thread = thread::Builder::new()
            .name("pom-plugin-strata-ui".into())
            .spawn(move || serve(listener, upstream, app, thread_token, thread_stop))
            .map_err(|error| error.to_string())?;
        Ok(Self {
            port,
            token,
            stop,
            thread: Some(thread),
        })
    }
}

impl Drop for UiServer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

fn serve(
    listener: TcpListener,
    upstream: Upstream,
    app: AppUpstream,
    token: String,
    stop: Arc<AtomicBool>,
) {
    let token = Arc::new(token);
    while !stop.load(Ordering::Acquire) {
        match listener.accept() {
            Ok((stream, _)) => {
                let upstream = Arc::clone(&upstream);
                let app = Arc::clone(&app);
                let token = Arc::clone(&token);
                let _ = thread::Builder::new()
                    .name("pom-plugin-strata-ui-request".into())
                    .spawn(move || {
                        let _ = handle(stream, &upstream, &app, &token);
                    });
            }
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                thread::sleep(Duration::from_millis(15));
            }
            Err(_) => break,
        }
    }
}

struct Request {
    method: String,
    target: String,
    headers: Vec<(String, String)>,
}

impl Request {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(key, _)| key.eq_ignore_ascii_case(name))
            .map(|(_, value)| value.as_str())
    }

    fn path(&self) -> &str {
        self.target.split(['?', '#']).next().unwrap_or_default()
    }
}

fn read_request(reader: &mut BufReader<TcpStream>) -> Option<Request> {
    let mut head = 0usize;
    let mut line = String::new();
    reader.read_line(&mut line).ok()?;
    head += line.len();
    let mut parts = line.split_whitespace();
    let method = parts.next()?.to_owned();
    let target = parts.next()?.to_owned();
    if !target.starts_with('/') {
        return None;
    }
    let mut headers = Vec::new();
    loop {
        let mut line = String::new();
        let read = reader.read_line(&mut line).ok()?;
        head += read;
        if read == 0 || head > MAX_HEAD_BYTES {
            return None;
        }
        let line = line.trim_end_matches(['\r', '\n']);
        if line.is_empty() {
            break;
        }
        let (name, value) = line.split_once(':')?;
        headers.push((name.trim().to_owned(), value.trim().to_owned()));
    }
    Some(Request {
        method,
        target,
        headers,
    })
}

fn respond(
    stream: &mut TcpStream,
    status: &str,
    content_type: &str,
    body: &[u8],
) -> std::io::Result<()> {
    write!(
        stream,
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n",
        body.len()
    )?;
    stream.write_all(body)
}

/// The prefix the POM mounted this server under, when it is one a URL can
/// safely carry: an absolute path of plain segments, without a trailing slash.
pub fn safe_prefix(prefix: &str) -> Option<&str> {
    let prefix = prefix.trim_end_matches('/');
    let plain = prefix.starts_with('/')
        && prefix.len() <= 512
        && prefix.split('/').skip(1).all(|segment| {
            !segment.is_empty()
                && segment != "."
                && segment != ".."
                && segment
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
        });
    plain.then_some(prefix)
}

/// Rebases the quoted absolute `/web/` paths of a page and loads the base
/// script before anything else.
pub fn rewrite_page(html: &str, prefix: &str) -> String {
    let rebased = html
        .replace("=\"/web/", &format!("=\"{prefix}/web/"))
        .replace("='/web/", &format!("='{prefix}/web/"));
    let tag = format!("<script src=\"{prefix}{BASE_SCRIPT_PATH}\"></script>");
    match rebased.find("<head>") {
        Some(head) => {
            let at = head + "<head>".len();
            format!("{}{tag}{}", &rebased[..at], &rebased[at..])
        }
        None => format!("{tag}{rebased}"),
    }
}

/// The app's page: it names its prefix (`__STRATA_HOST__`, read by the app's
/// API client) and loads the app script before anything else.
pub fn rewrite_app_page(html: &str, app_prefix: &str) -> String {
    let tag = format!(
        "<script>window.__STRATA_HOST__=\"{app_prefix}\";</script><script src=\"{app_prefix}{APP_SCRIPT_PATH}\"></script>"
    );
    match html.find("<head>") {
        Some(head) => {
            let at = head + "<head>".len();
            format!("{}{tag}{}", &html[..at], &html[at..])
        }
        None => format!("{tag}{html}"),
    }
}

/// How `forward` treats an HTML answer under a prefix.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Pages {
    /// Strata's own pages (`rewrite_page`).
    Strata,
    /// The app's page (`rewrite_app_page`).
    App,
}

/// Headers that describe one connection, carry the browser's credentials or
/// identity, or that this server sets itself.
fn is_dropped_header(name: &str) -> bool {
    [
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
        "host",
        "content-length",
        "accept-encoding",
        "authorization",
        "x-api-key",
        "cookie",
        "origin",
        "referer",
        TOKEN_HEADER,
    ]
    .iter()
    .any(|dropped| name.eq_ignore_ascii_case(dropped))
        || name.to_ascii_lowercase().starts_with("x-forwarded-")
        || name.to_ascii_lowercase().starts_with("sec-fetch-")
}

fn handle(
    stream: TcpStream,
    upstream: &Upstream,
    app: &AppUpstream,
    token: &str,
) -> std::io::Result<()> {
    stream.set_nonblocking(false)?;
    stream.set_read_timeout(Some(Duration::from_secs(30)))?;
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut stream = stream;
    let Some(request) = read_request(&mut reader) else {
        return respond(&mut stream, "400 Bad Request", "text/plain", b"bad request");
    };
    if request.header(TOKEN_HEADER) != Some(token) {
        return respond(
            &mut stream,
            "401 Unauthorized",
            "text/plain",
            b"unauthorized",
        );
    }
    let prefix = request
        .header(PREFIX_HEADER)
        .and_then(safe_prefix)
        .unwrap_or_default()
        .to_owned();
    if request.path() == BASE_SCRIPT_PATH {
        let script = BASE_SCRIPT.replace("__PREFIX__", &prefix);
        return respond(
            &mut stream,
            "200 OK",
            "text/javascript; charset=utf-8",
            script.as_bytes(),
        );
    }
    if request.path() == "/app" {
        let location = format!("{prefix}/app/");
        return write!(
            stream,
            "HTTP/1.1 308 Permanent Redirect\r\nLocation: {location}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
        );
    }
    if let Some(rest) = request.target.strip_prefix("/app/") {
        let app_prefix = format!("{prefix}/app");
        if request.path() == format!("/app{APP_SCRIPT_PATH}") {
            let script = format!(
                "{}\n{}",
                BASE_SCRIPT.replace("__PREFIX__", &app_prefix),
                APP_SCRIPT_EXTRA.replace("__PREFIX__", &app_prefix)
            );
            return respond(
                &mut stream,
                "200 OK",
                "text/javascript; charset=utf-8",
                script.as_bytes(),
            );
        }
        let Some(port) = app() else {
            let page = "<!doctype html><meta charset=utf-8><body style=\"font-family:sans-serif;padding:2rem\">\
                        <h3>Strata's app is not running</h3><p>It starts with the plugin once the Strata runtime \
                        and the agent are installed on this machine; the plugin screen's status says how far it is.</p></body>";
            return respond(
                &mut stream,
                "503 Service Unavailable",
                "text/html; charset=utf-8",
                page.as_bytes(),
            );
        };
        let inner = Request {
            method: request.method.clone(),
            target: format!("/{rest}"),
            headers: request.headers.clone(),
        };
        return forward(
            &mut stream,
            &mut reader,
            &inner,
            port,
            None,
            &app_prefix,
            Pages::App,
        );
    }
    let Some((port, key)) = upstream() else {
        let page = "<!doctype html><meta charset=utf-8><body style=\"font-family:sans-serif;padding:2rem\">\
                    <h3>Strata is not running</h3><p>Deploy a model on this machine with the Strata \
                    inference service, then reload this page.</p></body>";
        return respond(
            &mut stream,
            "503 Service Unavailable",
            "text/html; charset=utf-8",
            page.as_bytes(),
        );
    };
    forward(
        &mut stream,
        &mut reader,
        &request,
        port,
        Some(&key),
        &prefix,
        Pages::Strata,
    )
}

fn forward(
    stream: &mut TcpStream,
    reader: &mut BufReader<TcpStream>,
    request: &Request,
    port: u16,
    key: Option<&str>,
    prefix: &str,
    pages: Pages,
) -> std::io::Result<()> {
    if request.header("transfer-encoding").is_some() {
        return respond(
            stream,
            "411 Length Required",
            "text/plain",
            b"length required",
        );
    }
    let length = request
        .header("content-length")
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(0);
    let agent = ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(3))
        .timeout_read(Duration::from_secs(900))
        .redirects(0)
        .build();
    let url = format!("http://127.0.0.1:{port}{}", request.target);
    let mut call = agent.request(&request.method, &url);
    if let Some(key) = key {
        call = call.set("Authorization", &format!("Bearer {key}"));
    }
    for (name, value) in &request.headers {
        if !is_dropped_header(name) {
            call = call.set(name, value);
        }
    }
    let result = if length > 0 {
        call.set("Content-Length", &length.to_string())
            .send(reader.take(length))
    } else {
        call.call()
    };
    let response = match result {
        Ok(response) | Err(ureq::Error::Status(_, response)) => response,
        Err(_) => {
            return respond(
                stream,
                "502 Bad Gateway",
                "application/json",
                br#"{"error":"Strata did not answer"}"#,
            )
        }
    };
    let is_html = response
        .header("content-type")
        .is_some_and(|value| value.starts_with("text/html"));
    if is_html && !prefix.is_empty() {
        let status = format!("{} {}", response.status(), response.status_text());
        let mut body = String::new();
        response
            .into_reader()
            .take(8 << 20)
            .read_to_string(&mut body)?;
        return respond(
            stream,
            &status,
            "text/html; charset=utf-8",
            if pages == Pages::App {
                rewrite_app_page(&body, prefix)
            } else {
                rewrite_page(&body, prefix)
            }
            .as_bytes(),
        );
    }
    write!(
        stream,
        "HTTP/1.1 {} {}\r\n",
        response.status(),
        response.status_text()
    )?;
    for name in response.headers_names() {
        if is_dropped_header(&name) || name.eq_ignore_ascii_case("content-encoding") {
            continue;
        }
        if let Some(value) = response.header(&name) {
            write!(stream, "{name}: {value}\r\n")?;
        }
    }
    // The body ends when the connection closes: a stream (the chat's SSE)
    // passes through without being held in memory.
    stream.write_all(b"Connection: close\r\n\r\n")?;
    if request.method != "HEAD" {
        let mut body = response.into_reader();
        let mut buffer = [0u8; 16 * 1024];
        loop {
            let read = body.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            stream.write_all(&buffer[..read])?;
            stream.flush()?;
        }
    }
    stream.flush()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pages_load_the_base_script_first_and_web_paths_move_under_the_prefix() {
        let prefix = "/api/ui/plugins/strata/proxy";
        let page = rewrite_page(
            r#"<html><head><link href="/web/tokens.css"><script src="/web/monitor.js"></script></head></html>"#,
            prefix,
        );
        assert!(page.starts_with(
            r#"<html><head><script src="/api/ui/plugins/strata/proxy/_pom/base.js"></script>"#
        ));
        assert!(page.contains(r#"href="/api/ui/plugins/strata/proxy/web/tokens.css""#));
        assert!(page.contains(r#"src="/api/ui/plugins/strata/proxy/web/monitor.js""#));
        let relative = rewrite_page(r#"<head><link href="web/app.css"></head>"#, prefix);
        assert!(relative.contains(r#"href="web/app.css""#));
    }

    #[test]
    fn the_app_page_names_its_prefix_and_loads_the_app_script_first() {
        let page = rewrite_app_page(
            r#"<html><head><script src="./assets/index.js"></script></head></html>"#,
            "/api/ui/plugins/strata/proxy/app",
        );
        assert!(page.starts_with(
            r#"<html><head><script>window.__STRATA_HOST__="/api/ui/plugins/strata/proxy/app";</script><script src="/api/ui/plugins/strata/proxy/app/_pom/app.js"></script>"#
        ));
        assert!(page.contains(r#"src="./assets/index.js""#));
    }

    #[test]
    fn app_paths_go_to_the_app_without_strata_key() {
        let app = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let app_port = app.local_addr().unwrap().port();
        let seen = thread::spawn(move || {
            let (stream, _) = app.accept().unwrap();
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let request = read_request(&mut reader).unwrap();
            let mut stream = stream;
            respond(
                &mut stream,
                "200 OK",
                "text/html",
                b"<html><head></head></html>",
            )
            .unwrap();
            (
                request.target.clone(),
                request.header("authorization").map(str::to_owned),
            )
        });
        let server = UiServer::start(
            Arc::new(|| Some((1, "strata-key".into()))),
            Arc::new(move || Some(app_port)),
            "token-123".into(),
        )
        .unwrap();
        let mut stream = TcpStream::connect(("127.0.0.1", server.port)).unwrap();
        write!(
            stream,
            "GET /app/api/health?x=1 HTTP/1.1\r\nHost: x\r\n{TOKEN_HEADER}: token-123\r\n{PREFIX_HEADER}: /p\r\nAuthorization: Bearer browser\r\n\r\n"
        )
        .unwrap();
        let mut answer = String::new();
        stream.read_to_string(&mut answer).unwrap();
        let (target, authorization) = seen.join().unwrap();
        assert_eq!(target, "/api/health?x=1");
        assert_eq!(authorization, None);
        assert!(answer.starts_with("HTTP/1.1 200"));
        assert!(answer.contains(r#"window.__STRATA_HOST__="/p/app""#));
        let mut redirect = TcpStream::connect(("127.0.0.1", server.port)).unwrap();
        write!(redirect, "GET /app HTTP/1.1\r\nHost: x\r\n{TOKEN_HEADER}: token-123\r\n{PREFIX_HEADER}: /p\r\n\r\n").unwrap();
        let mut answer = String::new();
        redirect.read_to_string(&mut answer).unwrap();
        assert!(answer.starts_with("HTTP/1.1 308") && answer.contains("Location: /p/app/"));
    }

    #[test]
    fn the_browser_never_sends_credentials_or_its_origin_to_strata() {
        for name in [
            "Authorization",
            "x-api-key",
            "Cookie",
            "Origin",
            "Referer",
            "x-pom-plugin-token",
            "X-Forwarded-Prefix",
            "Sec-Fetch-Site",
            "Host",
        ] {
            assert!(is_dropped_header(name), "{name}");
        }
        for name in ["Content-Type", "Accept"] {
            assert!(!is_dropped_header(name), "{name}");
        }
    }

    #[test]
    fn only_a_plain_absolute_prefix_is_used() {
        assert_eq!(
            safe_prefix("/api/ui/plugins/strata/proxy/"),
            Some("/api/ui/plugins/strata/proxy")
        );
        for bad in ["", "/", "relative", "/a/../b", "/a//b", "/a\"b"] {
            assert_eq!(safe_prefix(bad), None, "{bad}");
        }
    }

    #[test]
    fn requests_need_the_token_and_strata_down_is_a_readable_page() {
        let server = UiServer::start(
            Arc::new(|| None),
            Arc::new(|| None),
            "secret-token-123".into(),
        )
        .unwrap();
        let ask = |token: Option<&str>| {
            let mut stream = TcpStream::connect(("127.0.0.1", server.port)).unwrap();
            let token = token
                .map(|token| format!("{TOKEN_HEADER}: {token}\r\n"))
                .unwrap_or_default();
            write!(stream, "GET / HTTP/1.1\r\nHost: x\r\n{token}\r\n").unwrap();
            let mut answer = String::new();
            stream.read_to_string(&mut answer).unwrap();
            answer
        };
        assert!(ask(None).starts_with("HTTP/1.1 401"));
        assert!(ask(Some("wrong")).starts_with("HTTP/1.1 401"));
        let down = ask(Some("secret-token-123"));
        assert!(down.starts_with("HTTP/1.1 503"));
        assert!(down.contains("Strata is not running"));
    }
}
