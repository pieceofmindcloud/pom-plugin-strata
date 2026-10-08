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

/// Where Strata answers right now, read on every request (the port changes
/// when another model is loaded).
pub type Upstream = Arc<dyn Fn() -> Option<(u16, String)> + Send + Sync>;

pub struct UiServer {
    pub port: u16,
    pub token: String,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl UiServer {
    pub fn start(upstream: Upstream, token: String) -> Result<Self, String> {
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
            .spawn(move || serve(listener, upstream, thread_token, thread_stop))
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

fn serve(listener: TcpListener, upstream: Upstream, token: String, stop: Arc<AtomicBool>) {
    let token = Arc::new(token);
    while !stop.load(Ordering::Acquire) {
        match listener.accept() {
            Ok((stream, _)) => {
                let upstream = Arc::clone(&upstream);
                let token = Arc::clone(&token);
                let _ = thread::Builder::new()
                    .name("pom-plugin-strata-ui-request".into())
                    .spawn(move || {
                        let _ = handle(stream, &upstream, &token);
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

fn handle(stream: TcpStream, upstream: &Upstream, token: &str) -> std::io::Result<()> {
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
    forward(&mut stream, &mut reader, &request, port, &key, &prefix)
}

fn forward(
    stream: &mut TcpStream,
    reader: &mut BufReader<TcpStream>,
    request: &Request,
    port: u16,
    key: &str,
    prefix: &str,
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
    let mut call = agent
        .request(&request.method, &url)
        .set("Authorization", &format!("Bearer {key}"));
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
            rewrite_page(&body, prefix).as_bytes(),
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
        let server = UiServer::start(Arc::new(|| None), "secret-token-123".into()).unwrap();
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
