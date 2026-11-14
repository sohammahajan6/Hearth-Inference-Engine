#include "ui/server.h"

// winsock2.h must come before windows.h (which core headers may include).
#include <winsock2.h>
#include <ws2tcpip.h>
// windows.h after winsock2.h, bcrypt.h after windows.h.
#include <windows.h>
//
#include <bcrypt.h>

#include <algorithm>
#include <chrono>
#include <exception>
#include <format>
#include <mutex>
#include <span>
#include <thread>
#include <vector>

#include "core/error.h"
#include "core/log.h"

namespace hearth::ui {
namespace {

constexpr std::string_view kSecurityHeaders =
    "Cache-Control: no-store\r\n"
    "X-Content-Type-Options: nosniff\r\n"
    "Referrer-Policy: no-referrer\r\n"
    "Content-Security-Policy: default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
    "img-src data:; connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'\r\n"
    "Connection: close\r\n";

std::string head_of(int status, std::string_view content_type, const std::size_t* length) {
  std::string h = std::format("HTTP/1.1 {} {}\r\nContent-Type: {}\r\n", status, status_text(status), content_type);
  if (length != nullptr) h += std::format("Content-Length: {}\r\n", *length);
  h += kSecurityHeaders;
  h += "\r\n";
  return h;
}

std::string json_error(std::string_view message) {
  std::string out = "{\"error\":\"";
  for (const char c : message) {
    if (c == '"' || c == '\\') out += '\\';
    if (static_cast<unsigned char>(c) < 0x20) continue;  // reasons are plain ASCII text
    out += c;
  }
  out += "\"}";
  return out;
}

class SocketConnection final : public Connection {
 public:
  explicit SocketConnection(SOCKET s) : s_(s) {}

  bool respond(int status, std::string_view content_type, std::string_view body) override {
    if (started_) return false;
    started_ = true;
    const std::size_t n = body.size();
    return write(head_of(status, content_type, &n)) && write(body);
  }
  bool begin_stream(std::string_view content_type) override {
    if (started_) return false;
    started_ = true;
    return write(head_of(200, content_type, nullptr));
  }
  bool send(std::string_view bytes) override { return started_ && write(bytes); }
  bool alive() const override { return alive_; }
  bool started() const noexcept { return started_; }

 private:
  bool write(std::string_view bytes) {
    std::span<const char> b(bytes.data(), bytes.size());
    while (alive_ && !b.empty()) {
      const int chunk = static_cast<int>(std::min<std::size_t>(b.size(), 1 << 20));
      const int sent = ::send(s_, b.data(), chunk, 0);
      if (sent <= 0) {
        alive_ = false;
        break;
      }
      b = b.subspan(static_cast<std::size_t>(sent));
    }
    return alive_;
  }

  SOCKET s_;
  bool started_ = false;
  bool alive_ = true;
};

// Ends a connection gracefully: no more sending, then read and discard what
// the client still sends (briefly, and at most 1 MiB) before closing. Closing
// with unread data makes Windows reset the connection, and the client could
// lose the response it has not read yet (a refused oversized request).
void close_gracefully(SOCKET s, DWORD drain_ms) {
  shutdown(s, SD_SEND);
  setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char*>(&drain_ms), sizeof drain_ms);
  char buf[4096];
  std::size_t total = 0;
  for (;;) {
    const int n = recv(s, buf, sizeof buf, 0);
    if (n <= 0) break;
    total += static_cast<std::size_t>(n);
    if (total > (1u << 20)) break;
  }
  closesocket(s);
}

bool ascii_iequal(std::string_view a, std::string_view b) noexcept {
  if (a.size() != b.size()) return false;
  for (std::size_t i = 0; i < a.size(); ++i) {
    char x = a[i], y = b[i];
    if (x >= 'A' && x <= 'Z') x = static_cast<char>(x - 'A' + 'a');
    if (y >= 'A' && y <= 'Z') y = static_cast<char>(y - 'A' + 'a');
    if (x != y) return false;
  }
  return true;
}

}  // namespace

std::string sse_event(std::string_view name, std::string_view json) {
  std::string out;
  out.reserve(name.size() + json.size() + 16);
  out += "event: ";
  out += name;
  out += "\ndata: ";
  out += json;
  out += "\n\n";
  return out;
}

std::string random_token() {
  unsigned char bytes[16] = {};
  if (BCryptGenRandom(nullptr, bytes, sizeof bytes, BCRYPT_USE_SYSTEM_PREFERRED_RNG) != 0) {
    fail(ErrorKind::kInternal, "BCryptGenRandom failed");
  }
  std::string t;
  for (const unsigned char b : bytes) t += std::format("{:02x}", b);
  return t;
}

struct Server::Impl {
  ServerOptions o;
  Handler handler;
  SOCKET listener = INVALID_SOCKET;
  std::uint16_t port = 0;
  std::atomic<bool> stopping{false};
  Refused refused;
  bool wsa = false;

  struct Worker {
    std::thread thread;
    std::shared_ptr<std::atomic<bool>> done;
  };
  std::vector<Worker> workers;

  // The request passed every check of server.h.
  bool allowed(const Request& r) {
    const std::string* host = r.header("host");
    const std::string a = std::format("127.0.0.1:{}", port), b = std::format("localhost:{}", port);
    if (host == nullptr || !(*host == a || ascii_iequal(*host, b))) {
      ++refused.host;
      return false;
    }
    if (const std::string* origin = r.header("origin")) {
      if (!(*origin == "http://" + a || ascii_iequal(*origin, "http://" + b))) {
        ++refused.origin;
        return false;
      }
    }
    const std::string* t = r.path == "/" ? r.query_param("t") : r.header("x-hearth-token");
    if (t == nullptr || !constant_time_equal(*t, o.token)) {
      ++refused.token;
      return false;
    }
    return true;
  }

  void serve(SOCKET s) {
    const DWORD rcv = static_cast<DWORD>(o.read_timeout_ms), snd = 30000;
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char*>(&rcv), sizeof rcv);
    setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, reinterpret_cast<const char*>(&snd), sizeof snd);
    SocketConnection c(s);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(o.read_timeout_ms);
    std::string buf;
    Request req;
    char chunk[8192];
    for (;;) {
      const ParseResult p = parse_request(buf, o.limits, req);
      if (p.status == ParseStatus::kComplete) break;
      if (p.status == ParseStatus::kBad) {
        ++refused.parse;
        log_debug("ui: refused a request: {} ({})", p.reason, p.http_status);
        c.respond(p.http_status, "application/json", json_error(p.reason));
        return;
      }
      if (std::chrono::steady_clock::now() > deadline) {
        c.respond(408, "application/json", json_error("the request did not arrive in time"));
        return;
      }
      const int n = recv(s, chunk, sizeof chunk, 0);
      if (n <= 0) return;  // closed, reset or timed out before a whole request
      buf.append(chunk, static_cast<std::size_t>(n));
    }
    if (!allowed(req)) {
      log_debug("ui: refused {} {} (host, origin or token)", req.method, req.path);
      c.respond(403, "application/json", json_error("forbidden"));
      return;
    }
    try {
      handler(req, c);
    } catch (const std::exception& e) {
      log_warn("ui: {} {} failed: {}", req.method, req.path, e.what());
      if (!c.started()) c.respond(500, "application/json", json_error(e.what()));
    }
  }

  void reap(bool all) {
    for (std::size_t i = 0; i < workers.size();) {
      if (all || workers[i].done->load()) {
        workers[i].thread.join();
        workers[i] = std::move(workers.back());
        workers.pop_back();
      } else {
        ++i;
      }
    }
  }
};

Server::Server(ServerOptions options, Handler handler) : impl_(std::make_unique<Impl>()) {
  Impl& m = *impl_;
  m.o = std::move(options);
  m.handler = std::move(handler);
  if (m.o.token.empty()) fail(ErrorKind::kInvalidArgument, "ui server: an access token is required");
  WSADATA wd{};
  if (WSAStartup(MAKEWORD(2, 2), &wd) != 0) fail(ErrorKind::kIo, "ui server: Winsock could not start");
  m.wsa = true;
  m.listener = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
  if (m.listener == INVALID_SOCKET) fail(ErrorKind::kIo, "ui server: socket() failed ({})", WSAGetLastError());
  const BOOL exclusive = TRUE;
  setsockopt(m.listener, SOL_SOCKET, SO_EXCLUSIVEADDRUSE, reinterpret_cast<const char*>(&exclusive), sizeof exclusive);
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);  // this PC only, never the network
  addr.sin_port = htons(m.o.port);
  if (bind(m.listener, reinterpret_cast<const sockaddr*>(&addr), sizeof addr) != 0) {
    const int err = WSAGetLastError();
    closesocket(m.listener);
    WSACleanup();
    fail(ErrorKind::kIo, "ui server: cannot listen on 127.0.0.1:{} (error {}{})", m.o.port, err,
         err == WSAEADDRINUSE ? ", the port is in use" : "");
  }
  int len = sizeof addr;
  getsockname(m.listener, reinterpret_cast<sockaddr*>(&addr), &len);
  m.port = ntohs(addr.sin_port);
  if (listen(m.listener, SOMAXCONN) != 0) {
    closesocket(m.listener);
    WSACleanup();
    fail(ErrorKind::kIo, "ui server: listen() failed");
  }
}

Server::~Server() {
  Impl& m = *impl_;
  m.reap(true);
  if (m.listener != INVALID_SOCKET) closesocket(m.listener);
  if (m.wsa) WSACleanup();
}

std::uint16_t Server::port() const noexcept { return impl_->port; }
void Server::stop() noexcept { impl_->stopping.store(true); }
const Server::Refused& Server::refused() const noexcept { return impl_->refused; }

void Server::run(const std::function<bool()>& keep_running) {
  Impl& m = *impl_;
  while (!m.stopping.load() && (!keep_running || keep_running())) {
    fd_set ready;
    FD_ZERO(&ready);
    FD_SET(m.listener, &ready);
    timeval tv{0, 200 * 1000};
    const int n = select(0, &ready, nullptr, nullptr, &tv);
    m.reap(false);
    if (n <= 0) continue;
    SOCKET c = accept(m.listener, nullptr, nullptr);
    if (c == INVALID_SOCKET) continue;
    if (static_cast<int>(m.workers.size()) >= m.o.max_connections) {
      ++m.refused.busy;
      SocketConnection busy(c);
      busy.respond(503, "application/json", json_error("too many connections"));
      close_gracefully(c, 100);
      continue;
    }
    auto done = std::make_shared<std::atomic<bool>>(false);
    m.workers.push_back({std::thread([&m, c, done] {
                           m.serve(c);
                           close_gracefully(c, 500);
                           done->store(true);
                         }),
                         done});
  }
  m.reap(true);  // the handlers end when the app stops their jobs
}

}  // namespace hearth::ui
