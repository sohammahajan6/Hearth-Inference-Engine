// src/ui/server: the security checks of ADR 0019 over real sockets on
// 127.0.0.1 (token, Host, Origin), the limits, streaming, the connection
// limit and the read timeout.
#include <doctest/doctest.h>

#include <winsock2.h>
#include <ws2tcpip.h>

#include <atomic>
#include <chrono>
#include <format>
#include <ostream>
#include <string>
#include <thread>

#include "ui/server.h"

using namespace hearth;

namespace {

constexpr const char* kToken = "0123456789abcdef0123456789abcdef";

struct Winsock {
  Winsock() {
    WSADATA wd{};
    REQUIRE(WSAStartup(MAKEWORD(2, 2), &wd) == 0);
  }
  ~Winsock() { WSACleanup(); }
  Winsock(const Winsock&) = delete;
  Winsock& operator=(const Winsock&) = delete;
};

SOCKET connect_to(std::uint16_t port) {
  SOCKET s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
  REQUIRE(s != INVALID_SOCKET);
  sockaddr_in a{};
  a.sin_family = AF_INET;
  a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  a.sin_port = htons(port);
  REQUIRE(connect(s, reinterpret_cast<const sockaddr*>(&a), sizeof a) == 0);
  const DWORD timeout = 10000;
  setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char*>(&timeout), sizeof timeout);
  return s;
}

std::string read_all(SOCKET s) {
  std::string out;
  char buf[4096];
  for (;;) {
    const int n = recv(s, buf, sizeof buf, 0);
    if (n <= 0) break;
    out.append(buf, static_cast<std::size_t>(n));
  }
  return out;
}

// Sends `raw` (in two parts, to exercise reassembly) and returns everything the server answers.
std::string round_trip(std::uint16_t port, const std::string& raw) {
  SOCKET s = connect_to(port);
  const std::size_t half = raw.size() / 2;
  send(s, raw.data(), static_cast<int>(half), 0);
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  send(s, raw.data() + half, static_cast<int>(raw.size() - half), 0);
  std::string out = read_all(s);
  closesocket(s);
  return out;
}

int status_of(const std::string& response) {
  if (response.size() < 12 || !response.starts_with("HTTP/1.1 ")) return 0;
  return std::stoi(response.substr(9, 3));
}

std::string body_of(const std::string& response) {
  const std::size_t i = response.find("\r\n\r\n");
  return i == std::string::npos ? std::string() : response.substr(i + 4);
}

// A server on a free port, run on its own thread for the test's scope.
struct Running {
  explicit Running(ui::ServerOptions o, ui::Handler h) : server(std::move(o), std::move(h)) {
    thread = std::thread([this] { server.run([this] { return !done.load(); }); });
  }
  ~Running() {
    done.store(true);
    thread.join();
  }
  Running(const Running&) = delete;
  Running& operator=(const Running&) = delete;
  std::uint16_t port() const { return server.port(); }
  ui::Server server;
  std::atomic<bool> done{false};
  std::thread thread;
};

ui::ServerOptions options() {
  ui::ServerOptions o;
  o.token = kToken;
  o.read_timeout_ms = 2000;
  return o;
}

}  // namespace

TEST_SUITE("quick") {
  TEST_CASE("ui server: only requests with the token, this PC's Host and our own Origin reach the handler") {
    const Winsock ws;
    std::atomic<int> handled{0};
    Running srv(options(), [&](const ui::Request& r, ui::Connection& c) {
      ++handled;
      if (r.path == "/stream") {
        CHECK(c.begin_stream("text/event-stream"));
        for (int i = 0; i < 3; ++i) CHECK(c.send(ui::sse_event("n", std::format("{{\"i\":{}}}", i))));
        return;
      }
      c.respond(200, "text/plain", r.method + " " + r.path + " " + r.body);
    });
    const std::uint16_t port = srv.port();
    const std::string host = std::format("Host: 127.0.0.1:{}\r\n", port);
    const std::string tok = std::string("X-Hearth-Token: ") + kToken + "\r\n";

    // The page: token in the query.
    std::string r = round_trip(port, std::format("GET /?t={} HTTP/1.1\r\n{}\r\n", kToken, host));
    CHECK(status_of(r) == 200);
    CHECK(body_of(r) == "GET / ");
    CHECK(r.find("Content-Security-Policy: default-src 'none';") != std::string::npos);
    CHECK(r.find("X-Content-Type-Options: nosniff") != std::string::npos);
    CHECK(r.find("Cache-Control: no-store") != std::string::npos);
    CHECK(status_of(round_trip(port, std::format("GET / HTTP/1.1\r\n{}\r\n", host))) == 403);
    CHECK(status_of(round_trip(port, std::format("GET /?t=wrong HTTP/1.1\r\n{}\r\n", host))) == 403);
    CHECK(status_of(round_trip(port, std::format("GET / HTTP/1.1\r\n{}{}\r\n", host, tok))) == 403);  // "/" wants ?t=

    // The API: token in the header, never in the query.
    CHECK(status_of(round_trip(port, std::format("GET /api/status HTTP/1.1\r\n{}{}\r\n", host, tok))) == 200);
    CHECK(status_of(round_trip(port, std::format("GET /api/status?t={} HTTP/1.1\r\n{}\r\n", kToken, host))) == 403);

    // Host: this PC only (DNS rebinding), localhost in any case.
    CHECK(status_of(round_trip(port, std::format("GET /api/x HTTP/1.1\r\nHost: evil.example\r\n{}\r\n", tok))) == 403);
    CHECK(status_of(round_trip(
              port, std::format("GET /api/x HTTP/1.1\r\nHost: 127.0.0.1:{}\r\n{}\r\n", port + 1, tok))) == 403);
    CHECK(status_of(round_trip(port, std::format("GET /api/x HTTP/1.1\r\nHost: LocalHost:{}\r\n{}\r\n", port, tok))) ==
          200);
    CHECK(status_of(round_trip(port, std::format("GET /api/x HTTP/1.0\r\n{}\r\n", tok))) == 403);  // no Host at all

    // Origin, when sent, must be this page's.
    CHECK(status_of(round_trip(port, std::format("POST /api/x HTTP/1.1\r\n{}{}Origin: http://evil.example\r\n"
                                                 "Content-Length: 0\r\n\r\n",
                                                 host, tok))) == 403);
    CHECK(status_of(round_trip(
              port, std::format("POST /api/x HTTP/1.1\r\n{}{}Origin: null\r\nContent-Length: 0\r\n\r\n", host, tok))) ==
          403);
    r = round_trip(port,
                   std::format("POST /api/x HTTP/1.1\r\n{}{}Origin: http://127.0.0.1:{}\r\nContent-Length: 5\r\n\r\n"
                               "a=b&c",
                               host, tok, port));
    CHECK(status_of(r) == 200);
    CHECK(body_of(r) == "POST /api/x a=b&c");

    // Parser refusals come back with their status.
    CHECK(status_of(round_trip(port, "GARBAGE\r\n\r\n")) == 400);
    CHECK(status_of(round_trip(
              port, std::format("POST /api/x HTTP/1.1\r\n{}Transfer-Encoding: chunked\r\n\r\n", host))) == 501);
    CHECK(status_of(round_trip(
              port, std::format("GET /api/x HTTP/1.1\r\n{}X: {}\r\n\r\n", host, std::string(20000, 'a')))) == 431);

    // A stream has no Content-Length and ends when the connection closes.
    r = round_trip(port, std::format("GET /stream HTTP/1.1\r\n{}{}\r\n", host, tok));
    CHECK(status_of(r) == 200);
    CHECK(r.find("Content-Length") == std::string::npos);
    CHECK(body_of(r) == "event: n\ndata: {\"i\":0}\n\nevent: n\ndata: {\"i\":1}\n\nevent: n\ndata: {\"i\":2}\n\n");

    CHECK(handled.load() == 5);  // the page, status, localhost, the POST and the stream; nothing refused
    const ui::Server::Refused& f = srv.server.refused();
    CHECK(f.token.load() == 4);
    CHECK(f.host.load() == 3);
    CHECK(f.origin.load() == 2);
    CHECK(f.parse.load() == 3);
  }

  TEST_CASE("ui server: a connection beyond the limit gets 503, and a silent client times out") {
    const Winsock ws;
    std::atomic<bool> release{false}, entered{false};
    ui::ServerOptions o = options();
    o.max_connections = 1;
    o.read_timeout_ms = 300;
    Running srv(o, [&](const ui::Request&, ui::Connection& c) {
      entered.store(true);
      while (!release.load()) std::this_thread::sleep_for(std::chrono::milliseconds(5));
      c.respond(200, "text/plain", "done");
    });
    const std::uint16_t port = srv.port();
    const std::string req =
        std::format("GET /api/x HTTP/1.1\r\nHost: 127.0.0.1:{}\r\nX-Hearth-Token: {}\r\n\r\n", port, kToken);
    SOCKET first = connect_to(port);
    send(first, req.data(), static_cast<int>(req.size()), 0);
    for (int i = 0; i < 400 && !entered.load(); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(5));
    REQUIRE(entered.load());
    CHECK(status_of(round_trip(port, req)) == 503);
    release.store(true);
    CHECK(body_of(read_all(first)) == "done");
    closesocket(first);
    CHECK(srv.server.refused().busy.load() == 1);

    // Half a request, then silence: the server gives up after its read timeout.
    // The first connection's slot frees once the server has reaped it (within its 200 ms loop).
    for (int i = 0; i < 100 && status_of(round_trip(port, req)) == 503; ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    SOCKET slow = connect_to(port);
    send(slow, "GET /api/x HTTP/1.1\r\n", 21, 0);
    const auto t0 = std::chrono::steady_clock::now();
    const std::string r = read_all(slow);
    const double waited = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    closesocket(slow);
    CHECK((r.empty() || status_of(r) == 408));
    CHECK(waited < 5.0);
  }

  TEST_CASE("ui server: a random token is 32 hex characters and differs every time") {
    const std::string a = ui::random_token(), b = ui::random_token();
    CHECK(a.size() == 32);
    CHECK(a.find_first_not_of("0123456789abcdef") == std::string::npos);
    CHECK(a != b);
  }
}
