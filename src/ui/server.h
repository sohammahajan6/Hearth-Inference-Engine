// The local app server: HTTP/1.1 on
// 127.0.0.1 only, one request per connection, each connection on its own
// thread. Every request passes these checks before the handler sees it:
//   - it parses (ui/http.h) within the limits;
//   - Host is 127.0.0.1:<port> or localhost:<port>, and Origin, if sent, is
//     the same origin (DNS rebinding, cross-site requests);
//   - "/" carries the access token as ?t=..., every other path as the
//     X-Hearth-Token header.
// A failed check is answered here (400/403/404/421/431/...) and logged at
// debug level; the handler never runs.
#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>

#include "ui/http.h"

namespace hearth::ui {

// The client side of one request, for the handler. Every send either writes
// all bytes or marks the connection dead (the client went away).
class Connection {
 public:
  virtual ~Connection() = default;
  // A whole response with Content-Length.
  virtual bool respond(int status, std::string_view content_type, std::string_view body) = 0;
  // A response whose body follows in send() calls and ends when the
  // connection closes (text/event-stream for progress and streamed text).
  virtual bool begin_stream(std::string_view content_type) = 0;
  virtual bool send(std::string_view bytes) = 0;
  // False once a send failed.
  virtual bool alive() const = 0;
};

// One server-sent event: "event: <name>\ndata: <json>\n\n". `json` must be one line.
std::string sse_event(std::string_view name, std::string_view json);

struct ServerOptions {
  std::uint16_t port = 0;  // 0: any free port
  std::string token;       // required on every request (see above)
  Limits limits;
  int read_timeout_ms = 10000;  // for the whole request to arrive
  int max_connections = 16;     // more at once get 503
};

using Handler = std::function<void(const Request&, Connection&)>;

class Server {
 public:
  // Binds 127.0.0.1 and listens. Throws hearth::Error(kIo) if it cannot.
  Server(ServerOptions options, Handler handler);
  ~Server();
  Server(const Server&) = delete;
  Server& operator=(const Server&) = delete;

  std::uint16_t port() const noexcept;
  // Accepts connections until stop() is called or `keep_running` returns
  // false (checked every 200 ms), then waits for the open connections to end.
  void run(const std::function<bool()>& keep_running);
  void stop() noexcept;  // any thread

  // Requests refused by the checks above, by kind (for tests and the log).
  struct Refused {
    std::atomic<int> parse{0}, host{0}, origin{0}, token{0}, busy{0};
  };
  const Refused& refused() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// A random token of 32 hex characters (128 bits) from the system's
// cryptographic generator.
std::string random_token();

}  // namespace hearth::ui
