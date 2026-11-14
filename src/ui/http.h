// The HTTP/1.1 request parser of the local app server.
//
// Input is untrusted: any program on this PC can connect to 127.0.0.1. The
// parser is strict (CRLF line ends, token characters in methods and header
// names, one Content-Length, no Transfer-Encoding) and bounded (Limits), and
// it is fuzzed (tests/fuzz/fuzz_http.cpp). It never allocates more than the
// limits allow, whatever the input claims.
#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace hearth::ui {

using Params = std::vector<std::pair<std::string, std::string>>;

struct Request {
  std::string method;  // "GET", "POST", ... (token characters)
  std::string path;    // the target up to '?', e.g. "/api/status" (not percent-decoded; routes are ASCII)
  Params query;        // the decoded query parameters, in order
  Params headers;      // names lowercased, values with surrounding spaces removed
  std::string body;

  // The first header or query parameter with this (lowercase) name, or null.
  const std::string* header(std::string_view name) const;
  const std::string* query_param(std::string_view name) const;
};

struct Limits {
  std::size_t max_head = 16 * 1024;    // request line + headers + blank line
  std::size_t max_body = 1024 * 1024;  // Content-Length
  std::size_t max_headers = 100;
};

enum class ParseStatus {
  kIncomplete,  // valid so far; more bytes are needed
  kComplete,    // a whole request; `consumed` bytes of the input
  kBad,         // answer with `http_status` and close
};

struct ParseResult {
  ParseStatus status = ParseStatus::kIncomplete;
  int http_status = 0;  // kBad: 400, 405, 413, 414, 431, 501 or 505
  std::string reason;   // kBad: why, for the log
  std::size_t consumed = 0;
};

// Parses one request from the start of `data` into `out`. Call again with
// more data while kIncomplete. kComplete fills every field of `out`.
ParseResult parse_request(std::string_view data, const Limits& limits, Request& out);

// Percent-decoding (and '+' as a space when `plus_is_space`). False if an
// escape is malformed or the result is not valid UTF-8.
bool percent_decode(std::string_view in, bool plus_is_space, std::string& out);

// application/x-www-form-urlencoded (also a URL's query): "a=1&b=x%20y".
// False if any part fails percent_decode. Empty parts are skipped.
bool form_decode(std::string_view text, Params& out);

// The value of `name` in decoded form parameters, or nullopt.
std::optional<std::string> param(const Params& params, std::string_view name);

// Equal, in time that depends only on the lengths (for the access token).
bool constant_time_equal(std::string_view a, std::string_view b) noexcept;

// "OK", "Bad Request", ... for the status codes this server sends.
const char* status_text(int code) noexcept;

}  // namespace hearth::ui
