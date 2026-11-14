#include "ui/http.h"

#include <cstdint>

#include "core/utf8.h"

namespace hearth::ui {
namespace {

// RFC 9110 "tchar": the characters of a method or a header name.
bool is_tchar(char c) noexcept {
  if ((c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) return true;
  switch (c) {
    case '!':
    case '#':
    case '$':
    case '%':
    case '&':
    case '\'':
    case '*':
    case '+':
    case '-':
    case '.':
    case '^':
    case '_':
    case '`':
    case '|':
    case '~':
      return true;
    default:
      return false;
  }
}

// A request target: visible ASCII only (no spaces, no controls, no fragment).
bool valid_target(std::string_view t) noexcept {
  if (t.empty() || t.front() != '/') return false;
  for (const char c : t) {
    const auto u = static_cast<unsigned char>(c);
    if (u <= 0x20 || u >= 0x7f || c == '#') return false;
  }
  return true;
}

// A header value: visible characters, spaces and tabs (and obs-text bytes).
bool valid_field_value(std::string_view v) noexcept {
  for (const char c : v) {
    const auto u = static_cast<unsigned char>(c);
    if ((u < 0x20 && c != '\t') || u == 0x7f) return false;
  }
  return true;
}

std::string_view trim(std::string_view s) noexcept {
  while (!s.empty() && (s.front() == ' ' || s.front() == '\t')) s.remove_prefix(1);
  while (!s.empty() && (s.back() == ' ' || s.back() == '\t')) s.remove_suffix(1);
  return s;
}

char lower(char c) noexcept { return (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c; }

int hex_value(char c) noexcept {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}

ParseResult bad(int status, std::string reason) {
  ParseResult r;
  r.status = ParseStatus::kBad;
  r.http_status = status;
  r.reason = std::move(reason);
  return r;
}

const std::string* find(const Params& p, std::string_view name) {
  for (const auto& [k, v] : p) {
    if (k == name) return &v;
  }
  return nullptr;
}

}  // namespace

const std::string* Request::header(std::string_view name) const { return find(headers, name); }
const std::string* Request::query_param(std::string_view name) const { return find(query, name); }

ParseResult parse_request(std::string_view data, const Limits& limits, Request& out) {
  // The head ends at the first empty line. Until it is seen, only check that
  // it cannot grow past the limit.
  const std::size_t head_end = data.find("\r\n\r\n");
  if (head_end == std::string_view::npos) {
    if (data.size() > limits.max_head) return bad(431, "request head larger than the limit");
    return {};
  }
  if (head_end + 4 > limits.max_head) return bad(431, "request head larger than the limit");
  const std::string_view head = data.substr(0, head_end + 2);  // every line ends with CRLF

  Request r;
  std::size_t pos = 0;
  auto next_line = [&](std::string_view& line) {
    const std::size_t e = head.find("\r\n", pos);
    if (e == std::string_view::npos) return false;
    line = head.substr(pos, e - pos);
    pos = e + 2;
    return true;
  };
  std::string_view line;
  if (!next_line(line)) return bad(400, "no request line");
  if (line.find('\n') != std::string_view::npos || line.find('\r') != std::string_view::npos) {
    return bad(400, "bare CR or LF in the request line");
  }
  // METHOD SP target SP HTTP/x.y, single spaces.
  const std::size_t s1 = line.find(' ');
  if (s1 == std::string_view::npos || s1 == 0) return bad(400, "malformed request line");
  const std::size_t s2 = line.find(' ', s1 + 1);
  if (s2 == std::string_view::npos || s2 == s1 + 1 || line.find(' ', s2 + 1) != std::string_view::npos) {
    return bad(400, "malformed request line");
  }
  const std::string_view method = line.substr(0, s1);
  const std::string_view target = line.substr(s1 + 1, s2 - s1 - 1);
  const std::string_view version = line.substr(s2 + 1);
  for (const char c : method) {
    if (!is_tchar(c)) return bad(400, "invalid character in the method");
  }
  if (version != "HTTP/1.1" && version != "HTTP/1.0") {
    if (version.starts_with("HTTP/")) return bad(505, "only HTTP/1.0 and HTTP/1.1 are supported");
    return bad(400, "malformed HTTP version");
  }
  if (target.size() > 4096) return bad(414, "request target too long");
  if (!valid_target(target)) return bad(400, "invalid request target");
  r.method = std::string(method);
  const std::size_t q = target.find('?');
  r.path = std::string(target.substr(0, q));
  if (q != std::string_view::npos && !form_decode(target.substr(q + 1), r.query)) {
    return bad(400, "malformed query string");
  }

  std::optional<std::size_t> content_length;
  while (next_line(line)) {
    if (line.find('\r') != std::string_view::npos || line.find('\n') != std::string_view::npos) {
      return bad(400, "bare CR or LF in a header");
    }
    if (line.empty()) return bad(400, "empty header line");  // cannot happen: the head ends at the first one
    if (line.front() == ' ' || line.front() == '\t') return bad(400, "obsolete header line folding");
    if (r.headers.size() >= limits.max_headers) return bad(431, "too many headers");
    const std::size_t colon = line.find(':');
    if (colon == std::string_view::npos || colon == 0) return bad(400, "malformed header");
    const std::string_view name = line.substr(0, colon);
    std::string lname;
    lname.reserve(name.size());
    for (const char c : name) {
      if (!is_tchar(c)) return bad(400, "invalid character in a header name");
      lname += lower(c);
    }
    const std::string_view value = trim(line.substr(colon + 1));
    if (!valid_field_value(value)) return bad(400, "invalid character in a header value");
    if (lname == "transfer-encoding") return bad(501, "Transfer-Encoding is not supported");
    if (lname == "content-length") {
      if (content_length) return bad(400, "more than one Content-Length");
      if (value.empty() || value.size() > 12) return bad(400, "malformed Content-Length");
      std::size_t n = 0;
      for (const char c : value) {
        if (c < '0' || c > '9') return bad(400, "malformed Content-Length");
        n = n * 10 + static_cast<std::size_t>(c - '0');
      }
      if (n > limits.max_body) return bad(413, "request body larger than the limit");
      content_length = n;
    }
    r.headers.emplace_back(std::move(lname), std::string(value));
  }
  if (version == "HTTP/1.1" && r.header("host") == nullptr) return bad(400, "HTTP/1.1 request without Host");

  const std::size_t body_start = head_end + 4;
  const std::size_t body_size = content_length.value_or(0);
  if (data.size() - body_start < body_size) return {};  // the body is still arriving
  r.body = std::string(data.substr(body_start, body_size));
  out = std::move(r);
  ParseResult done;
  done.status = ParseStatus::kComplete;
  done.consumed = body_start + body_size;
  return done;
}

bool percent_decode(std::string_view in, bool plus_is_space, std::string& out) {
  out.clear();
  out.reserve(in.size());
  for (std::size_t i = 0; i < in.size(); ++i) {
    const char c = in[i];
    if (c == '%') {
      if (in.size() - i < 3) return false;  // "%" needs two hex digits after it
      const int hi = hex_value(in[i + 1]);
      const int lo = hex_value(in[i + 2]);
      if (hi < 0 || lo < 0) return false;
      out += static_cast<char>(static_cast<unsigned char>(hi * 16 + lo));
      i += 2;
    } else if (c == '+' && plus_is_space) {
      out += ' ';
    } else {
      out += c;
    }
  }
  return valid_utf8(out);
}

bool form_decode(std::string_view text, Params& out) {
  out.clear();
  while (!text.empty()) {
    const std::size_t amp = text.find('&');
    const std::string_view part = text.substr(0, amp);
    text = amp == std::string_view::npos ? std::string_view() : text.substr(amp + 1);
    if (part.empty()) continue;
    const std::size_t eq = part.find('=');
    std::string k, v;
    if (!percent_decode(part.substr(0, eq), true, k)) return false;
    if (eq != std::string_view::npos && !percent_decode(part.substr(eq + 1), true, v)) return false;
    out.emplace_back(std::move(k), std::move(v));
  }
  return true;
}

std::optional<std::string> param(const Params& params, std::string_view name) {
  if (const std::string* v = find(params, name)) return *v;
  return std::nullopt;
}

bool constant_time_equal(std::string_view a, std::string_view b) noexcept {
  if (a.size() != b.size()) return false;
  unsigned char diff = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    diff = static_cast<unsigned char>(diff | (static_cast<unsigned char>(a[i]) ^ static_cast<unsigned char>(b[i])));
  }
  return diff == 0;
}

const char* status_text(int code) noexcept {
  switch (code) {
    case 200:
      return "OK";
    case 400:
      return "Bad Request";
    case 403:
      return "Forbidden";
    case 404:
      return "Not Found";
    case 405:
      return "Method Not Allowed";
    case 408:
      return "Request Timeout";
    case 409:
      return "Conflict";
    case 413:
      return "Content Too Large";
    case 414:
      return "URI Too Long";
    case 421:
      return "Misdirected Request";
    case 431:
      return "Request Header Fields Too Large";
    case 500:
      return "Internal Server Error";
    case 501:
      return "Not Implemented";
    case 503:
      return "Service Unavailable";
    case 505:
      return "HTTP Version Not Supported";
    default:
      return "Unknown";
  }
}

}  // namespace hearth::ui
