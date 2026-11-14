// Test runner entry point.
#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "core/log.h"
#include "core/platform.h"

namespace {

// doctest's console reporter without the MESSAGE notes (measurements and
// "skipped" remarks that tests print when they pass). Failures, their
// CAPTURE/INFO context and the summary are unchanged. scripts/test.ps1 uses
// it unless -Details is given.
struct NoNotesReporter : doctest::ConsoleReporter {
  using doctest::ConsoleReporter::ConsoleReporter;
  void log_message(const doctest::MessageData& m) override {
    if (m.m_severity == doctest::assertType::is_warn) return;  // MESSAGE(); FAIL_CHECK/FAIL still print
    doctest::ConsoleReporter::log_message(m);
  }
};
DOCTEST_REGISTER_REPORTER("no-notes", 1, NoNotesReporter);

bool env_is_1(const char* name) {
  char* v = nullptr;
  std::size_t n = 0;
  const bool on = _dupenv_s(&v, &n, name) == 0 && v != nullptr && std::strcmp(v, "1") == 0;
  std::free(v);
  return on;
}

}  // namespace

int main(int argc, char** argv) {
  // Same unattended-safe process setup as the product: a crash or failed
  // assert must end the run with an exit code, never wait on a dialog.
  hearth::init_process();
  // Tests that make the code warn or report an error on purpose catch those
  // lines with hearth::LogCapture and check their text. Any other warning or
  // error fails the run below. HEARTH_TEST_SHOW_EXPECTED=1 also prints the
  // caught lines, marked "(expected)".
  hearth::LogCapture::set_echo(env_is_1("HEARTH_TEST_SHOW_EXPECTED"));
  doctest::Context context;
  context.applyCommandLine(argc, argv);
  int code = context.run();
  if (context.shouldExit()) return code;
  const unsigned long long problems = hearth::log_problem_count();
  if (problems != 0) {
    std::fprintf(stderr,
                 "error: %llu unexpected warning/error line(s) were logged during the tests (shown above). A test "
                 "that causes one on purpose must catch it with hearth::LogCapture and check its text.\n",
                 problems);
    if (code == 0) code = 1;
  }
  return code;
}
