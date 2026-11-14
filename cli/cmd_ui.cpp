// `hearth ui`: the graphical app. A
// local web server on 127.0.0.1 serves one page; Microsoft Edge shows it as
// an app window (or the default browser opens it). This console window is
// the app's engine room: closing it quits.
#include <windows.h>
//
#include <objbase.h>
#include <shellapi.h>

#include <cstdio>
#include <filesystem>
#include <format>
#include <string>

#include "commands.h"
#include "core/error.h"
#include "core/log.h"
#include "core/platform.h"
#include "ui/command_line.h"
#include "ui/server.h"
#include "ui_app.h"

namespace hearth::cli {
namespace {

std::filesystem::path exe_dir() {
  wchar_t buf[MAX_PATH] = {};
  const DWORD n = GetModuleFileNameW(nullptr, buf, MAX_PATH);
  if (n == 0 || n == MAX_PATH) fail(ErrorKind::kIo, "cannot find the folder of hearth.exe");
  return std::filesystem::path(buf).parent_path();
}

// Microsoft Edge in app mode (a window without tabs or an address bar), if
// it is installed where Windows puts it; otherwise the default browser.
void open_window(const std::string& url) {
  const std::wstring wurl = path_from_utf8(url).wstring();
  for (const wchar_t* var : {L"ProgramFiles(x86)", L"ProgramFiles"}) {
    wchar_t base[MAX_PATH] = {};
    const DWORD n = GetEnvironmentVariableW(var, base, MAX_PATH);
    if (n == 0 || n >= MAX_PATH) continue;
    const std::filesystem::path edge =
        std::filesystem::path(base) / L"Microsoft" / L"Edge" / L"Application" / L"msedge.exe";
    std::error_code ec;
    if (!std::filesystem::is_regular_file(edge, ec)) continue;
    std::wstring cmd = ui::quote_argument(edge.wstring()) + L" " + ui::quote_argument(L"--app=" + wurl);
    STARTUPINFOW si{};
    si.cb = sizeof si;
    PROCESS_INFORMATION pi{};
    if (CreateProcessW(edge.c_str(), cmd.data(), nullptr, nullptr, FALSE, 0, nullptr, nullptr, &si, &pi)) {
      CloseHandle(pi.hThread);
      CloseHandle(pi.hProcess);
      return;
    }
  }
  const HRESULT co = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE);
  const auto r =
      reinterpret_cast<INT_PTR>(ShellExecuteW(nullptr, L"open", wurl.c_str(), nullptr, nullptr, SW_SHOWNORMAL));
  if (SUCCEEDED(co)) CoUninitialize();
  if (r <= 32) log_warn("ui: could not open a browser window; open this address yourself: {}", url);
}

}  // namespace

int cmd_ui(Args& args) {
  const auto port = static_cast<std::uint16_t>(args.int_or("port", 0, 0, 65535));
  const bool no_browser = args.flag("no-browser");
  UiOptions o;
  o.threads = static_cast<int>(args.int_or("threads", 4, 1, 64));
  o.ctx = static_cast<std::uint32_t>(args.int_or("ctx", 4096, 512, 131072));
  o.idle_minutes = static_cast<int>(args.int_or("idle-minutes", 10, 0, 24 * 60));
  o.draft = !args.flag("no-draft");
  const auto results = args.value("results");
  const auto url_file = args.value("url-file");
  args.finish();
  o.model_dirs = default_model_dirs();
  o.results_file = results ? path_from_utf8(*results) : exe_dir() / L"hearth-results.jsonl";
  o.models_file = exe_dir() / L"hearth-models.txt";

  ui::ServerOptions so;
  so.port = port;
  so.token = ui::random_token();
  UiApp app(o);
  ui::Server server(so, [&app](const ui::Request& r, ui::Connection& c) { app.handle(r, c); });
  const std::string url = std::format("http://127.0.0.1:{}/?t={}", server.port(), so.token);
  if (url_file) {
    // For scripts (scripts/ui-check.ps1): the address, written once the server listens.
    std::FILE* f = nullptr;
    if (_wfopen_s(&f, path_from_utf8(*url_file).c_str(), L"wb") != 0 || f == nullptr) {
      fail(ErrorKind::kIo, "cannot write {}", *url_file);
    }
    std::fprintf(f, "%s\n", url.c_str());
    std::fclose(f);
  }
  std::printf(
      "Hearth is running.\n\n"
      "  The app opens in its own window. If it does not, open this address in a browser:\n"
      "  %s\n\n"
      "  Keep this window open while you use the app. Close it (or press Ctrl+C) to quit.\n"
      "  The address only works on this PC, and only with the key at its end.\n\n",
      url.c_str());
  std::fflush(stdout);
  if (!no_browser) open_window(url);
  server.run([&app] { return app.keep_running(); });
  log_info("ui: closed");
  return 0;
}

}  // namespace hearth::cli
