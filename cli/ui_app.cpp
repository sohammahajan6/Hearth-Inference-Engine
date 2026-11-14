#include "ui_app.h"

#include <windows.h>

#include <shobjidl.h>

#include <algorithm>
#include <atomic>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <format>
#include <fstream>
#include <mutex>
#include <optional>
#include <random>
#include <span>
#include <string_view>
#include <thread>
#include <utility>

#include "api/engine_internal.h"
#include "core/cpu_features.h"
#include "core/error.h"
#include "core/json_writer.h"
#include "core/log.h"
#include "core/platform.h"
#include "core/utf8.h"
#include "gguf/types.h"
#include "gpu/context.h"
#include "hearth/engine.h"
#include "hearth/version.h"
#include "model/llama.h"
#include "model/llama_ref.h"
#include "tokenizer/chat.h"
#include "tokenizer/tokenizer.h"

namespace hearth::cli {
namespace {

using Clock = std::chrono::steady_clock;

std::filesystem::path exe_path() {
  std::wstring buf(MAX_PATH, L'\0');
  for (;;) {
    const DWORD n = GetModuleFileNameW(nullptr, buf.data(), static_cast<DWORD>(buf.size()));
    if (n == 0) fail(ErrorKind::kIo, "cannot find the path of hearth.exe");
    if (n < buf.size()) {
      buf.resize(n);
      return std::filesystem::path(buf);
    }
    buf.resize(buf.size() * 2);
  }
}

double gib(unsigned long long bytes) { return static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0); }

// The power source as the page reads it in /api/status: exactly "ac",
// "battery" or "unknown". Not to_string(), which writes "AC" for people and
// for the benchmark records; the page compares case-sensitively, and "AC"
// once made it show the battery warning on a machine that was plugged in.
const char* power_api_name(PowerSource s) noexcept {
  switch (s) {
    case PowerSource::kAc:
      return "ac";
    case PowerSource::kBattery:
      return "battery";
    case PowerSource::kUnknown:
      return "unknown";
  }
  return "unknown";
}

std::string json_message(std::string_view key, std::string_view message) {
  JsonWriter w(false);
  w.begin_object();
  w.field(key, message);
  w.end_object();
  return w.str();
}

// Answers with {"error": message} and the status.
void respond_error(ui::Connection& c, int status, std::string_view message) {
  c.respond(status, "application/json; charset=utf-8", json_message("error", message));
}

// What kind of weights a file holds: "q8_0" or "q4_0" (every backend),
// "k_quant" (any other quantized format: the `cpu` backend only; Q4_K/Q6_K
// with their own kernels, ADR 0021) or "full" (F32/F16/BF16, only the slow
// reference path).
std::string weight_kind(const gguf::File& f) {
  for (const gguf::TensorInfo& t : f.tensors()) {
    if (t.name == "blk.0.attn_q.weight") {
      const auto q8 = static_cast<std::uint32_t>(gguf::GgmlType::kQ8_0);
      const auto q4 = static_cast<std::uint32_t>(gguf::GgmlType::kQ4_0);
      const auto f32 = static_cast<std::uint32_t>(gguf::GgmlType::kF32);
      const auto f16 = static_cast<std::uint32_t>(gguf::GgmlType::kF16);
      const auto bf16 = static_cast<std::uint32_t>(gguf::GgmlType::kBF16);
      return t.type == q8                                       ? "q8_0"
             : t.type == q4                                     ? "q4_0"
             : t.type == f32 || t.type == f16 || t.type == bf16 ? "full"
                                                                : "k_quant";
    }
  }
  return "full";
}

// Whether `--backend gpu` can run the file: every matrix it multiplies with
// (the layers' and the LM head) is Q8_0, Q4_0, or a format converted to
// Q8_0 at load (Q4_1, Q5_0, Q5_1, Q2_K, Q3_K, Q5_K; ADR 0028). Full
// precision, Q4_K and Q6_K run on the CPU only. The embeddings are looked
// up on the CPU, so their type only counts when they are the LM head.
bool gpu_readable(const gguf::File& f) {
  const bool tied = f.find_tensor("output.weight") == nullptr;
  for (const gguf::TensorInfo& t : f.tensors()) {
    if (t.n_dims < 2 || (!tied && t.name == "token_embd.weight")) continue;
    switch (static_cast<gguf::GgmlType>(t.type)) {
      case gguf::GgmlType::kF32:
      case gguf::GgmlType::kF16:
      case gguf::GgmlType::kBF16:
      case gguf::GgmlType::kQ4_K:
      case gguf::GgmlType::kQ6_K:
        return false;
      default:
        break;
    }
  }
  return true;
}

}  // namespace

std::filesystem::path find_draft(const std::filesystem::path& model) {
  std::error_code ec;
  const auto model_size = std::filesystem::file_size(model, ec);
  if (ec) return {};
  std::filesystem::path best;
  std::uintmax_t best_size = 0;
  std::vector<std::string> vocab;  // the model's, read on first need
  for (const auto& e : std::filesystem::directory_iterator(model.parent_path(), ec)) {
    if (!e.is_regular_file(ec)) continue;
    std::wstring ext = e.path().extension().wstring();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](wchar_t ch) { return static_cast<wchar_t>(towlower(ch)); });
    if (ext != L".gguf") continue;
    const std::uintmax_t size = e.file_size(ec);
    if (ec || size > model_size / 10 * 3 || (!best.empty() && size >= best_size)) continue;
    try {
      const gguf::File f = gguf::File::open(e.path());
      if (weight_kind(f) != "q4_0") continue;
      if (vocab.empty()) {  // copied: the views point into the mapped file
        const gguf::File m = gguf::File::open(model);
        const auto& v = m.get_array("tokenizer.ggml.tokens", gguf::ValueType::kString).strings;
        vocab.assign(v.begin(), v.end());
      }
      if (!std::ranges::equal(f.get_array("tokenizer.ggml.tokens", gguf::ValueType::kString).strings, vocab)) continue;
    } catch (const Error&) {
      continue;  // not a usable model file
    }
    best = e.path();
    best_size = size;
  }
  return best;
}

namespace {

// Whether a file can run here, from its header alone (the tensor data is
// never read): the same checks loading does, so "ready" means it loads.
struct ModelCheck {
  std::string status;  // "ok" (CPU or GPU), "cpu" (CPU only) or "unsupported"
  std::string reason;  // why not, or what limits it, in plain words
  std::string arch;    // general.architecture, if readable
  std::string types;   // the matrices' number formats, e.g. "Q4_K + Q6_K"
};

// The engine's messages, in words for someone who did not write it.
std::string plain_reason(std::string_view why, std::string_view arch) {
  const auto has = [&](std::string_view s) { return why.find(s) != std::string_view::npos; };
  if (has("model architecture")) {
    return std::format("Model family \"{}\" is not supported yet (supported: Llama 3 and Qwen2.5 families).", arch);
  }
  if (has("tokenizer model 'llama'")) {
    return "Uses the SentencePiece tokenizer (Mistral, Llama 2 and similar models): not supported yet.";
  }
  if (has("tokenizer model") || has("pre-tokenizer")) {
    return "Uses a tokenizer this version does not support yet (" + std::string(why) + ").";
  }
  if (has("tensor") && has("type")) return "Uses a number format this version cannot read: " + std::string(why) + ".";
  if (has("chat template")) return "Its chat format is not supported yet: " + std::string(why) + ".";
  if (has("GGUF")) return "Not a valid GGUF model file: " + std::string(why) + ".";
  return std::string(why);
}

ModelCheck check_model(const std::filesystem::path& path) {
  ModelCheck c;
  try {
    const gguf::File f = gguf::File::open(path);
    if (const gguf::Value* a = f.find("general.architecture"); a != nullptr && a->type == gguf::ValueType::kString) {
      c.arch = std::string(a->s);
    }
    std::vector<std::string> types;
    for (const gguf::TensorInfo& t : f.tensors()) {
      if (t.n_dims < 2) continue;  // norms and biases are F32 in every file
      const std::string n = t.traits != nullptr ? t.traits->name : "?";
      if (std::find(types.begin(), types.end(), n) == types.end()) types.push_back(n);
    }
    for (const std::string& t : types) c.types += (c.types.empty() ? "" : " + ") + t;
    const model::LlamaConfig cfg = model::LlamaConfig::from_gguf(f);
    static_cast<void>(model::LlamaTensors::bind(f, cfg));
    const tokenizer::Tokenizer tok = tokenizer::Tokenizer::from_gguf(f);
    static_cast<void>(tokenizer::ChatTemplate::detect(f, tok));
    // Formats without kernels of their own are converted to Q8_0 as the
    // model loads (LlamaRef, ADR 0028): full speed, Q8_0's memory.
    bool converted = false;
    for (const std::string& t : types) {
      converted = converted ||
                  (t != "Q8_0" && t != "Q4_0" && t != "Q4_K" && t != "Q6_K" && t != "F32" && t != "F16" && t != "BF16");
    }
    const std::string note = converted ? "Some matrices are converted to 8-bit (Q8_0) while it loads: full speed, "
                                         "but about the memory of an 8-bit file."
                                       : "";
    if (weight_kind(f) == "full") {
      c.status = "cpu";
      c.reason =
          "Full precision: runs on the CPU's slow reference path. A quantized file (for example Q8_0 or "
          "Q4_K_M) is much faster.";
    } else if (!gpu_readable(f)) {
      c.status = "cpu";
      c.reason = "Runs on the CPU: the graphics chip cannot read its Q4_K/Q6_K matrices yet." +
                 (note.empty() ? std::string() : " " + note);
    } else {
      c.status = "ok";
      c.reason = note;
    }
  } catch (const Error& e) {
    c.status = "unsupported";
    c.reason = plain_reason(e.what(), c.arch);
  } catch (const std::exception& e) {
    c.status = "unsupported";
    c.reason = std::string("The file could not be read: ") + e.what();
  }
  return c;
}

struct ModelFile {
  std::filesystem::path path;
  std::string name;
  double size_gib = 0;
  bool added = false;  // chosen with "Add model", not found in a models folder
  ModelCheck check;
};

bool is_gguf(const std::filesystem::path& p) {
  std::wstring ext = p.extension().wstring();
  std::transform(ext.begin(), ext.end(), ext.begin(), [](wchar_t ch) { return static_cast<wchar_t>(towlower(ch)); });
  return ext == L".gguf";
}

// The checks, cached by path, size and time of last write: the page asks
// for the list every few seconds, and a vocabulary takes ~0.1 s to read.
class ModelChecks {
 public:
  ModelCheck get(const std::filesystem::path& p, std::uintmax_t size) {
    std::error_code ec;
    const auto when = std::filesystem::last_write_time(p, ec);
    const std::lock_guard<std::mutex> lock(mutex_);
    for (const Entry& e : cache_) {
      if (e.path == p && e.size == size && e.when == when) return e.check;
    }
    ModelCheck c = check_model(p);
    std::erase_if(cache_, [&](const Entry& e) { return e.path == p; });
    cache_.push_back({p, size, when, c});
    return c;
  }

 private:
  struct Entry {
    std::filesystem::path path;
    std::uintmax_t size;
    std::filesystem::file_time_type when;
    ModelCheck check;
  };
  std::mutex mutex_;
  std::vector<Entry> cache_;
};

// The .gguf files in `dirs`, then the added ones; each file once. Files that
// run first (GPU too, then CPU only), then the small ones.
std::vector<ModelFile> scan_models(const std::vector<std::filesystem::path>& dirs,
                                   const std::vector<std::filesystem::path>& added, ModelChecks& checks) {
  std::vector<ModelFile> out;
  std::vector<std::filesystem::path> seen;
  auto add = [&](const std::filesystem::path& p, bool from_list) {
    std::error_code ec;
    if (!std::filesystem::is_regular_file(p, ec) || !is_gguf(p)) return;
    const std::filesystem::path canon = std::filesystem::weakly_canonical(p, ec);
    if (std::find(seen.begin(), seen.end(), canon) != seen.end()) return;
    seen.push_back(canon);
    ModelFile m;
    m.path = canon;
    m.name = path_to_utf8(canon.filename());
    const std::uintmax_t size = std::filesystem::file_size(canon, ec);
    m.size_gib = gib(size);
    m.added = from_list;
    m.check = checks.get(canon, size);
    out.push_back(std::move(m));
  };
  for (const auto& d : dirs) {
    std::error_code ec;
    if (!std::filesystem::is_directory(d, ec)) continue;
    for (const auto& e : std::filesystem::directory_iterator(d, ec)) add(e.path(), false);
  }
  for (const auto& p : added) add(p, true);
  const auto rank = [](const ModelFile& m) { return m.check.status == "ok" ? 0 : m.check.status == "cpu" ? 1 : 2; };
  std::stable_sort(out.begin(), out.end(), [&](const ModelFile& a, const ModelFile& b) {
    if (rank(a) != rank(b)) return rank(a) < rank(b);
    return a.size_gib < b.size_gib;
  });
  return out;
}

// The files chosen with "Add model", one UTF-8 path per line.
std::vector<std::filesystem::path> read_added(const std::filesystem::path& file) {
  std::vector<std::filesystem::path> out;
  std::ifstream in(file, std::ios::binary);
  for (std::string line; std::getline(in, line);) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (!line.empty() && valid_utf8(line)) out.push_back(path_from_utf8(line));
  }
  return out;
}

void write_added(const std::filesystem::path& file, const std::vector<std::filesystem::path>& paths) {
  std::ofstream out(file, std::ios::binary | std::ios::trunc);
  for (const auto& p : paths) out << path_to_utf8(p) << '\n';
  if (!out) log_warn("ui: could not save the list of added models to {}", path_to_utf8(file));
}

// A top-level window of the app (the Edge window shows the page's title),
// to own the file dialog; nullptr if there is none.
HWND find_app_window() {
  struct Search {
    HWND found = nullptr;
  } search;
  EnumWindows(
      [](HWND h, LPARAM lp) -> BOOL {
        wchar_t title[64] = {};
        if (IsWindowVisible(h) && GetWindowTextW(h, title, 64) > 0 && std::wstring_view(title).starts_with(L"Hearth")) {
          // EnumWindows hands back the pointer passed below as its LPARAM.
          reinterpret_cast<Search*>(lp)->found = h;  // NOLINT(performance-no-int-to-ptr)
          return FALSE;
        }
        return TRUE;
      },
      reinterpret_cast<LPARAM>(&search));
  return search.found;
}

// Brings the file dialog in front when it opens. The click came from the
// browser, another process, so Windows would otherwise open it behind.
class DialogEvents final : public IFileDialogEvents {
 public:
  IFACEMETHODIMP QueryInterface(REFIID riid, void** out) override {
    if (riid == IID_IUnknown || riid == IID_IFileDialogEvents) {
      *out = static_cast<IFileDialogEvents*>(this);
      AddRef();
      return S_OK;
    }
    *out = nullptr;
    return E_NOINTERFACE;
  }
  IFACEMETHODIMP_(ULONG) AddRef() override { return ++refs_; }
  IFACEMETHODIMP_(ULONG) Release() override { return --refs_; }  // on the stack; never deleted by COM
  IFACEMETHODIMP OnFolderChange(IFileDialog* d) override {
    if (shown_) return S_OK;
    shown_ = true;
    IOleWindow* ole = nullptr;
    HWND h = nullptr;
    if (SUCCEEDED(d->QueryInterface(IID_PPV_ARGS(&ole)))) {
      ole->GetWindow(&h);
      ole->Release();
    }
    if (h == nullptr) return S_OK;
    const DWORD fg = GetWindowThreadProcessId(GetForegroundWindow(), nullptr);
    const DWORD me = GetCurrentThreadId();
    const bool attached = fg != 0 && fg != me && AttachThreadInput(me, fg, TRUE);
    SetForegroundWindow(h);
    BringWindowToTop(h);
    if (attached) AttachThreadInput(me, fg, FALSE);
    return S_OK;
  }
  IFACEMETHODIMP OnFileOk(IFileDialog*) override { return S_OK; }
  IFACEMETHODIMP OnFolderChanging(IFileDialog*, IShellItem*) override { return S_OK; }
  IFACEMETHODIMP OnSelectionChange(IFileDialog*) override { return S_OK; }
  IFACEMETHODIMP OnShareViolation(IFileDialog*, IShellItem*, FDE_SHAREVIOLATION_RESPONSE*) override { return S_OK; }
  IFACEMETHODIMP OnTypeChange(IFileDialog*) override { return S_OK; }
  IFACEMETHODIMP OnOverwrite(IFileDialog*, IShellItem*, FDE_OVERWRITE_RESPONSE*) override { return S_OK; }

 private:
  ULONG refs_ = 1;
  bool shown_ = false;
};

// Windows' own "Open" dialog for a .gguf file, on a thread of its own (a COM
// apartment for the dialog). nullopt if the user cancelled.
std::optional<std::filesystem::path> pick_gguf_file() {
  std::optional<std::filesystem::path> result;
  std::string error;
  std::thread t([&] {
    const HRESULT init = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE);
    IFileOpenDialog* d = nullptr;
    HRESULT hr = CoCreateInstance(CLSID_FileOpenDialog, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&d));
    if (SUCCEEDED(hr)) {
      const COMDLG_FILTERSPEC filters[] = {{L"GGUF models (*.gguf)", L"*.gguf"}, {L"All files", L"*.*"}};
      d->SetFileTypes(2, filters);
      d->SetTitle(L"Add a model file (.gguf)");
      FILEOPENDIALOGOPTIONS opts = 0;
      d->GetOptions(&opts);
      d->SetOptions(opts | FOS_FORCEFILESYSTEM | FOS_FILEMUSTEXIST | FOS_PATHMUSTEXIST);
      DialogEvents events;
      DWORD cookie = 0;
      const bool advised = SUCCEEDED(d->Advise(&events, &cookie));
      hr = d->Show(find_app_window());
      if (advised) d->Unadvise(cookie);
      if (SUCCEEDED(hr)) {
        IShellItem* item = nullptr;
        if (SUCCEEDED(d->GetResult(&item))) {
          PWSTR p = nullptr;
          if (SUCCEEDED(item->GetDisplayName(SIGDN_FILESYSPATH, &p))) {
            result = std::filesystem::path(p);
            CoTaskMemFree(p);
          }
          item->Release();
        }
      } else if (hr != HRESULT_FROM_WIN32(ERROR_CANCELLED)) {
        error = std::format("the file dialog failed (HRESULT 0x{:08X})", static_cast<std::uint32_t>(hr));
      }
      d->Release();
    } else {
      error = std::format("the file dialog is not available (HRESULT 0x{:08X})", static_cast<std::uint32_t>(hr));
    }
    if (SUCCEEDED(init)) CoUninitialize();
  });
  t.join();
  if (!error.empty()) fail(ErrorKind::kIo, "{}", error);
  return result;
}

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  const std::size_t n = v.size();
  return n % 2 == 1 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

std::string now_iso_local() {
  SYSTEMTIME t{};
  GetLocalTime(&t);
  return std::format("{:04}-{:02}-{:02} {:02}:{:02}", t.wYear, t.wMonth, t.wDay, t.wHour, t.wMinute);
}

// Integer and real form fields with bounds; a missing field takes `fallback`.
long long int_param(const ui::Params& p, std::string_view name, long long fallback, long long lo, long long hi) {
  const auto v = ui::param(p, name);
  if (!v || v->empty()) return fallback;
  long long x = 0;
  std::size_t i = 0;
  const bool neg = (*v)[0] == '-';
  if (neg) i = 1;
  if (i == v->size() || v->size() > 18) fail(ErrorKind::kInvalidArgument, "{} must be a whole number", name);
  for (; i < v->size(); ++i) {
    const char c = (*v)[i];
    if (c < '0' || c > '9') fail(ErrorKind::kInvalidArgument, "{} must be a whole number", name);
    x = x * 10 + (c - '0');
  }
  if (neg) x = -x;
  if (x < lo || x > hi) fail(ErrorKind::kInvalidArgument, "{} must be between {} and {}", name, lo, hi);
  return x;
}

double real_param(const ui::Params& p, std::string_view name, double fallback, double lo, double hi) {
  const auto v = ui::param(p, name);
  if (!v || v->empty()) return fallback;
  double x = 0;  // from_chars: a dot as the decimal point whatever the PC's region settings
  const auto [end, ec] = std::from_chars(v->data(), v->data() + v->size(), x);
  if (ec != std::errc() || end != v->data() + v->size() || !std::isfinite(x)) {
    fail(ErrorKind::kInvalidArgument, "{} must be a number", name);
  }
  if (x < lo || x > hi) fail(ErrorKind::kInvalidArgument, "{} must be between {} and {}", name, lo, hi);
  return x;
}

std::string text_param(const ui::Params& p, std::string_view name, std::size_t max_bytes, bool required) {
  auto v = ui::param(p, name);
  if (!v || v->empty()) {
    if (required) fail(ErrorKind::kInvalidArgument, "{} is empty", name);
    return {};
  }
  if (v->size() > max_bytes) fail(ErrorKind::kInvalidArgument, "{} is longer than {} bytes", name, max_bytes);
  return std::move(*v);
}

const char* stop_name(StopReason s) {
  switch (s) {
    case StopReason::kEndOfGeneration:
      return "end";
    case StopReason::kLimit:
      return "limit";
    case StopReason::kContextFull:
      return "full";
    case StopReason::kCancelled:
      return "stopped";
  }
  return "end";
}

}  // namespace

std::vector<std::filesystem::path> default_model_dirs() {
  std::vector<std::filesystem::path> dirs;
  const std::filesystem::path dir = exe_path().parent_path();
  dirs.push_back(dir / L"models");
  dirs.push_back(dir / L".." / L".." / L".." / L"models");  // build\<preset>\cli\ in the repository
  std::error_code ec;
  dirs.push_back(std::filesystem::current_path(ec) / L"models");
  return dirs;
}

struct UiApp::Impl {
  UiOptions o;
  std::filesystem::path exe = exe_path();

  // Machine facts, captured once.
  std::string cpu;
  int cores = 0, logical = 0;
  bool avx512 = false;
  std::vector<std::string> gpus;

  // Jobs: job_mutex is held for a whole job; state_mutex guards the fields
  // below it, which handle() reads for /api/status while a job runs.
  std::mutex job_mutex;
  std::mutex state_mutex;
  std::string job;  // "" when idle
  bool quit = false;
  Clock::time_point last_request = Clock::now();
  Clock::time_point started = Clock::now();

  struct Loaded {
    bool present = false;
    std::string model, mode, label, backend, gpu, kind;
    std::string draft;          // the draft model's file name, or empty
    bool gpu_possible = false;  // a Q8_0/Q4_0 file and a Vulkan GPU: the gpu mode can run
    double load_s = 0;          // how long loading took
    bool fresh = true;
    std::uint32_t used = 0, size = 0;
    bool gpu_lost = false;
  } loaded;  // guarded by state_mutex

  // Owned by the job holder.
  std::optional<Engine> engine;

  struct Reply {
    int prompt_tokens = 0, generated = 0;
    double prefill_s = 0, decode_s = 0;
    int draft_tokens = 0, draft_accepted = 0;
    std::string mode;
  };
  static constexpr std::size_t kMaxReplies = 200;
  std::vector<Reply> replies;  // this session's latest replies, guarded by state_mutex

  ModelChecks checks;
  std::mutex added_mutex;
  std::vector<std::filesystem::path> added;  // guarded by added_mutex

  std::vector<ModelFile> models() {
    std::vector<std::filesystem::path> a;
    {
      const std::lock_guard<std::mutex> lock(added_mutex);
      a = added;
    }
    return scan_models(o.model_dirs, a, checks);
  }

  explicit Impl(UiOptions opts) : o(std::move(opts)) {
    if (!o.models_file.empty()) added = read_added(o.models_file);
    const CpuFeatures& f = cpu_features();
    cpu = f.brand;
    cores = f.physical_cores;
    logical = f.logical_cores;
    avx512 = isa_supported(Isa::kAvx512);
    try {
      if (gpu::vulkan_available()) {
        for (const gpu::DeviceInfo& d : gpu::list_devices())
          gpus.push_back(d.name + " (driver " + d.driver_version_text + ")");
      }
    } catch (const std::exception& e) {
      log_warn("ui: could not list the GPUs: {}", e.what());
    }
  }

  // RAII for one job: refuses if another runs; on exit clears a pending
  // stop request before it marks the app idle (keep_running relies on that
  // order), and wakes nothing else.
  class Job {
   public:
    Job(Impl& m, const char* name) : m_(m), lock_(m.job_mutex, std::try_to_lock) {
      if (!lock_.owns_lock()) return;
      const std::lock_guard<std::mutex> s(m_.state_mutex);
      m_.job = name;
    }
    ~Job() {
      if (!lock_.owns_lock()) return;
      const std::lock_guard<std::mutex> s(m_.state_mutex);
      clear_cancel();
      m_.job.clear();
      m_.last_request = Clock::now();
    }
    Job(const Job&) = delete;
    Job& operator=(const Job&) = delete;
    bool owns() const noexcept { return lock_.owns_lock(); }

   private:
    Impl& m_;
    std::unique_lock<std::mutex> lock_;
  };

  std::string busy_message() {
    const std::lock_guard<std::mutex> s(state_mutex);
    return std::format("busy: {} is running. Wait for it or press Stop.", job.empty() ? std::string("a job") : job);
  }

  void stop_job() {
    const std::lock_guard<std::mutex> s(state_mutex);
    if (job.empty()) return;
    request_cancel();
  }

  void update_loaded() {
    const std::lock_guard<std::mutex> s(state_mutex);
    if (!engine) {
      loaded = Loaded{};
      return;
    }
    loaded.used = engine->context_used();
    loaded.size = engine->context_size();
    loaded.gpu_lost = engine->gpu_lost();
    loaded.backend = engine->backend_description();
  }

  // Loads `model` on `mode` (cpu or gpu). `with_draft`: also the draft model
  // of speculative decoding, if the folder has one (find_draft); the
  // benchmark measures the model alone. Holds the job.
  void load(const std::string& model, const std::string& mode, bool with_draft = true) {
    if (mode != "cpu" && mode != "gpu") fail(ErrorKind::kInvalidArgument, "mode must be cpu or gpu");
    const std::filesystem::path p = path_from_utf8(model);
    std::error_code ec;
    if (!std::filesystem::is_regular_file(p, ec)) fail(ErrorKind::kIo, "the model file {} does not exist", model);
    engine.reset();  // free the old model first: two would not fit in memory
    {
      const std::lock_guard<std::mutex> s(state_mutex);
      loaded = Loaded{};
    }
    const auto t0 = Clock::now();
    api::LoadOptions lo;
    lo.model = model;
    lo.threads = o.threads;
    lo.ctx = o.ctx;
    lo.isa = avx512 ? Isa::kAvx512 : Isa::kScalar;
    lo.gpu_fallback = true;  // interactive use: a GPU problem continues on the CPU
    std::string kind;
    bool gpu_file = false;
    {
      const gguf::File f = gguf::File::open(p);
      kind = weight_kind(f);
      gpu_file = kind != "full" && gpu_readable(f);
    }
    // The draft runs on the CPU, so only the cpu mode uses one.
    const std::filesystem::path draft =
        with_draft && o.draft && mode == "cpu" ? find_draft(p) : std::filesystem::path();
    if (!draft.empty()) lo.draft_model = path_to_utf8(draft);
    if (kind == "full") {
      lo.matmul = model::Matmul::kFp32;  // only the reference path runs full-precision files
    } else if (mode == "gpu" && gpu_file) {
      lo.matmul = model::Matmul::kGpu;
    } else {
      lo.matmul = model::Matmul::kQ8Activations;
    }
    Engine e = detail::EngineAccess::load(lo);
    const model::LlamaRef& llm = detail::EngineAccess::llama(e);
    const bool on_gpu = llm.matmul() == model::Matmul::kGpu;
    Loaded l;
    l.present = true;
    l.model = model;
    l.mode = on_gpu ? "gpu" : "cpu";
    l.kind = kind;
    l.gpu = e.gpu_device();
    l.gpu_possible = gpu_file && !gpus.empty();
    if (kind == "full") {
      l.label = "Processor (CPU), full-precision reference path: slow; pick a Q8_0 or Q4_0 file for speed";
    } else if (on_gpu) {
      l.label = "Graphics chip (GPU)";
    } else if (mode == "gpu" && !gpu_file) {
      l.label = "Processor (CPU): the graphics chip cannot read this file's Q4_K/Q6_K matrices yet";
    } else if (mode == "gpu") {
      l.label = "Processor (CPU): the graphics chip could not be used";
    } else {
      l.label = std::format("Processor (CPU), {} kernels", avx512 ? "AVX-512" : "scalar");
    }
    if (!draft.empty()) {
      l.draft = path_to_utf8(draft.filename());
      l.label += std::format(" + speculative decoding with {}", l.draft);
    }
    l.load_s = std::chrono::duration<double>(Clock::now() - t0).count();
    engine.emplace(std::move(e));
    {
      const std::lock_guard<std::mutex> s(state_mutex);
      loaded = std::move(l);
    }
    update_loaded();
    log_info("ui: loaded {} ({})", model, mode);
  }

  // ----- the routes -----

  void status(ui::Connection& c) {
    const MemoryStatus mem = memory_status();
    const std::vector<ModelFile> models = this->models();
    const BuildInfo& b = build_info();
    JsonWriter w(false);
    w.begin_object();
    w.field("version", b.version);
    w.field("build", b.build_type);
    w.field("commit", b.commit);
    w.field("uptime_s", std::chrono::duration<double>(Clock::now() - started).count());
    w.key("machine");
    w.begin_object();
    w.field("cpu", cpu);
    w.field("cores", cores);
    w.field("threads", logical);
    w.field("avx512", avx512);
    w.field("memory_gib", gib(mem.total_bytes));
    w.field("memory_free_gib", gib(mem.available_bytes));
    w.field("power", power_api_name(power_source()));
    w.field("engine_threads", o.threads);
    w.key("gpus");
    w.begin_array();
    for (const std::string& g : gpus) w.value(g);
    w.end_array();
    w.end_object();
    w.key("process");
    w.begin_object();
    w.field("working_set_gib", gib(mem.working_set_bytes));
    w.field("commit_gib", gib(mem.commit_bytes));
    w.end_object();
    w.key("models");
    w.begin_array();
    for (const ModelFile& m : models) {
      w.begin_object();
      w.field("path", path_to_utf8(m.path));
      w.field("name", m.name);
      w.field("size_gib", m.size_gib);
      w.field("added", m.added);
      w.field("status", m.check.status);
      w.field("reason", m.check.reason);
      w.field("arch", m.check.arch);
      w.field("types", m.check.types);
      w.end_object();
    }
    w.end_array();
    {
      const std::lock_guard<std::mutex> s(state_mutex);
      w.field("job", job);
      w.key("loaded");
      if (!loaded.present) {
        w.null();
      } else {
        w.begin_object();
        w.field("model", loaded.model);
        w.field("name", path_to_utf8(path_from_utf8(loaded.model).filename()));
        w.field("mode", loaded.mode);
        w.field("label", loaded.label);
        w.field("backend", loaded.backend);
        w.field("gpu", loaded.gpu);
        w.field("kind", loaded.kind);
        w.field("draft", loaded.draft);
        w.field("gpu_possible", loaded.gpu_possible);
        w.field("load_s", loaded.load_s);
        w.field("context_used", loaded.used);
        w.field("context_size", loaded.size);
        w.field("gpu_lost", loaded.gpu_lost);
        w.end_object();
      }
      w.key("replies");
      w.begin_array();
      for (const Reply& r : replies) {
        w.begin_object();
        w.field("prompt_tokens", r.prompt_tokens);
        w.field("generated", r.generated);
        w.field("prefill_s", r.prefill_s);
        w.field("decode_s", r.decode_s);
        w.field("draft_tokens", r.draft_tokens);
        w.field("draft_accepted", r.draft_accepted);
        w.field("mode", r.mode);
        w.end_object();
      }
      w.end_array();
    }
    w.end_object();
    c.respond(200, "application/json; charset=utf-8", w.str());
  }

  // The saved benchmarks: our own lines, one JSON object each, of kind
  // "bench". Anything else (lines of an older version, damage) is skipped.
  void history(ui::Connection& c) {
    std::string out = "[";
    std::ifstream in(o.results_file, std::ios::binary);
    bool first = true;
    for (std::string line; std::getline(in, line);) {
      if (!line.empty() && line.back() == '\r') line.pop_back();
      if (line.size() < 2 || line.front() != '{' || line.back() != '}' || !valid_utf8(line)) continue;
      if (line.find("\"kind\":\"bench\"") == std::string::npos) continue;
      if (!first) out += ',';
      out += line;
      first = false;
    }
    out += ']';
    c.respond(200, "application/json; charset=utf-8", out);
  }

  // Adds a model file by path; answers with its check, so the page can say
  // at once whether it runs. Unsupported files are listed too, with why.
  void add_model(const std::filesystem::path& path, ui::Connection& c) {
    std::error_code ec;
    if (!std::filesystem::is_regular_file(path, ec)) {
      return respond_error(c, 400, std::format("{} is not a file", path_to_utf8(path)));
    }
    if (!is_gguf(path)) {
      return respond_error(c, 400,
                           "Only .gguf model files can be added (other formats need converting with llama.cpp first).");
    }
    const std::filesystem::path canon = std::filesystem::weakly_canonical(path, ec);
    {
      const std::lock_guard<std::mutex> lock(added_mutex);
      if (std::find(added.begin(), added.end(), canon) == added.end()) {
        added.push_back(canon);
        if (!o.models_file.empty()) write_added(o.models_file, added);
      }
    }
    const ModelCheck chk = checks.get(canon, std::filesystem::file_size(canon, ec));
    JsonWriter w(false);
    w.begin_object();
    w.field("path", path_to_utf8(canon));
    w.field("name", path_to_utf8(canon.filename()));
    w.field("status", chk.status);
    w.field("reason", chk.reason);
    w.end_object();
    c.respond(200, "application/json; charset=utf-8", w.str());
  }

  void route_add(const ui::Params& p, ui::Connection& c) {
    std::string path = text_param(p, "model", 4096, true);
    // Explorer's "Copy as path" puts quotes around it.
    if (path.size() >= 2 && path.front() == '"' && path.back() == '"') path = path.substr(1, path.size() - 2);
    add_model(path_from_utf8(path), c);
  }

  void route_pick(ui::Connection& c) {
    const std::optional<std::filesystem::path> path = pick_gguf_file();
    if (!path) {
      c.respond(200, "application/json; charset=utf-8", "{\"cancelled\":true}");
      return;
    }
    add_model(*path, c);
  }

  void route_forget(const ui::Params& p, ui::Connection& c) {
    const std::filesystem::path path = path_from_utf8(text_param(p, "model", 4096, true));
    {
      const std::lock_guard<std::mutex> lock(added_mutex);
      std::erase_if(added, [&](const std::filesystem::path& a) { return a == path; });
      if (!o.models_file.empty()) write_added(o.models_file, added);
    }
    c.respond(200, "application/json; charset=utf-8", json_message("ok", "the model was removed from the list"));
  }

  void route_load(const ui::Params& p, ui::Connection& c) {
    const std::string model = text_param(p, "model", 4096, true);
    const std::string mode = ui::param(p, "mode").value_or("cpu");
    Job job_guard(*this, "loading the model");
    if (!job_guard.owns()) return respond_error(c, 409, busy_message());
    load(model, mode);
    status(c);
  }

  void route_reset(ui::Connection& c) {
    Job job_guard(*this, "starting a new conversation");
    if (!job_guard.owns()) return respond_error(c, 409, busy_message());
    if (engine) engine->reset();
    {
      const std::lock_guard<std::mutex> s(state_mutex);
      loaded.fresh = true;
    }
    update_loaded();
    c.respond(200, "application/json; charset=utf-8", json_message("ok", "the conversation was reset"));
  }

  void route_chat(const ui::Params& p, ui::Connection& c) {
    const std::string message = text_param(p, "message", 32 * 1024, true);
    const std::string system = text_param(p, "system", 8 * 1024, false);
    GenerateOptions g;
    g.temperature = static_cast<float>(real_param(p, "temperature", 0.7, 0.0, 2.0));
    g.max_new_tokens = static_cast<int>(int_param(p, "max_tokens", 512, 1, 4096));
    g.top_k = 40;
    g.top_p = 0.95f;
    g.seed = (std::uint64_t{std::random_device{}()} << 32) | std::random_device{}();
    Job job_guard(*this, "writing a reply");
    if (!job_guard.owns()) return respond_error(c, 409, busy_message());
    if (!engine) return respond_error(c, 409, "load a model first");
    Engine& eng = *engine;
    bool fresh = true;
    std::string mode;
    {
      const std::lock_guard<std::mutex> s(state_mutex);
      fresh = loaded.fresh;
      mode = loaded.mode;
    }
    c.begin_stream("text/event-stream; charset=utf-8");
    try {
      const ChatMessage m{"user", message};
      const std::vector<std::int32_t> ids =
          fresh ? eng.chat_prompt(system, std::span(&m, 1)) : eng.chat_next_turn(message);
      if (ids.size() > eng.context_size() - eng.context_used()) {
        c.send(
            ui::sse_event("error", json_message("message", "The conversation is full. Start a new chat to continue.")));
        return;
      }
      const auto t0 = Clock::now();
      const GenerateResult r = eng.generate(ids, g, [&](std::int32_t, std::string_view piece) {
        if (piece.empty()) return;
        if (!c.send(ui::sse_event("token", json_message("text", piece)))) request_cancel();  // the page was closed
      });
      const double total_s = std::chrono::duration<double>(Clock::now() - t0).count();
      if (!r.trailing_text.empty()) c.send(ui::sse_event("token", json_message("text", r.trailing_text)));
      if (eng.gpu_lost()) mode = "cpu";
      {
        const std::lock_guard<std::mutex> s(state_mutex);
        loaded.fresh = fresh && eng.context_used() == 0;
        replies.push_back({r.prompt_tokens, r.generated_tokens, r.prefill_seconds, r.decode_seconds, r.draft_tokens,
                           r.draft_accepted, mode});
        if (replies.size() > kMaxReplies) replies.erase(replies.begin());
      }
      update_loaded();
      JsonWriter w(false);
      w.begin_object();
      w.field("prompt_tokens", r.prompt_tokens);
      w.field("prefill_s", r.prefill_seconds);
      w.field("generated", r.generated_tokens);
      w.field("decode_s", r.decode_seconds);
      w.field("total_s", total_s);
      w.field("stop", stop_name(r.stop));
      w.field("mode", mode);
      w.field("context_used", eng.context_used());
      w.field("context_size", eng.context_size());
      w.field("gpu_lost", eng.gpu_lost());
      w.field("draft_tokens", r.draft_tokens);
      w.field("draft_accepted", r.draft_accepted);
      w.end_object();
      c.send(ui::sse_event("done", w.str()));
    } catch (const Error& e) {
      c.send(ui::sse_event("error", json_message("message", e.what())));
    }
  }

  // The benchmark: `hearth bench`'s method (cli/cmd_bench.cpp) on the loaded
  // model, or on the CPU and then the GPU. Reading: one forward pass over
  // `prompt` tokens (BOS, then pseudo-random IDs); writing: `gen` one-token
  // passes from an empty context. One warm-up each, then `reps` timed runs;
  // the median and the range are reported in tokens per second.
  void route_bench(const ui::Params& p, ui::Connection& c) {
    const int prompt = static_cast<int>(int_param(p, "prompt_tokens", 512, 16, 2048));
    const int gen = static_cast<int>(int_param(p, "gen_tokens", 128, 16, 512));
    const int reps = static_cast<int>(int_param(p, "reps", 3, 1, 9));
    const std::string which = ui::param(p, "backends").value_or("current");
    if (which != "current" && which != "both") {
      return respond_error(c, 400, "backends must be current or both");
    }
    Job job_guard(*this, "the benchmark");
    if (!job_guard.owns()) return respond_error(c, 409, busy_message());
    c.begin_stream("text/event-stream; charset=utf-8");
    auto progress = [&](std::string_view what, int done, int total) {
      JsonWriter w(false);
      w.begin_object();
      w.field("text", what);
      w.field("done", done);
      w.field("total", total);
      w.end_object();
      c.send(ui::sse_event("progress", w.str()));
    };
    std::string model_path, mode_before;
    bool gpu_possible = false, had_draft = false;
    {
      const std::lock_guard<std::mutex> s(state_mutex);
      model_path = loaded.model;
      mode_before = loaded.mode;
      gpu_possible = loaded.gpu_possible;
      had_draft = !loaded.draft.empty();
    }
    bool reloaded = false;
    try {
      if (build_info().build_type != std::string("Release")) {
        fail(ErrorKind::kInvalidArgument, "Benchmarks need the Release build (this is a {} build).",
             build_info().build_type);
      }
      if (power_source() != PowerSource::kAc) {
        fail(ErrorKind::kInvalidArgument,
             "Plug in the charger first. On battery, Windows slows the chips down, so the numbers would mislead.");
      }
      if (!engine || model_path.empty()) fail(ErrorKind::kInvalidArgument, "Load a model first.");
      if (which == "both" && !gpu_possible) {
        fail(ErrorKind::kUnsupported,
             "This model or PC cannot use the graphics chip (a Q8_0 or Q4_0 model and a Vulkan GPU are needed), so "
             "only the current engine can be measured.");
      }
      if (static_cast<std::uint32_t>(std::max(prompt, gen)) > engine->context_size()) {
        fail(ErrorKind::kInvalidArgument, "The test is longer than the context ({} tokens).", engine->context_size());
      }
      const std::vector<std::string> modes =
          which == "both" ? std::vector<std::string>{"cpu", "gpu"} : std::vector<std::string>{mode_before};
      const int total = static_cast<int>(modes.size()) * 2 * (reps + 1);
      int done = 0;
      const PerformanceMode perf;  // keep the PC awake and at full speed, as bench does
      std::uint64_t weight_bytes = 0;
      {
        const gguf::File f = gguf::File::open(path_from_utf8(model_path));
        weight_bytes = model::decode_weight_bytes(f);
      }
      struct Run {
        std::string backend, device, engine;
        double pp = 0, pp_min = 0, pp_max = 0, tg = 0, tg_min = 0, tg_max = 0;
      };
      std::vector<Run> runs;
      for (const std::string& mode : modes) {
        throw_if_cancelled();
        if (which == "both" || had_draft) {
          // The model alone, without a draft: the draft's memory would only
          // be in the way, and the passes below never use it.
          progress(std::format("Loading the model on the {}...", mode == "gpu" ? "graphics chip" : "processor"), done,
                   total);
          load(model_path, mode, false);
          reloaded = true;
        }
        if (!engine) fail(ErrorKind::kInternal, "no engine after loading");
        Engine& e = *engine;
        model::LlamaRef& llm = detail::EngineAccess::llama(e);
        if (mode == "gpu" && llm.matmul() != model::Matmul::kGpu) {
          fail(ErrorKind::kGpu, "The graphics chip could not be used for this model, so it was not measured.");
        }
        const std::uint32_t n_vocab = llm.config().n_vocab;
        std::vector<float> logits(n_vocab);
        std::vector<std::int32_t> tokens(static_cast<std::size_t>(std::max(prompt, gen)));
        for (std::size_t i = 0; i < tokens.size(); ++i) {
          tokens[i] = static_cast<std::int32_t>((static_cast<std::uint64_t>(i + 1) * 2654435761u) % n_vocab);
        }
        const tokenizer::Tokenizer& tok = detail::EngineAccess::tokenizer(e);
        if (tok.bos() >= 0) tokens[0] = tok.bos();
        const char* who = llm.matmul() == model::Matmul::kGpu ? "GPU" : "CPU";
        auto check_power = [&] {
          if (power_source() != PowerSource::kAc) {
            fail(ErrorKind::kInvalidArgument,
                 "The charger was unplugged during the benchmark, so the result was discarded. Plug it in and run it "
                 "again.");
          }
        };
        auto pp = [&] {
          throw_if_cancelled();
          check_power();
          llm.reset();
          const auto t0 = Clock::now();
          llm.forward(std::span(tokens).first(static_cast<std::size_t>(prompt)), logits, false);
          return std::chrono::duration<double>(Clock::now() - t0).count();
        };
        auto tg = [&] {
          throw_if_cancelled();
          check_power();
          llm.reset();
          const auto t0 = Clock::now();
          for (int i = 0; i < gen; ++i)
            llm.forward(std::span(tokens).subspan(static_cast<std::size_t>(i), 1), logits, false);
          return std::chrono::duration<double>(Clock::now() - t0).count();
        };
        auto measure = [&](const char* what, int n, auto&& once) {
          once();  // warm-up
          progress(std::format("{}: warm-up of {} done", who, what), ++done, total);
          std::vector<double> rate;
          for (int r = 0; r < reps; ++r) {
            rate.push_back(n / once());
            progress(std::format("{}: {} run {} of {}: {:.1f} tokens/s", who, what, r + 1, reps, rate.back()), ++done,
                     total);
          }
          return rate;
        };
        const std::vector<double> read = measure("reading", prompt, pp);
        const std::vector<double> write = measure("writing", gen, tg);
        llm.reset();
        if (e.gpu_lost()) {
          fail(ErrorKind::kDeviceLost, "The graphics chip failed during the benchmark, so the result was discarded.");
        }
        const auto [rlo, rhi] = std::minmax_element(read.begin(), read.end());
        const auto [wlo, whi] = std::minmax_element(write.begin(), write.end());
        const bool on_gpu = llm.matmul() == model::Matmul::kGpu;
        runs.push_back({on_gpu ? "gpu" : "cpu", on_gpu ? e.gpu_device() : cpu, e.backend_description(), median(read),
                        *rlo, *rhi, median(write), *wlo, *whi});
      }
      if (reloaded) {
        progress("Loading the model as it was...", done, total);
        load(model_path, mode_before);
        reloaded = false;
      } else if (engine) {
        engine->reset();  // the benchmark used the context: the conversation starts over
      }
      {
        const std::lock_guard<std::mutex> s(state_mutex);
        loaded.fresh = true;
      }
      update_loaded();
      JsonWriter w(false);
      w.begin_object();
      w.field("kind", "bench");
      w.field("date", now_iso_local());
      w.field("model", path_to_utf8(path_from_utf8(model_path).filename()));
      w.field("prompt_tokens", prompt);
      w.field("gen_tokens", gen);
      w.field("reps", reps);
      w.field("threads", o.threads);
      w.field("weight_gb", static_cast<double>(weight_bytes) / 1e9);
      w.field("version", std::format("{} ({})", build_info().version, build_info().commit));
      w.key("runs");
      w.begin_array();
      for (const Run& r : runs) {
        w.begin_object();
        w.field("backend", r.backend);
        w.field("device", r.device);
        w.field("engine", r.engine);
        w.field("pp_tps", r.pp);
        w.field("pp_tps_min", r.pp_min);
        w.field("pp_tps_max", r.pp_max);
        w.field("tg_tps", r.tg);
        w.field("tg_tps_min", r.tg_min);
        w.field("tg_tps_max", r.tg_max);
        w.field("decode_gbps", static_cast<double>(weight_bytes) * r.tg / 1e9);
        w.end_object();
      }
      w.end_array();
      w.end_object();
      const std::string result = w.str();
      {
        std::ofstream out(o.results_file, std::ios::binary | std::ios::app);
        if (out) out << result << '\n';
        if (!out) log_warn("ui: could not save the result to {}", path_to_utf8(o.results_file));
      }
      c.send(ui::sse_event("result", result));
    } catch (const Error& err) {
      const bool stopped = err.kind() == ErrorKind::kCancelled;
      // Leave the model loaded as the user had it.
      if (reloaded) {
        clear_cancel();
        try {
          load(model_path, mode_before);
        } catch (const Error& again) {
          log_warn("ui: could not reload {} after the benchmark: {}", model_path, again.what());
        }
      } else if (engine) {
        engine->reset();
        const std::lock_guard<std::mutex> s(state_mutex);
        loaded.fresh = true;
      }
      update_loaded();
      c.send(ui::sse_event(stopped ? "stopped" : "error",
                           json_message("message", stopped ? "The benchmark was stopped." : err.what())));
    }
  }

  void handle(const ui::Request& r, ui::Connection& c) {
    {
      const std::lock_guard<std::mutex> s(state_mutex);
      last_request = Clock::now();
    }
    const bool get = r.method == "GET", post = r.method == "POST";
    if (r.path == "/") {
      if (!get) return respond_error(c, 405, "use GET");
      const std::string_view page = ui_index_html();
      c.respond(200, "text/html; charset=utf-8", page);
      return;
    }
    if (r.path == "/api/status" || r.path == "/api/history") {
      if (!get) return respond_error(c, 405, "use GET");
      return r.path == "/api/status" ? status(c) : history(c);
    }
    static constexpr std::string_view kPost[] = {"/api/load", "/api/chat", "/api/reset", "/api/stop",  "/api/bench",
                                                 "/api/quit", "/api/add",  "/api/pick",  "/api/forget"};
    if (std::find(std::begin(kPost), std::end(kPost), r.path) == std::end(kPost)) {
      return respond_error(c, 404, "no such page");
    }
    if (!post) return respond_error(c, 405, "use POST");
    ui::Params p;
    if (!ui::form_decode(r.body, p)) return respond_error(c, 400, "the form data is malformed");
    try {
      if (r.path == "/api/stop") {
        stop_job();
        c.respond(200, "application/json; charset=utf-8", json_message("ok", "stop requested"));
      } else if (r.path == "/api/quit") {
        {
          const std::lock_guard<std::mutex> s(state_mutex);
          quit = true;
        }
        stop_job();
        c.respond(200, "application/json; charset=utf-8", json_message("ok", "the app is closing"));
      } else if (r.path == "/api/load") {
        route_load(p, c);
      } else if (r.path == "/api/reset") {
        route_reset(c);
      } else if (r.path == "/api/chat") {
        route_chat(p, c);
      } else if (r.path == "/api/add") {
        route_add(p, c);
      } else if (r.path == "/api/pick") {
        route_pick(c);
      } else if (r.path == "/api/forget") {
        route_forget(p, c);
      } else {
        route_bench(p, c);
      }
    } catch (const Error& e) {
      // The request's fault (bad options, a missing or unusable file) or ours.
      const ErrorKind k = e.kind();
      const bool client = k == ErrorKind::kInvalidArgument || k == ErrorKind::kIo || k == ErrorKind::kBadModel ||
                          k == ErrorKind::kUnsupported;
      respond_error(c, client ? 400 : 500, e.what());
    }
  }

  bool keep_running() {
    bool go = true;
    {
      const std::lock_guard<std::mutex> s(state_mutex);
      if (quit) go = false;
      // A stop request only sets the cancel flag while a job runs, and a job
      // clears it before it ends, so the flag here, with no job, is Ctrl+C.
      if (job.empty() && cancel_requested()) {
        log_info("ui: Ctrl+C: closing");
        go = false;
      }
      if (o.idle_minutes > 0 && job.empty() && Clock::now() - last_request > std::chrono::minutes(o.idle_minutes)) {
        log_info("ui: no request for {} minutes: closing", o.idle_minutes);
        go = false;
      }
    }
    if (!go) stop_job();
    return go;
  }
};

UiApp::UiApp(UiOptions options) : impl_(std::make_unique<Impl>(std::move(options))) {}
UiApp::~UiApp() = default;
void UiApp::handle(const ui::Request& request, ui::Connection& connection) { impl_->handle(request, connection); }
bool UiApp::keep_running() { return impl_->keep_running(); }

}  // namespace hearth::cli
