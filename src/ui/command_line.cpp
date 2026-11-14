#include "ui/command_line.h"

namespace hearth::ui {

std::wstring quote_argument(std::wstring_view arg) {
  if (!arg.empty() && arg.find_first_of(L" \t\n\v\"") == std::wstring_view::npos) return std::wstring(arg);
  std::wstring out = L"\"";
  std::size_t backslashes = 0;
  for (const wchar_t c : arg) {
    if (c == L'\\') {
      ++backslashes;
      continue;
    }
    if (c == L'"') {
      out.append(backslashes * 2 + 1, L'\\');  // each backslash doubled, then the quote escaped
    } else {
      out.append(backslashes, L'\\');
    }
    backslashes = 0;
    out += c;
  }
  out.append(backslashes * 2, L'\\');  // before the closing quote
  out += L'"';
  return out;
}

}  // namespace hearth::ui
