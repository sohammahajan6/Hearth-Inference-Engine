// Building a Windows command line for a child process (`hearth ui` opens
// its window with `msedge.exe --app=<address>`).
#pragma once

#include <string>
#include <string_view>

namespace hearth::ui {

// One argument quoted so that CommandLineToArgvW and the C runtime read it
// back unchanged: spaces and quotes are enclosed in quotes, a quote is
// escaped, and backslashes before a quote (or the closing one) are doubled.
std::wstring quote_argument(std::wstring_view arg);

}  // namespace hearth::ui
