// The hearth subcommands. Each takes the arguments after its name and
// returns the process exit code.
#pragma once

#include <string>

#include "args.h"

namespace hearth::cli {

int cmd_info(Args& args);
// `info --model FILE`: the model file's contents (cmd_model_info.cpp).
int model_info(const std::string& path, bool list_tensors, bool json);
int cmd_probe(Args& args);
int cmd_dump(Args& args);
int cmd_tokenize(Args& args);
int cmd_run(Args& args);
int cmd_chat(Args& args);
int cmd_ppl(Args& args);
int cmd_bench(Args& args);
int cmd_ui(Args& args);

}  // namespace hearth::cli
