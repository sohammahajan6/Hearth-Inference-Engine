# Third-party notices and license inventory

Every dependency here is free of charge, and the inventory records each one's license.
- **Policy**: no paid software, trials, cloud services, paid APIs or account sign-ups.
- **New dependencies** get a row here *before* they are first used.
- **Exact versions and hashes** are in `scripts/pins.psd1` and `requirements.txt`.

"Shipped" means the item ends up inside `hearth.exe` or a release package.

## Compiled into hearth.exe
| Component | Version | License | Use |
|-----------|---------|---------|-----|
| [volk](https://github.com/zeux/volk) | vulkan-sdk-1.4.363.0 | MIT | Loads Vulkan functions at run time |
| [Vulkan-Headers](https://github.com/KhronosGroup/Vulkan-Headers) | vulkan-sdk-1.4.363.0 | Apache-2.0 OR MIT | Vulkan API declarations |
| Tables derived from the Unicode Character Database | 15.1.0 | Unicode License V3 (text below) | The tokenizer's letter, number and whitespace classes (`src/tokenizer/unicode_tables.cpp`, generated) |

Requirement for the release package: include the license texts of volk and Vulkan-Headers, and the Unicode License V3 notice. MIT requires the copyright notice to travel with redistributed copies, and the Unicode license requires its notice to appear with the data or in its documentation.

## Test-only (never shipped)
| Component | Version | License | Use |
|-----------|---------|---------|-----|
| [doctest](https://github.com/doctest/doctest) | v2.5.3 | MIT | Unit-test framework |

## Build-time and development tools (never shipped)
| Tool | Version | License | Use |
|------|---------|---------|-----|
| [glslang](https://github.com/KhronosGroup/glslang) | vulkan-sdk-1.4.363.0 | BSD-3-Clause, BSD-2-Clause, MIT, Apache-2.0; its generated parser is GPL-3.0-or-later with the Bison exception | Compiles GLSL shaders to SPIR-V at build time. The Bison exception covers the parser output, and none of glslang's code ends up in hearth.exe. |
| [Vulkan-ValidationLayers](https://github.com/KhronosGroup/Vulkan-ValidationLayers) | vulkan-sdk-1.4.363.1 | Apache-2.0 | GPU correctness checking in tests |
| Its fetched dependencies: SPIRV-Tools, SPIRV-Headers, Vulkan-Utility-Libraries, Vulkan-Loader, Vulkan-Profiles, mimalloc, googletest | commits from the layer's `known_good.json` | Apache-2.0, MIT, BSD-3-Clause, BSD-2-Clause, BSL-1.0, CC-BY-4.0 (documentation) | Built only so the validation layer can be built |
| [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) | b11960 (= v0.8.0) | MIT | Correctness oracle and performance baselines: the official release zips, and `hetero-oracle-dump` and `hetero-oracle-chat` (`tools/oracle/`), built from the pinned source and linked statically. All are separate programs; llama.cpp is never linked into hearth. |
| llama.cpp's vendored libraries, compiled into `hetero-oracle-chat` through llama.cpp's `llama-common` library | as pinned in llama.cpp b11960's `vendor/` | nlohmann/json: MIT; cpp-httplib: MIT (linked by `llama-common`, built without SSL, never called by the oracle); sheredom/subprocess.h: public domain (Unlicense) | JSON parsing and llama.cpp's own chat-template rendering in the oracle. Not part of hearth. |
| gguf-py, from the same llama.cpp checkout | b11960 | MIT | Independent GGUF reader for cross-checks |
| MSVC Build Tools 14.44 + Windows SDK 10.0.22621 | 17.14 | Microsoft Visual Studio Build Tools license (free for individuals, as under Visual Studio Community terms) | Product compiler |
| clang 23 + clang-tidy + clang-format, from llvm-mingw | 23.1.3 | Apache-2.0 with LLVM exception; mingw-w64 runtime: ZPL and public domain | Sanitizer builds, lint and formatting |
| CMake | 4.0.3 | BSD-3-Clause | Build generator |
| Ninja | 1.13.2 | Apache-2.0 | Build executor |
| Git for Windows | 2.51 | GPL-2.0 | Version control (a tool only) |
| Python | 3.12.8 | PSF-2.0 | Tooling, analysis and charts |

## Python packages (.venv, hash-pinned in requirements.txt)
| Package | License |
|---------|---------|
| numpy | BSD-3-Clause |
| PyYAML | MIT |
| matplotlib | Matplotlib License (PSF-based, BSD-compatible) |
| contourpy, cycler, kiwisolver | BSD-3-Clause |
| fonttools, pyparsing | MIT |
| pillow | MIT-CMU (HPND) |
| packaging | Apache-2.0 OR BSD-2-Clause |
| python-dateutil | Apache-2.0 AND BSD-3-Clause |
| tzdata | Apache-2.0 |

## Models and data (downloaded by scripts, never redistributed by this project)
| Item | Source | License |
|------|--------|---------|
| Llama-3.2-1B-Instruct and Llama-3.2-3B-Instruct (GGUF BF16, and the derived F32/Q8_0/Q4_0) | `unsloth/*-GGUF` on Hugging Face (ungated mirror), pinned revision | **Llama 3.2 Community License.** Free of charge, with Meta's Acceptable Use Policy. Anyone redistributing the model must show "Built with Llama". The license also has a 700M-monthly-users clause. |
| Qwen2.5-0.5B-Instruct (GGUF F16, and the derived F32/Q8_0/Q4_0) | `Qwen/Qwen2.5-0.5B-Instruct-GGUF` on Hugging Face (Qwen's own repository, ungated), pinned revision | **Apache-2.0** (the model card's license). Free, including commercial use; keep the license and notices when redistributing.|
| WikiText-2 (raw) | `ggml-org/ci` dataset on Hugging Face, pinned revision | CC BY-SA 3.0 (Merity et al., 2016). Used only for perplexity evaluation. |
| Unicode Character Database 15.1.0: `UnicodeData.txt`, `PropList.txt` | `https://www.unicode.org/Public/15.1.0/ucd/`, pinned by sha256 | **Unicode License V3**, free of charge. `tools/gen_unicode_tables.py` derives the tokenizer's letter, number and White_Space tables from these files; the generated `src/tokenizer/unicode_tables.cpp` is committed. The license requires its copyright and permission notice to appear with the data or in its documentation; it is reproduced below. |

### Unicode License V3 (for the tables derived from the Unicode Character Database)
Copied verbatim from `https://www.unicode.org/license.txt` on 2026-11-04.
```
UNICODE LICENSE V3

COPYRIGHT AND PERMISSION NOTICE

Copyright © 1991-2026 Unicode, Inc.

NOTICE TO USER: Carefully read the following legal agreement. BY
DOWNLOADING, INSTALLING, COPYING OR OTHERWISE USING DATA FILES, AND/OR
SOFTWARE, YOU UNEQUIVOCALLY ACCEPT, AND AGREE TO BE BOUND BY, ALL OF THE
TERMS AND CONDITIONS OF THIS AGREEMENT. IF YOU DO NOT AGREE, DO NOT
DOWNLOAD, INSTALL, COPY, DISTRIBUTE OR USE THE DATA FILES OR SOFTWARE.

Permission is hereby granted, free of charge, to any person obtaining a
copy of data files and any associated documentation (the "Data Files") or
software and any associated documentation (the "Software") to deal in the
Data Files or Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, and/or sell
copies of the Data Files or Software, and to permit persons to whom the
Data Files or Software are furnished to do so, provided that either (a)
this copyright and permission notice appear with all copies of the Data
Files or Software, or (b) this copyright and permission notice appear in
associated Documentation.

THE DATA FILES AND SOFTWARE ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY
KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT OF
THIRD PARTY RIGHTS.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR HOLDERS INCLUDED IN THIS NOTICE
BE LIABLE FOR ANY CLAIM, OR ANY SPECIAL INDIRECT OR CONSEQUENTIAL DAMAGES,
OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,
ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THE DATA
FILES OR SOFTWARE.

Except as contained in this notice, the name of a copyright holder shall
not be used in advertising or otherwise to promote the sale, use or other
dealings in these Data Files or Software without prior written
authorization of the copyright holder.
```
