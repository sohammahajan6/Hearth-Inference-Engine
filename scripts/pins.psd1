# Every external input this project uses, pinned by content hash or commit.
# Scripts read this file; nothing is downloaded from a floating "latest".
# Changing a pin is a deliberate act: update the value here, re-run the
# fetch script, and record the change in CHANGELOG.md (and an ADR if it
# changes behaviour or tolerances).
@{
    # llama.cpp: the oracle (source build) and the two performance baselines
    # (official release zips). b11960 is the build behind stable release v0.8.0.
    LlamaCpp = @{
        Url    = 'https://github.com/ggml-org/llama.cpp.git'
        Tag    = 'b11960'
        Commit = '8642541b255bfb5dc39cd351a178be09013a55d6'
        Zips   = @(
            @{
                Name   = 'llama-b11960-bin-win-cpu-x64.zip'
                Dir    = 'llama-bin-cpu'
                Sha256 = '73f962f82850d244db47eaacb9d913637027b82399124324298576d58c5f4287'
            }
            @{
                Name   = 'llama-b11960-bin-win-vulkan-x64.zip'
                Dir    = 'llama-bin-vulkan'
                Sha256 = 'bac6914b5e83523eb008c6789eaa12669f93fbc196c3b9e8237b6672323b33e9'
            }
        )
        ReleaseBase = 'https://github.com/ggml-org/llama.cpp/releases/download/b11960'
    }

    # Source dependencies, cloned at a tag and verified against the commit.
    Git = @(
        @{ Name = 'Vulkan-Headers'; Url = 'https://github.com/KhronosGroup/Vulkan-Headers.git'
           Tag = 'vulkan-sdk-1.4.363.0'; Commit = '6802bb4733b63ed5efd3adb308a6c885ef180ea1'
           License = 'Apache-2.0 OR MIT' }
        @{ Name = 'volk'; Url = 'https://github.com/zeux/volk.git'
           Tag = 'vulkan-sdk-1.4.363.0'; Commit = '54fc0d7a72887cbfc5b030d26897b2b40ac53fb8'
           License = 'MIT' }
        @{ Name = 'glslang'; Url = 'https://github.com/KhronosGroup/glslang.git'
           Tag = 'vulkan-sdk-1.4.363.0'; Commit = 'e1b562a8bed273a02f30b59b66a5d499793cede5'
           License = 'BSD-3-Clause and others (see its LICENSE.txt)' }
        @{ Name = 'Vulkan-ValidationLayers'; Url = 'https://github.com/KhronosGroup/Vulkan-ValidationLayers.git'
           Tag = 'vulkan-sdk-1.4.363.1'; Commit = '317cff060893783f96d4174f50491e6374eb3714'
           License = 'Apache-2.0' }
        @{ Name = 'doctest'; Url = 'https://github.com/doctest/doctest.git'
           Tag = 'v2.5.3'; Commit = '2d0a9359a60c51affe2a9bebb1be1dca47868151'
           License = 'MIT' }
    )

    # Models: ungated Hugging Face mirrors at a fixed revision. sha256 values
    # come from the Hugging Face API (the LFS object id is the file's sha256).
    Models = @(
        @{ Set = '1b'; Repo = 'unsloth/Llama-3.2-1B-Instruct-GGUF'
           Revision = 'b69aef112e9f895e6f98d7ae0949f72ff09aa401'
           File = 'Llama-3.2-1B-Instruct-BF16.gguf'
           Sha256 = 'f5d05e90d179fa27c01a049be1574a0201bdc09ba0b345eb85e1dbf3b87610f3' }
        @{ Set = '3b'; Repo = 'unsloth/Llama-3.2-3B-Instruct-GGUF'
           Revision = 'e7d0997e49c9cb00d88b4c1a6a16aa894b0bbc31'
           File = 'Llama-3.2-3B-Instruct-BF16.gguf'
           Sha256 = '9b8dce13b6cbcd8b20037bd6383ee8e747b5034ca32f40b5b8ee2efa9ebf56b8' }
        # Qwen:
        # Qwen's own ungated GGUF repository, the F16 file, saved as Local.
        @{ Set = 'qwen0.5b'; Repo = 'Qwen/Qwen2.5-0.5B-Instruct-GGUF'
           Revision = '9217f5db79a29953eb74d5343926648285ec7e67'
           File = 'qwen2.5-0.5b-instruct-fp16.gguf'; Local = 'Qwen2.5-0.5B-Instruct-F16.gguf'
           Sha256 = '8e0ae26000627ed62de0e78e41860af70094558b9d2913385c842a6aa06cf3fc' }
    )

    # Files derived locally by scripts\fetch-models.ps1 (llama-quantize --pure
    # from the pinned llama.cpp release). Recorded on first derivation
    # (2026-11-03); a mismatch on re-derivation means the quantizer is not
    # reproducible and must be investigated before any comparison is trusted.
    Derived = @{
        'Llama-3.2-1B-Instruct-F32-pure.gguf'  = '61839d5dfb84b8f57b880ddcd12d2d14f4d5277af9d87e980e4c714b44218dc0'
        'Llama-3.2-1B-Instruct-Q8_0-pure.gguf' = '5cadbba719694451b37bcf883aede2f6ed24f585cac15b7e28dd0e568bba0476'
        'Llama-3.2-1B-Instruct-Q4_0-pure.gguf' = '7268cd950902931966e24e6f4e493e45b11a7204746852a1b19e7cb4b752437e'
        # llama-quantize's Q4_K_M recipe (not --pure), first derived 2026-11-13;
        # two runs gave the same bytes.
        'Llama-3.2-1B-Instruct-Q4_K_M.gguf'    = '717810cca515d7b632bb2816129d15873f7d3b737f1dec4a9993a27d116b38c6'
        # 3B set, first derived 2026-11-06.
        'Llama-3.2-3B-Instruct-Q8_0-pure.gguf' = '4378ff9c03a7adf707f500ad837e7214129218f2316ddb41c23d367d84e27a88'
        'Llama-3.2-3B-Instruct-Q4_0-pure.gguf' = 'dc269781e84f7ce67150329d92aa94e0fa3005b11db9eaf305a178e411d68b14'
        # Qwen set, first derived 2026-11-13 (Step Q1).
        'Qwen2.5-0.5B-Instruct-F32-pure.gguf'  = '5580787a2a366d26e4d7e468f601ff42917cb7e94da591f36cf30d22a73f6f5c'
        'Qwen2.5-0.5B-Instruct-Q8_0-pure.gguf' = '51b344f6dbf12a9e0d7557453f1b115e72277727231b3198dc2f3e5331fb7616'
        'Qwen2.5-0.5B-Instruct-Q4_0-pure.gguf' = 'fbf0f0348be09e5b163d25de24c8a8d499d708795eb6a64b8c3f5774428a0513'
    }

    # Unicode Character Database files for the tokenizer's \p{L}, \p{N} and
    # White_Space tables (tools/gen_unicode_tables.py). 15.1.0 is the version
    # whose general categories reproduce llama.cpp b11960's table exactly,
    # all 1,114,112 codepoints.
    Unicode = @{
        Version = '15.1.0'
        Base    = 'https://www.unicode.org/Public/15.1.0/ucd'
        Files   = @(
            @{ Name = 'UnicodeData.txt'; Sha256 = '2fc713e6a31a87c4850a37fe2caffa4218180fadb5de86b43a143ddb4581fb86' }
            @{ Name = 'PropList.txt'; Sha256 = '05672956317b6296bc2ec3d6cef1f6452b57ff4f2efc6dc55b0a19277d5fcfd1' }
        )
    }

    # Evaluation text, from the same dataset llama.cpp's own CI uses.
    Wikitext = @{
        Repo     = 'ggml-org/ci'
        Revision = '927b3642933080f1b0e811e2f916e14c292992f9'
        File     = 'wikitext-2-raw-v1.zip'
        Sha256   = 'ef7edb566e3e2b2d31b29c1fdb0c89a4cc683597484c3dc2517919c615435a11'
    }
}
