"""Check that no function is shared between an ISA-specific object and the rest.

Some translation units are compiled with AVX-512 or AVX2 flags: the
per-ISA kernels (src/cpu/*_avx512.cpp, *_avx2.cpp) and the tests' hardware
reference (tests/hw_reference_avx2.cpp). They are found from the build's
compile_commands.json: every object whose command has an /arch: flag.

Any function such an object defines with external linkage that another
object file also defines is necessarily an inline or template
function (two ordinary definitions would not link). The linker keeps one copy
of such a function for the whole program and may pick the ISA object's, which
a CPU without that ISA cannot run. This tool lists the
external function definitions of every object with `dumpbin /symbols` and
fails if any is defined both in an ISA object and in another object.

    python -I tools/check_isa_symbols.py BUILD_DIR --dumpbin PATH_TO_DUMPBIN

Exit code 0: no shared function; 1: shared functions found (listed); 2: usage
or tool error.
"""

import argparse
import collections
import json
import pathlib
import re
import subprocess
import sys

# One symbol-table line of a function defined in a section of this object:
#   039 00000000 SECT1A notype ()    External     | ?name@@... (signature)
DEF = re.compile(r"^\s*[0-9A-F]+ [0-9A-F]+ SECT[0-9A-F]+\s+notype \(\)\s+External\s+\| (\S+)(?: \((.*)\))?\s*$")


def defined_functions(dumpbin: str, obj: pathlib.Path) -> dict[str, str]:
    out = subprocess.run([dumpbin, "/nologo", "/symbols", str(obj)], capture_output=True, text=True, errors="replace")
    if out.returncode != 0:
        raise RuntimeError(f"dumpbin failed on {obj}: {out.stderr.strip() or out.stdout.strip()}")
    found = {}
    for line in out.stdout.splitlines():
        m = DEF.match(line)
        if m:
            found[m.group(1)] = m.group(2) or m.group(1)
    return found


def isa_objects(build_dir: pathlib.Path) -> set[pathlib.Path]:
    db = build_dir / "compile_commands.json"
    if not db.exists():
        raise RuntimeError(f"{db} not found (CMAKE_EXPORT_COMPILE_COMMANDS)")
    out = set()
    for e in json.loads(db.read_text(encoding="utf-8")):
        if "/arch:" in e["command"]:
            out.add((build_dir / e["output"]).resolve())
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("build_dir")
    ap.add_argument("--dumpbin", required=True)
    args = ap.parse_args()
    build = pathlib.Path(args.build_dir)
    objs = sorted(o.resolve() for o in build.rglob("*.obj"))
    isa_set = isa_objects(build)
    isa_objs = [o for o in objs if o in isa_set]
    if not isa_objs:
        print(f"check_isa_symbols: no ISA objects under {args.build_dir}", file=sys.stderr)
        return 2
    where = collections.defaultdict(set)
    names = {}
    for o in objs:
        for sym, sig in defined_functions(args.dumpbin, o).items():
            where[sym].add(o)
            names[sym] = sig
    shared = []
    for sym, files in sorted(where.items()):
        isa = files & isa_set
        other = files - isa
        if isa and other:
            shared.append((names[sym], sorted(f.name for f in isa), sorted(f.name for f in other)))
    isa_defs = sum(1 for files in where.values() if files & isa_set)
    print(f"check_isa_symbols: {len(objs)} objects, {len(isa_objs)} ISA objects defining {isa_defs} external functions")
    for sig, isa, other in shared:
        print(f"shared: {sig}\n  ISA objects:   {', '.join(isa)}\n  other objects: {', '.join(other)}")
    if shared:
        print(f"check_isa_symbols: {len(shared)} function(s) shared with ISA objects")
        return 1
    print("check_isa_symbols: no function is shared between an ISA object and another object")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RuntimeError as e:
        print(f"check_isa_symbols: {e}", file=sys.stderr)
        sys.exit(2)
