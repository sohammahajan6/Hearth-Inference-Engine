"""Write tests/fixtures/tokenizer/adversarial.txt: text aimed at every rule of
the Llama-3 pre-tokenizer and at the BPE.
scripts/tokenizer-check.ps1 compares hearth's token IDs on it with llama-tokenize's.

    .venv\\Scripts\\python -I tools\\make_tokenizer_corpus.py

Deterministic: the same file every time. Valid UTF-8, with CR LF, CR and LF
line endings mixed on purpose.
"""

import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "tests" / "fixtures" / "tokenizer" / "adversarial.txt"

parts = []


def add(*items):
    parts.extend(items)


# Contractions, in every case combination, and near misses.
for c in ["s", "t", "re", "ve", "m", "ll", "d"]:
    variants = {c, c.upper(), c.capitalize(), c[:-1] + c[-1].upper() if len(c) > 1 else c.upper()}
    for v in sorted(variants):
        add(f"it'{v} x'{v}y '{v} {v}'{v}\n")
add("don't DON'T Don'T rock'n'roll y'all o'clock I'M we'Re they'VE ''s '' ' 'x '1 '\t's\n")
add("ends with an apostrophe'")
add("\n")
# Words with every kind of prefix character.
add("(word _word $word \u2013word \u00abword\u00bb \tword \nword \rword  word\u00a0word \u2003word 1word\n")
# Numbers: ASCII, other scripts, superscripts, fractions, Roman numerals.
add("1 12 123 1234 12345 1234567 123456789012345 0.5 3,141,592 007 2024-11-04\n")
add("\u0661\u0662\u0663\u0664 \u0967\u0968\u0969 \u00b2\u00b3 \u00bd \u2167 \u216b \u2460\u2461 \uff11\uff12\uff13\n")
# Whitespace: runs, mixes, Unicode spaces, line endings, at line ends.
add("a  b   c    d\te\t\tf \t g\t \th\n")
add("crlf line\r\nanother\r\n\r\nlone cr\rafter\n\n\n  \n \r\n\t\n")
add("nbsp\u00a0x em\u2003x ideo\u3000x nel\u0085x ls\u2028x ps\u2029x ogham\u1680x thin\u2009x\n")
add("trailing spaces   \ntrailing tab\t\n  leading spaces\n")
add(" " * 37 + "after 37 spaces\n")
# Punctuation runs and special-token text, which must stay ordinary text.
add("!!! ... ?! -- --- \u2014 \u00ab\u00bb () [] {} <> !!\n\n !!\r\n ?\n")
add("<|eot_id|> <|begin_of_text|><|start_header_id|>user<|end_header_id|> <|reserved_special_token_0|>\n")
add("C:\\path\\to\\file.txt \\n \\t \\\\ \"quoted\" 'single' `tick` @user #tag 50% $5 a+b=c a/b a|b\n")
# Scripts: CJK, kana, Hangul, Arabic, Hebrew, Devanagari (combining marks), Thai, Greek, Cyrillic.
add("\u4e2d\u6587\u6d4b\u8bd5 \u3072\u3089\u304c\u306a\u30ab\u30bf\u30ab\u30ca \ud55c\uad6d\uc5b4 "
    "\u0627\u0644\u0639\u0631\u0628\u064a\u0629 \u05e2\u05d1\u05e8\u05d9\u05ea "
    "\u0939\u093f\u0928\u094d\u0926\u0940 \u0e20\u0e32\u0e29\u0e32\u0e44\u0e17\u0e22 "
    "\u03ac\u03bb\u03c6\u03b1 \u043f\u0440\u0438\u0432\u0435\u0442\n")
# Combining marks after letters and alone; decomposed accents.
add("e\u0301t\u0301e\u0301 caf\u00e9 cafe\u0301 \u0301alone a\u0308\u0304 n\u0303\n")
# Emoji: plain, skin tones, ZWJ families, flags, keycaps, variation selectors.
add("\U0001F44D \U0001F44D\U0001F3FD \U0001F468\u200d\U0001F469\u200d\U0001F467 \U0001F1EB\U0001F1F7 "
    "1\ufe0f\u20e3 \u2764\ufe0f text\U0001F600text\n")
# Mathematical alphanumerics (letters and digits outside the BMP).
add("\U0001D400\U0001D401\U0001D402 \U0001D7CE\U0001D7CF \U0001D49C\U0001D4B7\n")
# Zero-width and format characters, a BOM in the middle, private use, unassigned.
add("zero\u200bwidth zw\u200dj soft\u00adhyphen mid\ufeffbom pua\ue000\uf8ff u0378:\u0378 "
    "plane2:\U0002ffff tag\U000e0041\n")
# Characters that became letters or marks in Unicode 16.0 and later: llama.cpp
# b11960 (Unicode 15.1 classes) treats them as unassigned.
add("pepet\u0897x garay\U00010d50\U00010d51x kirat\U00016d43x\n")
# Long runs for the BPE: one very long word, a long digit run, a long punctuation run.
add("pneumonoultramicroscopicsilicovolcanoconiosis" * 20 + "\n")
add("9" * 1000 + "\n")
add("!?" * 300 + "\n")
add("x" * 2000 + " " + "ab" * 500 + "\n")
# Mixed case and alphanumerics.
add("CamelCaseWord snake_case_word kebab-case-word ALLCAPS MiXeD123abc abc123DEF 3D 4K4k\n")
add("The capital of France is Paris. The year 1776; the value 3.14159; e-mail: a.b@c.de\n")
# The very end: whitespace that is not followed by anything.
add("final words  \t ")

text = "".join(parts)
text.encode("utf-8")  # all of it must be valid UTF-8
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_bytes(text.encode("utf-8"))
print(f"wrote {OUT}: {len(text)} code points, {len(text.encode('utf-8'))} bytes")

# The tokenizer fuzzer's seeds (tests/fuzz/tokenizer_target.h): one stride
# byte, then text. Chunks of the corpus cut at line ends, plus invalid UTF-8.
SEEDS = ROOT / "tests" / "fuzz" / "seeds" / "tokenizer"
SEEDS.mkdir(parents=True, exist_ok=True)
for old in SEEDS.glob("*.bin"):
    old.unlink()
lines = text.encode("utf-8").split(b"\n")
chunk, chunks = b"", []
for line in lines:
    chunk += line + b"\n"
    if len(chunk) > 900:
        chunks.append(chunk)
        chunk = b""
chunks.append(chunk)
for i, c in enumerate(chunks):
    (SEEDS / f"corpus-{i:02d}.bin").write_bytes(bytes([i]) + c)
invalid = {
    "truncated-2byte": b"ok\xc3",
    "truncated-4byte-then-ascii": b"\xf0\x9f\x99x",
    "lone-continuations": b"\x80\x80 a",
    "surrogate": b"x\xed\xa0\x80y",
    "overlong-slash": b"\xc0\xaf",
}
for name, b in invalid.items():
    (SEEDS / f"invalid-{name}.bin").write_bytes(b"\x02" + b)
print(f"wrote {len(chunks) + len(invalid)} seeds to {SEEDS}")
