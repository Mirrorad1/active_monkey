#!/usr/bin/env python3
"""Generate ../math-data.js from the math/*.md reference files.

Single source of truth: the markdown files in this directory. This script parses
each file into structured concept records (Glossary / Example / Data / Programmer
parts kept as raw markdown) and emits `window.AM_MATH` for math.html to render
client-side with `marked`. Pure stdlib; run from anywhere:

    python3 math/build_math_data.py

It writes <repo-root>/math-data.js. Re-run whenever you edit a math/*.md file.
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "math-data.js")

# file -> (short label for the topic chip, track grouping)
FILE_META = {
    "00-orientation.md":                       ("Orientation",       "map"),
    "01-free-energy-and-active-inference.md":  ("Free energy",       "active-inference"),
    "02-bayesian-inference-and-learning.md":   ("Bayesian learning", "active-inference"),
    "03-information-theory.md":                ("Information theory", "foundations"),
    "04-probability-and-distributions.md":     ("Probability",       "foundations"),
    "05-evolutionary-dynamics.md":             ("Evolution",         "ecology"),
    "06-control-and-dynamical-systems.md":     ("Control & dynamics","control"),
    "07-statistics-and-experimental-method.md":("Statistics",        "foundations"),
}

# the four part markers, in order. value = key in the emitted record.
PART_MARKERS = [
    (re.compile(r"^\*\*Glossary\b.*?\*\*", re.S),  "glossary"),
    (re.compile(r"^\*\*Example\b.*?\*\*", re.S),   "example"),
    (re.compile(r"^\*\*Data\b.*?\*\*", re.S),      "data"),
    (re.compile(r"^\*\*▸ In programmer terms\b.*?\*\*", re.S), "programmer"),
]


def slugify(text):
    s = re.sub(r"[`*_$()/]", "", text.lower())
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s or "section"


def split_h2_sections(body):
    """Return list of (heading, section_body) split on lines starting with '## '."""
    sections = []
    cur_head = None
    cur_lines = []
    for line in body.splitlines():
        m = re.match(r"^##\s+(.*)$", line)
        if m:
            if cur_head is not None:
                sections.append((cur_head, "\n".join(cur_lines)))
            cur_head = m.group(1).strip()
            cur_lines = []
        elif cur_head is not None:
            cur_lines.append(line)
    if cur_head is not None:
        sections.append((cur_head, "\n".join(cur_lines)))
    return sections


def strip_marker(text):
    """Remove the leading bold marker (**Glossary.** etc.) from a part."""
    t = text.lstrip()
    m = re.match(r"^\*\*[^*]*\*\*\s*", t)
    if m:
        t = t[m.end():]
    return t.strip()


def parse_parts(section_body):
    """Split a concept section into the four parts by their markers."""
    # find marker positions in order
    positions = []  # (start_index, key)
    for marker_re, key in PART_MARKERS:
        # marker must appear at the start of a line
        pat = re.compile(r"(?m)^" + marker_re.pattern[1:])  # reuse but anchored multiline
        m = pat.search(section_body)
        if m:
            positions.append((m.start(), key))
    positions.sort()
    parts = {}
    for i, (start, key) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(section_body)
        chunk = section_body[start:end]
        # drop a trailing horizontal rule and surrounding whitespace
        chunk = re.sub(r"\n+---\s*$", "", chunk.strip())
        parts[key] = strip_marker(chunk)
    return parts


def extract_lead(pre_body):
    """The blockquote intro after the H1, before the first '## '."""
    quote_lines = []
    for line in pre_body.splitlines():
        s = line.strip()
        if s.startswith(">"):
            quote_lines.append(re.sub(r"^>\s?", "", line))
        elif s == "" and quote_lines:
            continue
        elif quote_lines:
            break
    return " ".join(l.strip() for l in quote_lines).strip()


def parse_file(path):
    raw = open(path, encoding="utf-8").read()
    lines = raw.splitlines()
    title = ""
    if lines and lines[0].startswith("# "):
        title = lines[0][2:].strip()
    # everything after the H1
    after_h1 = "\n".join(lines[1:])
    first_h2 = re.search(r"(?m)^##\s+", after_h1)
    pre = after_h1[: first_h2.start()] if first_h2 else after_h1
    lead = extract_lead(pre)
    concepts = []
    for heading, sec_body in split_h2_sections(after_h1):
        parts = parse_parts(sec_body)
        if not parts:
            continue
        concepts.append({
            "id": slugify(heading),
            "term": heading,
            "glossary": parts.get("glossary", ""),
            "example": parts.get("example", ""),
            "data": parts.get("data", ""),
            "programmer": parts.get("programmer", ""),
        })
    return title, lead, concepts


def main():
    files = []
    for fname in sorted(FILE_META.keys()):
        path = os.path.join(HERE, fname)
        if not os.path.exists(path):
            print("WARN missing", fname, file=sys.stderr)
            continue
        label, track = FILE_META[fname]
        title, lead, concepts = parse_file(path)
        files.append({
            "id": fname.replace(".md", ""),
            "file": fname,
            "label": label,
            "track": track,
            "title": title,
            "lead": lead,
            "concepts": concepts,
        })
    total_concepts = sum(len(f["concepts"]) for f in files)
    payload = {"files": files, "totalConcepts": total_concepts, "totalFiles": len(files)}
    header = (
        "/* math-data.js — GENERATED by math/build_math_data.py from math/*.md.\n"
        "   Do NOT edit by hand; edit the markdown and re-run the generator.\n"
        "   Single source of truth = the math/ reference files. */\n"
    )
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(header)
        fh.write("window.AM_MATH = ")
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write(";\n")
    print("wrote", OUT)
    print("files:", len(files), "concepts:", total_concepts)
    for f in files:
        print("  %-44s %2d concepts" % (f["file"], len(f["concepts"])))


if __name__ == "__main__":
    main()
