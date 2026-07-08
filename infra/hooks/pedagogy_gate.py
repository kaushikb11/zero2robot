#!/usr/bin/env python3
"""PreToolUse hook: mechanical enforcement of the pedagogical code doctrine.

Reads the tool call JSON from stdin; exit 2 blocks with a message.
Notes:
- LOC cap is only enforced on FULL-FILE writes (Write/create). Partial edits
  (str_replace) are LOC-checked in CI (check_loc_caps.py on the resulting file),
  because a fragment's line count says nothing about the file's.
- 'transformers' is forbidden EXCEPT where the chapter's meta.yaml grants
  `allow_transformers: true` (tiny-VLA chapters). The hook checks the sibling
  meta.yaml; CI re-verifies.
"""
import ast
import json
import os
import re
import sys

FORBIDDEN_ALWAYS = ["hydra", "omegaconf", "pytorch_lightning", "stable_baselines3", "gymnasium", "gym"]
FORBIDDEN_UNLESS_GRANTED = ["transformers"]
CAP = 450

def imported_top_modules(content):
    """Top-level module names imported in `content`.

    Uses ast so `import a, gym` / `import a as g` / `import a; import hydra`
    are all caught. `content` may be an Edit fragment that doesn't parse — then
    fall back to a per-statement scan (split on ';' and ',') so nothing slips."""
    try:
        tree = ast.parse(content)
    except SyntaxError:
        mods = set()
        for line in content.splitlines():
            for stmt in line.split(";"):
                s = stmt.strip()
                if s.startswith("from "):
                    m = re.match(r"from\s+([\w.]+)", s)
                    if m:
                        mods.add(m.group(1).split(".")[0])
                elif s.startswith("import "):
                    for part in s[len("import "):].split(","):
                        tok = part.strip().split(" as ")[0].strip()
                        if tok:
                            mods.add(tok.split(".")[0])
        return mods
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mods.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module.split(".")[0])
    return mods

def meta_grants(path, key):
    meta = os.path.join(os.path.dirname(path), "meta.yaml")
    try:
        with open(meta) as f:
            meta_text = f.read()
    except OSError:
        return False
    # Line-anchored (stdlib only — hook can't import yaml): a bare
    # `allow_transformers: true` KEY, not a substring inside a YAML comment.
    return bool(re.search(rf"^\s*{re.escape(key)}:\s*true\s*$", meta_text, re.M))

def main():
    data = json.load(sys.stdin)
    tool = data.get("tool_name", "")
    ti = data.get("tool_input", {})
    # Write sends file_path+content; Edit sends file_path+new_string;
    # NotebookEdit sends notebook_path (+new_source).
    path = ti.get("file_path", "") or ""
    notebook_path = ti.get("notebook_path", "") or ""
    content = ti.get("content") or ti.get("new_string") or ti.get("new_str") or ""

    # Protected-path block: check both file_path (Edit/Write) and notebook_path
    # (NotebookEdit) so notebooks/ and grader/hidden_seeds can't be reached via
    # any tool.
    for p in (path, notebook_path):
        if p and ("grader/hidden_seeds" in p or "/notebooks/" in p or p.startswith("notebooks/")):
            print(f"BLOCKED: {p} is protected (generated or secret).", file=sys.stderr)
            sys.exit(2)

    # Chapter code = the artifact plus any .py under the chapter dir
    # (exercises/**, demo/**) EXCEPT human-owned tests/. Filenames may contain
    # digits (e.g. rl_v2.py, exercises/ex1_gym.py).
    is_chapter = (
        re.search(r"curriculum/phase\d[^/]*/ch[\d.]+[^/]*/.*[a-z0-9_]+\.py$", path)
        and "/tests/" not in path
    )
    if is_chapter:
        mods = imported_top_modules(content)
        for mod in FORBIDDEN_ALWAYS:
            if mod in mods:
                print(f"BLOCKED: '{mod}' is forbidden in chapter code (pedagogical doctrine). See curriculum/CLAUDE.md.", file=sys.stderr)
                sys.exit(2)
        for mod in FORBIDDEN_UNLESS_GRANTED:
            if mod in mods and not meta_grants(path, "allow_transformers"):
                print(f"BLOCKED: '{mod}' requires `allow_transformers: true` in this chapter's meta.yaml (tiny-VLA chapters only).", file=sys.stderr)
                sys.exit(2)
        # LOC cap: only meaningful when the tool writes the whole file
        if tool in ("Write", "create_file") and content.count("\n") + 1 > CAP:
            print(f"BLOCKED: chapter artifact exceeds {CAP} LOC hard cap (target <=400). Simplify - splitting is not allowed.", file=sys.stderr)
            sys.exit(2)
    sys.exit(0)

if __name__ == "__main__":
    main()
