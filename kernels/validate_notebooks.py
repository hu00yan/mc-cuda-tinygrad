#!/usr/bin/env python3
"""validate_notebooks.py -- everything that can be checked about a kernel
directory without a token or a network round trip.

Checks, per kernel directory (any dir under kernels/ with a kernel-metadata.json):
  * kernel-metadata.json parses, has the fields the CLI reads, and its id/title
    agree (the CLI warns when title does not slugify to the id's slug)
  * code_file exists inside the directory
  * the .ipynb parses as JSON, is nbformat 4, has a kernelspec
  * EVERY code cell's source compiles as Python  (this is the one that catches
    a notebook that will not run at all)
  * code cells that call ! shell escapes and %magics are reported, because the
    Kaggle runner accepts them but `compile()` does not, so a cell using them
    is skipped by the check above rather than silently trusted

Usage:  .venv/bin/python kernels/validate_notebooks.py [dir ...]
Exit 0 if everything is valid, 1 otherwise.
"""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
NEEDED = ("id", "title", "code_file", "language", "kernel_type")


def fail(msg, errs):
    print(f"  FAIL  {msg}")
    errs.append(msg)


def ok(msg):
    print(f"  ok    {msg}")


def check_metadata(kdir, errs):
    path = os.path.join(kdir, "kernel-metadata.json")
    if not os.path.isfile(path):
        fail("kernel-metadata.json missing", errs)
        return None
    try:
        with open(path) as f:
            meta = json.load(f)
    except Exception as e:
        fail(f"kernel-metadata.json does not parse: {type(e).__name__}: {e}", errs)
        return None
    for key in NEEDED:
        if key not in meta:
            fail(f"metadata has no {key!r}", errs)
    ok(f"metadata parses, keys {sorted(meta)}")
    slug = str(meta.get("id", ""))
    title = str(meta.get("title", ""))
    want = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    have = slug.split("/")[-1]
    if want and want != have:
        fail(f"title {title!r} slugifies to {want!r} but id slug is {have!r} "
             f"(the CLI warns on this; slug is immutable once created)", errs)
    else:
        ok(f"title resolves to slug {have!r}")
    acc = []
    if meta.get("enable_gpu"):
        acc.append(f"GPU shape={meta.get('machine_shape') or 'default'}")
    if meta.get("enable_tpu"):
        acc.append("TPU")
    if not acc:
        acc.append("CPU only")
    ok(f"accelerator: {', '.join(acc)}  internet={meta.get('enable_internet')}")
    return meta


def check_notebook(kdir, meta, errs):
    if not meta:
        return
    code = meta.get("code_file", "")
    path = os.path.join(kdir, code)
    if not os.path.isfile(path):
        fail(f"code_file {code!r} not found in {kdir}", errs)
        return
    with open(path) as f:
        raw = f.read()
    try:
        nb = json.loads(raw)
    except Exception as e:
        fail(f"{code} is not valid JSON: {type(e).__name__}: {e}", errs)
        return
    if nb.get("nbformat") != 4:
        fail(f"{code} nbformat is {nb.get('nbformat')!r}, want 4", errs)
    else:
        ok(f"{code} parses, nbformat 4.{nb.get('nbformat_minor')}")
    if "kernelspec" not in nb.get("metadata", {}):
        fail(f"{code} has no kernelspec in metadata", errs)
    cells = nb.get("cells")
    if not isinstance(cells, list) or not cells:
        fail(f"{code} has no cells", errs)
        return
    ncode = 0
    for i, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        ncode += 1
        raw_src = cell.get("source", "")
        if isinstance(raw_src, list):
            # `kaggle kernels push` sends "".join(source) to the runner, so a
            # list whose elements do not carry their own newlines becomes one
            # long line and the notebook dies on cell 1. Check it explicitly.
            bad = [j for j, ln in enumerate(raw_src)
                   if j < len(raw_src) - 1 and not ln.endswith("\n")]
            if bad:
                fail(f"{code} cell {i}: source list entries {bad} have no trailing "
                     f"newline; the CLI joins these with \"\" and the cell will "
                     f"be one line on the runner", errs)
        src = "".join(raw_src) if isinstance(raw_src, list) else raw_src
        if not src.strip():
            fail(f"{code} cell {i} is empty", errs)
            continue
        # `!cmd` and `%magic` lines are valid notebook source that the Kaggle
        # runner understands but python's compiler does not.
        stripped = "\n".join("" if re.match(r"^\s*[!%]", ln) else ln
                             for ln in src.split("\n")).strip()
        if not stripped:
            ok(f"cell {i}: shell-escape only ({len(src.splitlines())} lines)")
            continue
        try:
            compile(src, f"{code}:cell{i}", "exec")
        except SyntaxError as e:
            fail(f"{code} cell {i} is not valid Python: line {e.lineno}: {e.msg}", errs)
        else:
            ok(f"cell {i}: compiles ({len(src.splitlines())} lines)")
    ok(f"{code}: {len(cells)} cells, {ncode} code")


def main(argv):
    dirs = argv[1:]
    if not dirs:
        root = ROOT
        dirs = sorted(
            os.path.join(root, d) for d in os.listdir(root)
            if os.path.isfile(os.path.join(root, d, "kernel-metadata.json")))
    errs = []
    for d in dirs:
        print(f"--- {os.path.relpath(d, os.path.dirname(ROOT))}")
        meta = check_metadata(d, errs)
        check_notebook(d, meta, errs)
    print()
    if errs:
        print(f"{len(errs)} problem(s)")
        return 1
    print(f"all {len(dirs)} kernel director(ies) valid")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
