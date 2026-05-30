#!/usr/bin/env python3
"""
prism_filetag.py - sentinel-tag package completeness checker (hardened).

Tag every required file once; the tag is baked in (inline for text, sidecar
for binary) so `grep -rl PRISM-TAG /` finds it anywhere. Manifest is the
contract; verify is the proof.

Commands:
  tag       register one file
  tag-all   register every untagged file under a tree, in one pass
  snapshot  write/refresh manifest.json from currently-tagged files
  verify    scan a tree, diff vs manifest (--strict also fails on UNTRACKED)
  repair    heal manifest: adopt new locations (MOVED) + refresh hashes
  gather    copy every manifest file (located via tag) into one dir
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import uuid

SENTINEL = "PRISM-TAG"
TAG_RE = re.compile(rb"PRISM-TAG:([0-9a-f-]{36}):([^\s:]+)")
SIDECAR = ".prismtag"
COMMENT = {
    ".py": "#",
    ".sh": "#",
    ".bash": "#",
    ".rs": "//",
    ".c": "//",
    ".cpp": "//",
    ".cc": "//",
    ".h": "//",
    ".hpp": "//",
    ".js": "//",
    ".jsx": "//",
    ".ts": "//",
    ".tsx": "//",
    ".go": "//",
    ".java": "//",
    ".kt": "//",
    ".swift": "//",
    ".toml": "#",
    ".yaml": "#",
    ".yml": "#",
    ".cfg": "#",
    ".conf": "#",
    ".ini": ";",
    ".sql": "--",
    ".lua": "--",
    ".hs": "--",
    ".html": "<!--",
    ".xml": "<!--",
    ".css": "/*",
    ".scss": "/*",
    ".r": "#",
    ".jl": "#",
    ".rb": "#",
    ".pl": "#",
}
SKIP_DIRS = {
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "target",
    ".mypy_cache",
    ".pytest_cache",
    "dist",
    "build",
    ".idea",
    ".vscode",
}
SELF = os.path.realpath(__file__)


def norm(path):
    return path.replace(os.sep, "/")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 16), b""):
            digest.update(block)
    return digest.hexdigest()


def comment_wrap(body, ext):
    marker = COMMENT.get(ext)
    if marker == "<!--":
        return f"<!-- {body} -->"
    if marker == "/*":
        return f"/* {body} */"
    if marker:
        return f"{marker} {body}"
    return body


def read_tag(path):
    try:
        with open(path, "rb") as handle:
            match = TAG_RE.search(handle.read(1 << 14))
            if match:
                return match.group(1).decode(), match.group(2).decode()
            return None
    except Exception:
        return None


def inject_inline(path, ext, body):
    with open(path, "rb") as handle:
        raw = handle.read()
    newline = "\r\n" if b"\r\n" in raw else "\n"
    text = raw.decode("utf-8", "surrogateescape")
    lines = text.split(newline)
    index = 1 if lines and (
        lines[0].startswith("#!") or lines[0].lstrip().startswith("<?xml")
    ) else 0
    lines.insert(index, comment_wrap(body, ext))
    with open(
        path,
        "w",
        encoding="utf-8",
        errors="surrogateescape",
        newline="",
    ) as handle:
        handle.write(newline.join(lines))


def tag_file(path, logical_id, quiet=False):
    if read_tag(path):
        if not quiet:
            print(f"skip (already tagged): {path}")
        return False
    ext = os.path.splitext(path)[1].lower()
    tag_uuid = str(uuid.uuid4())
    name = logical_id or os.path.basename(path)
    body = f"{SENTINEL}:{tag_uuid}:{name}"
    if ext in COMMENT:
        inject_inline(path, ext, body)
        where = "inline"
    else:
        with open(path + SIDECAR, "w", encoding="utf-8") as handle:
            handle.write(body + "\n")
        where = "sidecar"
    print(f"tagged ({where}) {name} -> {tag_uuid}  {norm(path)}")
    return True


def cmd_tag(args):
    tag_file(args.path, args.id)


def cmd_tag_all(args):
    count = 0
    for dirpath, dirnames, filenames in os.walk(args.root):
        dirnames[:] = [
            entry
            for entry in dirnames
            if entry not in SKIP_DIRS
            and not os.path.islink(os.path.join(dirpath, entry))
        ]
        for filename in filenames:
            full = os.path.join(dirpath, filename)
            if os.path.islink(full):
                continue
            if filename.endswith(SIDECAR):
                continue
            if os.path.realpath(full) == SELF:
                continue
            if not args.include_hidden and filename.startswith("."):
                continue
            if args.manifest and (
                os.path.realpath(full) == os.path.realpath(args.manifest)
            ):
                continue
            if tag_file(full, None, quiet=True):
                count += 1
    print(f"\ntag-all: {count} newly tagged under {norm(args.root)}")


def scan(root):
    found = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            entry
            for entry in dirnames
            if entry not in SKIP_DIRS
            and not os.path.islink(os.path.join(dirpath, entry))
        ]
        for filename in filenames:
            full = os.path.join(dirpath, filename)
            if os.path.islink(full):
                continue
            tag = read_tag(full)
            if not tag:
                continue
            tag_uuid, name = tag
            sidecar = filename.endswith(SIDECAR)
            target = full[:-len(SIDECAR)] if sidecar else full
            found.setdefault(tag_uuid, []).append(
                (norm(os.path.relpath(target, root)), name, sidecar, target)
            )
    return found


def cmd_snapshot(args):
    found = scan(args.root)
    manifest = {}
    for tag_uuid, hits in found.items():
        rel, name, sidecar, abspath = hits[0]
        manifest[tag_uuid] = {
            "name": name,
            "rel": rel,
            "sha256": None if sidecar else sha256(abspath),
        }
    json.dump({"sentinel": SENTINEL, "files": manifest}, sys.stdout, indent=2)
    print(file=sys.stderr)
    print(f"snapshot: {len(manifest)} files", file=sys.stderr)


def cmd_verify(args):
    manifest = json.load(open(args.manifest, encoding="utf-8"))["files"]
    found = scan(args.root)
    rows = []
    clean = True
    for tag_uuid, expected in manifest.items():
        hits = found.get(tag_uuid)
        if not hits:
            rows.append(("MISSING", expected["name"], expected["rel"], "-"))
            clean = False
            continue
        if len(hits) > 1:
            rows.append(
                ("DUPLICATE", expected["name"], hits[0][0], f"{len(hits)} copies")
            )
            clean = False
            continue
        rel, name, sidecar, abspath = hits[0]
        if rel != expected["rel"]:
            rows.append(("MOVED", expected["name"], rel, f"was {expected['rel']}"))
            clean = False
            continue
        if expected["sha256"] and sha256(abspath) != expected["sha256"]:
            rows.append(("MODIFIED", expected["name"], rel, "hash differs"))
            clean = False
            continue
        rows.append(("PRESENT", expected["name"], rel, "ok"))
    for tag_uuid, hits in found.items():
        if tag_uuid not in manifest:
            rel, name, sidecar, abspath = hits[0]
            rows.append(("UNTRACKED", name, rel, "not in manifest"))
            if args.strict:
                clean = False
    width = max((len(row[1]) for row in rows), default=4)
    for status, name, location, note in sorted(rows, key=lambda row: row[0]):
        print(f"{status:<10} {name:<{width}}  {location:<40} {note}")
    present = len([row for row in rows if row[0] == "PRESENT"])
    bad = [row for row in rows if row[0] not in ("PRESENT", "UNTRACKED")] + (
        [row for row in rows if row[0] == "UNTRACKED"] if args.strict else []
    )
    strict_note = " [strict]" if args.strict else ""
    print(
        f"\n{'PASS' if clean else 'FAIL'}: {present}/{len(manifest)} present, "
        f"{len(bad)} problem(s){strict_note}"
    )
    sys.exit(0 if clean else 1)


def cmd_repair(args):
    data = json.load(open(args.manifest, encoding="utf-8"))
    manifest = data["files"]
    found = scan(args.root)
    healed = 0
    skipped = []
    for tag_uuid, expected in manifest.items():
        hits = found.get(tag_uuid)
        if not hits:
            skipped.append(("MISSING", expected["name"]))
            continue
        if len(hits) > 1:
            skipped.append(("DUPLICATE", expected["name"]))
            continue
        rel, name, sidecar, abspath = hits[0]
        new_sha = None if sidecar else sha256(abspath)
        if rel != expected["rel"] or new_sha != expected["sha256"]:
            expected["rel"] = rel
            expected["sha256"] = new_sha
            print(f"healed {expected['name']} -> {rel}")
            healed += 1
    with open(args.manifest, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
    for status, name in skipped:
        print(f"left {status}: {name} (fix by hand)")
    print(f"\nrepair: {healed} healed, {len(skipped)} unresolved -> {args.manifest}")
    sys.exit(1 if skipped else 0)


def cmd_gather(args):
    manifest = json.load(open(args.manifest, encoding="utf-8"))["files"]
    found = scan(args.root)
    os.makedirs(args.out, exist_ok=True)
    missing = 0
    for tag_uuid, expected in manifest.items():
        hits = found.get(tag_uuid)
        if not hits:
            print(f"MISSING {expected['name']}")
            missing += 1
            continue
        destination = os.path.join(args.out, expected["rel"])
        os.makedirs(os.path.dirname(destination) or ".", exist_ok=True)
        shutil.copy2(hits[0][3], destination)
        print(f"copied {expected['rel']}")
    print(f"\n{len(manifest) - missing}/{len(manifest)} gathered into {args.out} ({missing} missing)")
    sys.exit(1 if missing else 0)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    tag_parser = subparsers.add_parser("tag")
    tag_parser.add_argument("path")
    tag_parser.add_argument("--id")
    tag_parser.set_defaults(fn=cmd_tag)

    tag_all_parser = subparsers.add_parser("tag-all")
    tag_all_parser.add_argument("--root", default=".")
    tag_all_parser.add_argument("--manifest")
    tag_all_parser.add_argument("--include-hidden", action="store_true")
    tag_all_parser.set_defaults(fn=cmd_tag_all)

    snapshot_parser = subparsers.add_parser("snapshot")
    snapshot_parser.add_argument("--root", default=".")
    snapshot_parser.set_defaults(fn=cmd_snapshot)

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--root", default=".")
    verify_parser.add_argument("--manifest", required=True)
    verify_parser.add_argument("--strict", action="store_true")
    verify_parser.set_defaults(fn=cmd_verify)

    repair_parser = subparsers.add_parser("repair")
    repair_parser.add_argument("--root", default=".")
    repair_parser.add_argument("--manifest", required=True)
    repair_parser.set_defaults(fn=cmd_repair)

    gather_parser = subparsers.add_parser("gather")
    gather_parser.add_argument("--root", default=".")
    gather_parser.add_argument("--manifest", required=True)
    gather_parser.add_argument("--out", required=True)
    gather_parser.set_defaults(fn=cmd_gather)

    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
