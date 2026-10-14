"""
links.py
~~~~~~~~
Parser for the links file (links.txt) — the list of what to crawl.

Format (one entry per line):

    # comment
    https://imi.pmf.kg.ac.rs/informatika-studije
    https://imi.pmf.kg.ac.rs/upis-studije | depth=1 | docs=true
    C:/Projects/imiChat/data/raw/konkurs2026.pdf

  • An entry is a URL or a local PDF / DOCX path (spaces allowed).
  • Options after " | ":  depth=N  (default 0 = only this page)
                          docs=true (also fetch linked PDF/DOCX files)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

_WINDOWS_PATH_RE = re.compile(r"^[a-zA-Z]:[\\/]")
_TRUE = {"1", "true", "yes", "da", "on"}


@dataclass
class LinkSpec:
    target: str             # URL or absolute local path
    depth: int = 0
    docs: bool = False
    is_file: bool = False
    line_no: int = 0


def _is_local_path(target: str) -> bool:
    if target.lower().startswith("file://"):
        return True
    if _WINDOWS_PATH_RE.match(target) or target.startswith(("/", "./", "../", ".\\", "..\\")):
        return True
    return urlsplit(target).scheme not in ("http", "https")


def parse_links_file(path: str | Path) -> list[LinkSpec]:
    """Read the links file and return the entries in file order."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Links file not found: {path}")

    specs: list[LinkSpec] = []
    seen: set[str] = set()
    for line_no, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        parts = [p.strip() for p in line.split(" | ")]
        target = parts[0]
        spec = LinkSpec(target=target, line_no=line_no)

        for option in parts[1:]:
            key, _, value = option.partition("=")
            key, value = key.strip().lower(), value.strip().lower()
            if key == "depth":
                try:
                    spec.depth = max(0, int(value))
                except ValueError:
                    raise ValueError(f"{path}:{line_no}: depth must be an integer, got {value!r}")
            elif key == "docs":
                spec.docs = value in _TRUE
            else:
                raise ValueError(f"{path}:{line_no}: unknown option {key!r} (use depth=N, docs=true)")

        if _is_local_path(target):
            local = target[7:] if target.lower().startswith("file://") else target
            local_path = Path(local)
            if not local_path.is_absolute():
                local_path = (path.parent / local_path).resolve()
            spec.target = str(local_path)
            spec.is_file = True

        if spec.target in seen:
            continue
        seen.add(spec.target)
        specs.append(spec)

    return specs
