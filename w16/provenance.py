"""Small runtime provenance helper for public experiment scripts."""

from __future__ import annotations

import importlib.metadata
import json
import sys


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "MISSING"


def print_provenance(packages: list[str]) -> None:
    record = {
        "argv": sys.argv,
        "executable": sys.executable,
        "python": sys.version.split()[0],
        "packages": {name: package_version(name) for name in packages},
    }
    print(json.dumps({"provenance": record}, sort_keys=True), flush=True)
