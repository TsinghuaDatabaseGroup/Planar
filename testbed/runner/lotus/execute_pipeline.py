#!/usr/bin/env python3
"""Execute one LOTUS pipeline with optional native runtime activation."""

from __future__ import annotations

import argparse
import os
import runpy
from pathlib import Path


def _enabled(name: str) -> bool:
    return os.getenv(name, "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("script", type=Path)
    args = parser.parse_args()

    cache_dir = os.environ["LOTUS_CACHE_DIR"]
    if _enabled("LOTUS_JOIN_OPTIMIZATION_ENABLED"):
        import join_optimizer_runtime

        join_optimizer_runtime.configure(cache_dir)

    runpy.run_path(str(args.script.resolve()), run_name="__main__")


if __name__ == "__main__":
    main()
