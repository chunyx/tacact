#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path


def main() -> None:
    pipeline_path = Path(__file__).with_name("hpo_pipeline.py")
    spec = importlib.util.spec_from_file_location("hpo_pipeline", pipeline_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load pipeline module: {pipeline_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()


if __name__ == "__main__":
    main()
