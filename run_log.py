from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_SAFE_PREFIX = re.compile(r"[^A-Za-z0-9_-]+")


class MLXDGemmaRunLogWriter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
                "canvas_state": ("MLX_DGEMMA_CANVAS_STATE",),
                "canvas_trace": ("MLX_DGEMMA_TRACE",),
                "run_metadata": ("MLX_DGEMMA_RUN_METADATA",),
                "filename_prefix": ("STRING", {"default": "mlx_diffusiongemma"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("artifact_path",)
    FUNCTION = "save"
    CATEGORY = "DiffusionGemma/MLX"
    OUTPUT_NODE = True

    def save(
        self,
        text: str,
        canvas_state: dict[str, Any],
        canvas_trace: dict[str, Any],
        run_metadata: dict[str, Any],
        filename_prefix: str,
    ):
        import folder_paths

        prefix = _SAFE_PREFIX.sub("_", filename_prefix).strip("_")
        if not prefix:
            raise ValueError("filename_prefix must contain a letter or number")
        output_dir = Path(folder_paths.get_output_directory()) / "diffusiongemma_mlx"
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        destination = output_dir / f"{prefix}_{timestamp}.json"
        document = {
            "schema_version": "mlx-diffusiongemma-run/v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "text": text,
            "canvas_state": canvas_state,
            "canvas_trace": canvas_trace,
            "run_metadata": run_metadata,
        }

        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{prefix}_", suffix=".tmp", dir=output_dir
        )
        try:
            with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
                json.dump(document, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, destination)
        except Exception:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise

        path = str(destination)
        return {"ui": {"artifact_path": [path]}, "result": (path,)}
