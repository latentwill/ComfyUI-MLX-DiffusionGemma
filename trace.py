from __future__ import annotations

import json
from typing import Any


class MLXDGemmaTrace:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trace": ("MLX_DGEMMA_TRACE",),
                "layer": ("INT", {"default": 19, "min": 0, "max": 29}),
                "scale": ("INT", {"default": 4, "min": 1, "max": 16}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("hidden_rms_heatmap", "summary_json")
    FUNCTION = "render"
    CATEGORY = "DiffusionGemma/MLX"
    OUTPUT_NODE = True

    def render(self, trace: dict[str, Any], layer: int, scale: int):
        import numpy as np
        import torch

        rows: list[list[float]] = []
        step_ids: list[dict[str, int | None]] = []
        for step in trace.get("hidden_steps", []):
            selected = next(
                (entry for entry in step.get("layers", []) if entry["layer"] == layer),
                None,
            )
            if selected is None:
                continue
            rows.append(selected["position_rms"])
            step_ids.append(
                {
                    "global_step": step.get("global_step"),
                    "canvas_index": step.get("canvas_index"),
                    "diffusion_step": step.get("diffusion_step"),
                }
            )

        if not rows:
            raise ValueError(
                f"Layer {layer} is absent from this trace; captured layers are "
                f"{trace.get('selected_layers', [])}"
            )

        width = max(len(row) for row in rows)
        values = np.full((len(rows), width), np.nan, dtype=np.float32)
        for index, row in enumerate(rows):
            values[index, : len(row)] = row

        finite = values[np.isfinite(values)]
        low = float(finite.min())
        high = float(finite.max())
        if high == low:
            normalized = np.zeros_like(values)
        else:
            normalized = (values - low) / (high - low)
        normalized = np.nan_to_num(normalized, nan=0.0)

        red = np.clip((normalized - 0.5) * 2.0, 0.0, 1.0)
        green = np.clip(normalized * 2.0, 0.0, 1.0)
        blue = np.clip(1.0 - normalized, 0.0, 1.0)
        image = np.stack((red, green, blue), axis=-1).astype(np.float32)
        if scale > 1:
            image = np.repeat(np.repeat(image, scale, axis=0), scale, axis=1)

        summary = {
            "schema_version": trace.get("schema_version"),
            "model": trace.get("model"),
            "layer": layer,
            "steps": len(rows),
            "positions": width,
            "minimum_rms": low,
            "maximum_rms": high,
            "trace_consistent": trace.get("trace_consistent"),
            "step_ids": step_ids,
        }
        return {
            "ui": {"summary_json": [json.dumps(summary, sort_keys=True)]},
            "result": (torch.from_numpy(image).unsqueeze(0), json.dumps(summary, sort_keys=True)),
        }
