from __future__ import annotations

import json
from typing import Any

from .client import normalize_base_url, request_json
from .run_log import MLXDGemmaRunLogWriter
from .trace import MLXDGemmaTrace


_DEFAULT_MODEL = "mlx-community/diffusiongemma-26B-A4B-it-4bit"
_DEFAULT_BASE_URL = "http://127.0.0.1:8080"

MLX_DGEMMA_MODEL = "MLX_DGEMMA_MODEL"
MLX_DGEMMA_CANVAS_STATE = "MLX_DGEMMA_CANVAS_STATE"
MLX_DGEMMA_TRACE = "MLX_DGEMMA_TRACE"
MLX_DGEMMA_RUN_METADATA = "MLX_DGEMMA_RUN_METADATA"
MLX_DGEMMA_REG_CONTROL_MEMORY = "REG_CONTROL_MEMORY"


def _parse_hidden_layers(value: str) -> list[int]:
    try:
        layers = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError(
            "hidden_layers must be a comma-separated list of integers"
        ) from exc
    if len(set(layers)) != len(layers):
        raise ValueError("hidden_layers must not contain duplicates")
    if any(layer < 0 or layer > 29 for layer in layers):
        raise ValueError("hidden_layers must be between 0 and 29")
    return sorted(layers)


def _serialize_reg_control_memory(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if hasattr(value, "as_dict"):
        value = value.as_dict()
    if not isinstance(value, dict):
        raise ValueError("reg_control_memory must be a REG control-memory object")
    if value.get("schema_version") != "reg-control-memory/v2":
        raise ValueError("reg_control_memory must use schema reg-control-memory/v2")
    return value


class MLXDGemmaLoader:
    DESCRIPTION = "Connects ComfyUI to one loopback-only MLX DiffusionGemma sidecar."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_url": ("STRING", {"default": _DEFAULT_BASE_URL}),
                "model": ("STRING", {"default": _DEFAULT_MODEL}),
                "timeout_seconds": ("INT", {"default": 10, "min": 1, "max": 60}),
            }
        }

    RETURN_TYPES = (MLX_DGEMMA_MODEL,)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "DiffusionGemma/MLX"

    def load(self, base_url: str, model: str, timeout_seconds: int):
        normalized = normalize_base_url(base_url)
        health = request_json(
            normalized,
            "/health",
            timeout_seconds=timeout_seconds,
        )
        if not health.get("ready"):
            raise RuntimeError("MLX sidecar is not ready")
        loaded_model = health.get("model")
        if loaded_model != model:
            raise RuntimeError(
                f"MLX sidecar loaded {loaded_model!r}, but the node requires {model!r}"
            )
        return (
            {
                "base_url": normalized,
                "model": model,
                "timeout_seconds": timeout_seconds,
                "backend": health.get("backend"),
            },
        )


class MLXDGemmaSampler:
    DESCRIPTION = (
        "Runs MLX DiffusionGemma through the loopback sidecar with optional "
        "REG denoising-time logit guidance."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (MLX_DGEMMA_MODEL,),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                    },
                ),
                "num_inference_steps": (
                    "INT",
                    {"default": 48, "min": 1, "max": 256},
                ),
                "t_min": (
                    "FLOAT",
                    {"default": 0.4, "min": 0.0, "max": 2.0, "step": 0.01},
                ),
                "t_max": (
                    "FLOAT",
                    {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.01},
                ),
                "entropy_bound": (
                    "FLOAT",
                    {"default": 0.1, "min": 0.0, "max": 100.0, "step": 0.001},
                ),
                "confidence": (
                    "FLOAT",
                    {"default": 0.005, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
                "gen_length": (
                    "INT",
                    {"default": 128, "min": 1, "max": 8192},
                ),
                "thinking": ("BOOLEAN", {"default": False}),
                "hidden_layers": ("STRING", {"default": "0,9,19,29"}),
                "sampler": (["entropy-bound", "confidence-threshold"],),
                "timeout_seconds": (
                    "INT",
                    {"default": 600, "min": 5, "max": 21600},
                ),
            },
            "optional": {
                "reg_control_memory": (MLX_DGEMMA_REG_CONTROL_MEMORY,),
            },
        }

    RETURN_TYPES = (
        "STRING",
        MLX_DGEMMA_CANVAS_STATE,
        MLX_DGEMMA_TRACE,
        "STRING",
        MLX_DGEMMA_RUN_METADATA,
    )
    RETURN_NAMES = ("text", "canvas_state", "canvas_trace", "frames", "run_metadata")
    OUTPUT_IS_LIST = (False, False, False, True, False)
    FUNCTION = "sample"
    CATEGORY = "DiffusionGemma/MLX"
    OUTPUT_NODE = True

    def sample(
        self,
        model: dict[str, Any],
        prompt: str,
        seed: int,
        num_inference_steps: int,
        t_min: float,
        t_max: float,
        entropy_bound: float,
        confidence: float,
        gen_length: int,
        thinking: bool,
        hidden_layers: str,
        sampler: str,
        timeout_seconds: int,
        reg_control_memory: Any | None = None,
    ):
        if not prompt.strip():
            raise ValueError("prompt must not be empty")
        layers = _parse_hidden_layers(hidden_layers)
        serialized_reg_control_memory = _serialize_reg_control_memory(reg_control_memory)
        payload = {
            "prompt": prompt,
            "seed": seed,
            "num_inference_steps": num_inference_steps,
            "t_min": t_min,
            "t_max": t_max,
            "entropy_bound": entropy_bound,
            "confidence": confidence,
            "gen_length": gen_length,
            "thinking": thinking,
            "hidden_layers": layers,
            "sampler": sampler,
        }
        if serialized_reg_control_memory is not None:
            payload["reg_control_memory"] = serialized_reg_control_memory
        response = request_json(
            model["base_url"],
            "/generate",
            payload=payload,
            timeout_seconds=timeout_seconds,
        )
        if response.get("model") != model["model"]:
            raise RuntimeError("MLX sidecar returned an unexpected model ID")

        text = response["text"]
        state = response["state"]
        frames = [frame["text"] for frame in response.get("frames", [])]
        run_metadata = {
            "schema_version": response.get("schema_version"),
            "model": response.get("model"),
            "request": response.get("request"),
            "runtime": response.get("runtime"),
            "trace_consistent": response.get("trace_consistent"),
        }
        state_json = json.dumps(state, sort_keys=True)
        metadata_json = json.dumps(run_metadata, sort_keys=True)
        return {
            "ui": {
                "text": [text],
                "canvas_state_json": [state_json],
                "run_metadata_json": [metadata_json],
            },
            "result": (text, state, response, frames, run_metadata),
        }


class MLXDGemmaLongSampler:
    DESCRIPTION = (
        "Runs bounded long-form generation through the MLX DiffusionGemma sidecar "
        "with optional REG denoising-time logit guidance."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (MLX_DGEMMA_MODEL,),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                    },
                ),
                "target_tokens": (
                    "INT",
                    {"default": 8192, "min": 1, "max": 65536},
                ),
                "segment_tokens": (
                    "INT",
                    {"default": 1536, "min": 1024, "max": 2048},
                ),
                "summary_tokens": (
                    "INT",
                    {"default": 256, "min": 1, "max": 2048},
                ),
                "max_segments": (
                    "INT",
                    {"default": 8, "min": 1, "max": 64},
                ),
                "num_inference_steps": (
                    "INT",
                    {"default": 48, "min": 1, "max": 256},
                ),
                "t_min": (
                    "FLOAT",
                    {"default": 0.4, "min": 0.0, "max": 2.0, "step": 0.01},
                ),
                "t_max": (
                    "FLOAT",
                    {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.01},
                ),
                "entropy_bound": (
                    "FLOAT",
                    {"default": 0.1, "min": 0.0, "max": 100.0, "step": 0.001},
                ),
                "confidence": (
                    "FLOAT",
                    {"default": 0.005, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
                "temperature": (
                    "FLOAT",
                    {"default": 0.2, "min": 0.0, "max": 2.0, "step": 0.01},
                ),
                "repetition_penalty": (
                    "FLOAT",
                    {"default": 1.0, "min": 1.0, "max": 2.0, "step": 0.01},
                ),
                "repetition_context_size": (
                    "INT",
                    {"default": 2048, "min": 1, "max": 65536},
                ),
                "repetition_guard": ("BOOLEAN", {"default": True}),
                "max_retries": (
                    "INT",
                    {"default": 2, "min": 0, "max": 16},
                ),
                "retry_seed_stride": (
                    "INT",
                    {"default": 1, "min": 1, "max": 0xFFFFFFFFFFFFFFFF},
                ),
                "thinking": ("BOOLEAN", {"default": False}),
                "timeout_seconds": (
                    "INT",
                    {"default": 21600, "min": 5, "max": 21600},
                ),
            },
            "optional": {
                "reg_control_memory": (MLX_DGEMMA_REG_CONTROL_MEMORY,),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("text", "summary", "segments", "metadata_json")
    OUTPUT_IS_LIST = (False, False, True, False)
    FUNCTION = "sample"
    CATEGORY = "DiffusionGemma/MLX"
    OUTPUT_NODE = True

    def sample(
        self,
        model: dict[str, Any],
        prompt: str,
        seed: int,
        target_tokens: int,
        segment_tokens: int,
        summary_tokens: int,
        max_segments: int,
        num_inference_steps: int,
        t_min: float,
        t_max: float,
        entropy_bound: float,
        confidence: float,
        temperature: float,
        repetition_penalty: float,
        repetition_context_size: int,
        repetition_guard: bool,
        max_retries: int,
        retry_seed_stride: int,
        thinking: bool,
        timeout_seconds: int,
        reg_control_memory: Any | None = None,
    ):
        if not prompt.strip():
            raise ValueError("prompt must not be empty")
        if not 1024 <= segment_tokens <= 2048:
            raise ValueError("segment_tokens must be between 1024 and 2048")
        serialized_reg_control_memory = _serialize_reg_control_memory(reg_control_memory)
        payload = {
            "prompt": prompt,
            "seed": seed,
            "target_tokens": target_tokens,
            "segment_tokens": segment_tokens,
            "summary_tokens": summary_tokens,
            "max_segments": max_segments,
            "num_inference_steps": num_inference_steps,
            "t_min": t_min,
            "t_max": t_max,
            "entropy_bound": entropy_bound,
            "confidence": confidence,
            "temperature": temperature,
            "repetition_penalty": repetition_penalty,
            "repetition_context_size": repetition_context_size,
            "repetition_guard": repetition_guard,
            "max_retries": max_retries,
            "retry_seed_stride": retry_seed_stride,
            "thinking": thinking,
        }
        if serialized_reg_control_memory is not None:
            payload["reg_control_memory"] = serialized_reg_control_memory
        response = request_json(
            model["base_url"],
            "/generate-long",
            payload=payload,
            timeout_seconds=timeout_seconds,
        )
        if response.get("model") != model["model"]:
            raise RuntimeError("MLX sidecar returned an unexpected model ID")

        text = response["text"]
        summary = response["summary"]
        segments = [segment["text"] for segment in response.get("segments", [])]
        metadata = dict(response)
        metadata.pop("text", None)
        metadata.pop("summary", None)
        metadata_json = json.dumps(metadata, sort_keys=True)
        return {
            "ui": {
                "text": [text],
                "summary": [summary],
                "segments": segments,
                "metadata_json": [metadata_json],
            },
            "result": (text, summary, segments, metadata_json),
        }


NODE_CLASS_MAPPINGS = {
    "MLXDGemmaLoader": MLXDGemmaLoader,
    "MLXDGemmaSampler": MLXDGemmaSampler,
    "MLXDGemmaLongSampler": MLXDGemmaLongSampler,
    "MLXDGemmaTrace": MLXDGemmaTrace,
    "MLXDGemmaRunLogWriter": MLXDGemmaRunLogWriter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MLXDGemmaLoader": "MLX DiffusionGemma Loader",
    "MLXDGemmaSampler": "MLX DiffusionGemma Sampler",
    "MLXDGemmaLongSampler": "MLX DiffusionGemma Long Sampler",
    "MLXDGemmaTrace": "MLX DiffusionGemma Trace",
    "MLXDGemmaRunLogWriter": "MLX DiffusionGemma Run Log Writer",
}
