# ComfyUI MLX DiffusionGemma

Four ComfyUI nodes that connect a workflow to a local MLX DiffusionGemma sidecar. The nodes can run a generation, return intermediate text frames, render hidden-layer activity, and save a complete run log.

The MLX model server is a separate process. This repository contains the ComfyUI client nodes only.

## Requirements

- ComfyUI
- A local MLX DiffusionGemma sidecar that implements the API described below
- The model configured in both the sidecar and the loader node
- NumPy and PyTorch, which are part of a standard ComfyUI installation

The default model is:

```text
mlx-community/diffusiongemma-26B-A4B-it-4bit
```

The sidecar uses MLX and therefore requires supported Apple silicon hardware. The ComfyUI nodes communicate with it through loopback HTTP only.

## Install

Clone this repository into the ComfyUI `custom_nodes` directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/latentwill/ComfyUI-MLX-DiffusionGemma.git
```

Restart ComfyUI. The nodes appear in the **DiffusionGemma/MLX** category.

The nodes do not add Python package requirements. They use the Python standard library and packages that ComfyUI already provides.

## Start the sidecar

Start the companion sidecar as a separate process. By default, the nodes expect it at:

```text
http://127.0.0.1:8080
```

Check it before you run a workflow:

```bash
curl http://127.0.0.1:8080/health
```

A ready sidecar returns a response in this form:

```json
{
  "ready": true,
  "model": "mlx-community/diffusiongemma-26B-A4B-it-4bit",
  "backend": "mlx"
}
```

The model value must match the model value in the loader node.

The client accepts only plain HTTP on `127.0.0.1`, `localhost`, or `::1`. It rejects remote hosts, credentials, query parameters, and URL fragments.

## Basic workflow

Add and connect these nodes:

```text
MLX DiffusionGemma Loader
            │
            ▼
MLX DiffusionGemma Sampler
       │             │
       ▼             ▼
MLX DiffusionGemma Trace    MLX DiffusionGemma Run Log Writer
```

1. Add **MLX DiffusionGemma Loader**.
2. Set `base_url` to the local sidecar URL.
3. Set `model` to the exact model ID returned by `/health`.
4. Connect the loader output to **MLX DiffusionGemma Sampler**.
5. Enter a prompt and configure the generation values.
6. Queue the workflow.
7. Optionally connect `canvas_trace` to **MLX DiffusionGemma Trace**.
8. Optionally connect `text`, `canvas_state`, `canvas_trace`, and `run_metadata` to **MLX DiffusionGemma Run Log Writer**.

## Nodes

### MLX DiffusionGemma Loader

Checks the sidecar health endpoint and creates the model handle used by the sampler.

| Input | Default | Purpose |
| --- | --- | --- |
| `base_url` | `http://127.0.0.1:8080` | Loopback URL for the sidecar |
| `model` | `mlx-community/diffusiongemma-26B-A4B-it-4bit` | Required model ID |
| `timeout_seconds` | `10` | Health request timeout |

The loader stops with an error if the sidecar is not ready or if the loaded model does not match.

### MLX DiffusionGemma Sampler

Sends one generation request to the sidecar.

| Input | Default | Purpose |
| --- | --- | --- |
| `prompt` | Empty | Generation prompt; must not be empty |
| `seed` | `0` | Reproducible random seed |
| `num_inference_steps` | `48` | Number of denoising steps |
| `t_min` | `0.4` | Lower schedule bound |
| `t_max` | `0.8` | Upper schedule bound |
| `entropy_bound` | `0.1` | Entropy-bound sampler threshold |
| `confidence` | `0.005` | Confidence-threshold sampler value |
| `gen_length` | `128` | Maximum generation length |
| `thinking` | `false` | Enables thinking mode in a compatible sidecar |
| `hidden_layers` | `0,9,19,29` | Comma-separated decoder layers to capture |
| `sampler` | `entropy-bound` | `entropy-bound` or `confidence-threshold` |
| `timeout_seconds` | `600` | Generation request timeout |

`hidden_layers` accepts unique layer numbers from 0 through 29. The node sorts the selected layers before it sends the request.

Outputs:

| Output | Type | Contents |
| --- | --- | --- |
| `text` | `STRING` | Final generated text |
| `canvas_state` | `MLX_DGEMMA_CANVAS_STATE` | Generation state and timing values |
| `canvas_trace` | `MLX_DGEMMA_TRACE` | Complete sidecar response and hidden-layer trace |
| `frames` | `STRING` list | Intermediate decoded denoising frames |
| `run_metadata` | `MLX_DGEMMA_RUN_METADATA` | Request, runtime, model, schema, and consistency data |

### MLX DiffusionGemma Trace

Renders the captured root-mean-square hidden activity as a ComfyUI image.

| Input | Default | Purpose |
| --- | --- | --- |
| `trace` | Required | `canvas_trace` output from the sampler |
| `layer` | `19` | Captured decoder layer to render |
| `scale` | `4` | Nearest-neighbor display scale |

The selected layer must also be present in the sampler `hidden_layers` input. The node returns an RGB heatmap and a JSON summary.

### MLX DiffusionGemma Run Log Writer

Writes the generation result and trace to one JSON file. Connect all four matching sampler outputs.

Files are saved under:

```text
ComfyUI/output/diffusiongemma_mlx/
```

The writer uses an atomic file replacement so that an interrupted write does not leave a partial final file. The JSON document contains the final text, canvas state, canvas trace, run metadata, schema version, and UTC creation time.

## Sidecar API contract

A compatible sidecar must provide these routes on the configured loopback address.

### `GET /health`

Returns:

- `ready`: Boolean readiness state
- `model`: Loaded model ID
- `backend`: Backend name

### `POST /generate`

Accepts a JSON object with the sampler inputs. It must return a JSON object with these fields:

- `schema_version`
- `model`
- `text`
- `state`
- `frames`
- `hidden_steps`
- `selected_layers`
- `trace_consistent`
- `request`
- `runtime`

Each `frames` entry must contain a `text` field. Each captured layer in `hidden_steps` must contain `layer` and `position_rms` fields.

## Troubleshooting

### `MLX sidecar is not ready`

Confirm that the sidecar process is running and that `/health` returns `"ready": true`.

### Loaded model does not match

Use the exact model ID returned by `/health` in the loader node.

### Connection refused or request failed

Confirm the sidecar address and port. The default is `http://127.0.0.1:8080`.

### Layer is absent from the trace

Add the layer number to the sampler `hidden_layers` input, run the sampler again, and then render that new trace.

### Generation request times out

Increase the sampler `timeout_seconds` value. The allowed range is 5 to 3600 seconds.
