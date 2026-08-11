# ComfyUI MLX DiffusionGemma

Four ComfyUI nodes for running MLX DiffusionGemma workflows. Generate text, inspect intermediate frames, render hidden-layer activity, and save complete run logs.

Install the nodes in ComfyUI, connect them in a workflow, and queue the workflow.

## Requirements

- ComfyUI
- A local MLX DiffusionGemma runtime available at the URL entered in the Loader node
- NumPy and PyTorch, which are part of a standard ComfyUI installation

The default model is:

```text
mlx-community/diffusiongemma-26B-A4B-it-4bit
```

The MLX runtime requires supported Apple silicon hardware.

## Install

Clone this repository into the ComfyUI `custom_nodes` directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/latentwill/ComfyUI-MLX-DiffusionGemma.git
```

Restart ComfyUI. The nodes appear in the **DiffusionGemma/MLX** category.

The nodes do not add Python package requirements. They use the Python standard library and packages that ComfyUI already provides.


## Basic workflow

Add and connect these nodes:

```text
MLX DiffusionGemma Loader
            │
            ▼
MLX DiffusionGemma Sampler
       │         │          │
       ▼         ▼          ▼
    Trace    Run Log    Preview Text
```

1. Add **MLX DiffusionGemma Loader**.
2. Set `base_url` to the URL of your local MLX runtime.
3. Set `model` to the model ID configured in that runtime.
4. Connect the loader output to **MLX DiffusionGemma Sampler**.
5. Enter a prompt and configure the generation values.
6. Queue the workflow.
7. Optionally connect `text` to **Preview as Text** (`PreviewAny`) to read the response in ComfyUI.
8. Optionally connect `canvas_trace` to **MLX DiffusionGemma Trace**.
9. Optionally connect `text`, `canvas_state`, `canvas_trace`, and `run_metadata` to **MLX DiffusionGemma Run Log Writer**.

## Nodes

### MLX DiffusionGemma Loader

Checks the configured MLX runtime and creates the model handle used by the sampler.

| Input | Default | Purpose |
| --- | --- | --- |
| `base_url` | `http://127.0.0.1:8080` | URL for the local MLX runtime |
| `model` | `mlx-community/diffusiongemma-26B-A4B-it-4bit` | Required model ID |
| `timeout_seconds` | `10` | Runtime check timeout |

The loader stops with an error if the runtime is unavailable or if the loaded model does not match.

### MLX DiffusionGemma Sampler

Sends one generation request to the configured MLX runtime.

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
| `thinking` | `false` | Enables thinking mode when supported by the runtime |
| `hidden_layers` | `0,9,19,29` | Comma-separated decoder layers to capture |
| `sampler` | `entropy-bound` | `entropy-bound` or `confidence-threshold` |
| `timeout_seconds` | `600` | Generation request timeout |

`hidden_layers` accepts unique layer numbers from 0 through 29. The node sorts the selected layers before it sends the request.

Outputs:

| Output | Type | Contents |
| --- | --- | --- |
| `text` | `STRING` | Final generated text |
| `canvas_state` | `MLX_DGEMMA_CANVAS_STATE` | Generation state and timing values |
| `canvas_trace` | `MLX_DGEMMA_TRACE` | Generation response and hidden-layer trace |
| `frames` | `STRING` list | Intermediate decoded denoising frames |
| `run_metadata` | `MLX_DGEMMA_RUN_METADATA` | Request, runtime, model, schema, and consistency data |

### Preview as Text

Uses ComfyUI's built-in `PreviewAny` node to display the Sampler's final `text` output. Connect the Sampler `text` output to its `source` input and queue the workflow.

The preview node also provides a `STRING` output.

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


## Troubleshooting

### Runtime unavailable

Confirm that the local MLX runtime is running and that `base_url` is correct. The default is `http://127.0.0.1:8080`.

### Loaded model does not match

Use the exact model ID configured by the local MLX runtime in the Loader node.

### Layer is absent from the trace

Add the layer number to the Sampler `hidden_layers` input, queue the workflow again, and render the new trace.

### Generation request times out

Increase the Sampler `timeout_seconds` value. The allowed range is 5 to 21600 seconds.
