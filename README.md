# Attention U-Net++: 0.6 m Four-Band

**Unified Platform tensor adapter plus the preserved original TIFF pipeline.**
Author: Vansh Arora, Civil Engineering, IIT Bombay. Guide: Prof. Gopal Patil.

## Quick Start

```bash
git clone https://github.com/vansharora665/Urban-Atlas-Native-0_6m.git
cd Urban-Atlas-Native-0_6m
python -m pip install -r requirements-adapter.txt
python preflight_test.py --full
```

```python
import torch
from adapter import get_model, preprocess

# raw_np: four bands in BLUE, GREEN, RED, NIR order, CHW or HWC, raw digital numbers.
model = get_model()  # eval-mode nn.Module; never downloads pretrained weights
x = preprocess(raw_np)  # (1, 4, H, W), float32, [0, 1]
with torch.inference_mode():
    logits = model(x)
    labels = logits.argmax(1)
```

Normalize each full scene once BEFORE tiling. For memory-bounded raster inference,
estimate scene-wide per-band limits and call `preprocess_with_limits` on each tile.
Do not calculate a new stretch per window. Use 512-pixel tiles and 128-pixel
overlap; blend probabilities before argmax. Preserve the TIFF transform, CRS and
valid-data mask outside this tensor adapter. The hosted app's in-memory full-scene
runner is NOT the original streaming runner and is not certified for 1 GB inputs.

## Files And Contract

- `adapter.py`: `get_model(weights_path=None)`, `preprocess(raw_np)`, optional streaming normalization helper.
- `deployment_model.py`: exact trained architecture and canonical output-channel permutation.
- `contract_runtime.py`: strict state loading, prefix cleanup and normalization.
- `model_weights.pth`: ordinary tensor-only FP32 state_dict, below 100,000,000 bytes.
- `model.pt`: standalone traced/frozen TorchScript, also below 100,000,000 bytes.
- `model_config.json`: real input schema, canonical classes, hashes and honestly scoped metrics.
- `export_checkpoint.py`: reproducible export from the checksum-pinned original bundle or a trusted checkpoint.
- `preflight_test.py`: real CPU forward checks. This is NOT an accuracy test.
- `pipeline_config.json`, `loader.py`, `worker.py`, `runtime/`: preserved original file-based pipeline.

No precision reduction, quantization, new training or accuracy improvement is
claimed. ZIP DEFLATE is lossless; all exported state tensors are checked equal to
the source. TorchScript and eager logits are numerically compared. See
`EXPORT_VERIFICATION.json` for sizes and measured numerical tolerance.

### Exact Unified Class Mapping

| Index | Meaning |
| --- | --- |
| 0 | Water |
| 1 | Vegetation |
| 2 | Barren Land |
| 3 | Built-up |
| 4 | Roads |

Training used `water, roads, built-up, barren, vegetation`. The exported network
permutes channels by `[0, 4, 3, 2, 1]`. **Do not remap the exported output again.**
The original TIFF pipeline retains its original IDs. Unknown/cloud/no-data is
255 in GIS outputs, not a sixth logit; the host must apply the input raster mask
and an explicit quality/uncertainty policy. An all-zero normalized tile is not
proof of water or any other class.

## What This Tensor Model Includes

The tensor adapter is the existing joint native-detail ResNet34 U-Net++ with scSE attention. Auxiliary road/boundary parameters are retained for strict restoration, but the interface returns only five semantic logits. CoANet is NOT silently fused into these logits; it remains the separate original road-surface option.

If `segmentation_models_pytorch` is unavailable (as in the inspected hosted app),
the adapter verifies the matching checkpoint and loads the exact `model.pt`
graph instead. This still uses the trained model, never a simulated or heuristic
substitute. Custom/prefixed checkpoints require the architecture dependencies.
TorchScript itself only needs PyTorch at inference time.

## Accuracy: Do Not Substitute Unrelated Scores

The raw tensor interface has no independently measured five-class test score.
Its OA, macro mIoU and class IoUs are `null`, not zero and not fabricated
defaults. Neutral `class_weights=1` are inference/ensemble settings, NOT a claim
about fixed training-loss weights.

The original refined Pune development result was OA 89.73% / mIoU 65.03%. CoANet's separate road IoUs were 41.25% on reused local development crops and 48.95% on a 48-tile LoveDA pilot. They use different references and are NOT metrics of this raw semantic adapter.

Same GSD alone does not guarantee accuracy on a new city, season or sensor.

## Original TIFF Pipeline (Preserved)

```bash
python -m pip install -r requirements.txt
python loader.py --verify --device cpu
python loader.py --input /path/to/image.tif --output /path/to/new_results --vectors
```

Input must be a projected metre-based GeoTIFF, four B/G/R/NIR bands at 0.6 m.
Vector shapefiles are generated only with `--vectors`.
CoANet writes a separate road layer; use --no-road-specialist to skip it. --zones produces a labelled cartographic derivative, not ground truth.

`pipeline_weights.bundle` is intentionally NOT included in new clones: it would
force the importer's shallow clone to download a second, large legacy model.
The original loader retrieves the unchanged, SHA256-pinned bundle from the
existing [combined v1.0.0 release](https://github.com/vansharora665/Urban-Atlas-Final-Models/releases/tag/v1.0.0)
when requested. Existing local bundles are retained. It is not `model_weights.pth`.

## Re-Export

```bash
# First run loader.py --verify if the original bundle is not present.
python export_checkpoint.py
python preflight_test.py --full
# Or supply your own TRUSTED architecture-compatible training checkpoint:
python export_checkpoint.py --source-checkpoint /path/to/checkpoint.pt
```

Never load untrusted pickled training checkpoints. Optimizer, scheduler, scaler
and arbitrary checkpoint metadata are not part of the tensor-only export. The
export fails rather than silently quantizing if an artifact is still >=100 MB.

## Hosted App Integration

Import this repository URL. Prefer `model_weights.pth` with `adapter.py` so that
the custom preprocessing hook is registered. Direct `model.pt` loading also
requires calling the same custom preprocessing function first.

The target app source inspected at revision
`f401dbec06d6330aa802cc2a67af30293aa0c8d6` contains host-side compatibility
issues independent of these checkpoints. See `PLATFORM_INTEGRATION.md`.
Do not report successful hosted deployment until the owner applies those fixes.
The sensor string for native data is truthfully four-band B/G/R/NIR, not RGB.

## Attribution And Rights

Keep `NOTICE.md` and upstream notices in `runtime/`. The original bundle contains
CoANet material with retained GPL-3.0 and non-commercial research notices.
This academic package is not a commercially cleared navigation/survey product.
Training imagery and labels are not redistributed.
