# Model Preparation

Stock ONNX models exported from PyTorch / TensorFlow / the ONNX Model Zoo
rarely run through the inference scheduler unchanged. They typically ship
with a dynamic batch dimension (`N`), training-mode `BatchNormalization`
ops, scaffolding nodes (`Cast`, `Squeeze` / `Unsqueeze` chains, `Shape` /
`Constant` for dynamic reshape), and tail operators the kernels don't
implement (`Softmax`, `Identity`, `ArgMax`). The scheduler exits with a
`SchedulerError` on the first unsupported op, so every model needs a
preparation pass before it can be fed to `inference_scheduler.py`.

This doc covers the standard preparation flow built around
`simplify_onnx.py`. For the full list of operators the scheduler accepts,
see [`INFERENCE_SCHEDULER.md`](INFERENCE_SCHEDULER.md#2-supported-onnx-operators).

---

## 1. The one-shot fix: `simplify_onnx.py`

`simplify_onnx.py` is the top-level utility for normalising an ONNX file.
It is enough for most ONNX Model Zoo / torchvision exports:

```bash
python simplify_onnx.py model.onnx --batch 1
```

The default output is `<stem>-simplified.onnx` next to the input.
Pipeline:

1. **Pin input shapes.** `--batch N` rewrites every input's dynamic first
   dim to `N`; `--input-shape NAME=D1,D2,…` (repeatable) handles inputs
   with non-batch dynamic dims, or any time you need a non-1 batch.
2. **`onnxsim.simplify`** — constant folding, dead-node removal, shape
   inference. As a side-effect it usually folds `BatchNormalization` (and
   sometimes also `Squeeze`/`Unsqueeze`, `Reshape`/`Flatten` chains, and
   `Cast` of constants) into surrounding ops.
3. **`onnxoptimizer.fuse_bn_into_conv`** — safety-net pass for any BN
   that onnxsim left in place. Disable with `--no-fuse-bn` if you want
   to inspect the un-fused output.
4. **`onnx.checker.check_model`** — fails loudly on any structural
   issue introduced by the rewrites.
5. **Optional smoke test** — `--check` runs the saved model through
   onnxruntime with a random-input feed and prints output min/max so
   you can sanity-check non-trivial transforms.

The script prints a node-count delta plus a per-op-type diff (changed
counts highlighted in yellow). Examples observed on the bundled models:

| Source | Nodes (before → after) | BN folded |
|---|---|---|
| `resnet50-v1-12.onnx` | 175 → 122 | 53 → 0 |
| `resnet18-v1-7.onnx` | 69 → 49 | 20 → 0 |
| `mobilenet_v1_1.0_224.onnx` | 78 → 59 | 13 → 0 |
| `mobilenetv2-12.onnx` | 105 → 100 | 0 (already folded by exporter) |
| `bertsquad-12.onnx` | 1167 → 753 | 0 |
| `lenet.onnx` | 18 → 9 | 0 |

---

## 2. Picking input shapes

`--batch N` is the shortcut: every input whose first dim is dynamic
(`dim_param` set, e.g. `'N'`) gets its first dim set to `N`. Other dims
are left as authored. If any non-first dim is also dynamic, the script
errors out — you have to use `--input-shape`:

```bash
# Single-input vision model with dynamic batch
python simplify_onnx.py resnet50-v1-12.onnx --batch 1
#   data: [N, 3, 224, 224]  →  [1, 3, 224, 224]

# Single-input model with non-batch dynamic dims
python simplify_onnx.py super-resolution-10.onnx \
    --input-shape input=1,1,224,224

# Multi-input model — repeat --input-shape per input
python simplify_onnx.py bertsquad-12.onnx \
    --input-shape unique_ids_raw_output___9:0=1 \
    --input-shape segment_ids:0=1,256 \
    --input-shape input_mask:0=1,256 \
    --input-shape input_ids:0=1,256
```

`--input-shape` always wins over `--batch` for the named input, so you
can mix the two: `--batch 1 --input-shape segment_ids:0=1,512` to keep
batch=1 everywhere but use a non-default sequence length.

---

## 3. Handling unsupported ops left after simplify

`simplify_onnx.py` only does mechanical rewrites — it cannot invent
support for an operator the scheduler doesn't implement. If the scheduler
still reports an unsupported op on a simplified model, the typical
remedies are:

| Symptom | Cause | Fix |
|---|---|---|
| `Softmax`, `LogSoftmax` in the tail | Classifier head | Strip the tail with `onnx.utils.extract_model(in, out, [model_input], [pre_softmax_tensor])`. The bundled `mobilenet_v1_1.0_224_no_softmax.onnx` was produced this way. |
| `Cast`, `Identity`, `Dropout` (eval-mode no-op) | Exporter artifacts | A second `simplify_onnx.py` pass usually removes them once shapes are pinned. |
| `Pad` with non-constant pads | Dynamic padding via `Shape`/`Slice` | Re-export the model with constant padding values, or rewrite via `onnx.compose` / `onnx-graphsurgeon`. |
| `Conv` with `group != 1 and group != in_channels` | Grouped convolution (not depthwise) | Not supported by ConvKernel. The model needs surgery to expand the grouped conv into multiple normal convs. |
| `MatMul` with `k > kMaxK` | Inner-dim larger than the platform's `kernels.matmul.max_k` | Either bump `max_k` in `platforms/<name>.json` and re-synthesise, or split the matmul along K (manual). |

`onnx.utils.extract_model` is the easiest way to keep just the
"interesting" portion of a network — see the bundled `mobilenet_v1_*`
sub-graph fixtures (`mobilenet_v1_input_to_avgpool.onnx`,
`mobilenet_v1_conv13_relu6_avgpool.onnx`, etc.) for examples of
intermediate-tensor extraction used to isolate scheduler / kernel bugs
without running the whole network.

---

## 4. Diagnosis workflow

When a model the scheduler refuses to load:

```bash
# 1. Pin shapes + simplify. Most exporter cruft disappears here.
python simplify_onnx.py model.onnx --batch 1 --check

# 2. Try the scheduler on the simplified file.
python inference_scheduler.py model-simplified.onnx --out-dir /tmp/out

# 3. If it fails, the SchedulerError names the offending op and tensor.
#    Inspect the simplified model around that node — `netron`, `onnx.helper`,
#    or `onnx-graphsurgeon` are all useful.
python - <<EOF
import onnx
m = onnx.load("model-simplified.onnx")
for i, n in enumerate(m.graph.node):
    if n.op_type in {"Softmax", "Cast", "Identity"}:
        print(i, n.op_type, n.name, "->", list(n.output))
EOF

# 4. Strip the offending region. For a clean tail-cut:
python - <<EOF
import onnx, onnx.utils
onnx.utils.extract_model(
    "model-simplified.onnx", "model-trimmed.onnx",
    input_names=["data"], output_names=["last_supported_tensor"])
EOF

# 5. Re-simplify the trimmed model (so the constant-folding pass sees the
#    new, shorter graph) and re-run the scheduler.
python simplify_onnx.py model-trimmed.onnx
python inference_scheduler.py model-trimmed-simplified.onnx --out-dir /tmp/out
```

This loop is fast because `simplify_onnx.py --check` exits in seconds for
all but the largest models.

---

## 5. Worked example — resnet50-v1-12

The stock model from the ONNX Model Zoo:

```text
resnet50-v1-12.onnx
  175 nodes, opset ai.onnx:12
  input  data: [N, 3, 224, 224]
  output resnetv17_dense0_fwd: [N, 1000]
  ops: Conv×53, BatchNormalization×53, Relu×49, Add×16,
       MaxPool×1, GlobalAveragePool×1, Flatten×1, Gemm×1
```

`BatchNormalization` is not in the scheduler's supported set, and the
batch dim is dynamic. One command fixes both:

```bash
python simplify_onnx.py resnet50-v1-12.onnx --batch 1 --check
```

Result:

```text
resnet50-v1-12-simplified.onnx
  122 nodes
  input  data: [1, 3, 224, 224]
  ops: Conv×53, Relu×49, Add×16, MaxPool×1, GlobalAveragePool×1,
       Flatten×1, Gemm×1
```

All 53 BNs were absorbed into the preceding Convs by onnxsim's constant
folding (the BN scale/bias became compile-time constants once the input
shape was pinned), Flatten + Gemm form the classifier head, and the
scheduler decomposes `Gemm` → `MatMul + Add` at load time. The file is
now ready:

```bash
python inference_scheduler.py resnet50-v1-12-simplified.onnx \
    --out-dir /tmp/resnet50_inference
```
