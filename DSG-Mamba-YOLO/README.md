# DSG-Mamba-YOLO

DSG-Mamba-YOLO is the seed and bud detector in
[PIS-SGRDet](../README.md). It builds on Mamba-YOLO and introduces grouped
bottleneck convolution (GBC) and dynamic spectral gating for feature extraction.

## Model implementation

The active model code is
[`ultralytics/nn/modules/mamba_yolo.py`](ultralytics/nn/modules/mamba_yolo.py):

- `SimpleStem` integrates the `GBC` feature-enhancement module.
- `SpectralGatingNetwork` implements the dynamic spectral filtering branch.
- `DynamicRGBlock` combines local and spectral features and is used by
  `VSSBlock` and `XSSBlock`.

The model architecture is configured in
[`Mamba-YOLO-T.yaml`](ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T.yaml).
Its original filename and the internal Python module names are preserved.
The dataset YAML overrides the model YAML's default class count during training.

## Environment and installation

Use Linux with an NVIDIA GPU, a CUDA-enabled PyTorch installation and a CUDA
toolkit/compiler compatible with that PyTorch build. The upstream environment
uses Python 3.11 and PyTorch 2.3.0. Select the PyTorch/CUDA build appropriate
for your hardware before compiling the extension.

From the repository root:

```bash
conda create -n dsg-mbyolo python=3.11 -y
conda activate dsg-mbyolo
cd DSG-Mamba-YOLO
# Install compatible CUDA-enabled torch and torchvision first.
pip install seaborn thop timm einops
pip install ./selective_scan
pip install -e .
```

Use this directory as the working directory for the commands below. The
editable installation uses the included modified `ultralytics` package.

## Dataset preparation

Prepare images and YOLO detection labels with matching filenames, for example:

```text
corn_dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

Each label row has the format `class x_center y_center width height`, with
coordinates normalized to the image dimensions.

Edit [`corn_data.yaml`](ultralytics/cfg/datasets/corn_data.yaml) for your data:

```yaml
path: /absolute/path/to/corn_dataset
train: images/train
val: images/val
test: images/test
names:
  0: seed
  1: bud
```

The distributed YAML retains example paths from the original development
environment, including a test entry pointing to validation images. Replace
these values before use and use a separate test split for independent testing.

## Training

For an explicit training-only run, use the local Python API:

```python
from ultralytics import YOLO

model = YOLO('ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T.yaml')
model.train(
    data='ultralytics/cfg/datasets/corn_data.yaml',
    imgsz=640,
    epochs=200,
    batch=8,
    optimizer='SGD',
    device=0,
    workers=0,
    amp=False,
    patience=0,
    seed=0,
    project='output_dir/corn',
    name='dsg-mbyolo',
)
```

These settings are an example, not a complete reproduction protocol for every
reported experiment. Use the saved configuration of the relevant experiment
when reproducing its results.

The supplied `dsg-mbyolo_train.py` is also retained as an application entry
point. Its current task dictionary eagerly invokes both training and
validation, so `--task` does not isolate those operations. Use the explicit
API example above for a training-only run. Its parser does not support `--amp`.

## Validation and prediction

Set the checkpoint and dataset paths to your own trained model:

```python
from ultralytics import YOLO

model = YOLO('output_dir/corn/dsg-mbyolo/weights/best.pt')

# Independent detection evaluation: requires a separate test split.
metrics = model.val(
    data='ultralytics/cfg/datasets/corn_data.yaml',
    split='test',
    imgsz=640,
    batch=8,
    conf=0.001,
    iou=0.7,
    device=0,
)

# Example inference settings; adjust for your application.
results = model.predict(
    source='/absolute/path/to/images',
    imgsz=640,
    conf=0.25,
    iou=0.7,
    device=0,
    save=True,
)
```

Here `conf` filters predictions and `iou` controls NMS. The NMS IoU is distinct
from the matching IoU used to compute detection metrics.

## Application scripts

| Script | Purpose | Configuration |
| --- | --- | --- |
| `dsg-mbyolo_train.py` | Training and validation entry point | CLI arguments; see the task-dispatch note above. |
| `dsg-mbyolo_val.py` | Validate a trained checkpoint | Edit checkpoint, dataset and evaluation settings in the script. |
| `dsg-mbyolo_predict.py` | Predict seeds and buds | Edit checkpoint, image source and inference settings. |
| `dsg-mbyolo_model_info.py` | Estimate parameters, FLOPs and forward-pass speed | Supports `--config`, `--device` and `--imgsz`. |
| `dsg-mbyolo_germination_rate.py` | Calculate germination rates from detections | Edit checkpoint, input directory and output paths. |
| `dsg-mbyolo_visualize_activation.py` | Visualize model activations | Configure the target layer and input/model settings. |

For example:

```bash
python dsg-mbyolo_model_info.py --config ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T.yaml --device 0 --imgsz 640
```

Run these filenames directly with `python`; their hyphens are intended for
script names, not Python import statements. The rename changes filenames and
documentation references only; internal model modules and classes are unchanged.

Model-information timing measures model forward passes, not the complete
stitching, preprocessing and germination-estimation pipeline. No trained
checkpoints or full datasets are included in this source distribution.

## Acknowledgments

This implementation is based on
[Mamba-YOLO](https://github.com/HZAI-ZJNU/Mamba-YOLO) and
[Ultralytics](https://github.com/ultralytics/ultralytics). The selective-scan
implementation originates from [VMamba](https://github.com/MzeroMiko/VMamba).
The included source is distributed with its [AGPL-3.0 license](LICENSE) and
component-specific notices.

Please also cite the upstream Mamba-YOLO work when using its implementation:

```bibtex
@misc{wang2024mambayolossmsbasedyolo,
  title={Mamba YOLO: SSMs-Based YOLO For Object Detection},
  author={Zeyu Wang and Chen Li and Huiying Xu and Xinzhong Zhu},
  year={2024},
  eprint={2406.05835},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2406.05835}
}
```
