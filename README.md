# FastViT BDD100K Object Detection

This repository trains and evaluates a FastViT-based bounding-box detector on the ten BDD100K detection classes. Instance segmentation, VOC, COCO-as-a-dataset, and image-classification workflows are intentionally out of scope.

## Setup

```bash
python -m venv .venv
pip install -r requirements.txt
```

Prepare the official BDD100K images separately, then convert the detection labels:

```bash
python -m tools.download_bdd100k --dest-dir ./data/bdd100k
```

Expected layout:

```text
data/bdd100k/
├── annotations/
│   ├── bdd100k_det_train_coco.json
│   └── bdd100k_det_val_coco.json
└── images/100k/
    ├── train/
    └── val/
```

## Train and evaluate

```bash
python train_bdd100k.py --data-dir ./data/bdd100k --model fastvit_sa12
python train_bdd100k.py --data-dir ./data/bdd100k --model SAME_VARIANT --resume PATH/last.pth
python train_bdd100k.py --data-dir ./data/bdd100k --model SAME_VARIANT --resume PATH/best.pth --eval-only --save-json
```

All FastViT variants remain available: `fastvit_t8`, `fastvit_t12`, `fastvit_s12`, `fastvit_sa12`, `fastvit_sa24`, `fastvit_sa36`, and `fastvit_ma36`.

Convenience launchers live in `scripts/`. The full pipeline computes five FPN anchor sizes, passes them to training, and benchmarks the resulting checkpoint:

```bash
bash scripts/train.sh fastvit_sa12 8 ./data/bdd100k
bash scripts/resume.sh PATH/last.pth 8 ./data/bdd100k fastvit_sa12
bash scripts/pipeline.sh fastvit_sa12 8 800 50 ./data/bdd100k
```

## Analysis and deployment tools

```bash
python -m tools.compute_map --json-dir PATH/json_results
python -m tools.visualize_failures --json-dir PATH/json_results
python -m tools.split_by_weather --help
python -m tools.evaluate_gradcam --help
python -m tools.benchmark --model fastvit_sa12 --checkpoint PATH/best.pth
python -m tools.quantize --model fastvit_sa12 --checkpoint PATH/best.pth --benchmark
```

See the [detailed usage guide](docs/usage_guide.md) for complete setup, training, resume, evaluation, analysis, and troubleshooting instructions. Architecture references include the [detailed BDD100K model architecture](docs/model_architecture_bdd100k.md) and the [architecture summary](docs/architecture.md). The [cleanup plan](docs/cleanup_plan.md) and [source review](docs/source_review.md) record the cleanup history.

## Attribution

The backbone is derived from Apple's [FastViT research implementation](https://github.com/apple/ml-fastvit) and the ICCV 2023 paper *FastViT: A Fast Hybrid Vision Transformer using Structural Reparameterization*. Preserve the upstream notices and verify licensing requirements before redistributing derived model code or weights.
