# CGPLR: Confidence-Guided Pseudo-Label Refinement

Official research code organization for the manuscript:

**Confidence-Guided Pseudo-Label Refinement for Unsupervised Cross-Domain Person Re-Identification**

Authors: Sundas Iqbal, Danish Ali, Qing Tian, Yuhui Zheng, Jianping Gou, and Weihua Ou

This repository contains a PyTorch implementation of CGPLR for unsupervised domain adaptation (UDA) in person re-identification. The method transfers knowledge from a labeled source dataset to an unlabeled target dataset by combining clustering-based pseudo-label generation, sample-level confidence estimation, confidence-guided pseudo-label refinement, and confidence-aware target-domain optimization.


## Repository Layout

```text
CGPLR/
|-- NN (1).pdf
|-- README.md
|-- setup.py
|-- examples/
|   |-- train_cgplr.py
|   `-- train_cgplr_teacher.py
`-- cgplr/
    |-- datasets/
    |-- evaluation_metrics/
    |-- evaluators.py
    |-- method/
    |   |-- source_training.py
    |   |-- clustering.py
    |   |-- confidence.py
    |   |-- refinement.py
    |   `-- objectives.py
    |-- models/
    |-- trainers.py
    `-- utils/
```

## Key Modules

`cgplr/method/source_training.py` implements the source-domain initialization objective: cross entropy plus triplet loss.

`cgplr/method/clustering.py` implements target feature normalization support, DBSCAN pseudo-label generation, valid-cluster filtering, and centroid computation.

`cgplr/method/confidence.py` implements sample-level confidence estimation:

- centroid-based confidence,
- local neighborhood consistency,
- final weighted confidence score.

`cgplr/method/refinement.py` implements confidence-guided pseudo-label refinement by preserving reliable labels and reassigning uncertain samples to the nearest centroid.

`cgplr/method/objectives.py` implements confidence-weighted cross entropy, confidence-weighted triplet loss, and the combined target-domain objective.

`cgplr/models/` contains the backbone networks, pooling layers, memory modules, and metric-learning losses used by the training scripts.

`cgplr/datasets/` contains dataset loaders for Market-1501, MSMT17, DukeMTMC-reID, and PersonX.

`cgplr/evaluators.py` and `cgplr/evaluation_metrics/` contain feature extraction and standard Re-ID evaluation metrics.

## Dataset Preparation

Place datasets under a local `data/` directory or pass another root with `--data-dir`.

Expected layout:

```text
data/
|-- market1501/
|-- msmt17/
|-- dukemtmcreid/
`-- personx/
```

The manuscript evaluates CGPLR mainly on:

- Market-1501 -> MSMT17
- MSMT17 -> Market-1501
- Market-1501 -> CUHK03-NP-D as an additional generalization setting

## Installation

Create an environment with PyTorch, torchvision, scikit-learn, scipy, Pillow, h5py, and FAISS. Then install the project package in editable mode:

```bash
pip install -e .
```

FAISS installation depends on the CUDA and platform setup. If `faiss-gpu` is not available through pip for the target system, install FAISS through conda or the official FAISS instructions.

## Main Training Entrypoints

Teacher-stage adaptation:

```bash
python examples/train_cgplr_teacher.py --dataset market1501 --data-dir ./data --logs-dir ./logs/teacher_market
```

Student CGPLR adaptation:

```bash
python examples/train_cgplr.py --dataset market1501 --data-dir ./data --logs-dir ./logs/cgplr_market
```

Important CGPLR hyperparameters:

```text
--confidence-alpha        balance between centroid confidence and neighborhood consistency, default 0.5
--confidence-temperature  temperature for centroid confidence, default 1.0
--confidence-neighbors    K nearest neighbors for local reliability, default 5
--confidence-threshold    high-confidence threshold, default 0.6
```

Other commonly used training settings:

```text
--height 256
--width 128
--batch-size 256
--epochs 50
--lr 0.00035
--eps 0.55
--k1 30
--k2 6
--num-instances 16
```

## Implementation Notes

- The package namespace is `cgplr`, matching the paper acronym.
- Generated Python bytecode caches are excluded from the source tree.
- The method modules are separated from the legacy training infrastructure so each manuscript component can be inspected, reused, and cited clearly.
- The example scripts now call the CGPLR confidence estimation and pseudo-label refinement modules after DBSCAN pseudo-label generation.

## Citation

If this repository supports your work, cite the manuscript:

@unpublished{iqbal2026cgplr,
  title  = {Confidence-Guided Pseudo-Label Refinement for Unsupervised Cross-Domain Person Re-Identification},
  author = {Iqbal, Sundas and Ali, Danish and Tian, Qing and Zheng, Yuhui and Gou, Jianping and Ou, Weihua},
  year   = {2026},
  note   = {Manuscript submitted to IEEE Transactions on Circuits and Systems for Video Technology}
}

Update the citation fields after acceptance with the final volume, pages, DOI, and publication year assigned by the journal.
