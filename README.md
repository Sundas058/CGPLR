<div align="center">

# CGPLR
### Confidence-Guided Pseudo-Label Refinement
#### Unsupervised Cross-Domain Person Re-Identification

**One encoder · Four reliability cues · Three supervision decisions**

[Quick start](#quick-start) · [Data](#datasets) · [Training](#training-and-evaluation) · [Experiments](#controlled-experiments) · [Reproducibility](#reproducibility)

</div>

---

CGPLR couples pseudo-label reliability, label refinement and supervision. A source-trained **ResNet-50** extracts target embeddings; centroid compatibility, neighborhood agreement, competing-centroid separation and temporal stability guide **preserve / reassign / defer** decisions. Preserved and reassigned samples receive weighted identity supervision. Deferred samples receive label-free weak/strong consistency.

```mermaid
flowchart LR
    S[Source training] --> F[Target embeddings]
    F --> C[DBSCAN]
    C --> R[Four-cue reliability]
    R --> D[Preserve / Reassign / Defer]
    D --> L[Selective learning]
    L --> F
```

| Encoder | Transfers | Embedding | Input |
|---|---|---|---|
| ResNet-50 + BN neck | M→MSMT, MSMT→M, M→CUHK03-NP-D | 2,048-D, L2 normalized | 256 × 128 |

## Manuscript results

| Transfer | Rank-1 (%) | mAP (%) |
|---|---:|---:|
| Market-1501 ? MSMT17 | 76.40 ? 0.20 | 47.60 ? 0.25 |
| MSMT17 ? Market-1501 | 95.10 ? 0.20 | 87.70 ? 0.25 |
| Market-1501 ? CUHK03-NP-D | 67.30 ? 0.20 | 40.70 ? 0.25 |

Values are the manuscript's reported mean ? sample SD over three seeds. Run-specific measurements are written to `metrics.json`.

## Quick start

Install a matching PyTorch/torchvision build for your CUDA environment, then run:

```bash
python -m pip install -e ".[jaccard,diagnostics]"
python examples/train_cgplr.py --help
python -m unittest discover -s tests -v
```

The `jaccard` extra supplies FAISS CPU nearest-neighbor search. The `diagnostics` extra supplies Matplotlib. CUDA is selected when available; `--device cpu` selects CPU execution. Commands run from the repository root.

## Datasets

```text
data/
├── market1501/Market-1501-v15.09.15/
│   ├── bounding_box_train/
│   ├── query/
│   └── bounding_box_test/
├── msmt17/MSMT17_V1/
│   ├── train/
│   ├── test/
│   ├── list_train.txt
│   ├── list_val.txt
│   ├── list_query.txt
│   └── list_gallery.txt
└── cuhk03-np-d/
    ├── images_detected/
    └── cuhk03_new_protocol_config_detected.mat
```

- **Market-1501:** standard train/query/gallery split.
- **MSMT17:** official list files; `list_train.txt` supplies training membership. `--msmt-combine-trainval` includes validation images explicitly. V2 uses `--msmt-version MSMT17_V2` with `mask_train_v2/` and `mask_test_v2/`. A restructured train/query/gallery layout is also supported.
- **CUHK03-NP-D:** detected images and the official 767/700 new protocol; the MAT file determines the split.

Use `--data-dir PATH` for another data root. Dataset images, trained checkpoints and raw run outputs are not bundled.

## Training and evaluation

The same [reference configuration](configs/reference.json) applies across transfers. CLI values override JSON values; named interventions apply after parsing. Each run uses a fresh output directory.

### 1 · Train the source encoder

```bash
python examples/train_cgplr.py --config configs/reference.json --stage source --dataset market1501 --imagenet --seed 1 --output-dir logs/source/market1501/seed_1
```

For MSMT17 source training, change the dataset to `msmt17` and use its corresponding source output directory.

### 2 · Adapt to the target

```bash
python examples/train_cgplr.py --config configs/reference.json --stage adapt --dataset msmt17 --checkpoint logs/source/market1501/seed_1/source.pth --seed 1 --output-dir logs/M_to_MSMT/full/seed_1
```

| Transfer | Source checkpoint dataset | Target `--dataset` |
|---|---|---|
| M→MSMT | `market1501` | `msmt17` |
| MSMT→M | `msmt17` | `market1501` |
| M→CUHK03-NP-D | `market1501` | `cuhk03-np-d` |

Adaptation removes target identity labels before training. Raw assignments and centroids populate temporal history. A fresh target classifier is initialized each round; encoder optimizer state persists. The final fixed-budget checkpoint is evaluated after adaptation.

### 3 · Evaluate a checkpoint

```bash
python examples/train_cgplr.py --config configs/reference.json --stage evaluate --dataset msmt17 --checkpoint logs/M_to_MSMT/full/seed_1/final.pth --output-dir logs/M_to_MSMT/evaluation/seed_1 --save-retrieval-features
```

Evaluation uses single-query all-gallery mAP and rank-1/5/10, excludes same-identity/same-camera gallery images, and performs no retrieval reranking. JSON retrieval metrics use fractions; multiply by 100 for percentages. Final checkpoints are selected by the fixed training budget, not target test performance.

## Reference settings

| Group | Configuration |
|---|---|
| Source / target budget | 60 source epochs / 50 target rounds; 200 iterations each |
| Batch | 64 samples; four hard-label instances per identity; deferred fraction 0.25 |
| Optimizer | Adam; LR 3.5e-4; weight decay 5e-4; LR ×0.1 every 20 epochs/rounds with updates |
| Clustering | DBSCAN eps 0.55, min_samples 4; precomputed k-reciprocal Jaccard; k1=30, k2=6 |
| Reliability | K=5; H=3; alpha=0.5; fusion weights (0.5, 0.3, 0.2) |
| Temperatures | Centroid 0.5; margin 0.1; reassignment 0.5; classifier 0.05 |
| Refinement | rho_H/rho_L=1; beta=0.5; evidence gain 0.1; neighbor support 0.6; reassignment factor 0.7 |
| Objectives | Squared-distance triplet margin 0.3; triplet weight 1; deferred weight 0.1 |
| Numerical constants | Dispersion floor 1e-6; denominator epsilon 1e-12 |
| Repeated seeds | 1, 2, 3 |

Weak augmentation uses bicubic resize, horizontal flip (0.5), padding 10, random crop and ImageNet normalization. Strong augmentation additionally uses color jitter (0.2/0.2/0.2/0.1, application probability 0.5), grayscale (0.1), and random erasing (0.5; area 0.02–0.2; aspect ratio 0.3–3.3). Evaluation uses resize and normalization.

## Controlled experiments

```bash
python examples/run_review_experiments.py --suite main --include-source
python examples/run_review_experiments.py --suite main --include-source --execute
python examples/run_review_experiments.py --suite mechanism --execute --plot
```

Without `--execute`, the runner writes a command manifest. Suites are `main`, `stagewise`, `mechanism`, `sensitivity`, `safety`, and `all`. The full matrix has 744 source/target commands; select the suite appropriate to the comparison. Source paths include the seed, and matched methods reuse the corresponding source checkpoint.

| Intervention | Change |
|---|---|
| `clustering` | Raw labels, unit weights, no selective consistency |
| `centroid`, `neighborhood` | Successive reliability components on raw labels |
| `refinement` | Selective decisions, unit hard-label weights, deferred samples discarded |
| `no_margin`, `no_temporal` | Remove the cue and renormalize active weights |
| `global_thresholds`, `forced_reassignment` | Change thresholding or reassignment acceptance |
| `discard_deferred` | Remove deferred entries from batches |
| `no_consistency` | Zero consistency coefficient; retain weak-view batch participation |
| `equal_weights` | Unit hard-label weights with consistency retained |
| `overlap_history` | Membership-overlap history in place of centroid correspondence |
| `candidate_guard` | Minimum candidate size 8 and mean reliability 0.5 |

Zero consistency and discarding deferred samples differ in batch composition and BN statistics. No-temporal and temporal-weight zero are equivalent under identical conditions. Optional candidate gates are not active in the full configuration: its minimum size/reliability are 1/0.

External competitor commands can be supplied with `--competitor-manifest`; the interface does not supply the competitor algorithm itself.

## Diagnostics and statistics

```bash
python examples/analyze_revision.py diagnose --run logs/M_to_MSMT/full/seed_1 --data-dir data --output logs/M_to_MSMT/diagnostics/seed_1 --plot
python examples/analyze_revision.py aggregate --runs logs/M_to_MSMT/full/seed_1 logs/M_to_MSMT/full/seed_2 logs/M_to_MSMT/full/seed_3 --output logs/M_to_MSMT/summary.json
```

Add `--baselines` with matched run directories for paired comparisons. Aggregation checks seed pairing, protocol fields, source hashes and environments. It reports mean, sample SD, paired differences, 95% t intervals and two-sided paired t-tests. P-values from the generic aggregator are unadjusted; the manuscript's four-comparison family uses Holm correction.

Post-hoc correctness uses per-round cluster-majority identity mapping, with smallest-ID tie breaking and a fixed raw mapping for refined labels. Calibration uses all valid clustered samples, ten equal-width ECE bins and AUROC. The reliability plot uses the final saved round; decision summaries average the last five available rounds. Target identities enter these diagnostics only after training.

The supplied manuscript diagnostic cohorts comprise 10,000 reassignment observations, separate temporal-event groups, and a matched 1,000-sample deferred cohort. Their exact analyzed windows and selection records are N/A in the supplied metadata. These fields are not inferred from the 50-round training budget. Definitions and counting conventions are given in [Protocol conventions](#protocol-conventions).

## Reproducibility

Each run records effective settings, library/device versions, source fingerprints, initialization checkpoint hash, dataset filenames, assignments and stage measurements.

| Artifact | Contents |
|---|---|
| `config.json`, `environment.json` | Effective arguments, versions, device and source-checkpoint SHA-256 |
| `source_fingerprints.json`, `dataset_manifest.json` | Code/config hashes and sample ordering |
| `source.pth`, `final.pth` | Encoder weights, configuration and timing metadata |
| `round_NNN.npz`, `rounds.json` | Raw/refined labels, cues, decisions, centroids, counts and timings |
| `metrics.json`, `timing.json` | Retrieval, throughput, parameters and memory |
| `retrieval_cases.json` | Query filenames, top gallery matches and correctness flags |
| Diagnostic outputs | Per-round calibration, size strata, topology events and cohort transitions |

`--save-features` retains round embeddings; `--save-retrieval-features` retains evaluation embeddings and IDs. CUDA stages use synchronization. Inference timing uses 10 warmups and 50 measured forwards at batch one by default. Encoding latency excludes decoding, transfer and gallery search. CPU RSS sampling excludes child processes; GPU allocator peaks are recorded separately.

## Protocol conventions

**Reliability and history.** Centroid compatibility is exponential in negative squared distance; structural reliability blends it with valid-neighbor label agreement. The margin cue uses the assigned-versus-competing centroid distance gap. Exact rectangular Hungarian matching aligns historical and current centroids. Unmatched/outlier comparisons contribute zero; history scores divide by the number of available snapshots. Empty history excludes the temporal cue from fusion, and active coefficients are renormalized to unit sum.

**Selective refinement.** Cluster thresholds use median +/- scaled MAD, a dispersion floor and clipping to [0,1]. Preservation takes scores at or above the upper threshold. Intermediate scores require both sufficient alternative evidence gain and a separate candidate-neighbor vote fraction to be reassigned. Other valid-cluster samples are deferred. Raw DBSCAN outliers are excluded from both hard-label and deferred batches for that round.

**Objectives and batches.** Preserved weights equal reliability; reassigned weights are eta_r times alternative evidence divided by alternative plus current evidence and epsilon. Weighted classification and batch-hard squared-distance triplet losses normalize participating anchor weights. Deferred consistency compares normalized strong features with detached normalized weak features. Both views share the encoder; the strong forward uses a full batch while the consistency term selects deferred entries. Hard-label batches use replacement where needed; an empty subset's quota is filled from the other subset. All-deferred zero-consistency batches update batch statistics without an optimizer step.

**Diagnostic denominators.** Preserved/reassigned/deferred rates divide by valid raw-cluster observations. Correction rate divides corrected originally wrong reassigned observations by originally wrong reassigned observations. Correct-to-wrong and wrong-to-correct cohort rates divide by the respective initial correctness groups; outlier fraction divides by the entire cohort. Empty conditional groups are N/A. Membership-overlap event groups are defined independently of centroid matching; zero pairwise agreement is an individual historical comparison, not necessarily zero history-average stability.

**Measurement conventions.** The deployed encoder contains 23,512,128 parameters (23,508,032 backbone; 4,096 neck). Temporary identity classifiers are excluded. Convolution/linear FLOPs count two operations per multiply-add and exclude BN, activations and pooling. Source plus adaptation wall time is total training time; clustering is already inside adaptation. Per-batch forward latency, amortized per-image latency and gallery-ranking time are distinct outputs.

## Repository map

```text
cgplr/
  datasets/          Dataset readers and official split handling
  models/encoder.py  ResNet-50 + BN neck
  method/            Reliability, history, refinement and objectives
  training.py        Source training, adaptation and evaluation
  diagnostics.py     Calibration, subset and retrieval metrics
  review_analysis.py Temporal/cohort measurements
configs/             Shared configuration
examples/            Training, experiment and diagnostic commands
tests/               Numerical and integration checks
```

## Authors

Sundas Iqbal · Danish Ali · Qing Tian · Yuhui Zheng · Jianping Gou · Weihua Ou

**Paper:** Confidence-Guided Pseudo-Label Refinement for Unsupervised Cross-Domain Person Re-Identification.
