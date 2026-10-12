# Controlled Handcrafted–Deep Feature Fusion for Image Classification

Reproducibility repository for the manuscript **“An Optimization Framework for Fusing Handcrafted and Deep Features in Image Classification.”**

## What this repository contains

The experimental pipeline combines five handcrafted feature blocks with a pretrained VGG16 representation:

| Block | Representation | Dimension |
|---|---|---:|
| F1 | GLCM | 24 |
| F2 | DWT | 28 |
| F3 | SWT | 28 |
| F4 | DWT–GLCM | 24 |
| F5 | SWT–GLCM | 24 |
| F6 | VGG16 `block5_conv3` + global average pooling | 512 |
| **Full** | Weighted concatenation | **640** |

Each block is standardized using development/training data only before scalar weighting. The full representation is evaluated with SVM, Random Forest, MLP, and 1D CNN classifiers.

## Optimization protocol

Four continuous block-weight search strategies are evaluated under the same conditions:

- Particle Swarm Optimization (PSO)
- Genetic Algorithm (GA)
- ACOR continuous ant-colony optimization
- Gaussian-process Bayesian Optimization

All optimizers use the same weight range `[0.05, 1.0]`, the same three inner stratified folds, the same Logistic Regression fitness classifier, and **150 objective-function evaluations**. Five fixed seeds are used: `11, 22, 33, 44, 55`.

## Evaluation design

- 5 stratified outer folds
- 3 inner folds for optimizer fitness
- 5 fixed seeds
- identical outer partitions across representations and optimizers
- fold-local preprocessing
- outer test folds excluded from scaling, optimization, tuning, and neural early stopping
- leave-one-block-out ablation with re-optimization
- RGB-versus-grayscale VGG control
- fixed feature-order permutation control for the 1D CNN
- class-wise precision, recall, and F1
- paired fold-level Wilcoxon analysis with Holm correction
- computational and storage profiling

## Datasets

The reported experiments use:

1. **Fruits** — 10,901 images, 6 classes.
2. **KTH-TIPS** — 810 images, 10 classes (81 images per class).

Dataset images are **not redistributed** in this repository. Exact image identifiers and outer-fold membership are provided in `data_splits/` so that the published partitions can be reconstructed from legally obtained copies of the datasets.

The public manifests intentionally omit the authors' original absolute workstation paths; `image_id`, filename, class, and fold assignment are preserved.

## Repository structure

```text
notebooks/     publication notebook used for the final pipeline
src/           executable single-file version of the same pipeline
config/        seeds, feature dimensions, optimizer and experiment settings
data_splits/   exact image identifiers and outer-fold assignments
results/       reviewer-ready outputs for Fruits and KTH-TIPS
docs/          detailed reproduction notes
scripts/       helper validation script
```

## Installation

Python 3.9+ is recommended. Create an isolated environment and install:

```bash
pip install -r requirements.txt
```

The publication code records the Python/platform and major package versions at runtime. Because the original run-time environment export was not retained with these files, `requirements.txt` states the software dependencies without inventing package versions that were not independently recorded.

## Running the pipeline

The exact final publication pipeline is provided in both notebook and Python forms:

```text
notebooks/publication_feature_fusion_single_cell_FINAL_FIXED.ipynb
src/publication_feature_fusion_pipeline.py
```

Set `DATASET_ROOT` near the top of the notebook/script to the local dataset directory, or leave it as `None` only if the intended dataset can be auto-detected in the runtime environment.

Then run:

```bash
python src/publication_feature_fusion_pipeline.py
```

For a full paper reproduction, keep:

```python
PUBLICATION_MODE = True
QUICK_TEST = False
CPU_SAFE_MODE = True
```

Do not use `QUICK_TEST=True` for manuscript results.

## Published split manifests

`data_splits/fruits_outer_splits.csv` and `data_splits/kth_tips_outer_splits.csv` contain the exact five outer-fold assignments used in the completed experiments. Roles are recorded as `development` or `test` for every image and fold.

## Results

Each dataset folder under `results/` contains CSV exports for:

- feature dimensions
- split information
- classifier tuning
- optimizer runs and learned weights
- raw model results
- model summary
- ablation
- class-wise metrics
- Wilcoxon tests
- McNemar outputs produced by the pipeline
- runtime profiling

A sanitized multi-sheet Excel workbook is also provided for convenience.

## Reproducibility note

The main purpose of this repository is to expose the **exact evaluation logic, seeds, partitions, optimizer budgets, feature dimensions, and reported result tables** used in the revised manuscript. Cross-study headline accuracies should not be interpreted as directly comparable unless dataset version and evaluation protocol are matched.

## Citation

If this code supports your research, please cite the associated manuscript after publication. The final bibliographic citation can be added here once the article is formally published.
