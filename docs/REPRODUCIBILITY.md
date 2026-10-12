# Reproducibility Notes

## Leakage controls

The pipeline separates the outer test fold from all development operations. Block-wise `StandardScaler` objects are fitted only on development/training data. Optimizer fitness is computed only within the development partition using three inner stratified folds. Downstream classifier tuning and neural early stopping are also restricted to development data.

## Optimizer fairness

PSO, GA, ACOR, and Bayesian Optimization operate over the same scalar coefficient domain and receive exactly 150 objective evaluations. They use the same Logistic Regression fitness classifier, metric, and inner partitions.

## Fusion definition

The six blocks are non-duplicated. The feature dimensions are 24 + 28 + 28 + 24 + 24 + 512 = 640. Fusion is weighted concatenation after block-wise standardization; it is not element-wise addition of unequal-dimensional vectors.

## Ablation

Each leave-one-block-out condition re-optimizes the remaining coefficients rather than reusing weights learned for the full representation. This separates block necessity from a weight vector fitted to a different feature set.

## Statistical unit

The five independent outer-fold means are used as paired observations for fold-level inference. Repeated seeds characterize stochastic variability but are not treated as 25 independent samples in the Wilcoxon test.

## Dataset split caveat

The completed Fruits experiments use image-level stratified folds. The supplied manifest contains augmentation-style filenames in the dataset. A future source/group-aware evaluation would be a stricter test of generalization when original-source grouping can be reconstructed reliably. The repository therefore reports the completed protocol transparently rather than describing it as source-group-held-out.

KTH-TIPS in this study is the 810-image, 10-class KTH-TIPS collection, not KTH-TIPS2b.

## Environment

The publication code writes an environment summary during execution (Python, platform, NumPy, pandas, SciPy, scikit-learn, TensorFlow, and PyWavelets versions). The original environment text file was not present in the retained artifact set used to prepare this public repository, so exact package versions are not fabricated here.
