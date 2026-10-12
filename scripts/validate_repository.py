from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

expected = {
    "fruits": 10901,
    "kth_tips": 810,
}

for ds, n in expected.items():
    manifest = pd.read_csv(ROOT / "data_splits" / f"{ds}_manifest.csv")
    splits = pd.read_csv(ROOT / "data_splits" / f"{ds}_outer_splits.csv")
    assert len(manifest) == n, (ds, len(manifest), n)
    assert splits["fold"].nunique() == 5, (ds, "outer folds", splits["fold"].nunique())
    assert set(splits["role"].unique()) == {"development", "test"}
    assert splits["image_id"].nunique() == n
    print(f"{ds}: {n} images, 5 outer folds, split manifest OK")

features = json.loads((ROOT / "config" / "feature_blocks.json").read_text())
assert sum(features[k] for k in ["F1_GLCM","F2_DWT","F3_SWT","F4_DWT_GLCM","F5_SWT_GLCM","F6_VGG_RGB"]) == 640
print("feature dimensions: 640-D full representation OK")

cfg = json.loads((ROOT / "config" / "experiment_config.json").read_text())
assert cfg["seeds"] == [11,22,33,44,55]
assert cfg["outer_folds"] == 5 and cfg["inner_folds"] == 3
assert cfg["optimizer_objective_evaluations"] == 150
print("experiment configuration OK")
