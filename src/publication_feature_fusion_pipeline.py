# ================================================================
# PUBLICATION-GRADE HANDCRAFTED + VGG16 FEATURE FUSION PIPELINE
# Resume-safe CPU edition v2.1
# Single-cell compatible: paste this entire file into one Colab/Jupyter cell.
# ================================================================

# --------------------------- USER CONFIG ---------------------------
# Set DATASET_ROOT to your dataset folder. Leave None for auto-detection.
# Expected generic layout: DATASET_ROOT/class_name/...images...
# KTH-TIPS2b layout is auto-detected when class/sample_a...sample_d folders exist.
DATASET_ROOT = None
PIPELINE_VERSION = "2.1-resume-safe"
OUTPUT_DIR = "publication_results_resume_safe"
RESUME = True
CPU_SAFE_MODE = True
FORCE_NEW_RUN_ON_SIGNATURE_MISMATCH = False

PUBLICATION_MODE = True
QUICK_TEST = False                       # True = smoke test only; NOT for paper results
MAX_IMAGES_PER_CLASS = None if not QUICK_TEST else 40

SEEDS = [11, 22, 33, 44, 55] if PUBLICATION_MODE else [11]
GENERIC_OUTER_FOLDS = 5
INNER_FOLDS = 3

# Equal optimizer budgets are a core fairness control.
OPT_EVAL_BUDGET = (150 if CPU_SAFE_MODE else 300) if PUBLICATION_MODE else 40
ABLATION_EVAL_BUDGET = OPT_EVAL_BUDGET
PSO_SWARM_SIZE = 20
PSO_INERTIA = 0.7
PSO_C1 = 1.5
PSO_C2 = 1.5
GA_POP_SIZE = 20
GA_CROSSOVER_RATE = 0.90
GA_MUTATION_RATE = 0.20
GA_MUTATION_SIGMA = 0.10
GA_TOURNAMENT = 3
GA_ELITES = 2
ACO_ARCHIVE_SIZE = 20
ACO_Q = 0.50
ACO_XI = 0.85
WEIGHT_BOUNDS = (0.05, 1.0)

RUN_FULL_ABLATION = True                 # leave-one-block-out; PSO re-optimized each time
RUN_INDIVIDUAL_FEATURE_BASELINES = True
RUN_RGB_VS_GRAY_VGG = True               # reviewer concern about color information
RUN_CNN_ORDER_SHUFFLE = True             # tests whether 1D-CNN depends on arbitrary feature locality
RUN_GA = True
RUN_BAYES = True
RUN_PSO = True
RUN_ACO = True  # continuous Ant Colony Optimization (ACOR) for fair reviewer comparison

HANDCRAFT_SIZE = (128, 128)
VGG_SIZE = (224, 224)
VGG_BATCH_SIZE = 8 if CPU_SAFE_MODE else 32
HANDCRAFT_CHUNK_SIZE = 64
GLCM_LEVELS = 32
GLCM_DISTANCES = [1]
GLCM_ANGLES = [0.0, np.pi/4, np.pi/2, 3*np.pi/4] if 'np' in globals() else None

NEURAL_MAX_EPOCHS = (60 if CPU_SAFE_MODE else 100) if PUBLICATION_MODE else 12
NEURAL_BATCH_SIZE = 16 if CPU_SAFE_MODE else 32
N_JOBS = 2 if CPU_SAFE_MODE else -1
TF_INTRA_THREADS = 2 if CPU_SAFE_MODE else 0
TF_INTER_THREADS = 1 if CPU_SAFE_MODE else 0
NEURAL_PATIENCE = 12 if PUBLICATION_MODE else 3

# Optimizer fitness classifier: deliberately fixed and identical for PSO/GA/Bayes.
# This isolates the optimizer comparison from downstream model choice.
FITNESS_MODEL_NAME = "LogisticRegression(C=1.0, lbfgs)"
FITNESS_METRIC = "accuracy"

# ------------------------- INSTALL / IMPORTS -----------------------
import os, sys, re, json, math, time, random, hashlib, platform, subprocess, warnings, pickle
# Limit numerical-library thread fan-out before NumPy/TensorFlow are imported.
if CPU_SAFE_MODE:
    os.environ.setdefault("OMP_NUM_THREADS", str(TF_INTRA_THREADS))
    os.environ.setdefault("MKL_NUM_THREADS", str(TF_INTRA_THREADS))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(TF_INTRA_THREADS))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(TF_INTRA_THREADS))
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", str(TF_INTRA_THREADS))
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", str(TF_INTER_THREADS))
from pathlib import Path
from collections import defaultdict
warnings.filterwarnings("ignore")


def _pip_install(packages):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + packages)

required = {
    "numpy": "numpy", "pandas": "pandas", "scipy": "scipy", "sklearn": "scikit-learn",
    "skimage": "scikit-image", "pywt": "PyWavelets", "PIL": "Pillow",
    "matplotlib": "matplotlib", "tqdm": "tqdm", "skopt": "scikit-optimize",
    "openpyxl": "openpyxl"
}
for module, package in required.items():
    try:
        __import__(module)
    except Exception:
        _pip_install([package])

try:
    import tensorflow as tf
except Exception:
    _pip_install(["tensorflow>=2.12"])
    import tensorflow as tf

import numpy as np
import pandas as pd
import pywt
import scipy
from scipy.stats import skew, kurtosis, wilcoxon, t as student_t, binomtest
from PIL import Image
from tqdm.auto import tqdm
from skimage.color import rgb2gray
from skimage.feature import graycomatrix, graycoprops
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             classification_report, confusion_matrix)
from sklearn.pipeline import Pipeline
from skopt import gp_minimize, Optimizer
from skopt.space import Real
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Input, Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import EarlyStopping, BackupAndRestore
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

GLCM_ANGLES = [0.0, np.pi/4, np.pi/2, 3*np.pi/4]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".jfif", ".webp"}
BLOCK_NAMES = ["F1_GLCM", "F2_DWT", "F3_SWT", "F4_DWT_GLCM", "F5_SWT_GLCM", "F6_VGG_RGB"]

OUT = Path(OUTPUT_DIR).resolve()
(OUT / "cache").mkdir(parents=True, exist_ok=True)
(OUT / "figures").mkdir(parents=True, exist_ok=True)
(OUT / "splits").mkdir(parents=True, exist_ok=True)
(OUT / "models").mkdir(parents=True, exist_ok=True)
(OUT / "checkpoints" / "tasks").mkdir(parents=True, exist_ok=True)
(OUT / "checkpoints" / "optimizer_state").mkdir(parents=True, exist_ok=True)
(OUT / "checkpoints" / "keras_backups").mkdir(parents=True, exist_ok=True)
(OUT / "cache" / "chunks").mkdir(parents=True, exist_ok=True)

# ------------------------- REPRODUCIBILITY -------------------------
def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

set_seed(SEEDS[0])

# Conservative CPU thread limits reduce RAM pressure and sustained thermal load on laptops.
if CPU_SAFE_MODE:
    try:
        tf.config.threading.set_intra_op_parallelism_threads(TF_INTRA_THREADS)
        tf.config.threading.set_inter_op_parallelism_threads(TF_INTER_THREADS)
    except Exception:
        pass

# Avoid TensorFlow grabbing all GPU memory.
for gpu in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass

# ------------------------- RESUME / ATOMIC I/O ---------------------
def _atomic_replace_bytes(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(payload); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_pickle(obj, path):
    _atomic_replace_bytes(path, pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))


def load_pickle(path, default=None):
    try:
        with open(path, "rb") as f: return pickle.load(f)
    except Exception:
        return default


def atomic_json(obj, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_csv(df, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def atomic_npz(path, **arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_npy(path, arr):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, arr); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def task_id(kind, *parts):
    raw = kind + "|" + "|".join(map(str, parts))
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def task_path(kind, *parts):
    return OUT / "checkpoints" / "tasks" / f"{kind}_{task_id(kind,*parts)}.pkl"


def update_status(last_task=""):
    files = list((OUT / "checkpoints" / "tasks").glob("*.pkl"))
    atomic_json({"last_completed_task": last_task, "completed_task_files": len(files),
                 "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")}, OUT / "checkpoint_status.json")

# ------------------------- DATASET DISCOVERY -----------------------
def image_files_under(path):
    path = Path(path)
    return [p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]


def score_dataset_root(path):
    path = Path(path)
    if not path.is_dir():
        return -1
    children = [d for d in path.iterdir() if d.is_dir() and not d.name.startswith(".")]
    if len(children) < 2 or len(children) > 200:
        return -1
    image_counts = []
    valid_children = []
    for d in children:
        n = sum(1 for p in d.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        if n > 0:
            valid_children.append(d)
            image_counts.append(n)
    if len(valid_children) < 2:
        return -1
    penalty = 0
    common_split_names = {"train", "training", "test", "testing", "val", "valid", "validation"}
    if set(d.name.lower() for d in valid_children).issubset(common_split_names):
        penalty = 10000
    return len(valid_children) * 100 + min(sum(image_counts), 100000) / 100 - penalty


def auto_detect_dataset_root():
    env = os.environ.get("DATASET_ROOT")
    if env and Path(env).exists():
        return Path(env).resolve()
    bases = [Path.cwd(), Path("/content"), Path("/kaggle/input"), Path("/mnt/data")]
    candidates = []
    seen = set()
    for base in bases:
        if not base.exists():
            continue
        for depth1 in [base] + [p for p in base.iterdir() if p.is_dir()][:100]:
            for cand in [depth1] + ([p for p in depth1.iterdir() if p.is_dir()][:100] if depth1.exists() else []):
                try:
                    rp = cand.resolve()
                except Exception:
                    continue
                if rp in seen or str(rp).startswith(str(OUT)):
                    continue
                seen.add(rp)
                s = score_dataset_root(rp)
                if s >= 0:
                    candidates.append((s, rp))
    if not candidates:
        raise FileNotFoundError(
            "Could not auto-detect a class-folder dataset. Set DATASET_ROOT at the top of the cell."
        )
    candidates.sort(reverse=True, key=lambda x: x[0])
    print("Top dataset-root candidates:")
    for s, p in candidates[:5]:
        print(f"  score={s:.1f}  {p}")
    return candidates[0][1]


DATA_ROOT = Path(DATASET_ROOT).expanduser().resolve() if DATASET_ROOT else auto_detect_dataset_root()
print(f"\nUsing dataset root: {DATA_ROOT}\n")


def infer_sample_token(rel_parts, filename):
    # Official KTH-TIPS2b commonly uses class/sample_a/...file.png.
    if len(rel_parts) >= 3:
        second = rel_parts[1].lower()
        if re.fullmatch(r"sample[_\- ]?[a-d]", second) or second in {"a", "b", "c", "d"}:
            letter = re.search(r"([a-d])$", second)
            return f"sample_{letter.group(1)}" if letter else f"sample_{second}"
    low = filename.lower()
    m = re.search(r"sample[_\- ]?([a-d])", low)
    if m:
        return f"sample_{m.group(1)}"
    return None


def build_manifest(root):
    rows = []
    root = Path(root)
    class_dirs = [d for d in root.iterdir() if d.is_dir()]
    for class_dir in sorted(class_dirs):
        files = sorted([p for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
        if MAX_IMAGES_PER_CLASS:
            files = files[:MAX_IMAGES_PER_CLASS]
        for p in files:
            rel = p.relative_to(root)
            sample = infer_sample_token(rel.parts, p.name)
            uid = str(rel).replace(os.sep, "/")
            rows.append({
                "image_id": uid,
                "path": str(p.resolve()),
                "class_name": class_dir.name,
                "sample_group": sample,
                "filename": p.name,
            })
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No images found under {root}")
    counts = df.groupby("class_name").size()
    if len(counts) < 2:
        raise RuntimeError("Need at least two classes.")
    print("Dataset summary:")
    print(counts.to_string())
    print(f"Total images: {len(df):,}; classes: {df['class_name'].nunique()}")
    groups = sorted(df["sample_group"].dropna().unique().tolist())
    kth_like = (df["class_name"].nunique() == 11 and set(groups) >= {"sample_a","sample_b","sample_c","sample_d"})
    if kth_like:
        print("Detected KTH-TIPS2b-like sample structure. Using four sample-held-out outer folds.")
        if not QUICK_TEST and len(df) != 4752:
            print(f"WARNING: standard KTH-TIPS2b has 4,752 images; this root has {len(df)}. Check dataset completeness.")
    else:
        print(f"No complete KTH-TIPS2b 4-sample structure detected. Using {GENERIC_OUTER_FOLDS}-fold stratified outer CV.")
    return df, kth_like

manifest, IS_KTH_TIPS2B = build_manifest(DATA_ROOT)
manifest.to_csv(OUT / "01_dataset_manifest.csv", index=False)

# Dataset fingerprint ensures feature cache matches the exact image collection.
fingerprint_src = "\n".join(f"{r.image_id}|{Path(r.path).stat().st_size}|{int(Path(r.path).stat().st_mtime)}" for r in manifest.itertuples())
DATA_FINGERPRINT = hashlib.sha256(fingerprint_src.encode()).hexdigest()[:16]
RUN_SIGNATURE_PAYLOAD = {
    "pipeline_version":PIPELINE_VERSION,"dataset_fingerprint":DATA_FINGERPRINT,"publication_mode":PUBLICATION_MODE,"quick_test":QUICK_TEST,
    "seeds":SEEDS,"generic_outer_folds":GENERIC_OUTER_FOLDS,"inner_folds":INNER_FOLDS,
    "optimizer_budget":OPT_EVAL_BUDGET,"ablation_budget":ABLATION_EVAL_BUDGET,"weight_bounds":WEIGHT_BOUNDS,
    "pso":[PSO_SWARM_SIZE,PSO_INERTIA,PSO_C1,PSO_C2],"ga":[GA_POP_SIZE,GA_CROSSOVER_RATE,GA_MUTATION_RATE,GA_MUTATION_SIGMA,GA_TOURNAMENT,GA_ELITES],
    "aco":[ACO_ARCHIVE_SIZE,ACO_Q,ACO_XI],"neural":[NEURAL_MAX_EPOCHS,NEURAL_BATCH_SIZE,NEURAL_PATIENCE],"vgg_batch":VGG_BATCH_SIZE,
    "run_pso":RUN_PSO,"run_ga":RUN_GA,"run_aco":RUN_ACO,"run_bayes":RUN_BAYES,
    "run_ablation":RUN_FULL_ABLATION,"run_rgb_gray":RUN_RGB_VS_GRAY_VGG,"run_cnn_shuffle":RUN_CNN_ORDER_SHUFFLE,
    "handcraft_size":HANDCRAFT_SIZE,"vgg_size":VGG_SIZE,"glcm_levels":GLCM_LEVELS
}
RUN_SIGNATURE = hashlib.sha256(json.dumps(RUN_SIGNATURE_PAYLOAD,sort_keys=True,default=str).encode()).hexdigest()[:20]
sig_path=OUT/"run_signature.json"
if sig_path.exists():
    old_sig=json.load(open(sig_path,encoding="utf-8")).get("run_signature")
    if old_sig!=RUN_SIGNATURE:
        if FORCE_NEW_RUN_ON_SIGNATURE_MISMATCH:
            raise RuntimeError("Run signature changed. Choose a new OUTPUT_DIR so old and new experiments cannot be mixed.")
        raise RuntimeError(f"Existing checkpoints belong to a different configuration ({old_sig} != {RUN_SIGNATURE}). Set a new OUTPUT_DIR before running.")
else:
    atomic_json({"run_signature":RUN_SIGNATURE,"payload":RUN_SIGNATURE_PAYLOAD},sig_path)
print(f"Resume mode: {RESUME}; run signature: {RUN_SIGNATURE}; output: {OUT}")
if (OUT/"checkpoint_status.json").exists():
    try: print("Previous checkpoint:", json.load(open(OUT/"checkpoint_status.json",encoding="utf-8")))
    except Exception: pass

# ------------------------- FEATURE EXTRACTION ----------------------
def load_rgb_gray(path):
    with Image.open(path) as im:
        im = im.convert("RGB")
        rgb_h = np.asarray(im.resize(HANDCRAFT_SIZE, Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
        gray = rgb2gray(rgb_h).astype(np.float32)
        rgb_vgg = np.asarray(im.resize(VGG_SIZE, Image.Resampling.BILINEAR), dtype=np.float32)
    return rgb_h, gray, rgb_vgg


def robust_entropy(x, bins=32):
    x = np.asarray(x, dtype=np.float64).ravel()
    if np.allclose(x.max(), x.min()):
        return 0.0
    hist, _ = np.histogram(x, bins=bins)
    p = hist.astype(float)
    p = p[p > 0]
    p /= p.sum()
    return float(-(p * np.log2(p)).sum())


def stats7(x):
    z = np.asarray(x, dtype=np.float64).ravel()
    vals = [
        np.mean(z), np.std(z), np.var(z), np.mean(z*z), robust_entropy(z),
        skew(z, bias=False, nan_policy="omit"), kurtosis(z, bias=False, nan_policy="omit")
    ]
    return np.nan_to_num(np.asarray(vals, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def quantize_for_glcm(gray, levels=GLCM_LEVELS):
    g = np.asarray(gray, dtype=np.float32)
    mn, mx = float(g.min()), float(g.max())
    if mx <= mn:
        return np.zeros_like(g, dtype=np.uint8)
    q = np.floor((g - mn) / (mx - mn + 1e-12) * (levels - 1)).astype(np.uint8)
    return q


def glcm_features(gray):
    q = quantize_for_glcm(gray)
    M = graycomatrix(q, distances=GLCM_DISTANCES, angles=GLCM_ANGLES,
                     levels=GLCM_LEVELS, symmetric=True, normed=True)
    props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
    out = []
    for prop in props:
        out.extend(graycoprops(M, prop).ravel().tolist())
    return np.nan_to_num(np.asarray(out, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def dwt_components(gray):
    LL, (LH, HL, HH) = pywt.dwt2(gray, "haar")
    return [LL, LH, HL, HH]


def swt_components(gray):
    # HANDCRAFT_SIZE is even, so level-1 SWT is valid.
    LL, (LH, HL, HH) = pywt.swt2(gray, "haar", level=1)[0]
    return [LL, LH, HL, HH]


def extract_handcrafted(path):
    t0 = time.perf_counter()
    _, gray, _ = load_rgb_gray(path)
    times = {}

    a = time.perf_counter(); f1 = glcm_features(gray); times["F1_GLCM"] = time.perf_counter()-a
    a = time.perf_counter(); dwt = dwt_components(gray); f2 = np.concatenate([stats7(b) for b in dwt]); times["F2_DWT"] = time.perf_counter()-a
    a = time.perf_counter(); swt = swt_components(gray); f3 = np.concatenate([stats7(b) for b in swt]); times["F3_SWT"] = time.perf_counter()-a
    a = time.perf_counter(); f4 = glcm_features(dwt[0]); times["F4_DWT_GLCM"] = time.perf_counter()-a
    a = time.perf_counter(); f5 = glcm_features(swt[0]); times["F5_SWT_GLCM"] = time.perf_counter()-a
    times["handcrafted_total"] = time.perf_counter()-t0
    return f1, f2, f3, f4, f5, times


def build_vgg_extractor():
    base = VGG16(weights="imagenet", include_top=False, input_shape=(*VGG_SIZE, 3))
    base.trainable = False
    x = base.get_layer("block5_conv3").output
    x = GlobalAveragePooling2D(name="block5_conv3_gap")(x)
    model = Model(base.input, x, name="VGG16_block5_conv3_GAP")
    try:
        weight_bytes = int(sum(int(w.numpy().nbytes) for w in model.weights))
        atomic_csv(pd.DataFrame([{"network":"VGG16 ImageNet","feature_layer":"block5_conv3","pooling":"GlobalAveragePooling2D","feature_dimension":512,"parameters":int(model.count_params()),"weights_size_mb":weight_bytes/1024**2}]), OUT/"04b_vgg_model_profile.csv")
    except Exception:
        pass
    return model


def extract_vgg_features(paths, grayscale=False):
    model = build_vgg_extractor()
    feats = []
    start = time.perf_counter()
    for i in tqdm(range(0, len(paths), VGG_BATCH_SIZE), desc=f"VGG16 {'GRAY' if grayscale else 'RGB'}"):
        batch_paths = paths[i:i+VGG_BATCH_SIZE]
        arr = []
        for p in batch_paths:
            with Image.open(p) as im:
                im = im.convert("RGB").resize(VGG_SIZE, Image.Resampling.BILINEAR)
                x = np.asarray(im, dtype=np.float32)
            if grayscale:
                g = np.dot(x[..., :3], [0.2989, 0.5870, 0.1140])
                x = np.stack([g, g, g], axis=-1)
            arr.append(x)
        arr = preprocess_input(np.stack(arr, axis=0))
        pred = model.predict(arr, verbose=0)
        feats.append(pred.astype(np.float32))
    elapsed = time.perf_counter() - start
    return np.vstack(feats), elapsed / len(paths), model


cache_npz = OUT / "cache" / f"features_{DATA_FINGERPRINT}.npz"
cache_profile = OUT / "cache" / f"feature_profile_{DATA_FINGERPRINT}.csv"
chunk_dir = OUT / "cache" / "chunks" / DATA_FINGERPRINT
chunk_dir.mkdir(parents=True, exist_ok=True)


def extract_handcrafted_resume_safe():
    n = len(manifest)
    for start_i in range(0, n, HANDCRAFT_CHUNK_SIZE):
        end_i = min(n, start_i + HANDCRAFT_CHUNK_SIZE)
        cp = chunk_dir / f"hc_{start_i:06d}_{end_i:06d}.npz"
        if RESUME and cp.exists():
            continue
        f1s,f2s,f3s,f4s,f5s, prof=[] ,[] ,[] ,[] ,[] ,[]
        for row in manifest.iloc[start_i:end_i].itertuples():
            a,b,c,d,e,times = extract_handcrafted(row.path)
            f1s.append(a); f2s.append(b); f3s.append(c); f4s.append(d); f5s.append(e)
            prof.append([times[k] for k in ["F1_GLCM","F2_DWT","F3_SWT","F4_DWT_GLCM","F5_SWT_GLCM","handcrafted_total"]])
        atomic_npz(cp, F1=np.asarray(f1s),F2=np.asarray(f2s),F3=np.asarray(f3s),F4=np.asarray(f4s),F5=np.asarray(f5s),profile=np.asarray(prof,dtype=np.float64))
        update_status(f"handcrafted images {start_i}:{end_i}")
    parts=[]
    for start_i in range(0,n,HANDCRAFT_CHUNK_SIZE):
        end_i=min(n,start_i+HANDCRAFT_CHUNK_SIZE)
        z=np.load(chunk_dir/f"hc_{start_i:06d}_{end_i:06d}.npz")
        parts.append(z)
    F=[np.vstack([z[f"F{i}"] for z in parts]) for i in range(1,6)]
    prof=np.vstack([z["profile"] for z in parts])
    return (*F,prof)


def extract_vgg_resume_safe(paths, grayscale=False):
    tag="gray" if grayscale else "rgb"
    n=len(paths); model=None; sec_rows=[]; chunks=[]
    for start_i in range(0,n,VGG_BATCH_SIZE):
        end_i=min(n,start_i+VGG_BATCH_SIZE)
        cp=chunk_dir/f"vgg_{tag}_{start_i:06d}_{end_i:06d}.npy"
        mp=chunk_dir/f"vgg_{tag}_{start_i:06d}_{end_i:06d}.json"
        if RESUME and cp.exists():
            arr=np.load(cp); chunks.append(arr)
            meta=json.load(open(mp)) if mp.exists() else {"sec_per_image":np.nan}
            sec_rows.extend([meta.get("sec_per_image",np.nan)]*len(arr))
            continue
        if model is None: model=build_vgg_extractor()
        arr=[]; t0=time.perf_counter()
        for pth in paths[start_i:end_i]:
            with Image.open(pth) as im:
                im=im.convert("RGB").resize(VGG_SIZE,Image.Resampling.BILINEAR)
                x=np.asarray(im,dtype=np.float32)
            if grayscale:
                g=np.dot(x[...,:3],[0.2989,0.5870,0.1140]); x=np.stack([g,g,g],axis=-1)
            arr.append(x)
        batch=preprocess_input(np.stack(arr,axis=0))
        pred=model.predict(batch,verbose=0).astype(np.float32)
        elapsed=time.perf_counter()-t0
        atomic_npy(cp,pred); atomic_json({"sec_per_image":elapsed/len(pred)},mp)
        chunks.append(pred); sec_rows.extend([elapsed/len(pred)]*len(pred))
        update_status(f"VGG-{tag} images {start_i}:{end_i}")
    if model is not None:
        tf.keras.backend.clear_session()
    return np.vstack(chunks), np.asarray(sec_rows,dtype=float)


if RESUME and cache_npz.exists():
    print(f"Loading cached raw features: {cache_npz}")
    c=np.load(cache_npz)
    F1,F2,F3,F4,F5,F6_RGB,F6_GRAY=[c[k] for k in ["F1","F2","F3","F4","F5","F6_RGB","F6_GRAY"]]
    feature_profile=pd.read_csv(cache_profile) if cache_profile.exists() else pd.DataFrame()
else:
    print("\nExtracting/resuming handcrafted features in atomic chunks...")
    F1,F2,F3,F4,F5,hprof=extract_handcrafted_resume_safe()
    paths=manifest["path"].tolist()
    print("Extracting/resuming VGG16 RGB features...")
    F6_RGB,rgb_secs=extract_vgg_resume_safe(paths,grayscale=False)
    if RUN_RGB_VS_GRAY_VGG:
        print("Extracting/resuming VGG16 grayscale-control features...")
        F6_GRAY,gray_secs=extract_vgg_resume_safe(paths,grayscale=True)
    else:
        F6_GRAY=np.empty_like(F6_RGB); gray_secs=np.full(len(paths),np.nan)
    feature_profile=pd.DataFrame(hprof,columns=["F1_GLCM","F2_DWT","F3_SWT","F4_DWT_GLCM","F5_SWT_GLCM","handcrafted_total"])
    feature_profile["VGG_RGB_sec_per_image"]=rgb_secs
    feature_profile["VGG_GRAY_sec_per_image"]=gray_secs
    atomic_csv(feature_profile,cache_profile)
    atomic_npz(cache_npz,F1=F1,F2=F2,F3=F3,F4=F4,F5=F5,F6_RGB=F6_RGB,F6_GRAY=F6_GRAY)

BLOCKS = [F1,F2,F3,F4,F5,F6_RGB]
assert all(len(b) == len(manifest) for b in BLOCKS)
feature_dims = pd.DataFrame({
    "feature_block": BLOCK_NAMES + (["F6_VGG_GRAY"] if RUN_RGB_VS_GRAY_VGG else []),
    "dimension": [x.shape[1] for x in BLOCKS] + ([F6_GRAY.shape[1]] if RUN_RGB_VS_GRAY_VGG else [])
})
feature_dims.to_csv(OUT / "03_feature_dimensions.csv", index=False)
if not feature_profile.empty:
    pd.DataFrame({"metric": feature_profile.columns,
                  "mean_seconds": [feature_profile[c].mean() for c in feature_profile.columns],
                  "std_seconds": [feature_profile[c].std() for c in feature_profile.columns]}).to_csv(
        OUT / "04_feature_extraction_profile.csv", index=False)

# ---------------------------- SPLITS -------------------------------
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(manifest["class_name"].values)
class_names = label_encoder.classes_.tolist()
sample_groups = manifest["sample_group"].fillna("").values


def make_outer_splits():
    if IS_KTH_TIPS2B:
        splits = []
        for held in ["sample_a","sample_b","sample_c","sample_d"]:
            test_idx = np.where(sample_groups == held)[0]
            train_idx = np.where(sample_groups != held)[0]
            if len(test_idx) == 0:
                continue
            splits.append((train_idx, test_idx, held))
        return splits
    skf = StratifiedKFold(n_splits=GENERIC_OUTER_FOLDS, shuffle=True, random_state=2026)
    return [(tr,te,f"fold_{i+1}") for i,(tr,te) in enumerate(skf.split(np.zeros(len(y)), y))]

outer_splits = make_outer_splits()

split_rows = []
for fold_id,(tr,te,fold_name) in enumerate(outer_splits):
    for idx in tr:
        split_rows.append({"fold":fold_id,"fold_name":fold_name,"image_id":manifest.iloc[idx].image_id,"role":"development"})
    for idx in te:
        split_rows.append({"fold":fold_id,"fold_name":fold_name,"image_id":manifest.iloc[idx].image_id,"role":"test"})
split_manifest = pd.DataFrame(split_rows)
split_manifest.to_csv(OUT / "02_split_manifest.csv", index=False)


def inner_split_indices(indices, seed):
    yy = y[indices]
    gg = sample_groups[indices]
    unique_nonempty = sorted(set(g for g in gg if g))
    if IS_KTH_TIPS2B and len(unique_nonempty) >= 3:
        cv = StratifiedGroupKFold(n_splits=min(INNER_FOLDS, len(unique_nonempty)), shuffle=False)
        return [(indices[a], indices[b]) for a,b in cv.split(np.zeros(len(indices)), yy, groups=gg)]
    cv = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=seed)
    return [(indices[a], indices[b]) for a,b in cv.split(np.zeros(len(indices)), yy)]

# ----------------------- SCALING / REPRESENTATION ------------------
def normalize_weights(w):
    w = np.asarray(w, dtype=np.float64)
    w = np.clip(w, WEIGHT_BOUNDS[0], WEIGHT_BOUNDS[1])
    s = w.sum()
    return (w / s).astype(np.float32) if s > 0 else np.ones_like(w, dtype=np.float32)/len(w)


def fit_block_scalers(train_idx, test_idx, active_blocks=None, alternate_vgg=None):
    active_blocks = list(range(6)) if active_blocks is None else list(active_blocks)
    tr_blocks, te_blocks, scalers = [], [], []
    for j in active_blocks:
        raw = BLOCKS[j]
        if j == 5 and alternate_vgg is not None:
            raw = alternate_vgg
        sc = StandardScaler()
        tr = sc.fit_transform(raw[train_idx]).astype(np.float32)
        te = sc.transform(raw[test_idx]).astype(np.float32)
        tr_blocks.append(tr); te_blocks.append(te); scalers.append(sc)
    return tr_blocks, te_blocks, scalers


def weighted_concat(block_list, weights=None):
    k = len(block_list)
    w = np.ones(k, dtype=np.float32)/k if weights is None else normalize_weights(weights)
    return np.concatenate([b * w[i] for i,b in enumerate(block_list)], axis=1).astype(np.float32)


def precompute_inner_scaled(outer_train_idx, seed, active_blocks, alternate_vgg=None):
    folds = []
    for inner_tr, inner_va in inner_split_indices(outer_train_idx, seed):
        trb,vab,_ = fit_block_scalers(inner_tr, inner_va, active_blocks=active_blocks, alternate_vgg=alternate_vgg)
        folds.append((trb, vab, y[inner_tr], y[inner_va]))
    return folds

# --------------------------- OPTIMIZERS ----------------------------
def make_fitness(outer_train_idx, seed, active_blocks, alternate_vgg=None):
    pre = precompute_inner_scaled(outer_train_idx, seed, active_blocks, alternate_vgg=alternate_vgg)
    def objective(w):
        ww = normalize_weights(w)
        scores = []
        for trb,vab,ytr,yva in pre:
            Xtr = weighted_concat(trb, ww)
            Xva = weighted_concat(vab, ww)
            clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=seed)
            clf.fit(Xtr, ytr)
            scores.append(accuracy_score(yva, clf.predict(Xva)))
        score = float(np.mean(scores))
        return score
    return objective


def _state_path(name, state_key):
    return OUT / "checkpoints" / "optimizer_state" / f"{name}_{task_id(name,state_key)}.pkl"


def optimize_pso(objective, dim, budget, seed, state_key=""):
    sp=_state_path("PSO",state_key); state=load_pickle(sp) if RESUME else None
    lo,hi=WEIGHT_BOUNDS
    if state:
        rng=np.random.default_rng(); rng.bit_generator.state=state["rng_state"]
        S=state["S"]; pos=state["pos"]; vel=state["vel"]; pbest=state["pbest"]; pbest_score=state["pbest_score"]
        gbest=state["gbest"]; gbest_score=state["gbest_score"]; calls=state["calls"]; i=state["i"]
    else:
        rng=np.random.default_rng(seed); S=min(PSO_SWARM_SIZE,budget); pos=rng.uniform(lo,hi,(S,dim)); vel=np.zeros_like(pos)
        pbest=pos.copy(); pbest_score=np.full(S,-np.inf); gbest=pos[0].copy(); gbest_score=-np.inf; calls=0; i=0
    while calls<budget:
        score=objective(pos[i]); calls+=1
        if score>pbest_score[i]: pbest_score[i]=score; pbest[i]=pos[i].copy()
        if score>gbest_score: gbest_score=score; gbest=pos[i].copy()
        i+=1
        if i>=S and calls<budget:
            r1=rng.random((S,dim)); r2=rng.random((S,dim))
            vel=PSO_INERTIA*vel+PSO_C1*r1*(pbest-pos)+PSO_C2*r2*(gbest-pos)
            pos=np.clip(pos+vel,lo,hi); i=0
        atomic_pickle({"rng_state":rng.bit_generator.state,"S":S,"pos":pos,"vel":vel,"pbest":pbest,
                       "pbest_score":pbest_score,"gbest":gbest,"gbest_score":gbest_score,"calls":calls,"i":i},sp)
    return normalize_weights(gbest),float(gbest_score),calls


def optimize_ga(objective, dim, budget, seed, state_key=""):
    sp=_state_path("GA",state_key); state=load_pickle(sp) if RESUME else None
    lo,hi=WEIGHT_BOUNDS
    if state:
        rng=np.random.default_rng(); rng.bit_generator.state=state["rng_state"]
        pop=state["pop"]; scores=state["scores"]; calls=state["calls"]; phase=state.get("phase","stable")
        new_pop=state.get("new_pop"); new_scores=state.get("new_scores")
    else:
        rng=np.random.default_rng(seed); pop_size=min(GA_POP_SIZE,budget); pop=rng.uniform(lo,hi,(pop_size,dim)); scores=np.full(pop_size,-np.inf); calls=0; phase="initial"; new_pop=None; new_scores=None
    pop_size=len(pop)
    # Resume/complete initial population one candidate at a time.
    if phase=="initial":
        for i in range(pop_size):
            if calls>=budget: break
            if np.isfinite(scores[i]): continue
            scores[i]=objective(pop[i]); calls+=1
            atomic_pickle({"rng_state":rng.bit_generator.state,"pop":pop,"scores":scores,"calls":calls,"phase":"initial"},sp)
        phase="stable"; atomic_pickle({"rng_state":rng.bit_generator.state,"pop":pop,"scores":scores,"calls":calls,"phase":phase},sp)
    def tournament():
        ids=rng.integers(0,pop_size,size=min(GA_TOURNAMENT,pop_size)); return pop[ids[np.argmax(scores[ids])]].copy()
    while calls<budget:
        if phase!="generation":
            order=np.argsort(scores)[::-1]; new_pop=[pop[i].copy() for i in order[:min(GA_ELITES,pop_size)]]; new_scores=[float(scores[i]) for i in order[:min(GA_ELITES,pop_size)]]; phase="generation"
            atomic_pickle({"rng_state":rng.bit_generator.state,"pop":pop,"scores":scores,"calls":calls,"phase":phase,"new_pop":new_pop,"new_scores":new_scores},sp)
        while len(new_pop)<pop_size and calls<budget:
            p1,p2=tournament(),tournament(); child=p1.copy()
            if rng.random()<GA_CROSSOVER_RATE:
                alpha=rng.random(dim); child=alpha*p1+(1-alpha)*p2
            mask=rng.random(dim)<GA_MUTATION_RATE
            child[mask]+=rng.normal(0,GA_MUTATION_SIGMA*(hi-lo),size=mask.sum()); child=np.clip(child,lo,hi)
            sc=objective(child); calls+=1; new_pop.append(child); new_scores.append(float(sc))
            atomic_pickle({"rng_state":rng.bit_generator.state,"pop":pop,"scores":scores,"calls":calls,"phase":"generation","new_pop":new_pop,"new_scores":new_scores},sp)
        if len(new_pop)<pop_size:
            order=np.argsort(scores)[::-1]
            while len(new_pop)<pop_size:
                j=order[len(new_pop)%len(order)]; new_pop.append(pop[j].copy()); new_scores.append(float(scores[j]))
        pop=np.asarray(new_pop); scores=np.asarray(new_scores); phase="stable"; new_pop=None; new_scores=None
        atomic_pickle({"rng_state":rng.bit_generator.state,"pop":pop,"scores":scores,"calls":calls,"phase":phase},sp)
    best=int(np.argmax(scores)); return normalize_weights(pop[best]),float(scores[best]),calls

def optimize_acor(objective, dim, budget, seed, state_key=""):
    # Continuous Ant Colony Optimization (ACOR): archive-based Gaussian sampling.
    sp=_state_path("ACO",state_key); state=load_pickle(sp) if RESUME else None
    lo,hi=WEIGHT_BOUNDS
    if state:
        rng=np.random.default_rng(); rng.bit_generator.state=state["rng_state"]
        archive=state["archive"]; scores=state["scores"]; calls=state["calls"]
    else:
        rng=np.random.default_rng(seed); k=min(ACO_ARCHIVE_SIZE,budget); archive=rng.uniform(lo,hi,(k,dim)); scores=np.empty(k); calls=0
        for i in range(k):
            if calls>=budget: break
            scores[i]=objective(archive[i]); calls+=1
            atomic_pickle({"rng_state":rng.bit_generator.state,"archive":archive,"scores":scores,"calls":calls},sp)
    k=len(archive)
    while calls<budget:
        order=np.argsort(scores)[::-1]; archive=archive[order]; scores=scores[order]
        ranks=np.arange(k); probs=np.exp(-(ranks**2)/(2*(ACO_Q*k)**2)); probs=probs/probs.sum()
        idx=int(rng.choice(k,p=probs)); mu=archive[idx]
        sigma=ACO_XI*np.sum(np.abs(archive-mu),axis=0)/max(k-1,1); sigma=np.maximum(sigma,1e-6)
        ant=np.clip(rng.normal(mu,sigma),lo,hi); sc=objective(ant); calls+=1
        archive=np.vstack([archive,ant]); scores=np.append(scores,sc)
        order=np.argsort(scores)[::-1][:k]; archive=archive[order]; scores=scores[order]
        atomic_pickle({"rng_state":rng.bit_generator.state,"archive":archive,"scores":scores,"calls":calls},sp)
    best=int(np.argmax(scores)); return normalize_weights(archive[best]),float(scores[best]),calls


def optimize_bayes(objective, dim, budget, seed, state_key=""):
    sp=_state_path("Bayesian",state_key); state=load_pickle(sp) if RESUME else None
    lo,hi=WEIGHT_BOUNDS; space=[Real(lo,hi,name=f"w{i+1}") for i in range(dim)]
    if state and isinstance(state,dict) and "optimizer" in state:
        opt=state["optimizer"]; calls=state["calls"]
    else:
        opt=Optimizer(space,base_estimator="GP",acq_func="EI",random_state=seed,n_initial_points=min(20,max(5,budget//5),budget)); calls=0
    while calls<budget:
        x=opt.ask(); score=objective(x); opt.tell(x,-score); calls+=1
        atomic_pickle({"optimizer":opt,"calls":calls},sp)
    vals=np.asarray(opt.yi,dtype=float); best=int(np.argmin(vals)); return normalize_weights(opt.Xi[best]),float(-vals[best]),calls


def run_optimizer(name, outer_train_idx, seed, active_blocks, budget, alternate_vgg=None, experiment="FULL"):
    completed=task_path("optimizer",fold_current,seed,name,experiment,tuple(active_blocks),budget)
    if RESUME and completed.exists():
        d=load_pickle(completed); return np.asarray(d["weights"],dtype=np.float32),d["score"],d["calls"],d["elapsed"]
    objective=make_fitness(outer_train_idx,seed,active_blocks,alternate_vgg=alternate_vgg)
    t0=time.perf_counter(); dim=len(active_blocks); sk=f"fold={fold_current}|seed={seed}|exp={experiment}|active={active_blocks}|budget={budget}"
    if name=="PSO": out=optimize_pso(objective,dim,budget,seed,sk)
    elif name=="GA": out=optimize_ga(objective,dim,budget,seed,sk)
    elif name=="ACO": out=optimize_acor(objective,dim,budget,seed,sk)
    elif name=="Bayesian": out=optimize_bayes(objective,dim,budget,seed,sk)
    else: raise ValueError(name)
    weights,score,calls=out; elapsed=time.perf_counter()-t0
    atomic_pickle({"weights":weights,"score":score,"calls":calls,"elapsed":elapsed},completed); update_status(f"optimizer {name} {experiment} fold={fold_current} seed={seed}")
    return weights,score,calls,elapsed

# ---------------------- CLASSIFIER TUNING --------------------------
def tune_reference_classifiers(outer_train_idx, seed):
    # Raw full fusion + pipeline scaler; tuned solely inside development data.
    Xraw = np.concatenate(BLOCKS, axis=1)[outer_train_idx]
    yy = y[outer_train_idx]
    local_splits = inner_split_indices(outer_train_idx, seed)
    pos = {idx:i for i,idx in enumerate(outer_train_idx)}
    cv_local = [(np.array([pos[z] for z in tr]), np.array([pos[z] for z in va])) for tr,va in local_splits]

    svm_pipe = Pipeline([("scale",StandardScaler()),("model",SVC())])
    svm_grid = [
        {"model__kernel":["linear"], "model__C":[0.1,1,10,100]},
        {"model__kernel":["rbf"], "model__C":[0.1,1,10,100], "model__gamma":["scale",0.01,0.1]}
    ]
    gs_svm = GridSearchCV(svm_pipe, svm_grid, scoring="accuracy", cv=cv_local, n_jobs=N_JOBS)
    gs_svm.fit(Xraw, yy)
    svm_params = {k.replace("model__",""):v for k,v in gs_svm.best_params_.items() if k.startswith("model__")}

    rf_pipe = Pipeline([("scale",StandardScaler()),("model",RandomForestClassifier(random_state=seed,n_jobs=N_JOBS))])
    rf_grid = {
        "model__n_estimators":[200,500],
        "model__max_depth":[None,20],
        "model__max_features":["sqrt",0.5],
        "model__min_samples_leaf":[1,2]
    }
    gs_rf = GridSearchCV(rf_pipe, rf_grid, scoring="accuracy", cv=cv_local, n_jobs=N_JOBS)
    gs_rf.fit(Xraw, yy)
    rf_params = {k.replace("model__",""):v for k,v in gs_rf.best_params_.items() if k.startswith("model__")}
    return svm_params, rf_params, float(gs_svm.best_score_), float(gs_rf.best_score_)

# ---------------------- DOWNSTREAM MODELS --------------------------
def make_val_split(outer_train_idx, seed):
    # Use first leakage-free inner fold as early-stopping validation.
    return inner_split_indices(outer_train_idx, seed)[0]


def keras_model_bytes(model):
    """Keras 3 / TensorFlow 2.21-safe model weight size in bytes."""
    total = 0
    for w in model.weights:
        try:
            total += int(w.numpy().nbytes)
        except Exception:
            # Fallback for variables that do not expose a NumPy array directly.
            shape = tuple(int(d) for d in w.shape)
            dtype = tf.as_dtype(w.dtype)
            total += int(np.prod(shape)) * int(dtype.size)
    return int(total)


def sklearn_model_bytes(model):
    try: return len(pickle.dumps(model))
    except Exception: return np.nan


def train_eval_model(model_name, Xtr, ytr, Xte, yte, outer_train_idx, seed, svm_params, rf_params, feature_permutation=None, keras_task_key=None):
    set_seed(seed)
    Xtr2,Xte2 = Xtr,Xte
    if feature_permutation is not None:
        Xtr2 = Xtr[:,feature_permutation]
        Xte2 = Xte[:,feature_permutation]
    t0 = time.perf_counter()
    params_count = np.nan

    if model_name == "SVM":
        m = SVC(**svm_params, probability=False, random_state=seed)
        m.fit(Xtr2,ytr)
        fit_sec = time.perf_counter()-t0
        p0=time.perf_counter(); pred=m.predict(Xte2); pred_sec=time.perf_counter()-p0
        size_bytes=sklearn_model_bytes(m)

    elif model_name == "RF":
        m = RandomForestClassifier(**rf_params, random_state=seed, n_jobs=N_JOBS)
        m.fit(Xtr2,ytr)
        fit_sec = time.perf_counter()-t0
        p0=time.perf_counter(); pred=m.predict(Xte2); pred_sec=time.perf_counter()-p0
        size_bytes=sklearn_model_bytes(m)

    elif model_name in {"MLP","CNN1D"}:
        # Map global outer-train indices to positions in Xtr.
        global_tr, global_va = make_val_split(outer_train_idx, seed)
        pos = {idx:i for i,idx in enumerate(outer_train_idx)}
        itr = np.array([pos[z] for z in global_tr]); iva=np.array([pos[z] for z in global_va])
        n_classes = len(class_names)
        if model_name == "MLP":
            m = Sequential([
                Input(shape=(Xtr2.shape[1],)),
                Dense(256, activation="relu"), Dropout(0.30),
                Dense(64, activation="relu"), Dropout(0.30),
                Dense(n_classes, activation="softmax")
            ])
            train_x,val_x = Xtr2[itr],Xtr2[iva]
            test_x = Xte2
        else:
            m = Sequential([
                Input(shape=(Xtr2.shape[1],1)),
                Conv1D(32,3,activation="relu"), MaxPooling1D(2), Dropout(0.20),
                Conv1D(64,3,activation="relu"), MaxPooling1D(2), Dropout(0.20),
                Flatten(), Dense(64,activation="relu"), Dropout(0.40),
                Dense(n_classes,activation="softmax")
            ])
            train_x,val_x = Xtr2[itr][...,None],Xtr2[iva][...,None]
            test_x = Xte2[...,None]
        m.compile(optimizer=Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        backup_dir=OUT/"checkpoints"/"keras_backups"/(keras_task_key or f"{model_name}_{seed}")
        backup_dir.mkdir(parents=True,exist_ok=True)
        cb=[EarlyStopping(monitor="val_loss",patience=NEURAL_PATIENCE,restore_best_weights=True),
            BackupAndRestore(backup_dir=str(backup_dir),delete_checkpoint=True)]
        m.fit(train_x,ytr[itr],validation_data=(val_x,ytr[iva]),epochs=NEURAL_MAX_EPOCHS,
              batch_size=NEURAL_BATCH_SIZE,verbose=0,callbacks=cb)
        fit_sec=time.perf_counter()-t0
        p0=time.perf_counter(); pred=np.argmax(m.predict(test_x,verbose=0),axis=1); pred_sec=time.perf_counter()-p0
        params_count=int(m.count_params()); size_bytes=keras_model_bytes(m)
        tf.keras.backend.clear_session()
    else:
        raise ValueError(model_name)

    acc=accuracy_score(yte,pred)
    prec,rec,f1,_=precision_recall_fscore_support(yte,pred,average="macro",zero_division=0)
    return {
        "accuracy":acc,"macro_precision":prec,"macro_recall":rec,"macro_f1":f1,
        "fit_sec":fit_sec,"predict_sec_total":pred_sec,"predict_ms_per_sample":pred_sec/len(yte)*1000,
        "model_size_bytes":size_bytes,"trainable_or_model_params":params_count,"pred":pred
    }

# ----------------------- EXPERIMENT RUNNER -------------------------
optimizer_rows=[]; model_rows=[]; class_rows=[]; pred_rows=[]; tuning_rows=[]; weight_rows=[]
models_to_run=["SVM","RF","MLP","CNN1D"]


def _ingest_existing_tasks():
    for fp in sorted((OUT/"checkpoints"/"tasks").glob("*.pkl")):
        d=load_pickle(fp)
        if not isinstance(d,dict): continue
        kind=d.get("kind")
        if kind=="model":
            model_rows.append(d["model_row"]); class_rows.extend(d["class_rows"]); pred_rows.extend(d["pred_rows"])
        elif kind=="tuning": tuning_rows.append(d["row"])
        elif kind=="optimizer_row":
            optimizer_rows.append(d["row"]); weight_rows.extend(d.get("weight_rows",[]))


def flush_raw_tables():
    atomic_csv(pd.DataFrame(optimizer_rows).drop_duplicates(subset=["fold","seed","optimizer","experiment"],keep="last") if optimizer_rows else pd.DataFrame(),OUT/"05_optimizer_runs.csv")
    atomic_csv(pd.DataFrame(model_rows).drop_duplicates(subset=["fold","seed","representation","optimizer","model"],keep="last") if model_rows else pd.DataFrame(),OUT/"06_model_results_raw.csv")
    atomic_csv(pd.DataFrame(class_rows).drop_duplicates(subset=["fold","seed","representation","optimizer","model","class_name"],keep="last") if class_rows else pd.DataFrame(),OUT/"09_classwise_metrics_raw.csv")
    atomic_csv(pd.DataFrame(pred_rows).drop_duplicates(subset=["fold","seed","representation","optimizer","model","image_id"],keep="last") if pred_rows else pd.DataFrame(),OUT/"predictions_raw.csv")
    atomic_csv(pd.DataFrame(tuning_rows).drop_duplicates(subset=["fold","seed"],keep="last") if tuning_rows else pd.DataFrame(),OUT/"classifier_tuning.csv")
    atomic_csv(pd.DataFrame(weight_rows).drop_duplicates(subset=["fold","seed","optimizer","experiment","feature_block"],keep="last") if weight_rows else pd.DataFrame(),OUT/"weights_raw.csv")


def add_class_metrics_rows(fold,seed,representation,optimizer,model,y_true,y_pred):
    rep=classification_report(y_true,y_pred,target_names=class_names,output_dict=True,zero_division=0)
    rows=[]
    for cls in class_names:
        d=rep[cls]; rows.append({"fold":fold,"seed":seed,"representation":representation,"optimizer":optimizer,
            "model":model,"class_name":cls,"precision":d["precision"],"recall":d["recall"],"f1":d["f1-score"],"support":d["support"]})
    return rows


def get_tuning(fold,fold_name,seed,outer_train_idx):
    tp=task_path("tuning",fold,seed)
    if RESUME and tp.exists():
        d=load_pickle(tp); return d["svm_params"],d["rf_params"],d["row"]
    svm_params,rf_params,svm_cv,rf_cv=tune_reference_classifiers(outer_train_idx,seed)
    row={"fold":fold,"fold_name":fold_name,"seed":seed,"svm_params":json.dumps(svm_params),"rf_params":json.dumps(rf_params),
         "svm_inner_cv_accuracy":svm_cv,"rf_inner_cv_accuracy":rf_cv}
    atomic_pickle({"kind":"tuning","svm_params":svm_params,"rf_params":rf_params,"row":row},tp); tuning_rows.append(row); update_status(f"classifier tuning fold={fold} seed={seed}")
    return svm_params,rf_params,row


def record_optimizer(fold,fold_name,seed,opt_name,experiment,active,w,cv_score,calls,elapsed):
    rp=task_path("optimizer_row",fold,seed,opt_name,experiment)
    if RESUME and rp.exists(): return
    row={"fold":fold,"fold_name":fold_name,"seed":seed,"optimizer":opt_name,"experiment":experiment,
         "objective_evaluations":calls,"inner_cv_accuracy":cv_score,"wall_sec":elapsed,
         **{f"w_{BLOCK_NAMES[j]}":float(w[k]) for k,j in enumerate(active)}}
    wr=[{"fold":fold,"seed":seed,"optimizer":opt_name,"experiment":experiment,"feature_block":BLOCK_NAMES[j],"weight":float(w[k])} for k,j in enumerate(active)]
    atomic_pickle({"kind":"optimizer_row","row":row,"weight_rows":wr},rp); optimizer_rows.append(row); weight_rows.extend(wr)


def evaluate_representation(fold,fold_name,seed,representation,optimizer_name,outer_train_idx,outer_test_idx,
                            active_blocks,weights,svm_params,rf_params,alternate_vgg=None,shuffle_cnn=False):
    trb,teb,_=fit_block_scalers(outer_train_idx,outer_test_idx,active_blocks,alternate_vgg=alternate_vgg)
    Xtr=weighted_concat(trb,weights); Xte=weighted_concat(teb,weights); ytr,yte=y[outer_train_idx],y[outer_test_idx]
    selected_models=["CNN1D"] if shuffle_cnn else models_to_run
    for model_name in selected_models:
        permutation=None; actual_rep=representation
        if shuffle_cnn:
            rng=np.random.default_rng(99173); permutation=rng.permutation(Xtr.shape[1]); actual_rep=representation+"__CNN_FIXED_SHUFFLE"
        mp=task_path("model",fold,seed,actual_rep,optimizer_name,model_name)
        if RESUME and mp.exists(): continue
        print(f"    model {model_name}: {actual_rep}")
        kkey=task_id("keras",fold,seed,actual_rep,optimizer_name,model_name)
        res=train_eval_model(model_name,Xtr,ytr,Xte,yte,outer_train_idx,seed,svm_params,rf_params,permutation,kkey)
        row={"fold":fold,"fold_name":fold_name,"seed":seed,"representation":actual_rep,"optimizer":optimizer_name,"model":model_name,
             "n_train":len(outer_train_idx),"n_test":len(outer_test_idx),"feature_dim":Xtr.shape[1],**{k:v for k,v in res.items() if k!="pred"}}
        cr=add_class_metrics_rows(fold,seed,actual_rep,optimizer_name,model_name,yte,res["pred"])
        pr=[{"fold":fold,"seed":seed,"representation":actual_rep,"optimizer":optimizer_name,"model":model_name,
             "image_id":manifest.iloc[idx].image_id,"true":int(true),"pred":int(pred)} for idx,true,pred in zip(outer_test_idx,yte,res["pred"])]
        atomic_pickle({"kind":"model","model_row":row,"class_rows":cr,"pred_rows":pr},mp)
        model_rows.append(row); class_rows.extend(cr); pred_rows.extend(pr); update_status(f"model {model_name} {actual_rep} fold={fold} seed={seed}"); flush_raw_tables()


if RESUME: _ingest_existing_tasks()
flush_raw_tables()

for fold,(outer_train_idx,outer_test_idx,fold_name) in enumerate(outer_splits):
    fold_current=fold
    print(f"\n{'='*80}\nOUTER FOLD {fold+1}/{len(outer_splits)} — test={fold_name} — train={len(outer_train_idx)} test={len(outer_test_idx)}\n{'='*80}")
    for seed in SEEDS:
        print(f"\nSeed {seed}: resume-safe experiment tasks...")
        svm_params,rf_params,_=get_tuning(fold,fold_name,seed,outer_train_idx)

        if RUN_INDIVIDUAL_FEATURE_BASELINES:
            for j,name in enumerate(BLOCK_NAMES):
                evaluate_representation(fold,fold_name,seed,f"{name}_ONLY","None",outer_train_idx,outer_test_idx,[j],[1.0],svm_params,rf_params)
        evaluate_representation(fold,fold_name,seed,"HANDCRAFTED_F1_F5_UNWEIGHTED","None",outer_train_idx,outer_test_idx,[0,1,2,3,4],None,svm_params,rf_params)
        evaluate_representation(fold,fold_name,seed,"FULL_F1_F6_UNWEIGHTED","None",outer_train_idx,outer_test_idx,[0,1,2,3,4,5],None,svm_params,rf_params)

        if RUN_RGB_VS_GRAY_VGG:
            evaluate_representation(fold,fold_name,seed,"F6_VGG_GRAY_ONLY","None",outer_train_idx,outer_test_idx,[5],[1.0],svm_params,rf_params,alternate_vgg=F6_GRAY)
            evaluate_representation(fold,fold_name,seed,"FULL_UNWEIGHTED_WITH_GRAY_VGG","None",outer_train_idx,outer_test_idx,[0,1,2,3,4,5],None,svm_params,rf_params,alternate_vgg=F6_GRAY)

        main_weights={}
        for opt_name,enabled in [("PSO",RUN_PSO),("GA",RUN_GA),("ACO",RUN_ACO),("Bayesian",RUN_BAYES)]:
            if not enabled: continue
            print(f"  Optimizing/resuming {opt_name} ({OPT_EVAL_BUDGET} objective evaluations)...")
            w,cv_score,calls,elapsed=run_optimizer(opt_name,outer_train_idx,seed,[0,1,2,3,4,5],OPT_EVAL_BUDGET,experiment="FULL_F1_F6")
            main_weights[opt_name]=w; record_optimizer(fold,fold_name,seed,opt_name,"FULL_F1_F6",[0,1,2,3,4,5],w,cv_score,calls,elapsed)
            evaluate_representation(fold,fold_name,seed,f"FULL_F1_F6_{opt_name.upper()}",opt_name,outer_train_idx,outer_test_idx,[0,1,2,3,4,5],w,svm_params,rf_params)
            flush_raw_tables()

        if RUN_CNN_ORDER_SHUFFLE and "PSO" in main_weights:
            evaluate_representation(fold,fold_name,seed,"FULL_F1_F6_PSO","PSO",outer_train_idx,outer_test_idx,[0,1,2,3,4,5],main_weights["PSO"],svm_params,rf_params,shuffle_cnn=True)

        if RUN_FULL_ABLATION and RUN_PSO:
            for removed in range(6):
                active=[j for j in range(6) if j!=removed]; exp=f"ABLATION_REMOVE_{BLOCK_NAMES[removed]}"
                print(f"  {exp}: re-optimizing/resuming PSO...")
                w,cv_score,calls,elapsed=run_optimizer("PSO",outer_train_idx,seed,active,ABLATION_EVAL_BUDGET,experiment=exp)
                record_optimizer(fold,fold_name,seed,"PSO",exp,active,w,cv_score,calls,elapsed)
                evaluate_representation(fold,fold_name,seed,exp,"PSO",outer_train_idx,outer_test_idx,active,w,svm_params,rf_params)
                flush_raw_tables()

        update_status(f"COMPLETED fold={fold} seed={seed}")

# Rebuild from authoritative task checkpoints one final time.
optimizer_rows=[]; model_rows=[]; class_rows=[]; pred_rows=[]; tuning_rows=[]; weight_rows=[]; _ingest_existing_tasks(); flush_raw_tables()

# --------------------- SUMMARY + STATISTICS ------------------------
results=pd.DataFrame(model_rows)
classes_df=pd.DataFrame(class_rows)
preds=pd.DataFrame(pred_rows)
opts=pd.DataFrame(optimizer_rows)
weights_df=pd.DataFrame(weight_rows)
tuning_df=pd.DataFrame(tuning_rows)


def ci95_from_foldmeans(g, metric):
    foldmeans=g.groupby("fold")[metric].mean().dropna().values
    n=len(foldmeans)
    mean=float(np.mean(foldmeans)) if n else np.nan
    if n<2: return mean,np.nan,np.nan,np.nan
    sd=float(np.std(foldmeans,ddof=1)); half=float(student_t.ppf(0.975,n-1)*sd/np.sqrt(n))
    return mean,sd,mean-half,mean+half

summary_rows=[]
for (rep,model),g in results.groupby(["representation","model"]):
    m,sd_fold,lo,hi=ci95_from_foldmeans(g,"accuracy")
    summary_rows.append({
        "representation":rep,"model":model,"n_fold_seed_runs":len(g),
        "accuracy_mean_all_runs":g.accuracy.mean(),"accuracy_sd_all_runs":g.accuracy.std(ddof=1),
        "accuracy_foldmean":m,"accuracy_fold_sd":sd_fold,"accuracy_ci95_low":lo,"accuracy_ci95_high":hi,
        "macro_f1_mean":g.macro_f1.mean(),"macro_f1_sd":g.macro_f1.std(ddof=1),
        "macro_precision_mean":g.macro_precision.mean(),"macro_recall_mean":g.macro_recall.mean(),
        "fit_sec_mean":g.fit_sec.mean(),"predict_ms_per_sample_mean":g.predict_ms_per_sample.mean(),
        "model_size_mb_mean":g.model_size_bytes.mean()/1024**2
    })
summary=pd.DataFrame(summary_rows).sort_values(["model","accuracy_mean_all_runs"],ascending=[True,False])
summary.to_csv(OUT/"07_model_summary.csv",index=False)

ablation_summary=summary[summary.representation.str.startswith("ABLATION_") | summary.representation.eq("FULL_F1_F6_PSO")].copy()
ablation_summary.to_csv(OUT/"08_ablation_summary.csv",index=False)

class_summary=(classes_df.groupby(["representation","model","class_name"],as_index=False)
               .agg(precision_mean=("precision","mean"),recall_mean=("recall","mean"),
                    f1_mean=("f1","mean"),f1_sd=("f1","std"),support_total=("support","sum")))
class_summary.to_csv(OUT/"09_classwise_metrics_summary.csv",index=False)

# Paired Wilcoxon tests on fold×seed pairs. Holm correction applied across all generated tests.
comparisons=[]
def paired_test(rep_a,rep_b,model,metric="accuracy"):
    # Outer folds are the independent experimental units; stochastic seeds are averaged within fold.
    a=(results[(results.representation==rep_a)&(results.model==model)]
       .groupby("fold",as_index=False)[metric].mean().rename(columns={metric:"a"}))
    b=(results[(results.representation==rep_b)&(results.model==model)]
       .groupby("fold",as_index=False)[metric].mean().rename(columns={metric:"b"}))
    z=a.merge(b,on="fold")
    if len(z)<2: return None
    diff=z.a-z.b
    if np.allclose(diff,0): stat,p=0.0,1.0
    else: stat,p=wilcoxon(z.a,z.b,zero_method="wilcox",alternative="two-sided")
    n=len(diff); dmean=float(diff.mean()); dsd=float(diff.std(ddof=1)) if n>1 else np.nan
    half=float(student_t.ppf(0.975,n-1)*dsd/np.sqrt(n)) if n>1 else np.nan
    return {"model":model,"metric":metric,"A":rep_a,"B":rep_b,"n_outer_fold_pairs":n,
            "mean_A":z.a.mean(),"mean_B":z.b.mean(),"mean_difference":dmean,
            "difference_ci95_low":dmean-half if n>1 else np.nan,"difference_ci95_high":dmean+half if n>1 else np.nan,
            "wilcoxon_stat":stat,"p_raw":p}


for model in models_to_run:
    pairs=[("FULL_F1_F6_PSO","FULL_F1_F6_UNWEIGHTED"),
           ("FULL_F1_F6_PSO","F6_VGG_RGB_ONLY"),
           ("FULL_F1_F6_PSO","FULL_F1_F6_GA"),
           ("FULL_F1_F6_PSO","FULL_F1_F6_BAYESIAN"),
           ("FULL_F1_F6_PSO","FULL_F1_F6_ACO")]
    if RUN_RGB_VS_GRAY_VGG:
        pairs += [("F6_VGG_RGB_ONLY","F6_VGG_GRAY_ONLY"),
                  ("FULL_F1_F6_UNWEIGHTED","FULL_UNWEIGHTED_WITH_GRAY_VGG")]
    for a,b in pairs:
        if a in set(results.representation) and b in set(results.representation):
            r=paired_test(a,b,model)
            if r: comparisons.append(r)
stats_df=pd.DataFrame(comparisons)
if not stats_df.empty:
    order=np.argsort(stats_df.p_raw.values)
    adjusted=np.empty(len(stats_df)); running=0
    m=len(stats_df)
    for rank,idx in enumerate(order):
        val=min(1.0,(m-rank)*stats_df.p_raw.iloc[idx])
        running=max(running,val); adjusted[idx]=running
    stats_df["p_holm"]=adjusted
stats_df.to_csv(OUT/"10_paired_statistics_wilcoxon.csv",index=False)

# Exact McNemar/binomial test for PSO vs unweighted, per seed, using pooled out-of-fold predictions.
mcnemar_rows=[]
for model in models_to_run:
    for seed in SEEDS:
        a=preds[(preds.representation=="FULL_F1_F6_PSO")&(preds.model==model)&(preds.seed==seed)][["image_id","true","pred"]].rename(columns={"pred":"a"})
        b=preds[(preds.representation=="FULL_F1_F6_UNWEIGHTED")&(preds.model==model)&(preds.seed==seed)][["image_id","pred"]].rename(columns={"pred":"b"})
        z=a.merge(b,on="image_id")
        if z.empty: continue
        aok=z.a==z.true; bok=z.b==z.true
        n10=int(np.sum(aok & ~bok)); n01=int(np.sum(~aok & bok)); n=n10+n01
        p=1.0 if n==0 else binomtest(min(n10,n01),n,0.5,alternative="two-sided").pvalue
        mcnemar_rows.append({"model":model,"seed":seed,"PSO_correct_unweighted_wrong":n10,
                             "PSO_wrong_unweighted_correct":n01,"discordant":n,"exact_p":p})
mcnemar_df=pd.DataFrame(mcnemar_rows)
mcnemar_df.to_csv(OUT/"11_mcnemar_exact.csv",index=False)

runtime_summary=(results.groupby(["representation","model"],as_index=False)
                 .agg(fit_sec_mean=("fit_sec","mean"),fit_sec_sd=("fit_sec","std"),
                      predict_ms_per_sample_mean=("predict_ms_per_sample","mean"),
                      model_size_mb_mean=("model_size_bytes",lambda x: np.nanmean(x)/1024**2),
                      feature_dim=("feature_dim","first")))
_feature_ms=float((feature_profile["handcrafted_total"].mean()+feature_profile["VGG_RGB_sec_per_image"].mean())*1000) if not feature_profile.empty else np.nan
runtime_summary["feature_extraction_ms_per_image"]=_feature_ms
runtime_summary["estimated_end_to_end_ms_per_image"]=runtime_summary["feature_extraction_ms_per_image"]+runtime_summary["predict_ms_per_sample_mean"]
runtime_summary["fused_vector_memory_bytes_per_sample_float32"]=runtime_summary["feature_dim"]*4
runtime_summary.to_csv(OUT/"12_runtime_summary.csv",index=False)

weights_summary=(weights_df.groupby(["optimizer","experiment","feature_block"],as_index=False)
                 .agg(weight_mean=("weight","mean"),weight_sd=("weight","std"))) if not weights_df.empty else pd.DataFrame()
weights_summary.to_csv(OUT/"13_weights_summary.csv",index=False)

# ----------------------- CONFUSION MATRICES ------------------------
# Aggregate OOF predictions across folds for first seed only to avoid counting each image five times.
plot_seed=SEEDS[0]
plot_reps=["F6_VGG_RGB_ONLY","FULL_F1_F6_UNWEIGHTED","FULL_F1_F6_PSO"]
for rep in plot_reps:
    for model in models_to_run:
        z=preds[(preds.representation==rep)&(preds.model==model)&(preds.seed==plot_seed)]
        if z.empty: continue
        cm=confusion_matrix(z.true,z.pred,labels=np.arange(len(class_names)),normalize="true")
        fig,ax=plt.subplots(figsize=(9,8))
        im=ax.imshow(cm,interpolation="nearest",aspect="auto")
        ax.set_title(f"Normalized confusion matrix\n{rep} + {model} (seed {plot_seed})")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_xticks(np.arange(len(class_names))); ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names,rotation=45,ha="right",fontsize=8); ax.set_yticklabels(class_names,fontsize=8)
        fig.colorbar(im,ax=ax,fraction=0.046,pad=0.04)
        fig.tight_layout()
        fig.savefig(OUT/"figures"/f"cm_{rep}_{model}.png",dpi=300,bbox_inches="tight")
        plt.close(fig)

# ------------------------- EXCEL WORKBOOK --------------------------
excel_path=OUT/"14_reviewer_ready_results.xlsx"
with pd.ExcelWriter(excel_path,engine="openpyxl") as writer:
    manifest.to_excel(writer,sheet_name="Dataset",index=False)
    feature_dims.to_excel(writer,sheet_name="Feature_Dimensions",index=False)
    split_manifest.to_excel(writer,sheet_name="Splits",index=False)
    tuning_df.to_excel(writer,sheet_name="Classifier_Tuning",index=False)
    opts.to_excel(writer,sheet_name="Optimizer_Runs",index=False)
    weights_summary.to_excel(writer,sheet_name="Weights_Summary",index=False)
    results.to_excel(writer,sheet_name="Raw_Model_Results",index=False)
    summary.to_excel(writer,sheet_name="Model_Summary",index=False)
    ablation_summary.to_excel(writer,sheet_name="Ablation",index=False)
    class_summary.to_excel(writer,sheet_name="Classwise",index=False)
    stats_df.to_excel(writer,sheet_name="Wilcoxon",index=False)
    mcnemar_df.to_excel(writer,sheet_name="McNemar",index=False)
    runtime_summary.to_excel(writer,sheet_name="Runtime",index=False)

# ---------------------- CONFIG / ENVIRONMENT -----------------------
config={
    "pipeline_version":PIPELINE_VERSION,"dataset_root":str(DATA_ROOT),"dataset_fingerprint":DATA_FINGERPRINT,"is_kth_tips2b_protocol":IS_KTH_TIPS2B,
    "classes":class_names,"n_images":len(manifest),"seeds":SEEDS,"generic_outer_folds":GENERIC_OUTER_FOLDS,
    "inner_folds":INNER_FOLDS,"optimizer_budget":OPT_EVAL_BUDGET,"ablation_budget":ABLATION_EVAL_BUDGET,
    "fitness_model":FITNESS_MODEL_NAME,"fitness_metric":FITNESS_METRIC,"weight_bounds":WEIGHT_BOUNDS,
    "pso":{"swarm":PSO_SWARM_SIZE,"inertia":PSO_INERTIA,"c1":PSO_C1,"c2":PSO_C2},
    "ga":{"population":GA_POP_SIZE,"crossover_rate":GA_CROSSOVER_RATE,"mutation_rate":GA_MUTATION_RATE,
          "mutation_sigma":GA_MUTATION_SIGMA,"tournament":GA_TOURNAMENT,"elites":GA_ELITES},
    "aco":{"method":"ACOR continuous ant-colony optimization","archive_size":ACO_ARCHIVE_SIZE,"q":ACO_Q,"xi":ACO_XI},
    "bayesian":{"method":"Gaussian-process Bayesian optimization (skopt.Optimizer ask/tell)","acquisition":"EI"},
    "vgg":{"network":"VGG16 ImageNet","layer":"block5_conv3","pooling":"GlobalAveragePooling2D","dimension":512,
           "main_input":"RGB","grayscale_control":RUN_RGB_VS_GRAY_VGG},
    "handcrafted_size":HANDCRAFT_SIZE,"glcm_levels":GLCM_LEVELS,"glcm_distances":GLCM_DISTANCES,
    "glcm_angles_radians":GLCM_ANGLES,"neural_epochs_max":NEURAL_MAX_EPOCHS,"neural_batch":NEURAL_BATCH_SIZE,
    "neural_patience":NEURAL_PATIENCE,"resume_enabled":RESUME,"cpu_safe_mode":CPU_SAFE_MODE,"n_jobs":N_JOBS,
    "tf_intra_threads":TF_INTRA_THREADS,"tf_inter_threads":TF_INTER_THREADS,"vgg_batch_size":VGG_BATCH_SIZE
}
with open(OUT/"00_config.json","w") as f: json.dump(config,f,indent=2,default=str)
with open(OUT/"environment.txt","w") as f:
    f.write(f"Python: {sys.version}\nPlatform: {platform.platform()}\n")
    f.write(f"NumPy: {np.__version__}\nPandas: {pd.__version__}\nScipy: {scipy.__version__}\n")
    f.write(f"Scikit-learn: {__import__('sklearn').__version__}\nTensorFlow: {tf.__version__}\nPyWavelets: {pywt.__version__}\n")
    f.write(f"GPU devices: {tf.config.list_physical_devices('GPU')}\n")


# ---------------- REVIEWER EXPERIMENTAL COMPLIANCE ----------------
reviewer_checklist = pd.DataFrame([
    [1,"Novelty framing","MANUSCRIPT","State novelty as optimization/validation framework, not invention of component methods."],
    [2,"Exact train/validation/test partitions","CODE+OUTPUT","02_split_manifest.csv + group-aware outer folds."],
    [3,"No test-set use in optimizer fitness","CODE+OUTPUT","Inner CV uses development data only; outer test used only for final evaluation."],
    [4,"Mathematical fusion definition","CODE+MANUSCRIPT","Scalar-weighted block concatenation after train-only scaling."],
    [5,"Duplicate handcrafted block F6 issue","CODE","Only six nonduplicated blocks; no F1–F5 combined duplicate block."],
    [6,"Optimizer parameters/seeds","CODE+OUTPUT","00_config.json records PSO/GA/ACOR/Bayesian settings and seeds."],
    [7,"Equal optimizer resources","CODE+OUTPUT","All optimizers use identical objective-evaluation budget and inner folds."],
    [8,"Multiple stochastic runs","CODE+OUTPUT","Five fixed seeds in publication mode; mean ± SD reported."],
    [9,"Statistical validation","OUTPUT","Wilcoxon + Holm correction; exact McNemar for paired predictions."],
    [10,"Avoid unsupported SOTA/definitive proof","MANUSCRIPT","Use protocol-matched descriptive claims only."],
    [11,"Dataset naming/protocol consistency","CODE+MANUSCRIPT","KTH-TIPS2b sample-held-out protocol auto-detected; manifest records dataset."],
    [12,"Recent methods/comparability","MANUSCRIPT/LITERATURE","Add recent deep/hybrid studies and mark protocol comparability; do not fabricate direct comparisons."],
    [13,"Optimizer dependence","OUTPUT+MANUSCRIPT","Report classifier/dataset-specific PSO/GA/ACOR/Bayesian results."],
    [14,"Analyze failures","OUTPUT+MANUSCRIPT","Raw/classwise results preserve poor-performing combinations for discussion."],
    [15,"Increment over VGG baseline","OUTPUT","F6 VGG-only vs unweighted and optimized fusion on identical outer splits."],
    [16,"Systematic feature ablation","CODE+OUTPUT","Leave-one-block-out PSO with re-optimization."],
    [17,"RGB vs grayscale","CODE+OUTPUT","RGB VGG vs grayscale VGG controls on identical splits."],
    [18,"VGG layer/dimension/scaling","CODE+OUTPUT","block5_conv3 + GAP=512D; train-only scaling."],
    [19,"1D-CNN locality claim","CODE+OUTPUT","Fixed feature-order permutation control plus MLP baseline."],
    [20,"Classifier hyperparameters","CODE+OUTPUT","SVM/RF tuned only inside development data and recorded."],
    [21,"Entropy interpretation","MANUSCRIPT","Do not infer discriminative ability from lower entropy alone; entropy omitted as evidence."],
    [22,"Runtime/memory/complexity","OUTPUT","Feature extraction, optimizer, fit, prediction latency, dimensions and model size recorded."],
    [23,"Class-specific analysis","OUTPUT","Per-class precision/recall/F1 and normalized confusion matrices."],
    [24,"Classifier consistency","MANUSCRIPT","Use SVM/RF/MLP/CNN consistently; remove KNN unless actually evaluated."],
    [25,"Language/formatting","MANUSCRIPT","Professional copy-edit after results are frozen."],
    [26,"Figure readability","OUTPUT+MANUSCRIPT","300-DPI main confusion matrices; move repetitive panels to supplement."],
    [27,"Limitations","MANUSCRIPT","Acknowledge dataset size, optimizer/classifier dependence, compute overhead, broader validation need."],
    [28,"Reproducibility repository","CODE+OUTPUT","Exact manifest, splits, seeds, configs, environment, checkpoints and tables are saved."],
], columns=["reviewer_comment","topic","handled_by","professional_action"])
reviewer_checklist.to_csv(OUT/"15_reviewer_compliance_checklist.csv",index=False)
with open(OUT/"15_reviewer_compliance_checklist.md","w",encoding="utf-8") as f:
    f.write("# Reviewer compliance checklist\n\n")
    for r in reviewer_checklist.itertuples(index=False):
        f.write(f"## Comment {r.reviewer_comment}: {r.topic}\n**Handled by:** {r.handled_by}\n\n{r.professional_action}\n\n")
with open(OUT/"RUN_COMPLETE.flag","w") as f: f.write(time.strftime("%Y-%m-%d %H:%M:%S"))

print("\n"+"="*80)
print("PIPELINE COMPLETE")
print("="*80)
print(f"Results directory: {OUT}")
print(f"Reviewer-ready workbook: {excel_path}")
print("\nTop result rows (descriptive only; interpret significance with the statistics sheets):")
print(summary.head(20).to_string(index=False))
print("\nIMPORTANT: Do not copy old manuscript numbers. Use only the outputs generated by this run.")
