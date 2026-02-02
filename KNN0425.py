import os
import re
import numpy as np

# Headless backend to avoid Tk-related errors on Windows
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd

from itertools import chain
from collections import defaultdict

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.metrics import pairwise_distances
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import normalize

# NEW: for saving CM bundles (to replot later without rerunning kNN)
import json
from datetime import datetime

# =============================================================================
#                                CONFIG
# =============================================================================

# Base folder containing all 8 datasets
DATA_BASE = 'D:/午/EXTRACTOR/RFF/0910'

# Choose group: 'Q941' for Q_matrix (channel-robust), 'S941' for S_matrix (channel-dependent)
GROUP = 'Q941'   # or 'S941'

# Distances
TRAIN_DISTS = ['0.1cm', '0.3cm', '0.5cm']                          # distances used to train the FE
UNSEEN_DISTS = ['0cm', '0.2cm', '0.4cm', '1cm']                     # unseen distances for FE
ALL_TEST_DISTS = ['0cm', '0.1cm', '0.2cm', '0.3cm', '0.4cm', '0.5cm', '1cm']

# Where to write results
OUTPUT_BASE = f'D:/午/EXTRACTOR/results_KNN/0913_{GROUP}_suite_font'

# What experiments to run (you can edit/extend this list)
# fairness_mode: 'fixed_support' | 'leave_one_out' | 'in_distance_allowed'
# unseen_support_source (for unknown-card): 'from_test' | 'from_known_dis'
#   - 'from_known_dis' pulls support for unseen cards from the *_unknownCard_knownDis root (recommended for fixed_support)
# fixed_support_dists: which distances to use as support for everyone in 'fixed_support' mode
EXPERIMENTS = [
    #{
    #    "name": "knownCard_unknownDis",
    #    "test_root_kind": "knownCard_unknownDis",
    #    "fairness_mode": "leave_one_out",
    #    "unseen_support_source": "from_known_dis",
    #    "fixed_support_dists": TRAIN_DISTS,
    #    "test_dists": UNSEEN_DISTS,      # evaluate per-distance here
    #    "run_all_merged": True,
    #    "include_train_leftover_in_eval": False,
    #},
    {
        "name": "unknownCard_unknownDis",
        "test_root_kind": "unknownCard_unknownDis",
        "fairness_mode": "fixed_support",      # <-- change this
        "unseen_support_source": "from_known_dis",  # <-- and this
        "fixed_support_dists": TRAIN_DISTS,    # unseen support pulled from *_unknownCard_knownDis at 0.1/0.3/0.5
        "test_dists": UNSEEN_DISTS,            # evaluate at 0/0.2/0.4/1 cm
        "run_all_merged": True,
        "include_train_leftover_in_eval": True,
    },
    # {
    #   "name": "unknownCard_knownDis",
    #   "test_root_kind": "unknownCard_knownDis",
    #   "fairness_mode": "in_distance_allowed",
    #   "unseen_support_source": "from_test",
    #   "fixed_support_dists": TRAIN_DISTS,
    #   "test_dists": TRAIN_DISTS,
    #   "run_all_merged": True,
    #   "include_train_leftover_in_eval": False,
    # },
]

# Enrollment & RNG
enroll_percentage = 0.002
random_seed = 42
use_center_1_percent = True
brand_mode = False
train_only_eval_mode = 'match_test_avg'  # 'match_test_avg' | 'match_test_median' | int | anything else => all leftovers

# ---- KNN / selection tuning (all in one place) ----
TUNING = {
    # core kNN
    "knn_metric": "cosine",        # try: 'cosine', 'euclidean', (or 'mahalanobis' with VI)
    "knn_algorithm": "brute",      # 'brute' is safest for cosine
    "knn_weights": "distance",     # 'distance' usually wins in few-shot; try 'uniform'
    #"knn_k_list": [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25],
    "knn_k_list": [25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51],

    # preprocessing
    "use_l2_norm": False,          # extractor already outputs L2-normalized embeddings
    "use_whitening": True,         # Ledoit-Wolf whitening on TRAIN split (recommended)
    "whiten_rescale_l2": False,    # re-L2 after whitening (usually not needed for your case)

    # enrollment centrality (for "center 1%")
    "center_metric": "cosine",

    # alternative decision heads (set one of these True to try)
    "use_classbalanced_vote": False,  # average neighbor scores per class, pick best
    "classbalanced_top_r": None,      # per-class top-r neighbors (None uses all k)
    "use_ncm": False,                 # Nearest Class Mean baseline instead of kNN
}

# =============================================================================
#                           Path helpers
# =============================================================================

def path_for(group: str, suffix: str) -> str:
    """Return full path for a dataset suffix under DATA_BASE."""
    return os.path.join(DATA_BASE, f'0910_{group}_{suffix}')

# Roots we will use
TRAIN_ROOT = path_for(GROUP, 'trained')
KNOWNCARD_UNKNOWNDIS_ROOT = path_for(GROUP, 'knownCard_unknownDis')
UNKNOWNCARD_UNKNOWNDIS_ROOT = path_for(GROUP, 'unknownCard_unknownDis')
UNKNOWNCARD_KNOWNDIS_ROOT = path_for(GROUP, 'unknownCard_knownDis')

# =============================================================================
#                           IO & preprocessing helpers
# =============================================================================

def load_rff_dataset(root_folder, distance_folders):
    X, y = [], []
    if isinstance(distance_folders, str):
        distance_folders = [distance_folders]
    for dist in distance_folders:
        dist_path = os.path.join(root_folder, dist)
        if not os.path.exists(dist_path):
            print(f"[WARNING] Distance folder not found: {dist_path}")
            continue
        for file in os.listdir(dist_path):
            if file.endswith('.npy'):
                label = file.replace('.npy', '')
                data = np.load(os.path.join(dist_path, file), allow_pickle=True)
                data = np.vstack(data) if isinstance(data, list) else data
                data = np.squeeze(data)
                if data.ndim == 1:
                    data = data.reshape(1, -1)
                elif data.ndim == 3:
                    data = data.reshape(data.shape[0], -1)
                X.append(data)
                y.extend([label] * data.shape[0])
    return (np.vstack(X), np.array(y, dtype=str)) if X else (np.empty((0, 512)), np.array([], dtype=str))

def build_pools_by_distance(root, dists, brand_mode=False):
    """Return dict: dist -> (dict label->list-of-arrays)."""
    pools = {}
    for d in dists:
        Xd, yd_raw = load_rff_dataset(root, d)
        yd = map_labels(yd_raw, brand_mode)
        dd = defaultdict(list)
        for x, y in zip(Xd, yd):
            dd[y].append(x)
        pools[d] = dd
    return pools

def extract_brand(lbl: str) -> str:
    m = re.match(r'[A-Za-z]+', lbl)
    return m.group(0) if m else lbl

def map_labels(labels: np.ndarray, brand_mode: bool) -> np.ndarray:
    if not brand_mode:
        return labels
    return np.array([extract_brand(l) for l in labels], dtype=str)

def select_1_percent_center(samples, percentage=0.01, metric='cosine'):
    """Pick the most 'central' subset under the specified metric."""
    n_samples = samples.shape[0]
    n_select = max(1, int(round(n_samples * float(percentage))))
    dists = pairwise_distances(samples, metric=metric)
    avg_dists = dists.mean(axis=1)
    sorted_idx = np.argsort(avg_dists)
    chosen_idx = sorted_idx[:n_select]
    return samples[chosen_idx], chosen_idx

def save_confusion_matrix(cm, labels, title, filename, output_folder, cmap):
    fig, ax = plt.subplots(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=cmap, ax=ax, xticks_rotation=45, colorbar=True)

    num_classes = len(labels)
    fontsize = 10
    for text in disp.text_.ravel():
        text.set_fontsize(fontsize)

    ax.set_xticks(np.arange(num_classes))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=fontsize)
    ax.set_yticks(np.arange(num_classes))
    ax.set_yticklabels(labels, fontsize=fontsize)

    ax.set_title(title, fontsize=16)
    plt.tight_layout(pad=2.0)
    os.makedirs(output_folder, exist_ok=True)
    plt.savefig(os.path.join(output_folder, filename), dpi=300)
    plt.close(fig)

def l2norm(X):
    return normalize(X, norm='l2', axis=1)

def fit_whitener(X_ref):
    """Return f(X) that whitens X using Ledoit-Wolf covariance fit on X_ref."""
    lw = LedoitWolf().fit(X_ref)
    S = lw.covariance_
    w, V = np.linalg.eigh(S + 1e-6*np.eye(S.shape[0], dtype=S.dtype))
    Winv = (V / np.sqrt(np.maximum(w, 1e-12))).dot(V.T)
    def f(X): return X.dot(Winv)
    return f

def preprocess_for_knn(Xtr, Xte):
    """Apply optional L2 and whitening according to TUNING."""
    Xtr_p, Xte_p = Xtr, Xte
    if TUNING["use_l2_norm"]:
        Xtr_p = l2norm(Xtr_p); Xte_p = l2norm(Xte_p)
    if TUNING["use_whitening"]:
        whitener = fit_whitener(Xtr_p)  # fit only on TRAIN split
        Xtr_p = whitener(Xtr_p); Xte_p = whitener(Xte_p)
        if TUNING["whiten_rescale_l2"]:
            Xtr_p = l2norm(Xtr_p); Xte_p = l2norm(Xte_p)
    return Xtr_p, Xte_p

def build_knn(k):
    return KNeighborsClassifier(
        n_neighbors=k,
        metric=TUNING["knn_metric"],
        weights=TUNING["knn_weights"],
        algorithm=TUNING["knn_algorithm"],
        n_jobs=-1
    )

def predict_classbalanced(knn, X, y_train, top_r=None, eps=1e-9):
    """Class-balanced voting over k neighbors."""
    dists, idx = knn.kneighbors(X, return_distance=True)
    y_train = np.asarray(y_train)
    preds = []
    for di, ii in zip(dists, idx):
        cls_scores = defaultdict(list)
        for d, j in zip(di, ii):
            c = y_train[j]
            score = 1.0 / (d + eps)
            cls_scores[c].append(score)
        if top_r is not None:
            for c in cls_scores:
                cls_scores[c] = sorted(cls_scores[c], reverse=True)[:top_r]
        best_c = max(cls_scores.items(), key=lambda kv: (np.mean(kv[1]), np.max(kv[1])))[0]
        preds.append(best_c)
    return np.array(preds, dtype=str)

def nearest_class_mean_predict(X_train, y_train, X_test, metric='cosine'):
    classes = sorted(set(y_train))
    protos = np.vstack([X_train[y_train == c].mean(axis=0, keepdims=True) for c in classes])
    if metric == 'cosine':
        if TUNING["use_l2_norm"]:
            protos = l2norm(protos); Xq = l2norm(X_test)
        else:
            Xq = X_test
        sims = Xq @ protos.T
        idx = np.argmax(sims, axis=1)
    else:
        d = pairwise_distances(X_test, protos, metric=metric)
        idx = np.argmin(d, axis=1)
    return np.array([classes[i] for i in idx], dtype=str)

def run_knn_grid(Xtr, ytr, Xte, yte, tag, outdir):
    """Sweep k, return best predictions."""
    Xtr_p, Xte_p = preprocess_for_knn(Xtr, Xte)

    results = []
    best = None

    for k in TUNING["knn_k_list"]:
        k_eff = min(k, max(1, Xtr_p.shape[0]))
        knn = build_knn(k_eff)
        knn.fit(Xtr_p, ytr)

        if TUNING["use_classbalanced_vote"]:
            yhat = predict_classbalanced(knn, Xte_p, ytr, top_r=TUNING["classbalanced_top_r"])
        elif TUNING["use_ncm"]:
            yhat = nearest_class_mean_predict(Xtr_p, ytr, Xte_p, metric=TUNING["knn_metric"])
        else:
            yhat = knn.predict(Xte_p)

        acc = accuracy_score(yte, yhat)
        results.append({
            "k": k_eff,
            "metric": TUNING["knn_metric"],
            "weights": TUNING["knn_weights"],
            "l2": TUNING["use_l2_norm"],
            "whiten": TUNING["use_whitening"],
            "acc": acc
        })
        if (best is None) or (acc > best["acc"]):
            best = {"k": k_eff, "yhat": yhat, "acc": acc}

    os.makedirs(outdir, exist_ok=True)
    pd.DataFrame(results).to_csv(os.path.join(outdir, f'grid_{tag}.csv'), index=False)
    return best

# =============================================================================
#                          NEW: CM bundle saver
# =============================================================================

def save_cm_bundle(y_true, y_pred, labels_order, outdir, prefix, meta_extra=None):
    """
    Store everything needed to re-plot a confusion matrix later without re-running kNN:
      - y_true / y_pred (strings)
      - labels_order used for cm axes
      - meta (knn settings, exp info, timestamp)
      - optional: the numeric CM itself for quick use

    Files written in `outdir` (prefix = 'random' or 'center'):
      <prefix>_cm_bundle.npz  : y_true, y_pred, labels_order (NumPy)
      <prefix>_pairs.csv      : two-column CSV (y_true, y_pred)
      <prefix>_meta.json      : experiment + tuning metadata
      <prefix>_cm.npy/.csv    : numeric confusion matrix for convenience
    """
    os.makedirs(outdir, exist_ok=True)

    # Ensure string dtype for safe round-trips
    y_true = np.asarray(y_true, dtype='U')
    y_pred = np.asarray(y_pred, dtype='U')
    labels_order = list(labels_order)

    # Core bundle (fast to load later)
    np.savez_compressed(
        os.path.join(outdir, f"{prefix}_cm_bundle.npz"),
        y_true=y_true,
        y_pred=y_pred,
        labels_order=np.array(labels_order, dtype='U'),
    )

    # Human-friendly pairs csv
    pd.DataFrame({'y_true': y_true, 'y_pred': y_pred}) \
      .to_csv(os.path.join(outdir, f"{prefix}_pairs.csv"), index=False)

    # Numeric CM (optional but handy)
    cm = confusion_matrix(y_true, y_pred, labels=labels_order)
    np.save(os.path.join(outdir, f"{prefix}_cm.npy"), cm)
    pd.DataFrame(cm, index=labels_order, columns=labels_order) \
      .to_csv(os.path.join(outdir, f"{prefix}_cm.csv"))

    # Rich metadata for reproducibility
    meta = {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'group': GROUP,
        'train_dists': list(TRAIN_DISTS),
        'unseen_dists': list(UNSEEN_DISTS),
        'all_test_dists': list(ALL_TEST_DISTS),
        'tuning': {
            'knn_metric': TUNING['knn_metric'],
            'knn_algorithm': TUNING['knn_algorithm'],
            'knn_weights': TUNING['knn_weights'],
            'use_l2_norm': TUNING['use_l2_norm'],
            'use_whitening': TUNING['use_whitening'],
            'whiten_rescale_l2': TUNING['whiten_rescale_l2'],
            'use_classbalanced_vote': TUNING['use_classbalanced_vote'],
            'classbalanced_top_r': TUNING['classbalanced_top_r'],
            'use_ncm': TUNING['use_ncm'],
        },
    }
    if meta_extra:
        # Ensure JSON-serializable
        def _to_jsonable(x):
            if isinstance(x, (set, tuple)):
                return list(x)
            return x
        meta.update({k: _to_jsonable(v) for k, v in meta_extra.items()})

    with open(os.path.join(outdir, f"{prefix}_meta.json"), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

# =============================================================================
#                                Main
# =============================================================================
rng = np.random.default_rng(random_seed)
os.makedirs(OUTPUT_BASE, exist_ok=True)

# Load TRAIN once (seen cards at TRAIN_DISTS)
X_train_full, y_train_full_raw = load_rff_dataset(TRAIN_ROOT, TRAIN_DISTS)
if X_train_full.size == 0:
    raise RuntimeError("[FATAL] Empty train features. Check TRAIN_ROOT or .npy files.")
y_train_full = map_labels(y_train_full_raw, brand_mode)
seen_labels = set(sorted(set(y_train_full)))

# Build master pool for seen labels from TRAIN
train_pool_seen_master = defaultdict(list)
for x, y in zip(X_train_full, y_train_full):
    train_pool_seen_master[y].append(x)

print(f"[INFO] TRAIN loaded: #samples={len(y_train_full)} | #seen labels={len(seen_labels)} | brand_mode={brand_mode}")

def evaluate_experiment(exp_cfg):
    name = exp_cfg["name"]
    test_root_kind = exp_cfg["test_root_kind"]
    fairness_mode = exp_cfg["fairness_mode"]
    unseen_support_source = exp_cfg["unseen_support_source"]
    fixed_support_dists = exp_cfg["fixed_support_dists"]
    run_all_merged = exp_cfg.get("run_all_merged", True)
    include_train_leftover_in_eval = exp_cfg.get("include_train_leftover_in_eval", False)
    test_dists = exp_cfg.get("test_dists", UNSEEN_DISTS)

    # Resolve roots
    if test_root_kind == "knownCard_unknownDis":
        test_root = KNOWNCARD_UNKNOWNDIS_ROOT
    elif test_root_kind == "unknownCard_unknownDis":
        test_root = UNKNOWNCARD_UNKNOWNDIS_ROOT
    elif test_root_kind == "unknownCard_knownDis":
        test_root = UNKNOWNCARD_KNOWNDIS_ROOT
    else:
        raise ValueError(f"Unknown test_root_kind: {test_root_kind}")

    out_dir = os.path.join(OUTPUT_BASE, f"{name}__{fairness_mode}__{unseen_support_source}")
    os.makedirs(out_dir, exist_ok=True)

    # Pre-build pools by distance for the test root
    TEST_POOLS_BY_DIST = build_pools_by_distance(test_root, test_dists, brand_mode=brand_mode)

    # For fixed_support using unknownCard_knownDis as source
    SUPPORT_POOLS_BY_DIST = None
    if fairness_mode == "fixed_support" and unseen_support_source == "from_known_dis":
        SUPPORT_POOLS_BY_DIST = build_pools_by_distance(UNKNOWNCARD_KNOWNDIS_ROOT, fixed_support_dists, brand_mode=brand_mode)

    # Helper: get unseen support pool for a label according to fairness policy
    def unseen_support_pool(lbl, current_dist, feat_dim):
        # Return ndarray [N, D] of candidate support for an unseen label
        if fairness_mode == "in_distance_allowed":
            arr = np.array(TEST_POOLS_BY_DIST.get(current_dist, {}).get(lbl, []))
            return arr if arr.size else np.empty((0, feat_dim))

        elif fairness_mode == "leave_one_out":
            # Use all test distances except the current
            other = [d for d in test_dists if d != current_dist]
            pooled_lists = [TEST_POOLS_BY_DIST.get(d, {}).get(lbl, []) for d in other]
            pooled = list(chain.from_iterable(pooled_lists))
            return np.array(pooled) if len(pooled) else np.empty((0, feat_dim))

        elif fairness_mode == "fixed_support":
            if unseen_support_source == "from_known_dis":
                pooled_lists = [SUPPORT_POOLS_BY_DIST.get(d, {}).get(lbl, []) for d in fixed_support_dists]
                pooled = list(chain.from_iterable(pooled_lists))
                return np.array(pooled) if len(pooled) else np.empty((0, feat_dim))
            elif unseen_support_source == "from_test":
                pooled_lists = [TEST_POOLS_BY_DIST.get(d, {}).get(lbl, []) for d in fixed_support_dists]
                pooled = list(chain.from_iterable(pooled_lists))
                return np.array(pooled) if len(pooled) else np.empty((0, feat_dim))
            else:
                raise ValueError(f"unseen_support_source not recognized: {unseen_support_source}")
        else:
            raise ValueError(f"fairness_mode not recognized: {fairness_mode}")

    # Shared evaluator (can run ALL merged + per-distance)
    def evaluate_distance_set(dist_name, dist_list):
        print("\n" + "="*70)
        print(f"[INFO] {name}: Evaluating TEST distance set: {dist_name} -> {dist_list if isinstance(dist_list, list) else [dist_list]}")
        print("="*70)

        dist_out = os.path.join(out_dir, str(dist_name))
        os.makedirs(dist_out, exist_ok=True)

        # Load this test set (subset of distances within test_root)
        X_test_full, y_test_full_raw = load_rff_dataset(test_root, dist_list)
        if X_test_full.size == 0:
            print(f"[WARN] No test samples for distance set '{dist_name}'. Skipping.")
            return

        y_test_full = map_labels(y_test_full_raw, brand_mode)
        test_labels = sorted(set(y_test_full))
        print(f"[INFO] #Test samples (set {dist_name}): {len(y_test_full)}")
        print(f"[INFO] Test labels (set {dist_name}): {test_labels}")

        # Build a label->list pool for this set only (current test distances)
        test_pool_all = defaultdict(list)
        for x, y in zip(X_test_full, y_test_full):
            test_pool_all[y].append(x)

        # Decide eval universe
        all_eval_labels = test_labels if not include_train_leftover_in_eval else sorted(set(test_labels) | seen_labels)

        # Per-label test counts and average (non-zero)
        test_count = {lbl: len(test_pool_all[lbl]) for lbl in all_eval_labels}
        nonzero_counts = [c for c in test_count.values() if c > 0]
        avg_test_per_label = np.mean(nonzero_counts) if nonzero_counts else 1.0
        K_target = max(1, int(round(avg_test_per_label * enroll_percentage)))

        # Determine global equal K with caps
        labels_seen_here   = [lbl for lbl in all_eval_labels if lbl in seen_labels]
        labels_unseen_here = [lbl for lbl in all_eval_labels if lbl not in seen_labels]

        caps_seen = [len(train_pool_seen_master[lbl]) for lbl in labels_seen_here]
        caps_unseen = [max(0, test_count[lbl] - 1) for lbl in labels_unseen_here]  # if support is drawn from test distances for this set

        # If we're NOT drawing unseen support from the current test distances (e.g., fixed_support/from_known_dis),
        # then unseen caps should be computed from that support source instead.
        if fairness_mode == "fixed_support" and unseen_support_source == "from_known_dis":
            # Each unseen label has sum over fixed_support_dists in UNKNOWNCARD_KNOWNDIS_ROOT
            # We won't compute exact cap here; with your data (8000 per dist) it's large. Keep K_target.
            pass
        elif fairness_mode == "leave_one_out":
            # unseen support is from OTHER test distances; cap is huge in your data; safe to ignore here.
            pass

        if labels_seen_here or labels_unseen_here:
            K_final = K_target
            if caps_seen:
                K_final = min(K_final, min(caps_seen))
            if caps_unseen and (fairness_mode == "in_distance_allowed"):
                # Only relevant when unseen support is drawn from the same set we are evaluating
                K_final = min(K_final, min(caps_unseen))
        else:
            raise RuntimeError(f"[FATAL] Set {dist_name}: No labels available to enroll.")

        if K_final < 1:
            raise RuntimeError(f"[FATAL] Set {dist_name}: Cannot enroll ≥1 sample/label while leaving enough for evaluation.")

        print(f"[INFO] Set {dist_name}: Equal per-label enrollment -> target={K_target}, final K={K_final}")
        print(f"[INFO] Set {dist_name}: Seen={labels_seen_here} | Unseen={labels_unseen_here}")

        # Compute cap for train-only eval leftovers (optional)
        if include_train_leftover_in_eval:
            if isinstance(train_only_eval_mode, int):
                train_only_eval_n = max(1, train_only_eval_mode)
            elif train_only_eval_mode == 'match_test_avg':
                train_only_eval_n = max(1, int(round(avg_test_per_label))) if nonzero_counts else 1
            elif train_only_eval_mode == 'match_test_median':
                med = int(round(np.median(nonzero_counts))) if nonzero_counts else 1
                train_only_eval_n = max(1, med)
            else:
                train_only_eval_n = None  # None => take all leftovers for train-only labels
            print(f"[INFO] Set {dist_name}: Train-only eval cap per label: {train_only_eval_n}")

        # Split builder
        def build_split(method="random"):
            X_tr_list, y_tr_list = [], []
            X_te_list, y_te_list = [], []

            feat_dim = X_test_full.shape[1]  # used for empty pools

            for lbl in all_eval_labels:
                if lbl in seen_labels:
                    pool = np.array(train_pool_seen_master[lbl])
                    N = pool.shape[0]
                    if N < K_final:
                        raise RuntimeError(f"[FATAL] Label {lbl} ({dist_name}): need {K_final} train samples, but only {N} available.")

                    # choose K_final
                    if method == "random":
                        idx = rng.permutation(N)
                        chosen_idx = idx[:K_final]
                        chosen = pool[chosen_idx]
                    else:
                        pct = K_final / N
                        chosen, chosen_idx = select_1_percent_center(pool, pct, metric=TUNING["center_metric"])
                        if chosen.shape[0] > K_final:
                            chosen = chosen[:K_final]
                            chosen_idx = chosen_idx[:K_final]
                        elif chosen.shape[0] < K_final:
                            need = K_final - chosen.shape[0]
                            remaining_idx = np.setdiff1d(np.arange(N), chosen_idx, assume_unique=False)
                            extra_idx = rng.permutation(remaining_idx)[:need]
                            chosen = np.vstack([chosen, pool[extra_idx]])
                            chosen_idx = np.concatenate([chosen_idx, extra_idx])

                    # train
                    X_tr_list.append(chosen)
                    y_tr_list.extend([lbl] * K_final)

                    # eval from TEST (if present)
                    te_pool_test = np.array(test_pool_all[lbl]) if lbl in test_pool_all else np.empty((0, pool.shape[1]))
                    if te_pool_test.size > 0:
                        X_te_list.append(te_pool_test)
                        y_te_list.extend([lbl] * te_pool_test.shape[0])

                    # optionally add TRAIN leftovers to eval
                    if include_train_leftover_in_eval:
                        leftover_idx = np.setdiff1d(np.arange(N), chosen_idx, assume_unique=False)
                        if leftover_idx.size > 0:
                            if test_count.get(lbl, 0) == 0:
                                m = leftover_idx.size if ('train_only_eval_n' not in locals() or train_only_eval_n is None) else min(leftover_idx.size, train_only_eval_n)
                                if m > 0:
                                    sel_idx = rng.permutation(leftover_idx)[:m]
                                    leftover_train = pool[sel_idx]
                                    X_te_list.append(leftover_train)
                                    y_te_list.extend([lbl] * leftover_train.shape[0])
                            else:
                                leftover_train = pool[leftover_idx]
                                X_te_list.append(leftover_train)
                                y_te_list.extend([lbl] * leftover_train.shape[0])

                else:
                    # Unseen label: build support pool per fairness policy
                    current_dist = dist_list if isinstance(dist_list, str) else None
                    upool = None

                    if fairness_mode in ("in_distance_allowed", "leave_one_out"):
                        if current_dist is None:
                            # ALL-merged case: allow using all test_dists as unseen support (except not meaningful for leave_one_out)
                            pooled_lists = [TEST_POOLS_BY_DIST.get(d, {}).get(lbl, []) for d in test_dists]
                            upool = np.array(list(chain.from_iterable(pooled_lists))) if sum(map(len, pooled_lists)) else np.empty((0, feat_dim))
                        else:
                            upool = unseen_support_pool(lbl, current_dist, feat_dim)

                    elif fairness_mode == "fixed_support":
                        # Use fixed distances either from test root or from *_unknownCard_knownDis (recommended)
                        # current_dist not used here
                        upool = unseen_support_pool(lbl, current_dist, feat_dim)

                    else:
                        raise ValueError("Unknown fairness_mode")

                    N = upool.shape[0]
                    if N < K_final + 1:
                        # If support pool is small (shouldn't happen in your data), relax to use all but 1
                        if N <= 1:
                            raise RuntimeError(f"[FATAL] Label {lbl} (unseen,{dist_name}): not enough support (N={N}).")
                        K_use = min(K_final, N - 1)
                    else:
                        K_use = K_final

                    if method == "random":
                        idx = rng.permutation(N)
                        tr_idx = idx[:K_use]
                        te_idx = idx[K_use:]
                        chosen = upool[tr_idx]
                        leftover = upool[te_idx]
                    else:
                        pct = K_use / N
                        chosen, chosen_idx = select_1_percent_center(upool, pct, metric=TUNING["center_metric"])
                        if chosen.shape[0] > K_use:
                            chosen = chosen[:K_use]
                            chosen_idx = chosen_idx[:K_use]
                        elif chosen.shape[0] < K_use:
                            need = K_use - chosen.shape[0]
                            remaining_idx = np.setdiff1d(np.arange(N), chosen_idx, assume_unique=False)
                            pad_idx = rng.permutation(remaining_idx)[:need]
                            chosen = np.vstack([chosen, upool[pad_idx]])
                            chosen_idx = np.concatenate([chosen_idx, pad_idx])
                        all_idx = np.arange(N)
                        te_idx = np.setdiff1d(all_idx, chosen_idx, assume_unique=False)
                        leftover = upool[te_idx]

                    X_tr_list.append(chosen)
                    y_tr_list.extend([lbl] * K_use)
                    X_te_list.append(leftover)
                    y_te_list.extend([lbl] * leftover.shape[0])

            X_tr = np.vstack(X_tr_list)
            y_tr = np.array(y_tr_list, dtype=str)
            X_te = np.vstack(X_te_list) if len(X_te_list) else np.empty((0, X_tr.shape[1]))
            y_te = np.array(y_te_list, dtype=str)
            return X_tr, y_tr, X_te, y_te

        # -------- RANDOM-BASED --------
        X_train_rand, y_train_rand, X_test_rand, y_test_rand = build_split(method="random")
        best_rand = run_knn_grid(X_train_rand, y_train_rand, X_test_rand, y_test_rand, tag="random", outdir=dist_out)

        labels_order = sorted(set(y_test_rand))
        cm_rand = confusion_matrix(y_test_rand, best_rand["yhat"], labels=labels_order)
        acc_rand = best_rand["acc"]

        save_confusion_matrix(
            cm_rand, labels_order,
            f'KNN ({TUNING["knn_metric"]},{TUNING["knn_weights"]}) best k={best_rand["k"]} | Acc: {acc_rand:.4f}',
            'confusion_matrix_random.png', dist_out, cmap='Blues'
        )

        # NEW: save everything needed to re-plot later
        save_cm_bundle(
            y_true=y_test_rand,
            y_pred=best_rand["yhat"],
            labels_order=labels_order,
            outdir=dist_out,
            prefix='random',
            meta_extra={
                'exp_name': name,
                'dist_set': dist_name,
                'mode': 'random',
                'best_k': int(best_rand['k']),
                'acc': float(acc_rand),
                'fairness_mode': fairness_mode,
                'unseen_support_source': unseen_support_source,
                'fixed_support_dists': list(fixed_support_dists),
                'include_train_leftover_in_eval': include_train_leftover_in_eval,
                'enroll_percentage': float(enroll_percentage),
                'use_center_1_percent': bool(use_center_1_percent),
                'brand_mode': bool(brand_mode),
                'train_only_eval_mode': str(train_only_eval_mode),
                'test_dists_evaluated': list(dist_list if isinstance(dist_list, list) else [dist_list]),
            }
        )

        pd.DataFrame({
            'Label': labels_order,
            'Accuracy': np.diag(cm_rand) / np.maximum(cm_rand.sum(axis=1), 1),
            'Total Samples': cm_rand.sum(axis=1),
            'Correct Predictions': np.diag(cm_rand)
        }).to_csv(os.path.join(dist_out, 'per_card_accuracy_random.csv'), index=False)

        # -------- CENTER-BASED --------
        if use_center_1_percent:
            X_train_ctr, y_train_ctr, X_test_ctr, y_test_ctr = build_split(method="center")
            best_ctr = run_knn_grid(X_train_ctr, y_train_ctr, X_test_ctr, y_test_ctr, tag="center", outdir=dist_out)

            labels_order_c = sorted(set(y_test_ctr))
            cm_center = confusion_matrix(y_test_ctr, best_ctr["yhat"], labels=labels_order_c)
            acc_center = best_ctr["acc"]

            save_confusion_matrix(
                cm_center, labels_order_c,
                f'KNN ({TUNING["knn_metric"]},{TUNING["knn_weights"]}) best k={best_ctr["k"]} | Acc: {acc_center:.4f}',
                'confusion_matrix_center.png', dist_out, cmap='Blues'
            )

            # NEW: save everything needed to re-plot later
            save_cm_bundle(
                y_true=y_test_ctr,
                y_pred=best_ctr["yhat"],
                labels_order=labels_order_c,
                outdir=dist_out,
                prefix='center',
                meta_extra={
                    'exp_name': name,
                    'dist_set': dist_name,
                    'mode': 'center',
                    'best_k': int(best_ctr['k']),
                    'acc': float(acc_center),
                    'fairness_mode': fairness_mode,
                    'unseen_support_source': unseen_support_source,
                    'fixed_support_dists': list(fixed_support_dists),
                    'include_train_leftover_in_eval': include_train_leftover_in_eval,
                    'enroll_percentage': float(enroll_percentage),
                    'use_center_1_percent': bool(use_center_1_percent),
                    'brand_mode': bool(brand_mode),
                    'train_only_eval_mode': str(train_only_eval_mode),
                    'test_dists_evaluated': list(dist_list if isinstance(dist_list, list) else [dist_list]),
                }
            )

            pd.DataFrame({
                'Label': labels_order_c,
                'Accuracy': np.diag(cm_center) / np.maximum(cm_center.sum(axis=1), 1),
                'Total Samples': cm_center.sum(axis=1),
                'Correct Predictions': np.diag(cm_center)
            }).to_csv(os.path.join(dist_out, 'per_card_accuracy_center.csv'), index=False)

    # Run ALL-merged first if requested
    if exp_cfg.get("run_all_merged", True):
        evaluate_distance_set('ALL', exp_cfg["test_dists"])

    # Then per-distance evaluation
    for d in exp_cfg["test_dists"]:
        evaluate_distance_set(d, d)

# =============================== Run all experiments ===============================
for exp in EXPERIMENTS:
    evaluate_experiment(exp)

print("\n[DONE]")