import os, re, json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau

from cfg import cfg
from dataset.tool import get_train_val_dataset
from utils.tool2 import fixed_seed, Gtrain
from pytorch_metric_learning.losses import TripletMarginLoss
from pytorch_metric_learning.miners import TripletMarginMiner

from model.resnet import ResNet                  # your extractor (unchanged)
from model.GRL_B import ExtractorWithGRL         # GRL + domain head wrapper
from model.head import TxHead                   # small TX classifier head

# ------- helpers to build label maps (strings -> indices) -------
def _natural_key(s: str):
    # sorts A1,A2,...,A10 naturally
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', str(s))]

def _parse_distance_cm(token: str) -> float:
    s = str(token).strip().lower()
    if s.endswith("cm"):
        s = s[:-2]
    return float(s)

def _flatten_to_base_and_indices(ds):
    """
    Follow .dataset recursively until we reach the base dataset.
    Compose indices so they reference the base dataset directly.
    Returns (base_dataset, base_indices_list).
    """
    base = ds
    idxs = None  # None means "all indices" so far
    while isinstance(base, Subset):
        if idxs is None:
            idxs = list(base.indices)
        else:
            idxs = [base.indices[i] for i in idxs]
        base = base.dataset
    if idxs is None:
        idxs = list(range(len(base)))
    return base, idxs

def _iter_samples_flat(ds_subset):
    """
    Iterate samples from the base dataset coordinate space.
    For RxAgnosticCISDataset we expect base.samples[i] = (path, label_str, distance_str).
    """
    base, idxs = _flatten_to_base_and_indices(ds_subset)
    if hasattr(base, "samples"):
        for i in idxs:
            yield base.samples[i]
    else:
        # Fallback: call __getitem__ (slower; may load .mat)
        for i in idxs:
            item = base[i]
            # Try to extract (path?, class, distance)
            if isinstance(item, (list, tuple)) and len(item) >= 3:
                yield (None, item[1], item[2])
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                yield (None, item[1], None)
            else:
                raise RuntimeError("Dataset item format unexpected; cannot build label maps.")

def build_maps_from_splits(train_subset, val_subset):
    """
    Build class_to_idx and dom_to_idx by flattening both subsets to the same base dataset.
    """
    cls_set, dom_set = set(), set()
    for _, y, d in _iter_samples_flat(train_subset):
        cls_set.add(str(y)); dom_set.add(str(d))
    for _, y, d in _iter_samples_flat(val_subset):
        cls_set.add(str(y)); dom_set.add(str(d))

    class_to_idx = {name: i for i, name in enumerate(sorted(cls_set, key=_natural_key))}
    dom_to_idx   = {name: i for i, name in enumerate(sorted(dom_set, key=lambda t: (_parse_distance_cm(t), str(t).lower())))}  # numeric cm sort
    return class_to_idx, dom_to_idx

def main():
    fixed_seed(cfg.get('seed', 1206))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    exp_name = 'GRLGaps_0928_3dis_AF16'
    save_path = os.path.join("D:/午/EXTRACTOR/save", exp_name)
    os.makedirs(save_path, exist_ok=True)

    print('start loading data')
    print('current device:', device)
    train_root = cfg['train_data_root']
    train_ratio, val_ratio = cfg['train_ratio'], cfg['val_ratio']
    batch_size = cfg['batch_size']

    train_set, val_set, _ = get_train_val_dataset(
        data_root=train_root, train_ratio=train_ratio, val_ratio=val_ratio
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print('end of loading data')
    print('Numbers of training data   =', len(train_loader.dataset))
    print('Numbers of validation data =', len(val_loader.dataset))
    print('Batch Size  =', batch_size)

    # --- maps (string labels -> indices). Use both splits to be safe ---
    class_to_idx, dom_to_idx = build_maps_from_splits(train_loader.dataset, val_loader.dataset)
    NUM_CLASSES, NUM_DOMAINS = len(class_to_idx), len(dom_to_idx)
    print("Domain map (distance → id):", dom_to_idx)   # ← moved here
    print("Class map (card → id):", class_to_idx)      # ← moved here
    print("Class map size:", NUM_CLASSES, "Domain map size:", NUM_DOMAINS)

    with open(os.path.join(save_path, "label_maps.json"), "w", encoding="utf-8") as f:
        json.dump({"class_to_idx": class_to_idx, "dom_to_idx": dom_to_idx}, f, ensure_ascii=False, indent=2)

    extractor = ResNet(img_channels=1).to(device)
    model     = ExtractorWithGRL(extractor, num_domains=NUM_DOMAINS, grl_alpha=0.0).to(device)
    tx_head   = TxHead(emb_dim=512, num_classes=NUM_CLASSES, hidden=128, dropout=0.1).to(device)

    margin = cfg['margin']
    criterion_metric = TripletMarginLoss(margin=margin)
    train_miner = TripletMarginMiner(margin=margin, type_of_triplets='all')
    val_miner   = TripletMarginMiner(margin=margin, type_of_triplets='all')

    lr = cfg['lr']
    optimizer = torch.optim.Adam(list(model.parameters()) + list(tx_head.parameters()), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=80, verbose=False, min_lr=5e-5)

    initial_epoch = 0
    num_epoch     = cfg['epoch']

    print("Start GRL training!!\n")
    Gtrain(model=model,
           tx_head=tx_head,
           train_loader=train_loader,
           val_loader=val_loader,
           initial_epoch=initial_epoch,
           num_epoch=num_epoch,
           save_path=save_path,
           device=device,
           criterion_metric=criterion_metric,
           criterion_domain=nn.CrossEntropyLoss(),
           criterion_tx=nn.CrossEntropyLoss(),
           optimizer=optimizer,
           train_miners=train_miner,
           val_miners=val_miner,
           scheduler=scheduler,
           lambda_domain=1.0,
           beta_tx=1.0,
           patience=2000,
           exp_name=exp_name,
           tb_every=20)

if __name__ == '__main__':
    main()