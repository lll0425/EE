from tqdm import tqdm
import torch
import torchmetrics
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from pytorch_metric_learning.losses import TripletMarginLoss, AngularLoss, NTXentLoss, NPairsLoss
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import random
import os
import time
from torchmetrics.classification  import BinaryF1Score
# from torchmetrics.functional import precision_recall
from sklearn.metrics import precision_recall_fscore_support ,f1_score, silhouette_score
from pytorch_metric_learning.miners import TripletMarginMiner
from sklearn.preprocessing import OneHotEncoder
from scipy.spatial.distance import pdist, squareform
from typing import Optional, Dict


# def confusion_matrix(preds,labels, conf_matrix) :
#     preds = torch.argmax(preds,1).long()
#     labels = torch.argmax(labels,1).long()
#     for p,t in zip(preds,labels):
#         conf_matrix[p][t] +=1

#     return conf_matrix

# def check_continue_training(saved_model_path,saved_optimizer_path):
#     if os.path.exists(saved_model_path) and os.path.exists(saved_optimizer_path):
#         model.load_state_dict(torch.load(saved_model_path))
#         optimizer.load_state_dict(torch.load(saved_optimizer_path))
#         print("continue training model.")
#         initial_epoch = 0
#     else:
#         print("No saved model or optimizer found.")
#         initial_epoch = 1
    
#     return initial_epoch

# class TripletLoss(nn.Module):
#     '''
#     reference:
#     https://blog.csdn.net/weixin_44538273/article/details/88409198    
#     '''   
    
#     def __init__(self, margin=0.3):
#         super(TripletLoss, self).__init__()
#         self.margin = margin
#         self.ranking_loss = nn.MarginRankingLoss(margin=margin)  # 获得一个简单的距离triplet函数

#     def forward(self, inputs, labels):

#         n = inputs.size(0)  # 获取batch_size
#         # Compute pairwise distance, replace by the official when merged
#         dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)  # 每个数平方后， 进行加和（通过keepdim保持2维），再扩展成nxn维
#         dist = dist + dist.t()  # 这样每个dis[i][j]代表的是第i个特征与第j个特征的平方的和
#         # dist.addmm_(1, -2, inputs, inputs.t())  # 然后减去2倍的 第i个特征*第j个特征 从而通过完全平方式得到 (a-b)^2
#         dist.addmm_(inputs, inputs.t(),beta=1,alpha=-2)      #new version
#         dist = dist.clamp(min=1e-12).sqrt()  # 然后开方

#         # For each anchor, find the hardest positive and negative
#         mask = labels.expand(n, n).eq(labels.expand(n, n).t())  # 这里dist[i][j] = 1代表i和j的label相同， =0代表i和j的label不相同
#         dist_ap, dist_an = [], []
#         for i in range(n):
#             dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))  # 在i与所有有相同label的j的距离中找一个最大的
#             dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))  # 在i与所有不同label的j的距离找一个最小的
#         dist_ap = torch.cat(dist_ap)  # 将list里的tensor拼接成新的tensor
#         dist_an = torch.cat(dist_an)

#         # Compute ranking hinge loss
#         y = torch.ones_like(dist_an)  # 声明一个与dist_an相同shape的全1tensor
#         loss = self.ranking_loss(dist_an, dist_ap, y)
#         return loss

def check_device(model):
    # 檢查模型中的每個參數的設備
    for name, param in model.named_parameters():
        print(f"{name}: {param.device}")

def fixed_seed(myseed:int):
    """Initial random seed
    Args:
        myseed: the seed for initial
    """
    np.random.seed(myseed)
    random.seed(myseed)
    torch.manual_seed(myseed)

    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(myseed)
        torch.cuda.manual_seed(myseed)

def load_parameters(model, path, optimizer=None, epoch:int=0):
    print(f'Loading model parameters from {path}...')
    param = torch.load(path)
    model.load_state_dict(param)

    if optimizer != None:
        optimizer.load_state_dict(torch.load(os.path.join(f"./save/optimizer_{epoch}.pt")))

    print("End of loading !!!")

def train(model, train_loader, val_loader, initial_epoch: int, num_epoch: int, save_path: str, device, criterion, optimizer, train_miners, val_miners, scheduler, patience: int = 1000, exp_name=""):
    import time
    from torch.utils.tensorboard import SummaryWriter
    from tqdm import tqdm
    from pytorch_metric_learning.losses import TripletMarginLoss
    from pytorch_metric_learning.miners import TripletMarginMiner

    start_train = time.time()
    writer = SummaryWriter(os.path.join("./runs", exp_name))
    os.makedirs(save_path, exist_ok=True)

    if isinstance(train_loader.dataset, torch.utils.data.Subset):
        base_dataset = train_loader.dataset.dataset
    else:
        base_dataset = train_loader.dataset

    class_names = sorted(list(set([label for _, label, _ in base_dataset.samples])))
    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}

    best_val_epoch = 0
    best_loss = 999
    no_improve_counter = 0

    margin = 0.5
    margin_increment = 0.05
    max_margin = 0.5
    dynamic_margin_patience = 200

    for epoch in range(initial_epoch, num_epoch):
        print(f'Epoch = {epoch}')
        start_time = time.time()

        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar("train/margin", float(margin), global_step=epoch)
        writer.add_scalar("train/learning_rate", float(current_lr), global_step=epoch)
        print(f"Current Learning Rate = {current_lr:.6f}")
        print(f"Current Margin = {margin:.2f}")

        model.train()
        print(f"Model is on device: {next(model.parameters()).device}")

        train_loss = 0.0
        num_train_triplet = 0

        for i, (CIS, label, _) in enumerate(tqdm(train_loader)):
            if isinstance(label, list):
                label = torch.tensor([class_to_idx[l] for l in label]).to(device)
            else:
                label = torch.tensor([class_to_idx[label]]).to(device)

            if isinstance(CIS, list):
                anchor = torch.stack([torch.tensor(x) for x in CIS]).to(device)
            else:
                anchor = CIS.to(device)

            outputs = model(anchor)
            triplets = train_miners(outputs, label)
            anchor_idx, positive_idx, negative_idx = triplets[0], triplets[1], triplets[2]

            if anchor_idx.numel() == 0:
                print(f"⚠️ Skipping training batch {i} in epoch {epoch} — no valid triplets")
                continue

            num_triplet = anchor_idx.size(0)
            loss, ap_dist, an_dist = criterion(outputs, label, (anchor_idx, positive_idx, negative_idx))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_train_triplet += num_triplet

        train_loss /= len(train_loader)
        writer.add_scalar("train/loss", float(train_loss), global_step=epoch)
        writer.add_scalar("train/num_triplets", int(num_train_triplet), global_step=epoch)

        with torch.no_grad():
            eval_loss = 0.0
            num_val_triplet = 0

            model.eval()

            for i, (CIS, label, _) in enumerate(tqdm(val_loader)):
                if isinstance(label, list):
                    label = torch.tensor([class_to_idx[l] for l in label]).to(device)
                else:
                    label = torch.tensor([class_to_idx[label]]).to(device)

                if isinstance(CIS, list):
                    anchor = torch.stack([torch.tensor(x) for x in CIS]).to(device)
                else:
                    anchor = CIS.to(device)

                outputs = model(anchor)
                val_triplets = val_miners(outputs, label)
                anchor_idx, positive_idx, negative_idx = val_triplets[0], val_triplets[1], val_triplets[2]

                if anchor_idx.numel() == 0:
                    print(f"⚠️ Skipping validation batch {i} in epoch {epoch} — no valid triplets")
                    continue

                eval_num_triplet = anchor_idx.size(0)
                loss, _, _ = criterion(outputs, label, (anchor_idx, positive_idx, negative_idx))

                eval_loss += loss.item()
                num_val_triplet += eval_num_triplet

            eval_loss /= len(val_loader)
            writer.add_scalar("val/loss", float(eval_loss), global_step=epoch)
            writer.add_scalar("val/num_val_triplets", int(num_val_triplet), global_step=epoch)

            if eval_loss < best_loss:
                best_loss = eval_loss
                best_val_epoch = epoch
                no_improve_counter = 0
            else:
                no_improve_counter += 1
                if no_improve_counter >= dynamic_margin_patience:
                    if margin < max_margin:
                        margin += margin_increment
                        margin = min(margin, max_margin)
                    no_improve_counter = 0
                    criterion = TripletMarginLoss(margin=margin)
                    train_miners = TripletMarginMiner(margin=margin, type_of_triplets='all')
                    val_miners = TripletMarginMiner(margin=margin, type_of_triplets='all')
                    writer.add_scalar("train/margin_change", float(margin), global_step=epoch)
                    print(f"Margin increased to {margin:.2f} after {dynamic_margin_patience} epochs without validation loss improvement.")

        scheduler.step(eval_loss)

        if eval_loss < best_loss:
            best_loss = eval_loss
            best_val_epoch = epoch
            torch.save(model.state_dict(), os.path.join(save_path, 'best.pt'))
            no_improve_counter = 0
        else:
            no_improve_counter += 1

        if no_improve_counter >= patience:
            print(f"Validation loss did not improve for {patience} epochs. Early stopping at epoch {epoch}.")
            break

        torch.save(model.state_dict(), os.path.join(save_path, f'{epoch}.pt'))
        torch.save(optimizer.state_dict(), os.path.join(save_path, f'optimizer_{epoch}.pt'))

        end_time = time.time()
        elapsed_time = end_time - start_time
        total_time = end_time - start_train
        print('=' * 24)
        print(f'Time for Epoch: {elapsed_time // 60:.0f} min {elapsed_time % 60:.1f} sec')
        print(f'Total Time: {total_time // 60:.0f} min {total_time % 60:.1f} sec')
        print('=' * 24 + '\n')

    print("End of training!")
    print(f"Best validation loss {best_loss:.6f} at epoch {best_val_epoch}")
    writer.close()

#########################################################################################################

def Ntrain(model, train_loader, val_loader, initial_epoch:int, num_epoch:int,
           save_path:str, device, criterion, optimizer, train_miners, val_miners,
           scheduler, patience: int = 2000, exp_name: str = "",
           class_names: Optional[Dict[int, str]] = None,
           tb_every: int = 20,
           classwise_every: Optional[int] = None,
           metric_chunk_size: int = 65536):
    """
    Aligned with your `train()`:
      - Batch can be (CIS,label) or (CIS,label,extra)
      - CIS may be Tensor or list/tuple/ndarray (stacked)
      - Labels may be strings/ints/tensors; strings are mapped to ints
      - Miner output is sliced to (a,p,n) before loss (same as your train())
      - AP/AN computed ONLY on logging epochs (every tb_every), in CPU chunks
      - Epoch-wise logging; writes every `tb_every` epochs
    """
    import os, time
    from collections import defaultdict
    import torch, numpy as np
    from torch.utils.tensorboard import SummaryWriter
    from tqdm import tqdm

    # ---------- helpers ----------
    def to_tensor_batch(x, device):
        if isinstance(x, torch.Tensor):
            return x.to(device)
        if isinstance(x, (list, tuple)):
            if len(x) == 0:
                raise ValueError("Empty CIS list/tuple in batch.")
            if isinstance(x[0], torch.Tensor):
                return torch.stack(x).to(device)
            else:
                return torch.stack([torch.as_tensor(e) for e in x]).to(device)
        return torch.as_tensor(x).to(device)

    def build_initial_mapping(loader):
        ds = loader.dataset
        if isinstance(ds, torch.utils.data.Subset):
            ds = ds.dataset
        mapping = {}
        labels = []
        if hasattr(ds, "samples"):
            try:
                for item in ds.samples:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        labels.append(item[1])
            except Exception:
                labels = []
        if labels:
            uniq = sorted(list(set(labels)))
            mapping = {name: idx for idx, name in enumerate(uniq)}
        return mapping

    def to_label_indices(y, device, class_to_idx):
        if isinstance(y, torch.Tensor):
            if y.dtype in (torch.int64, torch.int32, torch.int16, torch.uint8):
                return y.to(device).long().view(-1)
            y = y.tolist()
        if isinstance(y, (list, tuple, np.ndarray)):
            if len(y) == 0:
                return torch.empty((0,), dtype=torch.long, device=device)
            if isinstance(y[0], (int, np.integer)):
                return torch.as_tensor(y, dtype=torch.long, device=device).view(-1)
            mapped = []
            for elem in y:
                if isinstance(elem, (int, np.integer)):
                    mapped.append(int(elem))
                else:
                    key = str(elem)
                    if key not in class_to_idx:
                        class_to_idx[key] = len(class_to_idx)
                    mapped.append(class_to_idx[key])
            return torch.as_tensor(mapped, dtype=torch.long, device=device).view(-1)
        if isinstance(y, (int, np.integer)):
            return torch.tensor([int(y)], dtype=torch.long, device=device)
        key = str(y)
        if key not in class_to_idx:
            class_to_idx[key] = len(class_to_idx)
        return torch.tensor([class_to_idx[key]], dtype=torch.long, device=device)

    @torch.no_grad()
    def accumulate_pair_means(outputs, a, p, n, labels_long, do_classwise, chunk=65536):
        """Return (ap_mean, an_mean, count, cls_sum_ap?, cls_sum_an?, cls_cnt?). CPU-chunked."""
        if a.numel() == 0:
            if do_classwise:
                return float('nan'), float('nan'), 0, {}, {}, {}
            return float('nan'), float('nan'), 0, None, None, None

        E = outputs.detach().float().cpu()
        a_cpu, p_cpu, n_cpu = a.detach().cpu(), p.detach().cpu(), n.detach().cpu()
        labels_cpu = labels_long.detach().cpu()

        tot_ap = 0.0; tot_an = 0.0; tot_cnt = 0
        cls_sum_ap = defaultdict(float) if do_classwise else None
        cls_sum_an = defaultdict(float) if do_classwise else None
        cls_cnt    = defaultdict(int)   if do_classwise else None

        T = a_cpu.numel()
        for s in range(0, T, chunk):
            e = min(T, s + chunk)
            ai, pi, ni = a_cpu[s:e], p_cpu[s:e], n_cpu[s:e]
            ea, ep, en = E[ai], E[pi], E[ni]
            ap = (ea - ep).norm(p=2, dim=1)
            an = (ea - en).norm(p=2, dim=1)

            tot_ap += ap.sum().item()
            tot_an += an.sum().item()
            tot_cnt += ap.numel()

            if do_classwise:
                cls = labels_cpu[ai]
                for c in cls.unique():
                    m = (cls == c)
                    ci = int(c.item())
                    cls_sum_ap[ci] += ap[m].sum().item()
                    cls_sum_an[ci] += an[m].sum().item()
                    cls_cnt[ci]    += int(m.sum().item())

            del ea, ep, en, ap, an

        ap_mean = (tot_ap / tot_cnt) if tot_cnt > 0 else float('nan')
        an_mean = (tot_an / tot_cnt) if tot_cnt > 0 else float('nan')
        if do_classwise:
            return ap_mean, an_mean, tot_cnt, cls_sum_ap, cls_sum_an, cls_cnt
        return ap_mean, an_mean, tot_cnt, None, None, None

    # ---------- setup ----------
    if classwise_every is None:
        classwise_every = tb_every

    class_to_idx = build_initial_mapping(train_loader)

    writer = (SummaryWriter(os.path.join("./runs", exp_name), flush_secs=120, max_queue=5000)
              if exp_name else SummaryWriter(flush_secs=120, max_queue=5000))
    os.makedirs(save_path, exist_ok=True)

    best_val_epoch = 0
    best_loss = float("inf")
    no_improve_counter = 0
    start_train = time.time()

    for epoch in range(initial_epoch, num_epoch):
        t0 = time.time()
        lr_now = float(optimizer.param_groups[0]['lr'])
        print(f'epoch = {epoch}')
        print(f"Current Learning rate = {lr_now:.6f}")

        do_tb_log    = ((epoch - initial_epoch) % tb_every == 0)
        do_classwise = do_tb_log and ((epoch - initial_epoch) % classwise_every == 0)

        # --------------------- TRAIN ---------------------
        model.train()
        train_loss = 0.0
        train_ap_mean = float('nan')
        train_an_mean = float('nan')
        num_train_triplet = 0
        if do_tb_log and do_classwise:
            train_cls_sum_ap = defaultdict(float)
            train_cls_sum_an = defaultdict(float)
            train_cls_cnt    = defaultdict(int)

        for batch in tqdm(train_loader):
            if not isinstance(batch, (list, tuple)) or len(batch) < 2:
                raise TypeError("Each batch must be (CIS, label, ...)")
            CIS, label = batch[0], batch[1]

            anchor = to_tensor_batch(CIS, device)
            target = to_label_indices(label, device, class_to_idx)

            outputs = model(anchor)
            raw_indices = train_miners(outputs, target)
            if raw_indices is None or len(raw_indices) < 3:
                continue

            a, p, n = raw_indices[0], raw_indices[1], raw_indices[2]
            if a.numel() == 0:
                continue

            # loss uses (a,p,n) — same as your train()
            ret = criterion(outputs, target, (a, p, n))
            loss = ret[0] if isinstance(ret, (tuple, list)) else ret

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if do_tb_log:
                ap_m, an_m, cnt, cls_ap, cls_an, cls_cnt = accumulate_pair_means(
                    outputs, a, p, n, target, do_classwise, chunk=metric_chunk_size)

                if np.isfinite(ap_m) and np.isfinite(an_m):
                    if num_train_triplet == 0:
                        train_ap_mean = ap_m; train_an_mean = an_m
                    else:
                        total_prev = num_train_triplet
                        total_new  = total_prev + cnt
                        train_ap_mean = (train_ap_mean * total_prev + ap_m * cnt) / total_new
                        train_an_mean = (train_an_mean * total_prev + an_m * cnt) / total_new
                    num_train_triplet += cnt

                if do_classwise and cls_ap is not None:
                    for k, v in cls_ap.items(): train_cls_sum_ap[k] += v
                    for k, v in cls_an.items(): train_cls_sum_an[k] += v
                    for k, v in cls_cnt.items(): train_cls_cnt[k]    += v

            del outputs  # free GPU memory

        train_loss /= max(1, len(train_loader))

        # --------------------- VALIDATION ---------------------
        model.eval()
        eval_loss = 0.0
        val_ap_mean = float('nan')
        val_an_mean = float('nan')
        num_val_triplet = 0
        if do_tb_log and do_classwise:
            val_cls_sum_ap = defaultdict(float)
            val_cls_sum_an = defaultdict(float)
            val_cls_cnt    = defaultdict(int)

        with torch.no_grad():
            for batch in tqdm(val_loader):
                if not isinstance(batch, (list, tuple)) or len(batch) < 2:
                    raise TypeError("Each batch must be (CIS, label, ...)")
                CIS, label = batch[0], batch[1]

                anchor = to_tensor_batch(CIS, device)
                target = to_label_indices(label, device, class_to_idx)

                outputs = model(anchor)
                raw_indices = val_miners(outputs, target)
                if raw_indices is None or len(raw_indices) < 3:
                    continue

                a, p, n = raw_indices[0], raw_indices[1], raw_indices[2]
                if a.numel() == 0:
                    continue

                vloss = criterion(outputs, target, (a, p, n))
                vloss = vloss[0] if isinstance(vloss, (tuple, list)) else vloss
                eval_loss += vloss.item()

                if do_tb_log:
                    ap_m, an_m, cnt, cls_ap, cls_an, cls_cnt = accumulate_pair_means(
                        outputs, a, p, n, target, do_classwise, chunk=metric_chunk_size)

                    if np.isfinite(ap_m) and np.isfinite(an_m):
                        if num_val_triplet == 0:
                            val_ap_mean = ap_m; val_an_mean = an_m
                        else:
                            total_prev = num_val_triplet
                            total_new  = total_prev + cnt
                            val_ap_mean = (val_ap_mean * total_prev + ap_m * cnt) / total_new
                            val_an_mean = (val_an_mean * total_prev + an_m * cnt) / total_new
                        num_val_triplet += cnt

                    if do_classwise and cls_ap is not None:
                        for k, v in cls_ap.items(): val_cls_sum_ap[k] += v
                        for k, v in cls_an.items(): val_cls_sum_an[k] += v
                        for k, v in cls_cnt.items(): val_cls_cnt[k]    += v

                del outputs

        eval_loss /= max(1, len(val_loader))

        # build readable class tags once if not provided
        if do_tb_log and class_names is None and len(class_to_idx) > 0:
            inv = {v: k for k, v in class_to_idx.items()}
            class_names = {k: str(inv[k]) for k in inv.keys()}

        # --------------------- TensorBoard (every tb_every epochs) ---------------------
        if do_tb_log:
            writer.add_scalar("train/learning_rate", lr_now, epoch)
            writer.add_scalar("train/loss", float(train_loss), epoch)

            if np.isfinite(train_ap_mean):
                writer.add_scalar("train/train_ap_distance", float(train_ap_mean), epoch)
                writer.add_scalar("train/train_an_distance", float(train_an_mean), epoch)
                writer.add_scalar("triplet/num_train_triplet", int(num_train_triplet), epoch)

            writer.add_scalar("val/loss", float(eval_loss), epoch)
            if np.isfinite(val_ap_mean):
                writer.add_scalar("val/val_ap_distance", float(val_ap_mean), epoch)
                writer.add_scalar("val/val_an_distance", float(val_an_mean), epoch)
                writer.add_scalar("triplet/num_val_triplet", int(num_val_triplet), epoch)

            if do_classwise:
                for ci, cnt in sorted(train_cls_cnt.items()):
                    if cnt == 0: continue
                    tag = class_names.get(ci, str(ci)) if class_names else str(ci)
                    writer.add_scalar(f"train_class/AP_mean/{tag}", train_cls_sum_ap[ci] / cnt, epoch)
                    writer.add_scalar(f"train_class/AN_mean/{tag}", train_cls_sum_an[ci] / cnt, epoch)
                    writer.add_scalar(f"triplet/train_count/{tag}", cnt, epoch)

                for ci, cnt in sorted(val_cls_cnt.items()):
                    if cnt == 0: continue
                    tag = class_names.get(ci, str(ci)) if class_names else str(ci)
                    writer.add_scalar(f"val_class/AP_mean/{tag}", val_cls_sum_ap[ci] / cnt, epoch)
                    writer.add_scalar(f"val_class/AN_mean/{tag}", val_cls_sum_an[ci] / cnt, epoch)
                    writer.add_scalar(f"triplet/val_count/{tag}", cnt, epoch)

            writer.flush()

        # --------------------- Housekeeping ---------------------
        scheduler.step(eval_loss)

        t1 = time.time()
        elp_time = t1 - t0
        print('=' * 24)
        print('time = {} MIN {:.1f} SEC, total time = {} MIN {:.1f} SEC'.format(
            elp_time // 60, elp_time % 60, (t1 - start_train) // 60, (t1 - start_train) % 60))
        print(f"{'Training loss':<20} : {train_loss:.6f}")
        print(f"{'Validation  loss':<20} : {eval_loss:.6f}")
        print('=' * 24 + '\n')

        if eval_loss < best_loss:
            best_loss = eval_loss
            best_val_epoch = epoch
            torch.save(model.state_dict(), os.path.join(save_path, 'best.pt'))
            no_improve_counter = 0
        else:
            no_improve_counter += 1

        if no_improve_counter >= patience:
            print(f"Validation Loss did not improve for {patience} epochs. Early stopping at epoch {epoch}.")
            break

        torch.save(model.state_dict(), os.path.join(save_path, f'{epoch}.pt'))
        torch.save(optimizer.state_dict(), os.path.join(save_path, f'optimizer_{epoch}.pt'))

    print("End of training !!!")
    print(f"Best val loss {best_loss:.6f} on epoch {best_val_epoch}")
    writer.close()