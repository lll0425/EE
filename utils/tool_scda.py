import os
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from utils.tool2 import dann_lambda, build_maps_from_splits, _to_tensor_batch, _map_labels

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning Loss.
    比 Triplet Loss 更能有效處理 Batch 內的所有正負樣本對。
    """
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        # features: [B, D], labels: [B]
        device = features.device
        batch_size = features.shape[0]
        
        # 強制執行 L2 正規化 (對比學習核心)
        features = F.normalize(features, p=2, dim=1)
        
        # 計算相似度矩陣
        logits = torch.div(torch.matmul(features, features.T), self.temperature)
        
        # 數值穩定化處理
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # 建立 Mask：標記同類別樣本，並排除自己對自己
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1,
            torch.arange(batch_size).view(-1, 1).to(device), 0
        )
        mask = mask * logits_mask 

        # 計算 Log-Sum-Exp
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

        # 計算平均正樣本 Log 機率
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)
        loss = -mean_log_prob_pos.mean()
        return loss

def SCDA_train(model, tx_head, train_loader, val_loader, 
               initial_epoch, num_epoch, save_path, device,
               optimizer, scheduler, 
               lambda_domain=1.0, beta_tx=0.5, temp=0.07,
               exp_name="SCDA_RFFI", tb_every=20):
    
    os.makedirs(save_path, exist_ok=True)
    writer = SummaryWriter(os.path.join("./runs", exp_name))
    
    # 初始化損失函數 (取代 Triplet Loss)
    criterion_supcon = SupConLoss(temperature=temp)
    criterion_domain = nn.CrossEntropyLoss()
    criterion_tx = nn.CrossEntropyLoss()

    # 建立標籤映射 (使用 tool2.py 的工具)
    class_to_idx, dom_to_idx = build_maps_from_splits(train_loader.dataset, val_loader.dataset)
    NUM_CLASSES, NUM_DOMAINS = len(class_to_idx), len(dom_to_idx)
    
    best_val_total = float("inf")
    total_steps = len(train_loader) * (num_epoch - initial_epoch)

    for epoch in range(initial_epoch, num_epoch):
        model.train(); tx_head.train()
        tr_supcon = tr_dom = tr_tx = 0.0
        
        for bidx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            # Batch 結構: (CIS, class_label_str, distance_folder_str)
            CIS, y_cls_s, y_dom_s = batch[0], batch[1], batch[2] 
            
            x = _to_tensor_batch(CIS, device)
            y_cls = _map_labels(y_cls_s, class_to_idx, device)
            y_dom = _map_labels(y_dom_s, dom_to_idx, device)

            # 動態調整 GRL alpha
            curr_step = (epoch - initial_epoch) * len(train_loader) + bidx
            alpha = dann_lambda(curr_step / total_steps)

            # 前向傳播 (包含 GRL 分支)
            emb, dom_logits = model(x, alpha=alpha)
            tx_logits = tx_head(emb)

            # SCDA 損失計算
            loss_sc = criterion_supcon(emb, y_cls)      # 拉近同卡跨距離特徵
            loss_dom = criterion_domain(dom_logits, y_dom) # 對抗分支 (透過 GRL 洗掉距離資訊)
            loss_tx = criterion_tx(tx_logits, y_cls)    # 身份分類分支
            
            # 總損失整合
            loss = loss_sc + (lambda_domain * loss_dom) + (beta_tx * loss_tx)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            tr_supcon += loss_sc.item(); tr_dom += loss_dom.item(); tr_tx += loss_tx.item()

        # 驗證邏輯
        model.eval(); tx_head.eval()
        va_total = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x = _to_tensor_batch(batch[0], device)
                y_cls = _map_labels(batch[1], class_to_idx, device)
                y_dom = _map_labels(batch[2], dom_to_idx, device)
                
                emb, dom_logits = model(x, alpha=1.0) # 驗證時 GRL 不反轉
                va_total += (criterion_supcon(emb, y_cls) + 
                             lambda_domain * criterion_domain(dom_logits, y_dom)).item()
        
        va_total /= len(val_loader)
        if va_total < best_val_total:
            best_val_total = va_total
            torch.save(model.state_dict(), os.path.join(save_path, "best_scda_model.pt"))

        # TensorBoard 紀錄
        if epoch % tb_every == 0:
            writer.add_scalar("Loss/Train_SupCon", tr_supcon/len(train_loader), epoch)
            writer.add_scalar("Loss/Val_Total", va_total, epoch)
            writer.add_scalar("Param/GRL_Alpha", alpha, epoch)

    writer.close()
    print(f"訓練完成。最佳驗證損失: {best_val_total:.4f}")