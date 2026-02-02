# utils/tool.py
import os
import time
import torch
import numpy as np
import random
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

def fixed_seed(myseed: int):
    """確保實驗的可重複性，這對於論文對比實驗至關重要"""
    np.random.seed(myseed)
    random.seed(myseed)
    torch.manual_seed(myseed)
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(myseed)
        torch.cuda.manual_seed(myseed)

def train(model, train_loader, val_loader, num_epoch, save_path, device, 
          criterion, optimizer, miner, scheduler, exp_name=""):
    start_train = time.time()
    
    # 初始化 TensorBoard
    log_dir = os.path.join("./runs/", exp_name) if exp_name else "./runs/exp"
    writer = SummaryWriter(log_dir)
    os.makedirs(save_path, exist_ok=True)

    best_loss = float('inf')
    
    # 速度控制：每個 Epoch 最多跑 1200 個 Batch
    # A40 在 Batch Size 1024 下，這能保證一個 Epoch 在數分鐘內結束
    MAX_ITERS = 1200 

    for epoch in range(num_epoch):
        model.train()
        train_loss = 0.0
        num_train_triplets = 0
        
        print(f"\n--- Epoch {epoch}/{num_epoch} ---")
        pbar = tqdm(train_loader, desc="Training")
        
        for i, batch in enumerate(pbar):
            if i >= MAX_ITERS: break
            
            # CISDataset 回傳 (data, label_idx, distance)
            cis_data, labels = batch[0].to(device), batch[1].to(device)
            
            # 前向傳播提取嵌入向量 (Embeddings)
            embeddings = model(cis_data)
            
            # 挖掘難樣本並計算 Triplet Loss
            hard_pairs = miner(embeddings, labels)
            loss = criterion(embeddings, labels, hard_pairs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_train_triplets += hard_pairs[0].size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "triplets": hard_pairs[0].size(0)})

        # 驗證階段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for j, batch in enumerate(val_loader):
                if j >= 300: break # 驗證集也不需要全跑
                cis_data, labels = batch[0].to(device), batch[1].to(device)
                embeddings = model(cis_data)
                loss = criterion(embeddings, labels)
                val_loss += loss.item()

        avg_train = train_loss / min(len(train_loader), MAX_ITERS)
        avg_val = val_loss / min(len(val_loader), 300)
        
        # 記錄指標
        writer.add_scalar("Loss/Train", avg_train, epoch)
        writer.add_scalar("Loss/Val", avg_val, epoch)
        writer.add_scalar("Triplets/Count", num_train_triplets, epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]['lr'], epoch)

        print(f"Train Loss: {avg_train:.6f} | Val Loss: {avg_val:.6f}")
        
        # 更新學習率策略
        scheduler.step(avg_val)

        # 保存最佳模型
        if avg_val < best_loss:
            best_loss = avg_val
            torch.save(model.state_dict(), os.path.join(save_path, 'best.pt'))
            print(">>> 發現更佳模型，已更新 best.pt")

        # 定期保存
        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(save_path, f'model_epoch_{epoch}.pt'))

    writer.close()
    total_time = (time.time() - start_train) / 3600
    print(f"\n訓練結束！總耗時: {total_time:.2f} 小時")