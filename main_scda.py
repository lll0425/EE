import os
import torch
import argparse
import json
from datetime import datetime  # 新增：用於產生唯一時間標記
from cfg import cfg
from dataset.tool import get_train_val_dataset
from model.resnet import ResNet
from model.GRL_B import ExtractorWithGRL
from model.head import TxHead
from utils.tool_scda import SCDA_train
from utils.tool2 import fixed_seed, build_maps_from_splits

def main():
    # 1. 處理命令列參數
    parser = argparse.ArgumentParser(description='SCDA Training on A40')
    parser.add_argument('--gpu', type=int, default=1, help='GPU ID (0 or 1)')
    parser.add_argument('--tag', type=str, default='default_exp', help='實驗標籤 (例如: lr1e-4)')
    args = parser.parse_args()

    # 2. 環境設定
    fixed_seed(cfg.get('seed', 1206))
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f">>> 目前工作設備: {device}")
    
    # ⚠️ 邏輯升級：加入時間戳記，確保每次執行都是獨立資料夾
    # 格式：月日_時分 (例如 0202_1450)
    timestamp = datetime.now().strftime("%m%d_%H%M")
    exp_name = f'SCDA_RFFI_{args.tag}_{timestamp}'
    
    # 設定存儲路徑於專案根目錄下的 outputs
    save_path = os.path.join("outputs", exp_name) 
    os.makedirs(save_path, exist_ok=True)
    
    print(f">>> 實驗唯一 ID: {exp_name}")
    print(f">>> 數據與模型將存儲於: {save_path}")

    # 保存本次實驗的 cfg 參數 (確保實驗可重現性)
    with open(os.path.join(save_path, 'config_record.json'), 'w') as f:
        json.dump(cfg, f, indent=4)

    # 3. 載入資料
    print(">>> 正在讀取並過濾數據...")
    train_set, val_set, _ = get_train_val_dataset(
        data_root=cfg['train_data_root'], 
        train_ratio=cfg['train_ratio'], 
        val_ratio=cfg['val_ratio']
    )
    
    class_to_idx, dom_to_idx = build_maps_from_splits(train_set, val_set)
    NUM_CLASSES = len(class_to_idx)
    NUM_DOMAINS = len(dom_to_idx)

    # 5. DataLoader 優化 (針對 A40 提升效率)
    train_loader = torch.utils.data.DataLoader(
        train_set, 
        batch_size=cfg['batch_size'], 
        shuffle=True, 
        num_workers=8,      
        pin_memory=True     
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, 
        batch_size=cfg['batch_size'], 
        shuffle=False, 
        num_workers=8, 
        pin_memory=True
    )

    # 6. 初始化模型
    extractor = ResNet(img_channels=1).to(device)
    model = ExtractorWithGRL(extractor, num_domains=NUM_DOMAINS).to(device)
    tx_head = TxHead(emb_dim=512, num_classes=NUM_CLASSES).to(device)

    # 7. 優化器
    optimizer = torch.optim.Adam(list(model.parameters()) + list(tx_head.parameters()), lr=cfg['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=80)

    # 8. 啟動訓練
    SCDA_train(
        model=model,
        tx_head=tx_head,
        train_loader=train_loader,
        val_loader=val_loader,
        initial_epoch=0,
        num_epoch=cfg['epoch'],
        save_path=save_path,
        device=device,
        optimizer=optimizer,
        scheduler=scheduler,
        lambda_domain=cfg.get('lambda_domain', 1.0),
        beta_tx=cfg.get('beta_tx', 0.5),
        temp=cfg.get('temp', 0.07),
        exp_name=exp_name
    )

if __name__ == '__main__':
    main()