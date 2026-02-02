# main.py
import torch
from torch.utils.data import DataLoader
from pytorch_metric_learning.losses import TripletMarginLoss
from pytorch_metric_learning.miners import TripletMarginMiner
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils.tool import train, fixed_seed
from dataset.tool import get_train_val_dataset
from cfg import cfg
from model.resnet import ResNet

def main():
    # 實驗設定
    exp_name = 'ResNet_RFFI_ID1-10_A40_Optimized'
    save_dir = '/home/jovyan/jupyter/data_in/FE/EXTRACTOR/save'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # 1. 初始化模型與種子
    fixed_seed(cfg['seed'])
    model = ResNet().to(device)

    # 2. 載入資料 (設定 keep_ratio=0.1 確保數小時內跑完)
    print("正在準備資料集...")
    train_set, val_set, test_set = get_train_val_dataset(
        data_root=cfg['train_data_root'],
        train_ratio=cfg['train_ratio'],
        val_ratio=cfg['val_ratio'],
        keep_ratio=0.5  # A40 算力強，若 I/O 足夠可調至 0.2
    )

    # 3. 建立 DataLoader (A40 建議 num_workers 設定為 8-16)
    train_loader = DataLoader(
        train_set, 
        batch_size=cfg['batch_size'], 
        shuffle=True, 
        num_workers=8, 
        pin_memory=True, 
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_set, 
        batch_size=cfg['batch_size'], 
        shuffle=False, 
        num_workers=8, 
        pin_memory=True
    )

    # 4. 優化器與度量學習損失函數
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    criterion = TripletMarginLoss(margin=cfg['margin'])
    miner = TripletMarginMiner(margin=cfg['margin'], type_of_triplets="hard")
    
    # 學習率調整：當 Val Loss 停止下降時減半
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)

    # 5. 開始訓練
    print(f"開始訓練任務: {exp_name}")
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epoch=cfg['epoch'],
        save_path=save_dir,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        miner=miner,
        scheduler=scheduler,
        exp_name=exp_name
    )

if __name__ == '__main__':
    main()