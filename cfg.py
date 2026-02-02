# cfg.py
cfg = {
    'raw_root': '../raw/',
    'preprocessing_root': '../data',

    ## 模型配置 (Model Config)
    # 建議下修 batch_size 以避免 GPU 記憶體溢出 (OOM)，Q-matrix 較大時建議 128 或 256
    'batch_size': 1024, 
    'epoch': 50,      # SCDA 建議跑 100 輪左右以達到穩定效果

    ## 優化器與學習率 (Optimizer / LR)
    'lr': 0.002,

    ## 資料集配置 (Dataset Config)
    # 這裡指向原始目錄，因為篩選邏輯已經寫在 CISDataset.py 的 _gather_samples 裡面了
    'train_data_root': '/home/jovyan/jupyter/data_in/FE_out/RawIQ_to_Qmatrix_0to2cm/Q_matrix',
    'knn_train_data_root': '/home/jovyan/jupyter/data_in/FE/EXTRACTOR/save',

    ## 資料分割比例 (用於 main.py 或 GRLmain2.py)
    'train_ratio': 0.8,
    'val_ratio': 0.1,
    'split_ratio': 0.9, # 保留舊版相容性

    ## SCDA / 對抗學習專用參數 (New SCDA Params)
    # 這些參數對 SCDA 的邏輯可解釋性至關重要
    'lambda_domain': 1.0,  # 距離對抗權重 (λ): 強迫模型忘記距離資訊的力道
    'beta_tx': 0.5,        # 分類頭權重: 確保特徵仍具備卡片辨識能力的權重
    'temp': 0.07,          # 對比學習溫度係數: 控制特徵空間聚類的緊密程度

    ## 斷點續訓與存儲 (Resume / Save)
    # ⚠️ 嚴厲指正：開始新實驗（篩選卡片或改 Loss）時，必須設為 False
    'RESUME': False, 
    'saved_model_path': '/home/jovyan/jupyter/data_in/FE/EXTRACTOR/model',
    'saved_optimizer_path': '', # RESUME 為 False 時不需指定路徑

    ## 種子與雜項
    'seed': 1206,
    '13': 13,
}