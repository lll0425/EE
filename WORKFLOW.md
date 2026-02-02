# EXTRACTOR 完整運行流程指南

## 🎯 整體概述
```
訓練階段         提取階段          分類階段
  ↓              ↓               ↓
main.py    →   extract.py   →  KNN0425.py
  ↓              ↓               ↓
訓練模型          提取嵌入           分類結果
```

---

## 📋 詳細流程

### **階段 1️⃣: 訓練 (Training)**

#### 🚀 **運行指令**
```bash
python main.py
```

#### 📍 **`main.py` 執行流程**
```
main.py
    │
    ├─ 1. 設定 Device (GPU/CPU)
    │   └─ device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    │
    ├─ 2. 初始化 ResNet 模型
    │   └─ model = ResNet()
    │       └─ 調用 model/resnet.py 定義的 ResNet 架構
    │
    ├─ 3. 固定隨機種子
    │   └─ fixed_seed(seed=1206)  [在 utils/tool.py]
    │
    ├─ 4. 載入訓練資料
    │   └─ train_set, val_set = get_train_val_dataset(...)
    │       │
    │       └─ 調用 dataset/tool.py
    │           │
    │           └─ 返回 RxAgnosticCISDataset 實例
    │               │
    │               └─ 讀取 Q_matrix 資料 (位置: cfg['train_data_root'])
    │                   │
    │                   └─ 目錄結構:
    │                       ├─ 0cm/
    │                       │  ├─ A1.mat, A2.mat, ...
    │                       │  ├─ B1.mat, B2.mat, ...
    │                       │  └─ ...
    │                       ├─ 0.1cm/
    │                       │  └─ ...
    │                       └─ ...
    │
    ├─ 5. 創建 DataLoader
    │   ├─ train_loader = DataLoader(train_set, batch_size=1024, shuffle=True, ...)
    │   └─ val_loader = DataLoader(val_set, batch_size=1024, shuffle=True, ...)
    │
    ├─ 6. 設定優化器和損失函數
    │   ├─ optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    │   ├─ criterion = TripletMarginLoss(margin=0.5)
    │   ├─ miner = TripletMarginMiner(margin=0.5, type_of_triplets="hard")
    │   └─ scheduler = ReduceLROnPlateau(optimizer, ...)
    │
    └─ 7. 開始訓練 (調用 utils/tool.py 中的 train() 函數)
        │
        └─ train(model, train_loader, val_loader, ...)
            │
            ├─ 建立 TensorBoard 寫入器
            │
            └─ for epoch in range(num_epoch):  # 迴圈 2000 次
                │
                ├─ [訓練階段] model.train()
                │   │
                │   └─ for batch in train_loader:
                │       ├─ anchor_CIS, anchor_tx = batch
                │       ├─ out_anchor = model(anchor_CIS)  ← ResNet 前向傳播
                │       ├─ hard_pairs = miner(out_anchor, anchor_tx)  ← 挖掘難樣本對
                │       ├─ loss = criterion(out_anchor, anchor_tx, hard_pairs)  ← 計算 Triplet Loss
                │       ├─ optimizer.zero_grad()
                │       ├─ loss.backward()
                │       └─ optimizer.step()
                │
                ├─ [驗證階段] model.eval()
                │   │
                │   └─ with torch.no_grad():
                │       └─ for batch in val_loader:
                │           ├─ out_anchor = model(anchor_CIS)
                │           ├─ loss = criterion(out_anchor, anchor_tx)
                │           └─ 記錄驗證損失
                │
                ├─ 更新學習率
                │   └─ scheduler.step(val_loss)
                │
                ├─ TensorBoard 記錄
                │   ├─ train/loss
                │   ├─ val/loss
                │   ├─ train/learning_rate
                │   └─ triplet/num_train_triplet
                │
                └─ 存儲模型 (每個 epoch)
                    ├─ save/model_epoch.pt
                    ├─ save/optimizer_epoch.pt
                    └─ save/best.pt  (當 val_loss 最低時)
```

#### 📊 **ResNet 架構** (在 `model/resnet.py`)
```
输入: [1, 1, H, W]  (batch_size, channels, height, width)
  │
  ├─ Conv2d + BatchNorm + ReLU (層層堆疊)
  ├─ BasicBlock / BottleneckBlock
  ├─ ResNet Residual 連接
  │
  └─ 輸出: [batch_size, embedding_dim]  例: [batch, 512]
```

#### ⚙️ **TripletMarginLoss 原理**
```
給定 Triplet (anchor, positive, negative):
  - anchor: 參考樣本
  - positive: 同類別樣本
  - negative: 不同類別樣本

Loss = max(d(anchor, positive) - d(anchor, negative) + margin, 0)
其中:
  - d() = 歐幾里得距離
  - margin = 0.5

目標: 讓 positive 更接近 anchor，negative 遠離 anchor
```

---

### **階段 2️⃣: 特徵提取 (Feature Extraction)**

#### 🚀 **運行指令**
```bash
python extract.py
```

#### 📍 **`extract.py` 執行流程**
```
extract.py
    │
    ├─ 1. 定義路徑 (在函數 extration(dis=1))
    │   ├─ data_root: Q_matrix 測試資料 (需手動指定)
    │   └─ folder_path: 輸出嵌入向量的目錄 (RFF/)
    │
    ├─ 2. 設定設備
    │   └─ device = torch.device('cpu')  [可改為 'cuda:0']
    │
    ├─ 3. 載入訓練完的模型
    │   ├─ model = ResNet()
    │   ├─ model.load_state_dict(torch.load(saved_model_path))
    │   │   └─ saved_model_path 來自 cfg.py
    │   └─ model.eval()  ← 設為評估模式
    │
    ├─ 4. 準備資料集
    │   └─ dataset = CISDataset(data_root=data_root)
    │       └─ 讀取 Q_matrix .mat 文件
    │
    └─ 5. 開始提取 (迴圈遍歷資料集)
        │
        └─ with torch.no_grad():  ← 不計算梯度 (節省記憶體)
            │
            └─ for i in range(len(dataset)):
                │
                ├─ CIS, label, distance_folder = dataset[i]
                │   │
                │   └─ CIS.shape: [1, 1, H, W]
                │       label: 卡片名稱 (e.g., "A1", "B5")
                │       distance_folder: 距離 (e.g., "0cm", "0.1cm")
                │
                ├─ CIS = CIS.unsqueeze(0).to(device)  ← 變成 [1, 1, 1, H, W]
                │
                ├─ embeddings = model(CIS).cpu().numpy()
                │   │
                │   └─ embeddings.shape: [1, embedding_dim]
                │       例: [1, 512]
                │
                ├─ 按 distance_folder + label 分組存儲
                │   │
                │   └─ if (label 改變 OR distance_folder 改變):
                │       ├─ 將前一組的嵌入堆疊成 numpy array
                │       │   └─ shape: [num_samples_of_this_label, embedding_dim]
                │       │       例: [30, 512]  (如果該標籤有30個樣本)
                │       │
                │       └─ 存儲為 .npy 文件
                │           └─ RFF/[distance]/[label].npy
                │
                └─ RFF.append(embeddings)  ← 積累同一 label 的嵌入
```

#### 🗂️ **輸出結構**
```
RFF/
├─ 0cm/
│  ├─ A1.npy  [shape: (30, 512)]  = 30 個樣本的嵌入向量
│  ├─ A2.npy
│  ├─ B1.npy
│  └─ ...
├─ 0.1cm/
│  ├─ A1.npy
│  └─ ...
└─ ...
```

#### ⚡ **核心代碼片段**
```python
# 沒有梯度計算，加快速度
with torch.no_grad():
    for i in range(len(dataset)):
        CIS, label, distance_folder = dataset[i]
        CIS = CIS.unsqueeze(0).to(device)
        
        # 模型前向傳播，輸出嵌入向量
        embeddings = model(CIS).cpu().numpy()  # [1, 512]
        
        RFF.append(embeddings)
        
        # 當 label 或 distance 改變時，存儲前一組的結果
        if (previous_label != label or previous_distance != distance_folder):
            arr = np.stack(RFF, axis=0)  # [N, 512]
            np.save(f'RFF/{previous_distance}/{previous_label}.npy', arr)
```

---

### **階段 3️⃣: KNN 分類 (Classification)**

#### 🚀 **運行指令**
```bash
python KNN0425.py
```

#### 📍 **`KNN0425.py` 執行流程** (假設)
```
KNN0425.py
    │
    ├─ 1. 載入訓練用的嵌入向量
    │   └─ 從 RFF/ 讀取所有 .npy 文件
    │       └─ 標籤對應 (e.g., "A1" → label 0)
    │
    ├─ 2. 準備訓練集 & 測試集
    │   ├─ X_train: [N_train, 512]  嵌入向量
    │   ├─ y_train: [N_train]  標籤索引
    │   └─ X_test, y_test 同理
    │
    ├─ 3. 訓練 KNN 模型
    │   └─ knn = KNeighborsClassifier(n_neighbors=k)
    │       └─ 計算距離矩陣 (訓練樣本之間的距離)
    │
    ├─ 4. 預測測試集
    │   └─ y_pred = knn.predict(X_test)
    │       └─ 對每個測試樣本，找最近的 k 個訓練樣本
    │           └─ 用多數表決決定標籤
    │
    └─ 5. 評估效果
        ├─ accuracy = accuracy_score(y_test, y_pred)
        ├─ precision, recall, f1 = precision_recall_fscore_support(...)
        └─ 打印分類報告
```

---

### **階段 3+ (可選): GRL 訓練和提取** 

#### 🚀 **運行指令 (Domain Adaptation)**
```bash
python GRLmain2.py    # 訓練帶有 Gradient Reversal Layer
python extractGRL.py  # 提取 GRL 模型的嵌入
```

#### 📍 **GRL 架構** (在 `model/GRL_B.py`)
```
Q_matrix 輸入
    │
    ├─ ResNet (特徵提取器)
    │   └─ 輸出: [batch, 512] embedding
    │
    ├─ [分支 1] TX 分類頭
    │   ├─ 全連接層
    │   └─ 輸出: [batch, num_classes] logits
    │
    └─ [分支 2] 域分類頭 (with GRL)
        ├─ Gradient Reversal Layer
        │   └─ alpha 參數控制反轉強度
        ├─ 全連接層
        └─ 輸出: [batch, num_domains] logits

目標:
  - 最大化 TX 分類精度
  - 最小化域分類精度 (防止過度擬合特定距離)
```

---

## 🔑 **關鍵配置參數** (在 `cfg.py`)

```python
cfg = {
    # 資料路徑
    'train_data_root': 'D:/午/EXTRACTOR/dataset/0929_AF16_low1MHzDB/Q_matrix',
    'saved_model_path': 'D:/午/EXTRACTOR/save/GRLGaps_0928_3dis_AF16/model_320.pt',
    
    # 訓練參數
    'batch_size': 1024,           # 批次大小
    'epoch': 2000,                # 訓練輪數
    'lr': 0.001,                  # 學習率
    'margin': 0.5,                # Triplet Loss 邊距
    'seed': 1206,                 # 隨機種子
    
    # 數據分割
    'split_ratio': 0.9,           # 訓練/驗證比例
    'train_ratio': 0.8,
    'val_ratio': 0.1,
}
```

---

## 🎬 **完整運行工作流程**

### 場景 A: 從頭訓練

```bash
# 步驟 1: 確保 Q_matrix 在正確位置
# 路徑: D:/午/EXTRACTOR/dataset/0929_AF16_low1MHzDB/Q_matrix/
#      ├─ 0cm/  (或其他距離)
#      │  ├─ A1.mat
#      │  ├─ A2.mat
#      │  └─ ...

# 步驟 2: 修改 cfg.py 的路徑
nano cfg.py
# 修改: 'train_data_root' 指向你的 Q_matrix 路徑

# 步驟 3: 訓練模型
python main.py
# 輸出: ./save/model_0.pt, model_1.pt, ..., best.pt

# 步驟 4: 提取嵌入向量
python extract.py
# 修改: extract.py 中的 data_root (測試資料路徑)
#      folder_path (輸出路徑)
# 輸出: ./RFF/[distance]/[label].npy

# 步驟 5: KNN 分類
python KNN0425.py
# 輸出: 分類精度、F1 分數等
```

### 場景 B: 使用預訓練模型

```bash
# 步驟 1: 直接提取 (使用已訓練的模型)
python extract.py
# 修改: 
#   - data_root (你的測試 Q_matrix 路徑)
#   - saved_model_path (指向已訓練模型)
#   - folder_path (輸出路徑)

# 步驟 2: 分類
python KNN0425.py
```

---

## 📊 **數據流向圖**

```
Raw Data (.mat)
    │
    ├─ main.py 訓練階段
    │   ├─ 讀取 Q_matrix → ResNet → 計算 Loss → 優化參數
    │   └─ 保存: ./save/best.pt
    │
    ├─ extract.py 提取階段
    │   ├─ 讀取 Q_matrix → ResNet(best.pt) → 嵌入向量
    │   └─ 保存: ./RFF/[distance]/[label].npy
    │
    └─ KNN0425.py 分類階段
        ├─ 讀取 ./RFF/*.npy → KNN 模型 → 預測
        └─ 輸出: 精度、混淆矩陣等

```

---

## ⚙️ **常見問題排查**

| 問題 | 原因 | 解決方案 |
|------|------|--------|
| `No module named 'mat73'` | 缺少依賴 | `pip install mat73 scipy` |
| `CUDA out of memory` | 批次過大或 GPU 不足 | 減少 `batch_size` 或改用 CPU |
| `FileNotFoundError` | 路徑錯誤 | 檢查 `cfg.py` 中的路徑是否正確 |
| `shape mismatch` | Q_matrix 維度不符 | 確保 Q_matrix 的高度/寬度一致 |
| `val_loss 不下降` | 學習率太小或數據不足 | 增加 `lr` 或檢查訓練數據 |

---

## 💾 **檔案說明**

```
EXTRACTOR/
├─ main.py                ← ⭐ 訓練主程序
├─ extract.py             ← ⭐ 特徵提取
├─ extractGRL.py          ← GRL 特徵提取 (可選)
├─ KNN0425.py             ← KNN 分類
├─ GRLmain2.py            ← GRL 訓練 (可選)
├─ cfg.py                 ← ⚙️  配置文件
├─ cfg_ex.py              ← GRL 配置
│
├─ model/
│  ├─ resnet.py           ← ResNet 架構定義
│  ├─ GRL_B.py            ← GRL wrapper
│  └─ head.py             ← 分類頭
│
├─ dataset/
│  ├─ tool.py             ← 數據加載工具
│  └─ CISDataset.py       ← 數據集類定義
│
├─ utils/
│  ├─ tool.py             ← 訓練工具 (train 函數)
│  └─ tool2.py            ← GRL 訓練工具
│
├─ save/                  ← 🗂️  儲存訓練好的模型
│  ├─ model_0.pt
│  ├─ model_1.pt
│  └─ best.pt
│
└─ RFF/                   ← 🗂️  儲存提取的嵌入向量
    ├─ 0cm/
    │  ├─ A1.npy
    │  └─ ...
    └─ ...
```

---

## 📝 **总結: 你现在需要做什么**

既然你已經有 Q_matrix 了:

```
1. ✅ 修改 cfg.py
   └─ 'train_data_root' 指向你的 Q_matrix 路徑

2. ✅ 運行訓練
   └─ python main.py
   └─ 等待 ~2000 個 epoch (可能需要數小時)

3. ✅ 提取嵌入向量
   └─ python extract.py
   └─ 修改 data_root 和 folder_path

4. ✅ 分類和評估
   └─ python KNN0425.py
   └─ 查看精度結果
```

**祝你成功！** 🚀
