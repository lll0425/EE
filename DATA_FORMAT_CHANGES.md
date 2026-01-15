# Q_matrix 數據格式變更分析

## 📊 **核心變更對比**

### **原本方式 (Q_martix.py)：每個樣本一個檔案**
```
Q_matrix/
├─ 0cm/
│  ├─ A1/
│  │  ├─ 1.mat   [shape: (F_sel, T-1)]
│  │  ├─ 2.mat   [shape: (F_sel, T-1)]
│  │  ├─ 3.mat   [shape: (F_sel, T-1)]
│  │  └─ ... (12000 個檔案)
│  ├─ B1/
│  │  ├─ 1.mat
│  │  ├─ 2.mat
│  │  └─ ...
│  └─ ...
└─ ...
```

**特點：**
- ✅ 每個樣本獨立存儲
- ❌ 非常耗磁盤空間 (12000 個 .mat 文件 × 多張卡)
- ❌ I/O 操作慢 (讀取需要打開 12000 次文件)
- ❌ 難以管理

---

### **新方式 (Q_matrix_2.py)：每張卡一個檔案**
```
Q_matrix/
├─ 0cm/
│  ├─ A1/
│  │  └─ Q_all.mat   [shape: (N, F_sel, T-1)] 其中 N=12000
│  ├─ B1/
│  │  └─ Q_all.mat   [shape: (N, F_sel, T-1)]
│  └─ ...
└─ ...
```

**特點：**
- ✅ 每張卡只有 1 個文件
- ✅ 節省磁盤空間 (單個 .mat 文件 vs 12000 個)
- ✅ I/O 操作快 (一次讀取所有樣本)
- ✅ 更好管理

---

## 🔧 **Q_matrix_2.py 的關鍵改變**

### **1. 新增輸出選項**
```python
# 新增的配置項
OUTPUT_FLOAT16 = False  # 可選 float16 來進一步省空間
DO_COMPRESS    = True   # 啟用 mat 文件壓縮
```

### **2. 改變存儲方式**

#### **舊方式（Q_martix.py）**
```python
# 逐個存儲：每個樣本一個 .mat 檔
for i_idx, i in enumerate(sampleIdx, start=1):
    Q_cpu = Q_matrix.detach().cpu().numpy()  # shape: [F_sel, T-1]
    savemat(os.path.join(savePath_Q, f"{i_idx}.mat"), 
            {"Q_matrix": Q_cpu})
    # 結果：1.mat, 2.mat, 3.mat, ..., 12000.mat
```

#### **新方式（Q_matrix_2.py）**
```python
# 先創建一個大數組，累積所有樣本
N = len(sampleIdx)
F_sel = int(selectedMask_np.sum())
T = (signalLen - nfft) // hop_size + 1

# 創建 buffer（提前分配空間）
Q_all = np.empty((N, F_sel, T - 1), dtype=out_dtype)  # 3D array
valid = np.zeros((N,), dtype=np.uint8)  # 追蹤有效性

# 在迴圈中填充 buffer
for i_idx, i in enumerate(sampleIdx, start=1):
    ...
    Q_all[i_idx - 1, :, :] = Q_cpu.astype(out_dtype, copy=False)
    valid[i_idx - 1] = 1
    ...

# 最後一次性存儲整個 buffer
savemat(
    os.path.join(savePath_Q, "Q_all.mat"),
    {
        "Q_all": Q_all,                    # [12000, F_sel, T-1]
        "valid": valid,                    # [12000]
        "sampleIdx": np.asarray(sampleIdx, dtype=np.int32),
        "freqs_Hz": freqs_sel,
        "fs": np.float32(fs),
        "nfft": np.int32(nfft),
        "hop": np.int32(hop_size),
        "win": np.int32(window_size),
        "Q_in_dB": np.uint8(1 if Q_in_dB else 0),
    },
    do_compression=DO_COMPRESS
)
```

### **3. 新增元數據存儲**

Q_matrix_2.py 額外存儲了以下信息在 `Q_all.mat` 中：

| 字段 | 說明 | 用途 |
|------|------|------|
| `Q_all` | 形狀 `(N, F_sel, T-1)` 的 Q 矩陣 | 主要數據 |
| `valid` | 形狀 `(N,)` 的有效性標記 | 標記哪些樣本成功處理 |
| `sampleIdx` | 原始樣本索引 | 追蹤對應的原始數據 |
| `freqs_Hz` | 選中的頻率 | 頻率軸標籤 |
| `fs` | 採樣率 (12.5 MHz) | 信號參數 |
| `nfft` | FFT 大小 (560) | STFT 參數 |
| `hop` | hop size (56) | STFT 參數 |
| `win` | 窗口大小 (560) | STFT 參數 |
| `Q_in_dB` | 是否使用 dB 單位 | 數據單位信息 |

---

## ❓ **你需要改什麼？**

### **關鍵問題：EXTRACTOR 期望什麼格式？**

讓我檢查 CISDataset.py 的預期格式...

```python
# 從 dataset/CISDataset.py 看
for file in sorted(os.listdir(card_path), key=sort_key):
    if file.endswith('.mat'):
        file_path = os.path.join(card_path, file)
        self.samples.append((file_path, card_idx, first_layer_folder))

# __getitem__ 時：
CIS = mat73.loadmat(CIS_path, use_attrdict=True)['sample_data']
```

**EXTRACTOR 期望的格式：**
- ✅ 讀取 `.mat` 文件
- ✅ 從 `.mat` 中提取 `sample_data` 或類似的鍵
- ✅ 返回形狀為 `[1, 1, H, W]` 的 2D 數據

---

## ⚠️ **重要：兼容性檢查**

### **如果你用 Q_matrix_2.py 的格式，EXTRACTOR 需要改動：**

#### **現有 CISDataset.py 的問題：**
```python
# 原代碼假設
for file in sorted(os.listdir(card_path), key=sort_key):
    if file.endswith('.mat'):
        # 對每個 .mat 文件調用 __getitem__
        
# 現在只有 1 個 Q_all.mat，但裡面有 12000 個樣本！
```

#### **需要的改動方案：**

**方案 A：修改 CISDataset.py 以支持新格式（推薦）**

```python
class CISDataset_v2(Dataset):
    def __init__(self, data_root):
        self.data_root = data_root
        self.samples = []

        # 遍歷距離資料夾 → 卡片資料夾 → Q_all.mat
        for first_layer_folder in sorted(os.listdir(data_root), key=sort_key):
            first_layer_path = os.path.join(data_root, first_layer_folder)
            if os.path.isdir(first_layer_path):
                for card_idx, card_folder in enumerate(sorted(os.listdir(first_layer_path), key=sort_key)):
                    card_path = os.path.join(first_layer_path, card_folder)
                    if os.path.isdir(card_path):
                        # 查找 Q_all.mat
                        q_all_path = os.path.join(card_path, "Q_all.mat")
                        if os.path.exists(q_all_path):
                            # 讀取 meta 資訊
                            try:
                                mat_data = loadmat(q_all_path)
                                N = mat_data['Q_all'].shape[0]  # 樣本數
                                
                                # 為每個樣本創建條目
                                for sample_idx in range(N):
                                    self.samples.append({
                                        'q_all_path': q_all_path,
                                        'sample_idx': sample_idx,
                                        'label': card_idx,
                                        'distance_folder': first_layer_folder,
                                        'valid': mat_data['valid'][sample_idx]
                                    })
                            except Exception as e:
                                print(f"Failed to load {q_all_path}: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        
        # 載入 Q_all.mat（可使用緩存避免重複載入）
        mat_data = loadmat(sample_info['q_all_path'])
        Q_all = mat_data['Q_all']  # [N, F_sel, T-1]
        
        # 提取單個樣本
        Q_sample = Q_all[sample_info['sample_idx'], :, :]  # [F_sel, T-1]
        
        # 添加通道維度以匹配原格式 [1, F_sel, T-1]
        CIS = torch.from_numpy(Q_sample).unsqueeze(0).float()
        
        return (CIS, 
                sample_info['label'], 
                sample_info['distance_folder'])
```

**或者**

**方案 B：轉換回舊格式（不推薦，因為會失去優勢）**

```python
# 用 Q_matrix_2.py 的反轉：從 Q_all.mat 解包成逐檔
def unpack_q_all():
    for dist in os.listdir(Q_matrix_new):
        for card in os.listdir(os.path.join(Q_matrix_new, dist)):
            q_all_path = os.path.join(Q_matrix_new, dist, card, "Q_all.mat")
            mat_data = loadmat(q_all_path)
            Q_all = mat_data['Q_all']  # [N, F_sel, T-1]
            
            # 解包成逐檔
            out_dir = os.path.join(Q_matrix_old, dist, card)
            os.makedirs(out_dir, exist_ok=True)
            for i in range(Q_all.shape[0]):
                savemat(
                    os.path.join(out_dir, f"{i+1}.mat"),
                    {"sample_data": Q_all[i]}
                )
```

---

## 📋 **建議方案（最優）**

### **步驟 1: 確認你用的是哪個腳本**
```bash
# 檢查生成的 Q_matrix 結構
ls D:\TACC\data\Q_martix\0cm\A1\
# 看看是有多個 .mat 還是只有 Q_all.mat
```

### **步驟 2: 如果用的是 Q_matrix_2.py（新格式）**

**選項 1：修改 CISDataset.py** (推薦)
- 改用上面提供的 `CISDataset_v2` 或類似邏輯
- 優勢：保留磁盤空間節省和 I/O 性能提升
- 缺點：需要修改 EXTRACTOR 代碼

**選項 2：在 train 前轉換格式**
- 使用方案 B 解包成舊格式
- 優勢：不需改 EXTRACTOR
- 缺點：浪費磁盤空間，I/O 變慢

### **步驟 3: 修改 cfg.py 的路徑**
```python
'train_data_root': 'D:\TACC\data\Q_martix\Q_matrix',  # 指向新生成的路徑
```

---

## 🎯 **最小改動方案**

如果你想保持 EXTRACTOR 完全不變：

1. **確保 Q_matrix_2.py 的輸出目錄中**
   - 每個卡的目錄下 **只有 1 個檔案：Q_all.mat**

2. **在訓練前執行轉換腳本**
   ```python
   # convert_q_matrix.py
   import os
   from scipy.io import loadmat, savemat
   import numpy as np
   
   def unpack_q_matrix():
       source_dir = r'D:\TACC\data\Q_martix'
       output_dir = r'D:\TACC\data\Q_martix_unpacked'
       
       for dist_folder in os.listdir(source_dir):
           dist_path = os.path.join(source_dir, dist_folder)
           if not os.path.isdir(dist_path):
               continue
           
           for card_folder in os.listdir(dist_path):
               card_path = os.path.join(dist_path, card_folder)
               q_all_path = os.path.join(card_path, 'Q_all.mat')
               
               if os.path.exists(q_all_path):
                   mat_data = loadmat(q_all_path)
                   Q_all = mat_data['Q_all']  # [N, F_sel, T-1]
                   
                   # 解包
                   out_dir = os.path.join(output_dir, dist_folder, card_folder)
                   os.makedirs(out_dir, exist_ok=True)
                   
                   for i in range(Q_all.shape[0]):
                       savemat(
                           os.path.join(out_dir, f'{i+1}.mat'),
                           {'sample_data': Q_all[i]},
                           do_compression=True
                       )
                   print(f"✓ Unpacked {dist_folder}/{card_folder}: {Q_all.shape[0]} samples")
   
   if __name__ == '__main__':
       unpack_q_matrix()
   ```

3. **在 cfg.py 中指向解包的目錄**
   ```python
   'train_data_root': r'D:\TACC\data\Q_martix_unpacked\Q_matrix'
   ```

---

## ✅ **最終檢查清單**

- [ ] 確認你運行的是 `Q_matrix_2.py` 還是 `Q_martix.py`
- [ ] 檢查生成的 Q_matrix 的實際結構
  - 如果是舊格式（多個 .mat 檔）：**無需改動 EXTRACTOR**
  - 如果是新格式（Q_all.mat）：**選擇方案 A 或 B**
- [ ] 測試新的資料格式是否能被 EXTRACTOR 正確讀取
- [ ] 更新 cfg.py 中的路徑配置
- [ ] 運行 `python main.py` 開始訓練

---

## 📝 **小結**

| 方面 | Q_martix.py (舊) | Q_matrix_2.py (新) | 對 EXTRACTOR 的影響 |
|------|-----------------|-------------------|--------------------|
| **存儲方式** | 每樣本一個 .mat | 每卡一個 Q_all.mat | ⚠️ 需要改 CISDataset |
| **磁盤空間** | 很大 | 較小 | - |
| **I/O 速度** | 慢（12000 次打開） | 快（1 次打開） | ✅ 訓練更快 |
| **元數據** | 無 | 完整 | ✅ 更多信息可用 |
| **兼容性** | 100% | 需要轉換或改代碼 | ❌ 需要處理 |

