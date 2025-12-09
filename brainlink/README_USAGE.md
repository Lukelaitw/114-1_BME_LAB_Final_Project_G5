# BrainLink 原始數據讀取 -> 分類器 -> 遊戲控制 使用說明

> **語言版本選擇 / Language Selection**
> 
> - 🇹🇼 [繁體中文 (Traditional Chinese)](README_USAGE.md) ← 當前版本
> - 🇺🇸 [English](README_USAGE_EN.md)

## 功能概述

這個腳本 (`read_brainlink_raw.py`) 實現了完整的數據流：
1. **從 BrainLink 設備讀取原始 EEG 數據**
2. **使用 CTNet 分類器進行即時分類**（放鬆/專注）
3. **偵測眨眼動作**
4. **將控制信號發送到遊戲**

## 前置條件

### 1. 安裝依賴套件
```bash
# 確保已安裝必要的 Python 套件
pip install cushy-serial numpy torch
```

### 2. 確認模型文件存在
確保以下路徑存在模型文件：
```
server_client/Loso_C_heads_2_depth_8_0/
  ├── model_1.pth
  ├── model_2.pth
  └── ... (其他模型文件)
```

### 3. 連接 BrainLink 設備
- **Windows**: 配對藍牙後，在「裝置管理員」中找到 COM 端口（選擇「輸出」端口）
- **Mac**: 配對後通常為 `/dev/cu.BrainLink_Pro` 或 `/dev/cu.BrainLink_Lite`

### 4. 啟動遊戲（如果需要控制遊戲）
遊戲需要以 socket 模式啟動：
```bash
cd balance_game
python main.py --socket-input --socket-port 4789
```

## 基本使用方法

### Windows 系統
```bash
# 從項目根目錄執行
python brainlink/read_brainlink_raw.py --serial-port COM3
```

### Mac 系統
```bash
# 從項目根目錄執行
python brainlink/read_brainlink_raw.py --serial-port /dev/cu.BrainLink_Pro
```

## 命令行參數

### 必需參數
無（有預設值）

### 可選參數

| 參數 | 說明 | 預設值 | 範例 |
|------|------|--------|------|
| `--serial-port` | BrainLink 串口路徑 | `COM3` | `--serial-port COM3` (Windows)<br>`--serial-port /dev/cu.BrainLink_Pro` (Mac) |
| `--baud` | 串口波特率 | `115200` | `--baud 57600` |
| `--game-host` | 遊戲 socket 主機地址 | `127.0.0.1` | `--game-host 192.168.1.100` |
| `--game-port` | 遊戲 socket 端口 | `4789` | `--game-port 5000` |
| `--no-game` | 不連接遊戲，僅進行分類 | `False` | `--no-game` |

## 使用範例

### 範例 1: 基本使用（Windows）
```bash
python brainlink/read_brainlink_raw.py --serial-port COM3
```

### 範例 2: Mac 系統，自定義波特率
```bash
python brainlink/read_brainlink_raw.py \
    --serial-port /dev/cu.BrainLink_Pro \
    --baud 57600
```

### 範例 3: 僅進行分類，不連接遊戲
```bash
python brainlink/read_brainlink_raw.py \
    --serial-port COM3 \
    --no-game
```

### 範例 4: 自定義遊戲地址和端口
```bash
python brainlink/read_brainlink_raw.py \
    --serial-port COM3 \
    --game-host 127.0.0.1 \
    --game-port 5000
```

## 如何找到正確的串口

### Windows 系統

1. **方法一：裝置管理員**
   - 開啟「裝置管理員」
   - 展開「連接埠 (COM 和 LPT)」
   - 尋找 BrainLink 相關的 COM 端口（通常是 `COM3`, `COM4` 等）
   - **重要**: 選擇「輸出」端口（參考 BrainLinkParser README）

2. **方法二：使用 Python 列出所有端口**
   ```python
   import serial.tools.list_ports
   ports = list(serial.tools.list_ports.comports())
   for p in ports:
       print(f"{p.device}: {p.description}")
   ```

### Mac 系統

1. **列出所有串口**
   ```bash
   ls /dev/cu.*
   ```

2. **常見的 BrainLink 端口名稱**
   - `/dev/cu.BrainLink_Pro` (BrainLink Pro)
   - `/dev/cu.BrainLink_Lite` (BrainLink Lite)
   - `/dev/cu.BrainLink` (通用)

## 輸出說明

### 初始化階段
```
[初始化] 正在載入 CTNet 分類器...
[初始化] 分類器載入完成
[GameController] 已連接到遊戲 127.0.0.1:4789
[串口] 正在連接 COM3 (波特率: 115200)...
[啟動] 開始讀取 BrainLink 數據...
[提示] 每 1 秒輸出狀態；若有眨眼會顯示提示。Ctrl+C 結束。
```

### 運行階段

**狀態輸出（每秒一次）**：
```
[狀態] Attention =  45 , Meditation =  60  --> 放鬆中
[狀態] Attention =  70 , Meditation =  30  --> 專注中 （偵測到眨眼）
```

**分類結果輸出**：
```
  → 預測狀態 = 放鬆, prob(放鬆,專注) = [0.85 0.15]
  → [控制] 放鬆
  → 偵測狀態 = 眨眼（raw spike），CTNet prob(放鬆,專注) = [0.75 0.25]
  → [控制] 眨眼 + 放鬆
```

## 控制邏輯

### 狀態映射

| 偵測狀態 | 遊戲動作 |
|---------|---------|
| **放鬆** | 向左傾斜 (`lean: "放鬆"`) |
| **專注** | 向右傾斜 (`lean: "專注"`) |
| **眨眼** | 觸發跳躍 (`jump: true` → `jump: false`) + 同時發送當前 CTNet 分類結果作為傾斜 |

### 分類參數

- **窗口大小**: 1000 個樣本（約 2 秒，假設 500 Hz 取樣率）
- **步長**: 300 個樣本（約 0.6 秒）
- **眨眼偵測**: 振幅超過 150.0 的樣本數 >= 10 個

## 故障排除

### 問題 1: 無法連接串口
```
[串口] 正在連接 COM3 (波特率: 115200)...
錯誤: [Errno 2] No such file or directory: 'COM3'
```

**解決方法**:
- 確認 BrainLink 設備已配對並開啟
- 檢查串口名稱是否正確
- Windows: 檢查「裝置管理員」中的 COM 端口號
- Mac: 使用 `ls /dev/cu.*` 列出可用端口

### 問題 2: 無法連接遊戲
```
[GameController] 警告：無法連接到遊戲 127.0.0.1:4789
  請確認遊戲已啟動並使用 --socket-input 參數
```

**解決方法**:
- 確認遊戲已啟動
- 確認遊戲使用了 `--socket-input` 參數
- 檢查端口號是否匹配（預設 4789）
- 使用 `--no-game` 參數跳過遊戲連接，僅進行分類

### 問題 3: 分類器載入失敗
```
[初始化] 正在載入 CTNet 分類器...
錯誤: 找不到模型文件
```

**解決方法**:
- 確認 `server_client/Loso_C_heads_2_depth_8_0/` 目錄存在
- 確認目錄中包含 `model_*.pth` 文件
- 檢查文件權限

### 問題 4: 沒有分類結果輸出
```
[狀態] 資料尚未更新
```

**可能原因**:
- 數據尚未累積足夠（需要至少 1000 個樣本）
- BrainLink 設備未正確傳輸數據
- 檢查串口連接是否正常

**解決方法**:
- 等待幾秒鐘讓數據累積
- 檢查 BrainLink 設備是否正常工作
- 確認 `onRaw` 回調是否被調用

## 進階配置

### 調整眨眼偵測靈敏度

編輯 `read_brainlink_raw.py` 中的參數：

```python
# 眨眼偵測參數
BLINK_AMP_THRESHOLD = 150.0          # 調高 = 更難偵測，調低 = 更容易偵測
BLINK_MIN_SAMPLES = int(0.02 * FS)   # 至少 20 ms 以上都很大才當眨眼
```

### 調整分類窗口和步長

```python
WIN_SAMPLES = 1000         # 窗口大小（樣本數）
STRIDE_SAMPLES = 300       # 步長（樣本數）
```

## 停止程序

按 `Ctrl+C` 停止程序，程序會自動：
- 關閉遊戲連接
- 關閉串口連接
- 清理資源

## 注意事項

1. **取樣率**: 目前假設 BrainLink 取樣率為 500 Hz，如果實際不同，需要調整 `FS` 參數
2. **模型路徑**: 確保從項目根目錄執行，否則模型路徑可能無法正確解析
3. **遊戲連接**: 如果遊戲未啟動，程序仍會繼續運行，但不會發送控制信號
4. **數據緩衝**: Raw 數據會累積在內存中，長時間運行請注意內存使用

## 相關文件

- `server_client/eeg_server_ctnet.py` - 參考實現（從 Biopac socket 接收數據）
- `server_client/GAME_CONTROL_README.md` - 遊戲控制協議說明
- `BrainLinkParser-Python/README.md` - BrainLinkParser 使用說明
