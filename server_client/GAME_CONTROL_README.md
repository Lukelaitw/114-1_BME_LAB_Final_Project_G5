# 遊戲控制整合說明

這個專案已整合了將 EEG 預測狀態傳輸到遊戲的功能。

## 功能概述

`eeg_server_ctnet.py` 現在可以：
1. 接收 BIOPAC 的 EEG 數據
2. 使用 CTNet 模型進行即時分類
3. 根據預測結果自動發送控制信號到遊戲

## 使用方法

### 1. 啟動遊戲（必須先啟動）

在一個終端中啟動遊戲並開啟 socket 輸入：

```bash
source .venv/bin/activate
python main.py --socket-input --socket-port 4789
```

### 2. 啟動 EEG 分類服務器

在另一個終端中啟動分類服務器：

```bash
source .venv/bin/activate
python eeg_server_ctnet.py
```

預設會嘗試連接到 `127.0.0.1:4789`。如果遊戲在不同的主機或端口，可以使用：

```bash
python eeg_server_ctnet.py --game-host 127.0.0.1 --game-port 4789
```

如果只想進行分類而不連接遊戲，可以使用：

```bash
python eeg_server_ctnet.py --no-game
```

### 3. 測試連接

在啟動遊戲和服務器之前，可以使用測試腳本驗證連接：

```bash
python test_game_control.py
```

這會進入互動模式，可以手動發送控制指令測試。

## 控制邏輯

目前預設的控制映射如下：

- **放鬆** → 向左傾斜 (`lean: -0.5`)
- **專注** → 向右傾斜 (`lean: 0.5`)
- **眨眼** → 觸發跳躍 (`jump: true` 然後 `jump: false`)

### 自訂控制邏輯

可以在 `eeg_server_ctnet.py` 的 `GameController.update_control()` 方法中修改控制邏輯：

```python
def update_control(self, label, prob=None):
    if label == '放鬆':
        # 自訂放鬆狀態的控制
        self.send_command({"lean": -0.5})
    elif label == '專注':
        # 自訂專注狀態的控制
        self.send_command({"lean": 0.5})
    # ... 其他狀態
```

## 遊戲控制指令格式

遊戲接收的 JSON 指令格式：

```json
{"lean": -0.35}      // 傾斜值，範圍 -1.0（左）到 1.0（右）
{"jump": true}       // 觸發跳躍
{"jump": false}      // 停止跳躍
{"reset": true}      // 重置控制
```

可以在同一個訊息中包含多個指令：

```json
{"lean": 0.1, "jump": true}
```

## 測試腳本使用

`test_game_control.py` 提供互動式測試：

```bash
python test_game_control.py
```

可用指令：
- `lean <value>` - 設定傾斜值（-1.0 到 1.0）
- `jump` - 觸發跳躍
- `reset` - 重置控制
- `test` - 測試連接
- `demo` - 執行示範序列
- `quit` - 退出

也可以只測試連接：

```bash
python test_game_control.py --test-only
```

## 故障排除

### 連接被拒絕

如果看到 `Connection refused` 錯誤：

1. 確認遊戲已啟動並使用 `--socket-input` 參數
2. 確認端口號碼正確（預設 4789）
3. 使用 `test_game_control.py` 測試連接

### 沒有控制信號

如果分類正常但沒有發送控制信號：

1. 檢查 `GameController` 是否成功連接（查看啟動時的日誌）
2. 確認 `CLASS_NAMES` 中的類別名稱與 `update_control()` 中的判斷一致
3. 檢查是否有預測結果產生（查看日誌輸出）

### 修改類別名稱

如果模型的類別名稱不同，請修改 `eeg_server_ctnet.py` 中的：

```python
CLASS_NAMES = ['放鬆', '專注']  # 改為你的類別名稱
```

並相應地更新 `update_control()` 方法中的判斷邏輯。

## 程式碼結構

- `eeg_server_ctnet.py` - 主服務器，包含：
  - `GameController` 類：管理與遊戲的 socket 連接
  - `OnlineCTNet` 類：線上 EEG 分類
  - `main()` 函數：主程序邏輯

- `test_game_control.py` - 測試工具，用於驗證連接和控制功能

## 注意事項

1. **啟動順序**：必須先啟動遊戲，再啟動 EEG 服務器
2. **端口衝突**：確保遊戲和服務器使用的端口不衝突
3. **數據格式**：確保 BIOPAC 數據格式符合 `parse_lines_to_values()` 的預期
4. **模型路徑**：確認模型目錄 `Loso_C_heads_2_depth_8_0` 存在且包含模型文件


