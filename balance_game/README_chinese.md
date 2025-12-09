# 平衡遊戲使用說明

> **語言版本選擇 / Language Selection**
> 
> - 🇹🇼 [繁體中文 (Traditional Chinese)](README_chinese.md) ← 當前版本
> - 🇺🇸 [English](README.md)

## 首次使用：從專案根目錄

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 之後進入虛擬環境

```bash
source .venv/bin/activate
```

## 編譯應用程式

```bash
python -m compileall balance_game
```

## 使用兩個終端運行應用程式

### 終端 1：啟動遊戲

```bash
source .venv/bin/activate
python main.py --socket-input --socket-port 4789
```

### 終端 2：啟動 BrainLink 橋接（可選）

```bash
source .venv/bin/activate
python tools/brainlink_serial_bridge.py \
    --serial-port /dev/cu.BrainLink_Lite \
    --profile assets/blink_energy_profile.json \
    --game-port 4789 \
    --verbose \
    --debug-sensors
```

## 透過 BrainLink 眨眼觸發跳躍

遊戲可以透過內建的 ThinkGear socket 服務對 BrainLink / NeuroSky 的眨眼事件做出反應。

1. 配對並啟動 BrainLink 頭戴式設備，使用官方的 ThinkGear Connector（或相容服務）。  
   確保它在 `127.0.0.1:13854` 上串流 JSON 封包。
2. 啟動支援眨眼的遊戲：

   ```bash
   python main.py --brainlink
   ```

可選參數：

- `--blink-threshold <value>` – 改變觸發跳躍所需的眨眼強度（預設 55）。
- `--brainlink-host <host>` / `--brainlink-port <port>` – 連接到非預設的 ThinkGear socket。

您仍然可以使用鍵盤傾斜/跳躍；成功的眨眼動作就像按下跳躍鍵一樣。

## 透過 JSON socket 外部控制

如果您的 ML 模型或 AutoHotKey 腳本已經解釋了 BrainLink 數據，您可以將產生的控制信號直接串流到遊戲中。

1. 啟動遊戲並啟用 socket 監聽器（預設為 `127.0.0.1:4789`）：

   ```bash
   python main.py --socket-input
   ```

   使用 `--socket-host` / `--socket-port` 來改變綁定地址。

2. 從您的管道中，開啟一個 TCP 連接到該地址並發送換行分隔的 JSON 訊息，例如：

   ```json
   {"lean": -0.35}
   {"jump": true}
   {"jump": false}
   ```

   - `lean` 接受 `-1.0`（極左）到 `1.0`（極右）之間的值。
   - `jump` 就像按下和釋放跳躍鍵；短脈衝就足夠了。
   - 如果您願意，可以在一個訊息中包含兩個欄位：`{"lean": 0.1, "jump": true}`。
   - 可選的 `{"reset": true}` 將控制返回到鍵盤基準。

Socket 層與鍵盤和眨眼輸入堆疊，因此您可以隨時回退到手動控制。

## 眨眼能量訓練 + BrainLink 橋接

1. **生成能量設定檔（只需一次）**

   ```bash
   python tools/train_blink_energy.py \
       --datasets ~/Downloads/BME_Lab_BCI_training/bci_dataset_114-1 \
                 ~/Downloads/BME_Lab_BCI_training/bci_dataset_113-2 \
       --output assets/blink_energy_profile.json
   ```

   這會讀取各受試者的 `S*/3.txt`（含 20 秒睜眼／20 秒閉眼循環），計算開眼與閉眼的能量分佈並輸出建議的能量閾值。結果會寫進 `assets/blink_energy_profile.json`，後續橋接程式與即時偵測會自動讀取。

2. **啟動遊戲的 socket 監聽器**

   ```bash
   python main.py --socket-input
   ```

3. **執行 BrainLink → 模型 → 遊戲的橋接腳本**

   ```bash
   python tools/brainlink_socket_bridge.py \
       --thinkgear-host 127.0.0.1 --thinkgear-port 13854 \
       --game-port 4789 \
       --profile assets/blink_energy_profile.json \
       --model-module your_ml_module
   ```

   - `--profile` 指向上一步產生的能量設定，會驅動 `EnergyBlinkDetector` 讀取 raw EEG（需先開啟 ThinkGear Connector）。
   - `--model-module` 是選填的 Python 模組，需提供 `predict(packet: dict) -> dict`，可以在裡面載入專注/放鬆模型並輸出 `{"lean": …, "jump": …}`。若未指定，預設用冥想值對應傾斜，眨眼則由能量檢測決定。
   - 若您的模型也要外送 JSON，可直接在 `predict` 回傳字典即可。

4. 橋接腳本會把每次眨眼（能量短暫下降）轉成 `{"jump": true}` 的 JSON 指令送進遊戲的 socket。您也可以在自訂模組中利用 `packet["rawEeg"]` 自行處理特徵。

## 直接使用 BrainLinkParser 連接 BrainLink（不用 ThinkGear Connector）

1. **安裝需求（只需一次）**：
   ```bash
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **找到 BrainLink 的序列埠**：使用 `ls /dev/cu.*` 找到 BrainLink 的序列埠（例如 `/dev/cu.BrainLink_Lite`）。

3. **啟動遊戲 socket**：
   ```bash
   python main.py --socket-input
   ```

4. **使用 `tools/brainlink_serial_bridge.py` 直接解析 BrainLink 的串列資料並送進遊戲**：
   ```bash
   python tools/brainlink_serial_bridge.py \
       --serial-port /dev/cu.BrainLink_Lite \
       --profile assets/blink_energy_profile.json \
       --game-port 4789 \
       --verbose \
       --model-module your_ml_module   # 若沒有可省略
   ```

   - 腦波 raw 資料會經 `EnergyBlinkDetector` 做能量尖峰偵測 → 觸發 jump。
   - `--model-module` 可定義 `predict(packet: dict) -> dict`，回傳 `{"lean": …}` 等欄位；未指定時預設用 attention 值轉 lean。
   - 沒有 profile 時會 fallback 用 `blinkStrength >= threshold` 判斷眨眼。

> 如果橋接程式顯示 `Connection refused`，代表您還沒啟動 `python main.py --socket-input`；請先開遊戲 socket 再啟動橋接。

## 鍵盤控制

當遊戲運行時，您可以使用以下鍵盤控制：

- `A` / `←`：向左傾斜
- `D` / `→`：向右傾斜
- `Space` / `↑`：跳躍

## 故障排除

### 連接被拒絕

如果看到 `Connection refused` 錯誤：

1. 確認遊戲已啟動並使用 `--socket-input` 參數
2. 確認端口號碼正確（預設 4789）
3. 檢查防火牆設定

### 無法偵測眨眼

1. 確認 BrainLink 設備已正確連接
2. 檢查 `blink_energy_profile.json` 是否存在
3. 調整 `--blink-threshold` 參數

## 相關文件

- [專案主 README](../README_chinese.md)
- [遊戲控制整合說明](../server_client/GAME_CONTROL_README.md)
- [BrainLink 使用說明](../brainlink/README_USAGE.md)
