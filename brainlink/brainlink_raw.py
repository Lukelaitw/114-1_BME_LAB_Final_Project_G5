# read_brainlink_raw.py - 從 BrainLink 讀取原始數據 -> 分類器 -> 遊戲控制
import sys
import os
import json
import numpy as np
from pathlib import Path

from cushy_serial import CushySerial
from BrainLinkParser import BrainLinkParser
from inference import CTNetEnsembleInference
import time
import socket

# ========== 配置參數 ==========
# 遊戲控制 socket 設定
GAME_HOST = '127.0.0.1'
GAME_PORT = 4789

# CTNet 參數（參考 eeg_server_ctnet.py）
FS = 500                   # BrainLink 取樣率：500 Hz（需要確認實際值）
WIN_SAMPLES = 1000         # CTNet window（約 2 秒）
STRIDE_SAMPLES = 300       # 每隔約 0.6 秒出一次結果

# 眨眼偵測參數
BLINK_AMP_THRESHOLD = 150.0          # 振幅門檻：|EEG| 超過這個就視為「很大」
BLINK_MIN_SAMPLES = int(0.02 * FS)   # 至少 20 ms 以上都很大才當眨眼（約 10 點）
BLINK_REFRACTORY = 0.25              # 250 ms 冷卻時間

# CTNet 的兩類（放鬆 / 專注），眨眼我們額外用 rule 判定
CLASS_NAMES = ['放鬆', '專注']

# ========== GameController 類 ==========
class GameController:
    """
    管理與遊戲 socket 的連接，並發送控制信號
    """
    
    def __init__(self, host='127.0.0.1', port=4789):
        self.host = host
        self.port = port
        self.socket = None
        self.connected = False
        self.last_lean = 0.0
        self.last_jump_state = False
        
    def connect(self):
        """連接到遊戲 socket"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.socket.connect((self.host, self.port))
            self.connected = True
            print(f"[GameController] 已連接到遊戲 {self.host}:{self.port}")
            return True
        except ConnectionRefusedError:
            print(f"[GameController] 警告：無法連接到遊戲 {self.host}:{self.port}")
            print(f"  請確認遊戲已啟動並使用 --socket-input 參數")
            self.connected = False
            return False
        except Exception as e:
            print(f"[GameController] 連接錯誤: {e}")
            self.connected = False
            return False
    
    def send_command(self, command_dict):
        """
        發送 JSON 控制指令到遊戲
        
        Parameters:
        -----------
        command_dict : dict
            包含控制指令的字典，例如：
            - {"lean": -0.35}  # 傾斜值，範圍 -1.0 到 1.0
            - {"jump": true}   # 觸發跳躍
            - {"jump": false}  # 停止跳躍
            - {"reset": true}  # 重置控制
        """
        if not self.connected or self.socket is None:
            return False
        
        try:
            # 將字典轉換為 JSON 字串並加上換行符
            message = json.dumps(command_dict, ensure_ascii=False) + "\n"
            self.socket.sendall(message.encode('utf-8'))
            return True
        except Exception as e:
            print(f"[GameController] 發送指令失敗: {e}")
            self.connected = False
            return False
    
    def update_control(self, label, prob=None, ctnet_label=None):
        """
        根據預測的狀態更新遊戲控制（直接使用當前預測結果）
        
        Parameters:
        -----------
        label : str
            預測的類別標籤（例如：'放鬆', '專注', '眨眼'）
        prob : np.ndarray, optional
            各類別的機率分佈
        ctnet_label : str, optional
            CTNet 的分類結果（'放鬆' 或 '專注'），用於眨眼時同時發送 lean 資訊
        """
        if not self.connected:
            return
        
        # 眨眼處理：同時發送 jump 和 lean（使用 CTNet 的分類結果）
        if label == '眨眼' or 'blink' in label.lower():
            # ① 發送 jump 指令（眨眼觸發跳躍）
            jump_sent = self.send_command({"jump": True})
            time.sleep(0.1)  # 短暫延遲
            self.send_command({"jump": False})
            self.last_jump_state = False
            
            # ② 同時發送 lean 資訊（使用 CTNet 的分類結果：放鬆或專注）
            if ctnet_label and ctnet_label in ('放鬆', '專注'):
                lean_sent = self.send_command({"lean": ctnet_label})
                if ctnet_label == '放鬆':
                    self.last_lean = -0.5
                elif ctnet_label == '專注':
                    self.last_lean = 0.5
                print(f"  → [控制] {label} + {ctnet_label} (jump: ✓, lean: ✓)")
            else:
                print(f"  → [控制] {label} (jump: ✓, lean: ✗ 無 CTNet 分類結果)")
            return
        
        # 直接使用當前預測結果
        if label == '放鬆':
            # 放鬆狀態：向左傾斜
            self.send_command({"lean": "放鬆"})
            self.last_lean = -0.5
            print(f"  → [控制] {label}")
            
        elif label == '專注':
            # 專注狀態：向右傾斜
            self.send_command({"lean": "專注"})
            self.last_lean = 0.5
            print(f"  → [控制] {label}")
    
    def reset(self):
        """重置遊戲控制"""
        if self.connected:
            self.send_command({"reset": True})
            self.last_lean = 0.0
            self.last_jump_state = False
    
    def close(self):
        """關閉連接"""
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
        self.connected = False
        self.socket = None


# ========== OnlineCTNet 類 ==========
class OnlineCTNet:
    """
    用 CTNetEnsembleInference 做「線上」推論的小包裝。
    每次收到新的 samples，就更新 buffer，
    滿足 stride 才用最後 WIN_SAMPLES 筆做一次分類。
    """

    def __init__(self):
        # ⚠️ 這裡的參數完全照 eeg_server_ctnet.py
        model_dir = str( ".."/ "server_client" / "Loso_C_heads_2_depth_8_0")
        self.inferencer = CTNetEnsembleInference(
            model_dir=model_dir,
            dataset_type='C',
            heads=2, emb_size=16, depth=8,
            eeg1_f1=8, eeg1_kernel_size=64, eeg1_D=2,
            eeg1_pooling_size1=8, eeg1_pooling_size2=8,
            eeg1_dropout_rate=0.25, flatten_eeg1=240
        )

        # 內部 buffer：存 1D 浮點數列（累積所有 sample）
        self.buffer = []

        # 上一次做分類時 buffer 的長度（用來實現 stride）
        self.last_pred_pos = 0

    def append_and_maybe_predict(self, new_values):
        """
        new_values: list[float]，這次從 BrainLink 來的新 sample
        回傳： (label_str, prob_array) 或 (None, None) 如果目前樣本還不夠
        """

        # 1. 加入 buffer
        self.buffer.extend(new_values)

        # 2. 若樣本還不夠做第一個 1000-window，就先不算
        if len(self.buffer) < WIN_SAMPLES:
            return None, None

        # 3. 控制 stride：若距離上一次分類不到 STRIDE_SAMPLES，就暫時不算
        if len(self.buffer) - self.last_pred_pos < STRIDE_SAMPLES:
            return None, None

        # 4. 取「最後 WIN_SAMPLES 筆」當作一個 window
        data = np.array(self.buffer[-WIN_SAMPLES:], dtype=np.float32)

        # 5. 用 CTNetEnsembleInference 來跑這個 window
        def gen():
            yield data

        last_result = None
        for result in self.inferencer.predict_realtime(
            gen(),
            window_size=len(data),
            stride=len(data),
            smoothing_window=3,   # 輕微平滑，避免太抖動
            callback=None
        ):
            last_result = result

        if last_result is None:
            return None, None

        pred = int(last_result['prediction'])
        prob = np.array(last_result['probability'], dtype=np.float32)

        # 更新 last_pred_pos：這次已經用到 buffer 的最後一點
        self.last_pred_pos = len(self.buffer)

        # 轉成文字 label（只有放鬆 / 專注兩種）
        if 0 <= pred < len(CLASS_NAMES):
            label = CLASS_NAMES[pred]
        else:
            label = f"cls_{pred}"

        return label, prob


# ========== 眨眼偵測函數 ==========
def detect_blink_from_block(values,
                            amp_threshold=BLINK_AMP_THRESHOLD,
                            min_samples=BLINK_MIN_SAMPLES):
    """
    用 raw data 的振幅來判斷這一批 samples 裡有沒有眨眼。
    values: list[float] 或 1D np.array
    Rule: 若 |x| > amp_threshold 的樣本數 >= min_samples，就視為眨眼。
    """
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return False

    over = np.abs(arr) > float(amp_threshold)
    n_over = int(np.count_nonzero(over))

    return n_over >= int(min_samples)


# ========== 全局變數 ==========
# 用於累積 raw 數據的 buffer
raw_buffer = []
raw_buffer_lock = False  # 簡單的鎖機制

# 已移除 BrainLink 的 attention/meditation 判斷

# 眨眼偵測相關
_last_blink_time = 0.0
_blink_detected_in_window = False


# ========== BrainLink 回調函數 ==========
def onRaw(raw: int):
    """累積 raw 數據到 buffer，用於分類和眨眼偵測"""
    global raw_buffer, raw_buffer_lock
    
    if not raw_buffer_lock:
        raw_buffer.append(float(raw))


def onEEG(data):
    """不再使用 BrainLink 的 attention/meditation 判斷"""
    return


def onExtendEEG(data):
    return


def onGyro(x, y, z):
    return


def onRR(rr1, rr2, rr3):
    return


# ========== 主程序 ==========
def main():
    import argparse
    
    # 解析命令列參數
    parser = argparse.ArgumentParser(description='BrainLink 原始數據讀取 -> 分類器 -> 遊戲控制')
    parser.add_argument('--serial-port', type=str, default='COM3',
                        help='BrainLink 串口（Windows: COM3, Mac: /dev/cu.BrainLink_Pro）')
    parser.add_argument('--baud', type=int, default=115200,
                        help='串口波特率（預設: 115200）')
    parser.add_argument('--game-host', type=str, default=GAME_HOST,
                        help=f'遊戲 socket 主機地址（預設: {GAME_HOST}）')
    parser.add_argument('--game-port', type=int, default=GAME_PORT,
                        help=f'遊戲 socket 端口（預設: {GAME_PORT}）')
    parser.add_argument('--no-game', action='store_true',
                        help='不連接遊戲，僅進行分類')
    args = parser.parse_args()
    
    # 初始化分類器
    print("[初始化] 正在載入 CTNet 分類器...")
    classifier = OnlineCTNet()
    print("[初始化] 分類器載入完成")
    
    # 初始化遊戲控制器
    game_controller = None
    if not args.no_game:
        game_controller = GameController(
            host=args.game_host, 
            port=args.game_port
        )
        # 嘗試連接遊戲（如果失敗會繼續運行，但不會發送控制信號）
        game_controller.connect()
    
    # 初始化 BrainLink 解析器
    parser = BrainLinkParser(onEEG, onExtendEEG, onGyro, onRR, onRaw)
    
    # 初始化串口
    print(f"[串口] 正在連接 {args.serial_port} (波特率: {args.baud})...")
    serial = CushySerial(args.serial_port, args.baud)
    
    @serial.on_message()
    def handle_serial_message(msg: bytes):
        parser.parse(msg)
    
    print("[啟動] 開始讀取 BrainLink 數據...")
    print("[提示] 每 1 秒輸出狀態；若有眨眼會顯示提示。Ctrl+C 結束。")
    
    last_print = time.time()
    last_process_time = time.time()
    process_interval = 0.1  # 每 100ms 處理一次數據
    
    try:
        while True:
            now = time.time()
            
            # 定期處理 raw_buffer 中的數據
            if now - last_process_time >= process_interval:
                global raw_buffer, raw_buffer_lock, _blink_detected_in_window
                
                # 鎖定 buffer，複製數據
                raw_buffer_lock = True
                if raw_buffer:
                    current_raw = raw_buffer.copy()
                    raw_buffer.clear()
                else:
                    current_raw = []
                raw_buffer_lock = False
                
                if current_raw:
                    # ① 對 current_raw 進行標準化（Z-score normalization）
                    current_raw_array = np.array(current_raw, dtype=np.float32)
                    if len(current_raw_array) > 0:
                        raw_mean = np.mean(current_raw_array)
                        raw_std = np.std(current_raw_array)
                        if raw_std > 0:  # 避免除以零
                            current_raw_normalized = (current_raw_array - raw_mean) / raw_std
                        else:
                            current_raw_normalized = current_raw_array
                        current_raw = current_raw_normalized.tolist()
                    
                    # ② 先做「眨眼偵測」（用標準化後的 data）
                    blink = detect_blink_from_block(current_raw)
                    
                    if blink:
                        _blink_detected_in_window = True
                    
                    # ③ 丟進 CTNet 線上分類器（放鬆 / 專注）
                    label, prob = classifier.append_and_maybe_predict(current_raw)
                    
                    # 還沒累積到可以分類就先不處理
                    if label is None:
                        last_process_time = now
                        time.sleep(0.01)
                        continue
                    
                    # ④ 整合成「狀態」，如果偵測到眨眼就覆蓋狀態
                    if blink:
                        state_label = "眨眼"
                        # 顯示格式與 eeg_server_ctnet.py 一致，顯示放鬆和專注的機率
                        if prob is not None and len(prob) >= 2:
                            print(
                                f"  → 偵測狀態 = {state_label}（raw spike， 預測狀態 = {label},"
                                f"CTNet prob(放鬆={prob[0]:.3f}, 專注={prob[1]:.3f})"
                            )
                        else:
                            print(
                                f"  → 偵測狀態 = {state_label}（raw spike），"
                                f"CTNet prob(放鬆,專注) = {prob}"
                            )
                    else:
                        state_label = label
                        print(
                            f"  → 預測狀態 = {state_label}, "
                            f"prob(放鬆,專注) = {prob}"
                        )
                    
                    # 發送控制信號到遊戲
                    if game_controller:
                        # 如果未連接，嘗試重新連接
                        if not game_controller.connected:
                            print(f"  → 嘗試重新連接到遊戲...")
                            game_controller.connect()
                        
                        if game_controller.connected:
                            # 眨眼時需要傳遞 CTNet 的分類結果（label）作為 ctnet_label
                            if blink:
                                game_controller.update_control(state_label, prob, ctnet_label=label)
                            else:
                                game_controller.update_control(state_label, prob)
                        else:
                            print(f"  → 警告：無法連接到遊戲，控制信號未發送")
                
                last_process_time = now
            
            # 每秒輸出一次狀態（用於顯示）
            if now - last_print >= 1.0:
                blink_text = "（偵測到眨眼）" if _blink_detected_in_window else ""
                print(f"[狀態] 運行中... {blink_text}")
                
                # 重置 1 秒窗口的眨眼旗標
                _blink_detected_in_window = False
                last_print = now
            
            time.sleep(0.005)
    
    except KeyboardInterrupt:
        print("\n[停止] 收到中斷信號，正在關閉...")
    finally:
        if game_controller:
            game_controller.close()
        serial.close()
        print("[停止] 已關閉所有連接")


if __name__ == "__main__":
    main()
