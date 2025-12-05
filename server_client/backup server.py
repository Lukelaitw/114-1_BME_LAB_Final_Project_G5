# eeg_server_ctnet.py  (Own PC：使用 CTNet Ensemble 做即時分類)
import socket
import numpy as np
import json
import time

from inference import CTNetEnsembleInference  # 從你們 repo 來

HOST = '0.0.0.0'
PORT = 50007

# 遊戲控制 socket 設定（預設值，可通過參數修改）
GAME_HOST = '127.0.0.1'
GAME_PORT = 4789

FS = 500                   # Biopac 取樣率：500 Hz
WIN_SAMPLES = 1000         # 參考原 example：window_size=1000（約 2 秒）
STRIDE_SAMPLES = 500       # 滑動步長 500（約 1 秒出一次結果）

# 你要的類別名稱，先用兩類示範（依你實際模型改）
CLASS_NAMES = ['放鬆', '專注']  # 如果有第三類（眨眼）再加上去

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
    
    def update_control(self, label, prob=None):
        """
        根據預測的狀態更新遊戲控制（直接使用當前預測結果）
        
        Parameters:
        -----------
        label : str
            預測的類別標籤（例如：'放鬆', '專注', '眨眼'）
        prob : np.ndarray, optional
            各類別的機率分佈
        """
        if not self.connected:
            return
        
        # 眨眼直接處理
        if label == '眨眼' or 'blink' in label.lower():
            self.send_command({"jump": True})
            time.sleep(0.1)  # 短暫延遲
            self.send_command({"jump": False})
            self.last_jump_state = False
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


class OnlineCTNet:
    """
    用 CTNetEnsembleInference 做「線上」推論的小包裝。
    做法：每次拿到新的 samples，就用 predict_realtime 在那段資料上跑一次，
    並取最後一個 window 的結果。
    """

    def __init__(self):
        # ⚠️ 這裡的參數完全照 inference_example.py
        model_dir = "Loso_C_heads_2_depth_8_0"
        self.inferencer = CTNetEnsembleInference(
            model_dir=model_dir,
            dataset_type='C',
            heads=2, emb_size=16, depth=8,
            eeg1_f1=8, eeg1_kernel_size=64, eeg1_D=2,
            eeg1_pooling_size1=8, eeg1_pooling_size2=8,
            eeg1_dropout_rate=0.25, flatten_eeg1=240
        )

        # 內部 buffer：存 1D 浮點數列
        self.buffer = []

        # 上一次做分類時 buffer 的長度（用來實現 stride）
        self.last_pred_pos = 0

    def append_and_maybe_predict(self, new_values):
        """
        new_values: list[float]，這次從 Biopac 來的新 sample（大約 1 秒 500 筆）
        回傳： (label_str, prob_array) 或 (None, None) 如果目前樣本還不夠
        """

        # 1. 先把這次的 samples 加到 buffer
        self.buffer.extend(new_values)

        # 2. 若樣本還不夠做第一個 1000-window，就先不算
        if len(self.buffer) < WIN_SAMPLES:
            return None, None

        # 3. 控制 stride：如果距離上一次分類不到 STRIDE_SAMPLES，就暫時不算
        if len(self.buffer) - self.last_pred_pos < STRIDE_SAMPLES:
            return None, None

        # 4. 取「最後 WIN_SAMPLES 筆」當作一個 window
        data = np.array(self.buffer[-WIN_SAMPLES:], dtype=np.float32)

        # 5. 用 CTNetEnsembleInference 來跑這個 window
        #    利用 predict_realtime，但這裡只給它一個 chunk，
        #    並設定 window_size = len(data), stride = len(data)，所以只會產生 1 個 window。
        def gen():
            yield data

        last_result = None
        for result in self.inferencer.predict_realtime(
            gen(),
            window_size=len(data),
            stride=len(data),
            smoothing_window=3,   # 不需平滑，直接單次預測
            callback=None
        ):
            last_result = result

        if last_result is None:
            return None, None

        pred = int(last_result['prediction'])
        prob = np.array(last_result['probability'], dtype=np.float32)

        # 更新 last_pred_pos：這次已經用到 buffer 的最後一點
        self.last_pred_pos = len(self.buffer)

        # 轉成文字 label
        if 0 <= pred < len(CLASS_NAMES):
            label = CLASS_NAMES[pred]
        else:
            label = f"cls_{pred}"

        return label, prob


def parse_lines_to_values(lines):
    """
    把 new_lines（每行一筆 raw text）轉成 float list。
    假設每行形式類似：
        "123.45"
      或 "0.123  0.456  0.789"（即最後一欄是我們要的值）
    有非數字會自動略過。
    """
    values = []
    for ln in lines:
        parts = ln.strip().split()
        if not parts:
            continue
        try:
            v = float(parts[-1])  # 取最後一欄
            values.append(v)
        except ValueError:
            continue
    return values


def main():
    import argparse
    
    # 解析命令列參數
    parser = argparse.ArgumentParser(description='EEG 分類服務器，連接遊戲控制')
    parser.add_argument('--game-host', type=str, default=GAME_HOST,
                        help=f'遊戲 socket 主機地址（預設: {GAME_HOST}）')
    parser.add_argument('--game-port', type=int, default=GAME_PORT,
                        help=f'遊戲 socket 端口（預設: {GAME_PORT}）')
    parser.add_argument('--no-game', action='store_true',
                        help='不連接遊戲，僅進行分類')
    args = parser.parse_args()
    
    classifier = OnlineCTNet()
    
    # 初始化遊戲控制器
    game_controller = None
    if not args.no_game:
        game_controller = GameController(
            host=args.game_host, 
            port=args.game_port
        )
        # 嘗試連接遊戲（如果失敗會繼續運行，但不會發送控制信號）
        game_controller.connect()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        s.bind((HOST, PORT))
        s.listen(1)
        print(f"[Server] Listening on {HOST}:{PORT} ...")
        print(f"[Server] 等待 BIOPAC 連接...")

        conn, addr = s.accept()
        print(f"[Server] Connected by {addr}")
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        buffer = ""
        SEP = "\n===END===\n"
        block_id = 0

        # 儲存「累積文字行數」，辨識每包是不是重新開始
        total_line_count = 0

        try:
            with conn:
                while True:
                    data = conn.recv(4096)
                    if not data:
                        print("[Server] Connection closed")
                        break

                    buffer += data.decode("utf-8", errors="ignore")

                    while SEP in buffer:
                        block, buffer = buffer.split(SEP, 1)
                        block = block.strip()
                        if not block:
                            continue

                        block_id += 1

                        # 這包 BIOPAC 給你的所有行（從 0 秒到現在）
                        all_lines = [ln for ln in block.splitlines() if ln.strip()]

                        # 若行數突然變少，代表 BIOPAC 重新開始錄影，把計數歸零
                        if total_line_count > len(all_lines):
                            total_line_count = 0

                        # 取「這次新增的那一段行」
                        new_lines = all_lines[total_line_count:]
                        total_line_count = len(all_lines)

                        if not new_lines:
                            print(f"[Server] Block {block_id}: 沒有新資料")
                            continue

                        # 轉成 float list（每個元素是一個 sample）
                        new_values = parse_lines_to_values(new_lines)
                        print(
                            f"[Server] Block {block_id}: 新增 {len(new_values)} 筆樣本"
                        )

                        # 丟進 CTNet 線上分類器
                        label, prob = classifier.append_and_maybe_predict(new_values)
                        if label is not None:
                            print(
                                f"  → 預測狀態 = {label}, "
                                f"prob = {prob}"
                            )
                            
                            # 發送控制信號到遊戲
                            if game_controller:
                                # 如果未連接，嘗試重新連接
                                if not game_controller.connected:
                                    print(f"  → 嘗試重新連接到遊戲...")
                                    game_controller.connect()
                                
                                if game_controller.connected:
                                    game_controller.update_control(label, prob)
                                else:
                                    print(f"  → 警告：無法連接到遊戲，控制信號未發送")
        except KeyboardInterrupt:
            print("\n[Server] 收到中斷信號，正在關閉...")
        finally:
            if game_controller:
                game_controller.close()
            print("[Server] 已關閉")


if __name__ == "__main__":
    main()
