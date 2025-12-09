from cushy_serial import CushySerial
from BrainLinkParser import BrainLinkParser
import time
import os
import socket
import json
import argparse

# -------- 遊戲控制 socket 設定（預設值）--------
GAME_HOST = '127.0.0.1'
GAME_PORT = 4789

# -------- 最新的專注/放鬆 --------
latest_attention = None
latest_meditation = None

# -------- 眨眼偵測參數（可微調）--------
BLINK_THRESHOLD = 1000       # 抓太多誤判就調高；抓不到就調低
BLINK_REFRACTORY = 0.25     # 250 ms 冷卻時間

_last_blink_time = 0.0
_blink_detected_in_window = False

# -------- 遊戲控制相關變數 --------
_last_lean_state = None  # 記錄上一次的傾斜狀態，避免重複發送

def onRaw(raw: int):
    """用 raw 的瞬間大振幅做簡易眨眼偵測（只記錄有沒有）"""
    global _last_blink_time, _blink_detected_in_window

    now = time.time()
    if abs(raw) >= BLINK_THRESHOLD and (now - _last_blink_time) >= BLINK_REFRACTORY:
        _blink_detected_in_window = True
        _last_blink_time = now

def onEEG(data):
    global latest_attention, latest_meditation
    latest_attention = data.attention
    latest_meditation = data.meditation

def onExtendEEG(data):
    return

def onGyro(x, y, z):
    return

def onRR(rr1, rr2, rr3):
    return

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
    
    def update_control(self, label, state_label=None):
        """
        根據 BrainLink 的 attention/meditation 狀態更新遊戲控制
        
        Parameters:
        -----------
        label : str
            狀態標籤（例如：'放鬆', '專注', '眨眼'）
        state_label : str, optional
            BrainLink 的分類結果（'放鬆' 或 '專注'），用於眨眼時同時發送 lean 資訊
        """
        if not self.connected:
            return
        
        # 眨眼處理：同時發送 jump 和 lean（使用 BrainLink 的 attention/meditation 分類結果）
        if label == '眨眼' or 'blink' in label.lower():
            # ① 發送 jump 指令（眨眼觸發跳躍）
            self.send_command({"jump": True})
            time.sleep(0.1)  # 短暫延遲
            self.send_command({"jump": False})
            self.last_jump_state = False
            
            # ② 同時發送 lean 資訊（使用 BrainLink 的分類結果：放鬆或專注）
            if state_label and state_label in ('放鬆', '專注'):
                self.send_command({"lean": state_label})
                if state_label == '放鬆':
                    self.last_lean = -0.5
                elif state_label == '專注':
                    self.last_lean = 0.5
                print(f"  → [控制] {label} + {state_label} (jump: ✓, lean: ✓)")
            else:
                print(f"  → [控制] {label} (jump: ✓, lean: ✗ 無狀態分類結果)")
            return
        
        # 直接使用當前 BrainLink 的分類結果
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

parser = BrainLinkParser(onEEG, onExtendEEG, onGyro, onRR, onRaw)

# ========== 解析命令列參數 ==========
arg_parser = argparse.ArgumentParser(description='BrainLink 數據讀取並發送到遊戲')
arg_parser.add_argument('--serial-port', type=str, default=None,
                        help='BrainLink 串口（例如：/dev/cu.BrainLink_Lite 或 COM3）')
arg_parser.add_argument('--baud', type=int, default=115200,
                        help='串口波特率（預設: 115200）')
arg_parser.add_argument('--game-host', type=str, default=GAME_HOST,
                        help=f'遊戲 socket 主機地址（預設: {GAME_HOST}）')
arg_parser.add_argument('--game-port', type=int, default=GAME_PORT,
                        help=f'遊戲 socket 端口（預設: {GAME_PORT}）')
arg_parser.add_argument('--no-game', action='store_true',
                        help='不連接遊戲，僅顯示狀態')
args = arg_parser.parse_args()

# 更新遊戲設定
GAME_HOST = args.game_host
GAME_PORT = args.game_port

# ========== 初始化遊戲控制器 ==========
game_controller = None
if not args.no_game:
    game_controller = GameController(host=GAME_HOST, port=GAME_PORT)
    # 嘗試連接遊戲（如果失敗會繼續運行，但不會發送控制信號）
    game_controller.connect()

# ========== 檢查可用的串列端口 ==========
print("=" * 60)
print("檢查可用的串列端口...")
print("=" * 60)
try:
    import serial.tools.list_ports
    available_ports = list(serial.tools.list_ports.comports())
    print(f"找到 {len(available_ports)} 個串列端口:")
    for port in available_ports:
        print(f"  - {port.device}: {port.description}")
    
    # 查找 BrainLink 相關端口
    brainlink_ports = [p for p in available_ports if 'BrainLink' in p.description or 'BrainLink' in p.device]
    if brainlink_ports:
        print(f"\n找到 {len(brainlink_ports)} 個可能的 BrainLink 端口:")
        for port in brainlink_ports:
            print(f"  - {port.device}: {port.description}")
    else:
        print("\n⚠ 未找到 BrainLink 相關端口，將嘗試常見的端口名稱")
except Exception as e:
    print(f"無法列出端口: {e}")

# ========== 嘗試連接串列端口 ==========
# 如果命令列指定了端口，優先使用
if args.serial_port:
    possible_ports = [args.serial_port]
else:
    # 嘗試連接的端口列表（按優先順序）
    possible_ports = [
        '/dev/cu.BrainLink_Lite',
        '/dev/cu.BrainLink_Pro',
        '/dev/cu.BrainLink',
        'COM3',  # Windows 常見端口
        'COM4',
    ]
    
    # 如果找到 BrainLink 端口，優先使用
    try:
        import serial.tools.list_ports
        available_ports = list(serial.tools.list_ports.comports())
        brainlink_ports = [p for p in available_ports if 'BrainLink' in p.description or 'BrainLink' in p.device]
        if brainlink_ports:
            possible_ports = [p.device for p in brainlink_ports] + possible_ports
    except:
        pass

print("\n" + "=" * 60)
print("嘗試連接串列端口...")
print("=" * 60)

serial = None
connected_port = None

for port_name in possible_ports:
    # 檢查端口是否存在（macOS/Linux）或嘗試連接（Windows）
    if os.path.exists(port_name) or port_name.startswith('COM'):
        print(f"嘗試連接: {port_name}")
        try:
            serial = CushySerial(port_name, args.baud)
            connected_port = port_name
            print(f"✓ 成功連接到 {port_name} (波特率: {args.baud})")
            break
        except Exception as e:
            print(f"✗ 連接失敗: {e}")
            continue

if serial is None:
    print("\n✗ 無法連接到任何端口！")
    print("請檢查:")
    print("1. BrainLink 設備是否已開啟")
    print("2. 設備是否已配對到電腦")
    print("3. 端口名稱是否正確")
    print("\n常見的端口名稱:")
    print("  - /dev/cu.BrainLink_Lite (BrainLink Lite - macOS)")
    print("  - /dev/cu.BrainLink_Pro (BrainLink Pro - macOS)")
    print("  - COM3, COM4 (Windows)")
    print("\n請手動修改程式碼中的端口名稱，或使用命令列參數指定端口")
    exit(1)

@serial.on_message()
def handle_serial_message(msg: bytes):
    parser.parse(msg)

print("Start... 每 1 秒輸出狀態；若有眨眼會顯示提示。Ctrl+C 結束。")
if game_controller and game_controller.connected:
    print(f"[遊戲] 已連接到遊戲，將發送控制信號")
elif not args.no_game:
    print(f"[遊戲] 未連接到遊戲，僅顯示狀態（遊戲可能未啟動）")

last_print = time.time()

try:
    while True:
        now = time.time()

        if now - last_print >= 1.0:
            # 狀態文字
            if latest_attention is not None and latest_meditation is not None:
                state = "專注中" if latest_attention >= latest_meditation else "放鬆中"
                # 轉換為 update_control 使用的格式（'專注' 或 '放鬆'）
                state_label = "專注" if latest_attention >= latest_meditation else "放鬆"

                blink_text = "（偵測到眨眼）" if _blink_detected_in_window else ""

                print(
                    f"Attention = {latest_attention:3d} , "
                    f"Meditation = {latest_meditation:3d}  --> 狀態：{state} {blink_text}"
                )
                
                # 發送控制信號到遊戲
                if game_controller and game_controller.connected:
                    # 處理眨眼：優先發送眨眼控制
                    if _blink_detected_in_window:
                        # 眨眼時同時發送 jump 和 lean（使用當前的 BrainLink 狀態分類結果）
                        game_controller.update_control("眨眼", state_label=state_label)
                    else:
                        # 只在狀態改變時發送，避免重複發送
                        if _last_lean_state != state:
                            game_controller.update_control(state_label)
                            _last_lean_state = state
            else:
                blink_text = "（偵測到眨眼）" if _blink_detected_in_window else ""
                print(f"狀態資料尚未更新 {blink_text}")
                
                # 即使沒有狀態資料，如果有眨眼也發送控制
                if _blink_detected_in_window:
                    if game_controller and game_controller.connected:
                        game_controller.update_control("眨眼")

            # 重置 1 秒窗口的眨眼旗標
            _blink_detected_in_window = False
            last_print = now

        time.sleep(0.005)

except KeyboardInterrupt:
    print("Stop.")
    if game_controller:
        game_controller.close()
    serial.close()
