from cushy_serial import CushySerial
from BrainLinkParser import BrainLinkParser
import time

# -------- 最新的專注/放鬆 --------
latest_attention = None
latest_meditation = None

# -------- 眨眼偵測參數（可微調）--------
BLINK_THRESHOLD = 1000       # 抓太多誤判就調高；抓不到就調低
BLINK_REFRACTORY = 0.25     # 250 ms 冷卻時間

_last_blink_time = 0.0
_blink_detected_in_window = False

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

parser = BrainLinkParser(onEEG, onExtendEEG, onGyro, onRR, onRaw)

# ★ 用你確認過的「連出」COM port
serial = CushySerial("COM3", 115200)

@serial.on_message()
def handle_serial_message(msg: bytes):
    parser.parse(msg)

print("Start... 每 1 秒輸出狀態；若有眨眼會顯示提示。Ctrl+C 結束。")

last_print = time.time()

try:
    while True:
        now = time.time()

        if now - last_print >= 1.0:
            # 狀態文字
            if latest_attention is not None and latest_meditation is not None:
                state = "專注中" if latest_attention >= latest_meditation else "放鬆中"

                blink_text = "（偵測到眨眼）" if _blink_detected_in_window else ""

                print(
                    f"Attention = {latest_attention:3d} , "
                    f"Meditation = {latest_meditation:3d}  --> 狀態：{state} {blink_text}"
                )
            else:
                blink_text = "（偵測到眨眼）" if _blink_detected_in_window else ""
                print(f"狀態資料尚未更新 {blink_text}")

            # 重置 1 秒窗口的眨眼旗標
            _blink_detected_in_window = False
            last_print = now

        time.sleep(0.005)

except KeyboardInterrupt:
    print("Stop.")
    serial.close()
