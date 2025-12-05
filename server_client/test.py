import sys
import os

# 添加 BrainLinkParser-Python 目錄到 Python 路徑
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'BrainLinkParser-Python'))

from cushy_serial import CushySerial
from BrainLinkParser import BrainLinkParser

def onRaw(raw):
    print(f"[RAW] raw = {raw}", flush=True)
    return

def onEEG(data):
    print("attention = " + str(data.attention) +
          " meditation = " + str(data.meditation) +
          " delta = " + str(data.delta) +
          " theta = " + str(data.theta) +
          " lowAlpha = " + str(data.lowAlpha) +
          " highAlpha = " + str(data.highAlpha) +
          " lowBeta = " + str(data.lowBeta) +
          " highBeta = " + str(data.highBeta) +
          " lowGamma = " + str(data.lowGamma) +
          " highGamma = " + str(data.highGamma), flush=True)
    return

def onExtendEEG(data):
    print("ap = " + str(data.ap) +
          " battery = " + str(data.battery) +
          " version = " + str(data.version) +
          " gnaw = " + str(data.gnaw) +
          " temperature = " + str(data.temperature) +
          " heart = " + str(data.heart), flush=True)
    return

def onGyro(x, y, z):
    print("x = " + str(x) + " y = " + str(y) + " z = " + str(z), flush=True)
    return

def onRR(rr1, rr2, rr3):
    print("rr1 = " + str(rr1) + " rr2 = " + str(rr2) + " rr3 = " + str(rr3), flush=True)
    return

parser = BrainLinkParser(onEEG, onExtendEEG, onGyro, onRR, onRaw)

# 檢查可用的串列端口
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

# 嘗試連接的端口列表（按優先順序）
possible_ports = [
    '/dev/cu.BrainLink_Lite',
    '/dev/cu.BrainLink_Pro',
    '/dev/cu.BrainLink',
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
    if os.path.exists(port_name):
        print(f"嘗試連接: {port_name}")
        try:
            serial = CushySerial(port_name, 115200)
            connected_port = port_name
            print(f"✓ 成功連接到 {port_name}")
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
    print("  - /dev/cu.BrainLink_Lite (BrainLink Lite)")
    print("  - /dev/cu.BrainLink_Pro (BrainLink Pro)")
    exit(1)

# 設置消息處理
message_count = 0
@serial.on_message()
def handle_serial_message(msg: bytes):
    global message_count
    message_count += 1
    print(f"[串列消息 #{message_count}] 收到 {len(msg)} 字節: {msg.hex()[:50]}...", flush=True)
    try:
        parser.parse(msg)
    except Exception as e:
        print(f"[錯誤] 解析消息時出錯: {e}", flush=True)

print("\n" + "=" * 60)
print("串列通訊已啟動，等待數據...")
print("=" * 60)
print("提示: 如果沒有數據，請檢查:")
print("  1. 設備是否正在發送數據")
print("  2. 設備是否需要啟動命令")
print("  3. 波特率是否正確 (115200)")
print("  4. 設備是否正確佩戴")
print("=" * 60)
print("按 Ctrl+C 停止程式\n")

# 保持程式運行
try:
    import time
    last_count = 0
    while True:
        time.sleep(2)
        if message_count > last_count:
            print(f"[狀態] 已收到 {message_count} 條消息", flush=True)
            last_count = message_count
        elif message_count == 0:
            print("[狀態] 等待數據中... (已等待 {} 秒)".format(int(time.time() - start_time) if 'start_time' in locals() else 0), flush=True)
            if 'start_time' not in locals():
                start_time = time.time()
except KeyboardInterrupt:
    print("\n\n正在停止程式...", flush=True)
    serial.close()
    print(f"程式已停止。總共收到 {message_count} 條消息", flush=True)
