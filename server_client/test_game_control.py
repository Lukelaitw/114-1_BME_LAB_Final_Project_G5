#!/usr/bin/env python3
"""
測試腳本：用於測試與遊戲的 socket 連接和控制信號發送

使用方法：
    python test_game_control.py

或者指定不同的主機和端口：
    python test_game_control.py --host 127.0.0.1 --port 4789
"""

import socket
import json
import time
import argparse


def send_game_command(host, port, command_dict):
    """
    發送單個控制指令到遊戲
    
    Parameters:
    -----------
    host : str
        遊戲 socket 主機地址
    port : int
        遊戲 socket 端口
    command_dict : dict
        控制指令字典，例如 {"lean": -0.35} 或 {"jump": True}
    
    Returns:
    --------
    bool
        是否成功發送
    """
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.connect((host, port))
        
        message = json.dumps(command_dict, ensure_ascii=False) + "\n"
        sock.sendall(message.encode('utf-8'))
        
        sock.close()
        return True
    except ConnectionRefusedError:
        print(f"❌ 無法連接到遊戲 {host}:{port}")
        print("   請確認遊戲已啟動並使用 --socket-input 參數")
        return False
    except Exception as e:
        print(f"❌ 發送指令失敗: {e}")
        return False


def test_connection(host, port):
    """測試與遊戲的連接"""
    print(f"正在測試連接到 {host}:{port}...")
    
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        sock.connect((host, port))
        sock.close()
        print(f"✅ 成功連接到遊戲 {host}:{port}")
        return True
    except ConnectionRefusedError:
        print(f"❌ 無法連接到遊戲 {host}:{port}")
        print("   請確認遊戲已啟動並使用 --socket-input 參數")
        return False
    except Exception as e:
        print(f"❌ 連接測試失敗: {e}")
        return False


def interactive_test(host, port):
    """互動式測試模式"""
    print("\n" + "="*60)
    print("互動式遊戲控制測試")
    print("="*60)
    print(f"目標: {host}:{port}")
    print("\n可用指令:")
    print("  1. lean <value>  - 設定傾斜值 (-1.0 到 1.0)")
    print("  2. jump         - 觸發跳躍")
    print("  3. reset        - 重置控制")
    print("  4. test         - 測試連接")
    print("  5. demo         - 執行示範序列")
    print("  6. quit         - 退出")
    print("="*60 + "\n")
    
    while True:
        try:
            cmd = input("> ").strip().lower()
            
            if cmd == "quit" or cmd == "q":
                break
            elif cmd == "test":
                test_connection(host, port)
            elif cmd.startswith("lean "):
                try:
                    value = float(cmd.split()[1])
                    value = max(-1.0, min(1.0, value))  # 限制範圍
                    if send_game_command(host, port, {"lean": value}):
                        print(f"✅ 已發送: lean = {value}")
                except (ValueError, IndexError):
                    print("❌ 無效的數值，請輸入 -1.0 到 1.0 之間的數字")
            elif cmd == "jump":
                if send_game_command(host, port, {"jump": True}):
                    print("✅ 已發送: jump = True")
                    time.sleep(0.1)
                    send_game_command(host, port, {"jump": False})
                    print("✅ 已發送: jump = False")
            elif cmd == "reset":
                if send_game_command(host, port, {"reset": True}):
                    print("✅ 已發送: reset = True")
            elif cmd == "demo":
                print("執行示範序列...")
                if not test_connection(host, port):
                    continue
                
                # 示範序列：左右傾斜 + 跳躍
                print("  → 向左傾斜...")
                send_game_command(host, port, {"lean": -0.5})
                time.sleep(1)
                
                print("  → 向右傾斜...")
                send_game_command(host, port, {"lean": 0.5})
                time.sleep(1)
                
                print("  → 跳躍...")
                send_game_command(host, port, {"jump": True})
                time.sleep(0.1)
                send_game_command(host, port, {"jump": False})
                time.sleep(0.5)
                
                print("  → 重置...")
                send_game_command(host, port, {"reset": True})
                print("✅ 示範序列完成")
            else:
                print("❌ 未知指令，請輸入 'help' 查看可用指令")
                
        except KeyboardInterrupt:
            print("\n\n收到中斷信號，退出...")
            break
        except EOFError:
            break


def main():
    parser = argparse.ArgumentParser(description='測試遊戲 socket 連接和控制')
    parser.add_argument('--host', type=str, default='127.0.0.1',
                        help='遊戲 socket 主機地址（預設: 127.0.0.1）')
    parser.add_argument('--port', type=int, default=4789,
                        help='遊戲 socket 端口（預設: 4789）')
    parser.add_argument('--test-only', action='store_true',
                        help='僅測試連接，不進入互動模式')
    args = parser.parse_args()
    
    if args.test_only:
        test_connection(args.host, args.port)
    else:
        interactive_test(args.host, args.port)


if __name__ == "__main__":
    main()


