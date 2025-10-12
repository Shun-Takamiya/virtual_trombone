# send_sensor_data.py

import socket
import time
import math

# 送信先のIPアドレスとポート番号
HOST = '127.0.0.1'  # ローカルホスト（自分自身）
PORT = 50007       # 任意のポート番号

# ソケットを作成
# socket.AF_INET: IPv4を使用
# socket.SOCK_DGRAM: UDP通信を使用
client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

print("UDP Sender started")

# データを送信し続ける
try:
    count = 0
    while True:
        # --- ここでセンサーデータを取得・処理する ---
        # 今回はサンプルとして，オブジェクトが円を描くような座標を生成
        x = math.sin(count * 0.1) * 5.0
        y = math.cos(count * 0.1) * 5.0
        z = 0.0
        
        # Unityで扱いやすいように，カンマ区切りの文字列にする
        message = f"{x},{y},{z}"
        
        # データをエンコードして送信
        client.sendto(message.encode('utf-8'), (HOST, PORT))
        
        print(f"Sent: {message}")
        
        count += 1
        time.sleep(0.02) # 50Hz (秒間50回) で送信

except KeyboardInterrupt:
    print("Sender stopped.")
finally:
    client.close()