# -*- coding: utf-8 -*-
# 最終版: 3つのセンサーデータを統合し、UnityへUDP送信するプログラム

# ### ライブラリのインポート ###
import cv2  # カメラ映像の処理、ウィンドウ表示用 (OpenCV)
import mediapipe as mp  # 顔ランドマーク検出用 (Google MediaPipe)
import numpy as np  # 数値計算、配列操作用
import serial  # Arduinoとのシリアル通信用
import threading  # Arduino受信処理をバックグラウンドで動かすため
import time  # 処理の待機用
import socket  # UnityへUDP通信を行うため

# --- 1. 各種設定 ---

# ### Arduino & 連携設定 ###
# Arduinoが接続されているシリアルポート名を指定します。
# Macの例: '/dev/tty.usbmodemXXXX'
# Windowsの例: 'COM3'
SERIAL_PORT = 'COM5'
BAUD_RATE = 9600  # Arduinoの `Serial.begin()` で設定した値と合わせます。

# ### UnityへのUDP送信設定 ###
UDP_HOST = "127.0.0.1"  # 自分自身（PC）のIPアドレス。Unityも同じPCで動かすのでこれでOK。
UDP_PORT = 50007       # Unity側と合わせるポート番号（部屋番号のようなもの）。
# UDP通信を行うためのソケットを作成します。
udp_client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# ### スレッド間でのデータ共有設定 ###
# Arduino受信スレッドとメインスレッドで安全にデータを共有するための仕組みです。
shared_data = {
    'distance': 0.0,      # 距離センサーの値を格納
    'airflow': 0.0,       # 風量センサーの値を格納
    'stop_thread': False  # スレッドを停止させるためのフラグ
}
data_lock = threading.Lock()  # 複数のスレッドが同時にデータにアクセスするのを防ぐ「鍵」

# --- 2. 表示スタイル & MediaPipe 定数 ---
# --- 唇検出の表示スタイル設定 ---
INNER_LIP_LINE_THICKNESS = 4
INNER_LIP_LINE_COLOR = (0, 255, 0)
OUTER_LIP_LINE_THICKNESS = 2
OUTER_LIP_LINE_COLOR = (255, 0, 0)
INNER_LIP_POINT_RADIUS = 3
INNER_LIP_POINT_COLOR = (0, 0, 255)
OUTER_LIP_POINT_RADIUS = 3
OUTER_LIP_POINT_COLOR = (0, 255, 255)

# MediaPipe 定数 ---
mp_face_mesh = mp.solutions.face_mesh
LIP_INNER_INDICES = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
LIP_OUTER_INDICES = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]

# --- テキスト表示設定 ---
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 0.8
TEXT_THICKNESS = 2
TEXT_COLOR_A_T = (0, 255, 0)
TEXT_COLOR_R_T = (0, 255, 255)
TEXT_COLOR_C_T = (255, 0, 0)
TEXT_COLOR_DISTANCE = (255, 255, 0)      # 距離テキストの色 (シアン)
TEXT_COLOR_SUM = (0, 165, 255)          # 合計値テキストの色 (オレンジ)


# --- 3. 関数定義 ---

def get_lip_landmarks(face_landmarks, image_shape):
    """MediaPipeの検出結果から唇の座標リストを取得する関数"""
    h, w, _ = image_shape
    inner_lip_coords = np.array([(face_landmarks.landmark[i].x * w, face_landmarks.landmark[i].y * h) for i in LIP_INNER_INDICES], dtype=np.int32)
    outer_lip_coords = np.array([(face_landmarks.landmark[i].x * w, face_landmarks.landmark[i].y * h) for i in LIP_OUTER_INDICES], dtype=np.int32)
    return inner_lip_coords, outer_lip_coords


def calculate_features(inner_lip_coords, outer_lip_coords):
    """唇の座標リストから面積や縦横比などの特徴量を計算する関数"""
    def polygon_area(coords):
        if coords.shape[0] < 3: return 0
        x, y = coords[:, 0], coords[:, 1]
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    opening_area = polygon_area(inner_lip_coords)
    contact_area = polygon_area(outer_lip_coords)
    lip_ys, lip_xs = inner_lip_coords[:, 1], inner_lip_coords[:, 0]
    lip_height = lip_ys.max() - lip_ys.min()
    lip_width = lip_xs.max() - lip_xs.min()
    aspect_ratio = lip_height / lip_width if lip_width > 0 else 0
    return {"A(t)": opening_area, "C(t)": contact_area, "R(t)": aspect_ratio}


def draw_visualizations(image, inner_coords, outer_coords, features, distance, airflow, total_value):
    """PCのプレビューウィンドウに検出結果や数値を描画する関数"""

    # Args:
        # image: 描画対象の画像
        # inner_coords (np.array): 内唇の座標リスト
        # outer_coords (np.array): 外唇の座標リスト
        # features (dict): 計算された特徴量
    
    # 輪郭線の描画
    cv2.polylines(image, [inner_coords], True, INNER_LIP_LINE_COLOR, INNER_LIP_LINE_THICKNESS)
    cv2.polylines(image, [outer_coords], True, OUTER_LIP_LINE_COLOR, OUTER_LIP_LINE_THICKNESS)
    
    # ランドマーク点の描画
    for point in inner_coords: cv2.circle(image, tuple(point), radius=INNER_LIP_POINT_RADIUS, color=INNER_LIP_POINT_COLOR, thickness=-1)
    for point in outer_coords: cv2.circle(image, tuple(point), radius=OUTER_LIP_POINT_RADIUS, color=OUTER_LIP_POINT_COLOR, thickness=-1)
    
    # 計算結果のテキスト描画
    cv2.putText(image, f"Opening Area A(t): {features['A(t)']:.2f}", (30, 50), TEXT_FONT, TEXT_SCALE, TEXT_COLOR_A_T, TEXT_THICKNESS)
    cv2.putText(image, f"Distance (cm): {distance:.2f}", (30, 90), TEXT_FONT, TEXT_SCALE, TEXT_COLOR_DISTANCE, TEXT_THICKNESS)
    cv2.putText(image, f"Air Flow: {airflow:.2f}", (30, 130), TEXT_FONT, TEXT_SCALE, (255, 0, 255), TEXT_THICKNESS)
    cv2.putText(image, f"TOTAL VALUE: {total_value:.2f}", (30, 170), TEXT_FONT, TEXT_SCALE, TEXT_COLOR_SUM, TEXT_THICKNESS)


def arduino_handler(shared_data, lock):
    """
    バックグラウンドでArduinoからのデータを受信し続ける専門の関数。
    この関数全体が独立したスレッドとして動作します。
    """
    try:
        # シリアルポートへの接続を試みます。
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)  # 接続が安定するまで少し待ちます。
        print(f"Arduino on {SERIAL_PORT} connected.")
    except serial.SerialException as e:
        # ポートが見つからない、または開けない場合のエラー処理。
        print(f"Error connecting to Arduino: {e}")
        return # 関数を終了させ、スレッドを停止します。

    # `stop_thread`フラグがFalseである限り、無限にループします。
    while not shared_data['stop_thread']:
        # シリアルポートにデータが到着しているか確認します。
        if ser.in_waiting > 0:
            try:
                # データを1行読み込み、UTF-8形式の文字列に変換し、前後の空白や改行を削除します。
                line = ser.readline().decode('utf-8').strip()
                
                # 受信した文字列をカンマで分割してリストにします (例: "15.7,32.5" -> ["15.7", "32.5"])
                parts = line.split(',')

                # 分割した結果がちょうど2つの要素であるかを確認します（データ破損対策）。
                if len(parts) == 2:
                    # それぞれの要素を浮動小数点数（float）に変換します。
                    distance_val = float(parts[0])
                    airflow_val = float(parts[1])
                    
                    # 共有データを安全に更新します（ロックを使用）。
                    with lock:
                        shared_data['distance'] = distance_val
                        shared_data['airflow'] = airflow_val
            except (ValueError, UnicodeDecodeError):
                # データが数値に変換できない、または文字コードがおかしい場合は無視して次のループに進みます。
                pass
    
    # ループが終了したら、シリアルポートを閉じます。
    ser.close()
    print("Arduino connection closed.")


def main():
    """メインの処理を実行する関数"""
    
    # --- Arduino受信スレッドを開始 ---
    # `arduino_handler`関数をバックグラウンドで実行するためのスレッドを作成します。
    arduino_thread = threading.Thread(target=arduino_handler, args=(shared_data, data_lock))
    arduino_thread.daemon = True  # メインプログラムが終了したら、このスレッドも強制的に終了させます。
    arduino_thread.start() # スレッドを開始します。
    
    # --- カメラとMediaPipeの準備 ---
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    # PCに接続されたデフォルトのカメラ（通常は0番）を起動します。
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("エラー: カメラを起動できませんでした。")
        return

    # --- メインループ (カメラ処理 & データ送信) ---
    while cap.isOpened():
        # カメラから1フレーム分の映像を取得します。
        success, image = cap.read()
        if not success: continue

        # 映像を左右反転させて、鏡のように見せます。
        image = cv2.flip(image, 1)
        # OpenCVはBGR色空間ですが、MediaPipeはRGB色空間で処理するため、変換します。
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # MediaPipeで顔ランドマークを検出します。
        results = face_mesh.process(image_rgb)

        # 顔が検出された場合のみ、以下の処理を行います。
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # 唇の座標を取得し、特徴量を計算します。
                inner_coords, outer_coords = get_lip_landmarks(face_landmarks, image.shape)
                features = calculate_features(inner_coords, outer_coords)
                
                # Arduinoスレッドから最新のセンサーデータを安全に取得します。
                with data_lock:
                    current_distance = shared_data['distance']
                    current_airflow = shared_data['airflow']
                
                # 3つの主要なデータを合計します。
                total_value = features['A(t)'] + current_distance + current_airflow
                
                # PCのプレビューウィンドウに結果を描画します。
                draw_visualizations(image, inner_coords, outer_coords, features, current_distance, current_airflow, total_value)

                # ### UnityへUDP送信 ###
                try:
                    # 送信するデータをカンマ区切りの一つの文字列にまとめます。
                    # フォーマット: 開口面積,距離,風量,合計値
                    message = f"{features['A(t)']},{current_distance},{current_airflow},{total_value}"
                    # 文字列をバイト列に変換し、指定したホストとポートに送信します。
                    udp_client.sendto(message.encode('utf-8'), (UDP_HOST, UDP_PORT))
                except Exception as e:
                    print(f"UDP送信エラー: {e}")

        # PC上でプレビューウィンドウを表示します。
        cv2.imshow('Sensor Fusion Monitor', image)
        
        # 'q'キーが押されたらループを抜けます。
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    # --- 後処理 ---
    print("プログラムを終了します...")
    # Arduino受信スレッドに終了を通知します。
    shared_data['stop_thread'] = True
    # スレッドが完全に終了するのを待ちます。
    arduino_thread.join()
    # 使用したリソースを解放します。
    face_mesh.close()
    cap.release()
    cv2.destroyAllWindows()


# このスクリプトが直接実行された場合に`main`関数を呼び出します。
if __name__ == '__main__':
    main()