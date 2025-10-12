# -*- coding: utf-8 -*-
# 2つのプログラムを統合し、Arduinoの距離と唇の開口度を合計して表示するプログラム

import cv2
import mediapipe as mp
import numpy as np
import serial
import threading
import time

# --- 1. Arduino & 連携設定 ---
# ★★★ 自分の環境に合わせてArduinoのシリアルポート名を設定 ★★★
SERIAL_PORT = '/dev/tty.usbmodem1301'  # Macの例
# SERIAL_PORT = 'COM3'                  # Windowsの例
BAUD_RATE = 9600

# 2つの処理（スレッド）間でデータを安全にやり取りするための設定
shared_data = {'distance': 0.0, 'airflow': 0.0, 'stop_thread': False}
data_lock = threading.Lock()

# --- 2. 唇検出の表示スタイル設定 ---
INNER_LIP_LINE_THICKNESS = 4
INNER_LIP_LINE_COLOR = (0, 255, 0)
OUTER_LIP_LINE_THICKNESS = 2
OUTER_LIP_LINE_COLOR = (255, 0, 0)
INNER_LIP_POINT_RADIUS = 3
INNER_LIP_POINT_COLOR = (0, 0, 255)
OUTER_LIP_POINT_RADIUS = 3
OUTER_LIP_POINT_COLOR = (0, 255, 255)

# --- 3. テキスト表示設定 ---
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 0.8
TEXT_THICKNESS = 2
TEXT_COLOR_A_T = (0, 255, 0)
TEXT_COLOR_R_T = (0, 255, 255)
TEXT_COLOR_C_T = (255, 0, 0)
TEXT_COLOR_DISTANCE = (255, 255, 0)      # 距離テキストの色 (シアン)
TEXT_COLOR_SUM = (0, 165, 255)          # 合計値テキストの色 (オレンジ)

# --- 4. MediaPipe 定数 ---
mp_face_mesh = mp.solutions.face_mesh
LIP_INNER_INDICES = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
LIP_OUTER_INDICES = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]


# --- 関数定義 (唇検出) ---
def get_lip_landmarks(face_landmarks, image_shape):
    h, w, _ = image_shape
    inner_lip_coords = np.array([(face_landmarks.landmark[i].x * w, face_landmarks.landmark[i].y * h) for i in LIP_INNER_INDICES], dtype=np.int32)
    outer_lip_coords = np.array([(face_landmarks.landmark[i].x * w, face_landmarks.landmark[i].y * h) for i in LIP_OUTER_INDICES], dtype=np.int32)
    return inner_lip_coords, outer_lip_coords

def calculate_features(inner_lip_coords, outer_lip_coords):
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
    cv2.polylines(image, [inner_coords], True, INNER_LIP_LINE_COLOR, INNER_LIP_LINE_THICKNESS)
    cv2.polylines(image, [outer_coords], True, OUTER_LIP_LINE_COLOR, OUTER_LIP_LINE_THICKNESS)
    for point in inner_coords: cv2.circle(image, tuple(point), INNER_LIP_POINT_RADIUS, INNER_LIP_POINT_COLOR, -1)
    for point in outer_coords: cv2.circle(image, tuple(point), OUTER_LIP_POINT_RADIUS, OUTER_LIP_POINT_COLOR, -1)
    
    # 既存のテキスト表示
    cv2.putText(image, f"Opening Area A(t): {features['A(t)']:.2f}", (30, 50), TEXT_FONT, TEXT_SCALE, TEXT_COLOR_A_T, TEXT_THICKNESS)
    cv2.putText(image, f"Aspect Ratio R(t): {features['R(t)']:.2f}", (30, 80), TEXT_FONT, TEXT_SCALE, TEXT_COLOR_R_T, TEXT_THICKNESS)
    cv2.putText(image, f"Contact Area C(t): {features['C(t)']:.2f}", (30, 110), TEXT_FONT, TEXT_SCALE, TEXT_COLOR_C_T, TEXT_THICKNESS)

    # ★★ 追加: 距離と合計値を表示 ★★
    cv2.putText(image, f"Distance (cm): {distance:.2f}", (30, 150), TEXT_FONT, TEXT_SCALE, TEXT_COLOR_DISTANCE, TEXT_THICKNESS)
    cv2.putText(image, f"Airflow (N): {airflow:.2f}", (30, 190), TEXT_FONT, TEXT_SCALE, TEXT_COLOR_DISTANCE, TEXT_THICKNESS)
    cv2.putText(image, f"SUM (Area + Dist): {total_value:.2f}", (30, 230), TEXT_FONT, TEXT_SCALE, TEXT_COLOR_SUM, TEXT_THICKNESS)


# --- Arduino通信のための関数 (これがスレッドで動く) ---
def arduino_handler(shared_data, lock):
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2) # 接続安定待機
        print(f"Arduino on {SERIAL_PORT} connected.")
    except serial.SerialException as e:
        print(f"Error connecting to Arduino: {e}")
        return

    while not shared_data['stop_thread']:
        if ser.in_waiting > 0:
            try:
                line = ser.readline().decode('utf-8').strip()
                # 受信した文字列をカンマで分割
                parts = line.split(',')

                # 分割した結果が2つの要素であるか確認
                if len(parts) == 2:
                    distance_val = float(parts[0])
                    airflow_val = float(parts[1])
                    
                    # 共有データを安全に更新
                    with lock:
                        shared_data['distance'] = distance_val
                        shared_data['airflow'] = airflow_val
            except (ValueError, UnicodeDecodeError):
                # 不正なデータを無視
                pass
    
    ser.close()
    print("Arduino connection closed.")


def main():
    # --- Arduinoスレッドを開始 ---
    arduino_thread = threading.Thread(target=arduino_handler, args=(shared_data, data_lock))
    arduino_thread.daemon = True # メインプログラム終了時に自動で閉じる
    arduino_thread.start()
    
    # --- カメラとMediaPipeの準備 ---
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("エラー: カメラを起動できませんでした．")
        return

    # --- メインループ (カメラ処理) ---
    while cap.isOpened():
        success, image = cap.read()
        if not success: continue

        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)

        current_distance = 0
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                inner_coords, outer_coords = get_lip_landmarks(face_landmarks, image.shape)
                features = calculate_features(inner_coords, outer_coords)
                
                # Arduinoスレッドから最新の距離データを安全に取得
                with data_lock:
                    current_distance = shared_data['distance']
                    current_airflow = shared_data['airflow']
                
                # ★★ 合計値を計算 ★★
                # ここでは「開口部の面積 A(t)」と「距離」を合計しています
                total_value = features['A(t)'] + current_distance + current_airflow
                
                # 描画
                draw_visualizations(image, inner_coords, outer_coords, features, current_distance, current_airflow, total_value)

        cv2.imshow('Combined Sensor Measurement', image)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    # --- 後処理 ---
    print("Exiting program...")
    shared_data['stop_thread'] = True # スレッドに終了を通知
    arduino_thread.join() # スレッドが終了するのを待つ
    face_mesh.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()