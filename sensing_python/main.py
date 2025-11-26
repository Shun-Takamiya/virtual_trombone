# -*- coding: utf-8 -*-
# 最終版: 論文の数式に基づき、基本値と調整値でtotal_valueを計算するプログラム
# ★修正版: 音の切り替え時のノイズ除去(フェード) + 現在の音名表示機能 + 距離センサの平滑化(移動平均)

# ### ライブラリのインポート ###
import cv2          # カメラ映像の処理、ウィンドウ表示用 (OpenCV)
import mediapipe as mp  # 顔ランドマーク検出用 (Google MediaPipe)
import numpy as np  # 数値計算、配列操作用
import serial       # Arduinoとのシリアル通信用
import threading    # Arduino受信処理をバックグラウンドで動かすため
import time         # 処理の待機用
import socket       # UnityへUDP通信を行うため
import pygame       # 音声再生用
import os           # ファイルパスの操作用
from collections import deque # ★追加: データの履歴を保持するためのキュー

# --- 1. 各種設定 ---

# ### Arduino & 連携設定 
# Arduinoが接続されているシリアルポート名を指定します。
# Macの例: '/dev/tty.usbmodemXXXX'
# Windowsの例: 'COM3'
# ※ご自身の環境に合わせて変更してください
SERIAL_PORT = '/dev/tty.usbmodem1301'
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

# ### 音声再生に関する設定 ###
# 再生する音源ファイルリスト
SOUND_FILES = [
    '1_ra#.wav', '3_2_do.wav', '5_2_re.wav', '7_2_mi.wav',
    '8_2_fa.wav', '10_2_so.wav', '12_ra.wav'
]

# ★追加: 画面に表示する音名のリスト (SOUND_FILESに対応)
NOTE_NAMES = [
    "Si (Ti)",  # 0: 1_ra#.wav (Low Si/La#)
    "Do",       # 1: 3_2_do.wav
    "Re",       # 2: 5_2_re.wav
    "Mi",       # 3: 7_2_mi.wav
    "Fa",       # 4: 8_2_fa.wav
    "So",       # 5: 10_2_so.wav
    "La"        # 6: 12_ra.wav
]

# 音量の調整 (0.0から1.0の範囲)
SOUND_VOLUME = 1.0

# ★追加設定: フェード処理の時間（ミリ秒）
# これにより音の切り替えが滑らかになり、プチプチ音を防ぎます
FADE_OUT_MS = 400  # 前の音を消す際にかける時間 (長くすることでより滑らかに)
FADE_IN_MS = 350    # 次の音を鳴らす際にかける時間

# ★追加設定: センサー値の平滑化（移動平均）設定
# 距離センサーの値が急に変わるのを防ぐため、直近の数回分の平均値を使います。
# 値を大きくすると滑らかになりますが、反応が少し遅れます（遅延）。
DISTANCE_SMA_WINDOW = 0  # 5〜10くらいがおすすめ (0にすると平滑化なし)

# ### 変更: 論文の数式に基づくパラメータ設定 ###
# 式(2)の係数kに相当。口の面積と風量が「調整値」に与える影響を決めます。
K_A = 0.05  # 口の面積A(t)に対する係数
K_AIRFLOW = 1.0  # 風量に対する係数

# 式(3)の係数δFに相当。「調整値」が最終的なtotal_valueに与える影響の度合いを決めます。
ADJUSTMENT_INFLUENCE = 1.0

# 全体の値をスケールアップするための乗数
TOTAL_VALUE_MULTIPLIER = 8.0

# 距離センサーの有効範囲設定 (cm)
DISTANCE_MIN = 5.0
DISTANCE_MAX = 47.0

# 元の値の最大値（この値が700にマッピングされる）
# 注意: 上記の係数を変更した場合、total_valueの最大値も変わるため、この値の調整が必要になることがあります。
ORIGINAL_TOTAL_VALUE_MAX = 300.0
# スケーリング後の表示・送信される値の最大値
SCALED_TOTAL_VALUE_MAX = 750.0

# この値以下では音を鳴らしません (無音)。
SOUND_THRESHOLD = 10
SAMPLE_RATE = 44100
BITSIZE = -16
CHANNELS = 2
BUFFER = 2048

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
TEXT_COLOR_DISTANCE = (255, 255, 0)
TEXT_COLOR_SUM = (0, 165, 255)
TEXT_COLOR_NOTE = (0, 0, 255) # ★追加: 音名表示用の色（赤）

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


def draw_visualizations(image, inner_coords, outer_coords, features, distance, airflow, total_value, note_name):
    """PCのプレビューウィンドウに検出結果や数値を描画する関数"""
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

    # ★追加: 現在の音名を描画（音が鳴っている時だけ表示）
    if note_name:
        cv2.putText(image, f"NOTE: {note_name}", (30, 220), TEXT_FONT, 1.5, TEXT_COLOR_NOTE, 3)
    else:
        cv2.putText(image, "MUTE", (30, 220), TEXT_FONT, 1.0, (100, 100, 100), 2)


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

    # ★追加: 平滑化用のバッファ（キュー）
    # maxlenを指定することで、新しいデータが入ると古いデータが自動的に消えるようになります
    distance_history = deque(maxlen=DISTANCE_SMA_WINDOW)

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
                    raw_distance = float(parts[0]) # 生の距離データ
                    airflow_val = float(parts[1])
                    
                    # ★追加: 移動平均フィルタの適用
                    # 履歴に追加
                    distance_history.append(raw_distance)
                    
                    # 現在のバッファ内の平均を計算
                    # 最初の数回でバッファが埋まるまでは、あるだけのデータの平均を使います
                    if len(distance_history) > 0:
                        smoothed_distance = sum(distance_history) / len(distance_history)
                    else:
                        smoothed_distance = raw_distance

                    # 共有データを安全に更新します（ロックを使用）。
                    with lock:
                        shared_data['distance'] = smoothed_distance # 平滑化済みの値を入れる
                        shared_data['airflow'] = airflow_val

            except (ValueError, UnicodeDecodeError):
                # データが数値に変換できない、または文字コードがおかしい場合は無視して次のループに進みます。
                pass
    
    # ループが終了したら、シリアルポートを閉じます。
    ser.close()
    print("Arduino connection closed.")


def main():
    """メインの処理を実行する関数"""
    
    # ### Pygameミキサーの初期化と音源ファイルの読み込み ###
    pygame.mixer.init(frequency=SAMPLE_RATE, size=BITSIZE, channels=CHANNELS, buffer=BUFFER)
    
    # ★追加: チャンネル数を増やす
    # フェードアウト(前の音)とフェードイン(次の音)が重なる瞬間に、
    # チャンネルが足りなくなって音が消えるのを防ぐため、十分な数を確保します。
    pygame.mixer.set_num_channels(32)
    
    sound_folder = 'trombone_sounds'
    sounds = []
    # ファイルを一つずつ読み込んでsoundsリストに追加
    for file_name in SOUND_FILES:
        path = os.path.join(sound_folder, file_name)
        if os.path.exists(path):
            sound = pygame.mixer.Sound(path)
            sound.set_volume(SOUND_VOLUME)
            sounds.append(sound)
        else:
            print(f"エラー: 音源ファイルが見つかりません: {path}")
            print("プログラムを終了します。soundsフォルダとWAVファイルを確認してください。")
            return

    current_note_index = -1
    
    # --- Arduino受信スレッドを開始 ---
    arduino_thread = threading.Thread(target=arduino_handler, args=(shared_data, data_lock))
    arduino_thread.daemon = True
    arduino_thread.start()
    
    # --- カメラとMediaPipeの準備 ---
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
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
                
                # ### 変更: 論文の数式に基づいてtotal_valueを計算 ###

                # 1. 式(1)に基づき、距離から「基本の値」を計算
                # 変更: ユーザー要望により、反転せず距離センサーの値をそのまま起点として使用
                base_value = 0.0
                if current_distance < DISTANCE_MIN:
                    # 近すぎる場合は最小値(DISTANCE_MIN)に固定
                    base_value = DISTANCE_MIN
                elif current_distance > DISTANCE_MAX:
                    # 遠すぎる場合は最大値(DISTANCE_MAX)に固定
                    base_value = DISTANCE_MAX
                else:
                    # 有効範囲内では、現在の距離をそのまま使用
                    base_value = current_distance
                
                # 2. 式(2)に基づき、口と息から「調整値」を計算
                adjustment_value = (features['A(t)'] * K_A) + (current_airflow * K_AIRFLOW)

                
                
                # 3. 式(3)に基づき、「基本の値」と「調整値」を統合して最終的なtotal_valueを計算
                total_value = base_value* TOTAL_VALUE_MULTIPLIER  + (adjustment_value * ADJUSTMENT_INFLUENCE)
                # print(f"total_value: {total_value:.2f}, Distance_Base: {base_value:.2f}, Mouth_Adjust: {features['A(t)'] * K_A:.2f}")

                # total_valueを0-700の範囲にスケーリング
                scaled_total_value = (total_value / ORIGINAL_TOTAL_VALUE_MAX) * SCALED_TOTAL_VALUE_MAX
                scaled_total_value = max(0, min(scaled_total_value, SCALED_TOTAL_VALUE_MAX))
                
                # total_valueに応じて音を鳴らす
                # total_valueに応じて音を鳴らす（変更：ポジション(base_value)によるif分岐）
                note_index_to_play = -1

                # まず、息を吹き込んでいるか（音が鳴る条件を満たしているか）をチェック
                if current_airflow > 0.3 and total_value > SOUND_THRESHOLD:
                    # 次に、スライドの位置(base_value)によって音階を決定
                    # ★修正: base_value (距離) が小さいほど、手前なので高い音になるようにロジックを反転
                    # base_value = 距離そのもの (5cm 〜 47cm)
                    
                    if base_value < 12.0:       # 最も手前 (近い)
                        note_index_to_play = 6  # 最高音: ラ ('12_ra.wav')
                    elif base_value < 19.0:
                        note_index_to_play = 5  # ソ ('10_so.wav')
                    elif base_value < 26.0:
                        note_index_to_play = 4  # ファ ('8_fa.wav')
                    elif base_value < 33.0:
                        note_index_to_play = 3  # ミ ('7_mi.wav')
                    elif base_value < 40.0:
                        note_index_to_play = 2  # レ ('5_re.wav')
                    elif base_value < 47.0:
                        note_index_to_play = 1  # ド ('3_do.wav')
                    else:                       # 最も奥 (遠い)
                        note_index_to_play = 0  # 最低音: シ ('2_si.wav')

                # ★追加: 現在の音名を取得
                current_note_name = ""
                if note_index_to_play != -1:
                    current_note_name = NOTE_NAMES[note_index_to_play]

                # PCのプレビューウィンドウに結果を描画します。
                # ★修正: current_note_name を渡すように変更
                draw_visualizations(image, inner_coords, outer_coords, features, current_distance, current_airflow, scaled_total_value, current_note_name)

                # 鳴らすべき音が現在の音と違う場合のみ、音を切り替える
                if note_index_to_play != current_note_index:
                    
                    # ★変更点: クロスフェード処理
                    # pygame.mixer.stop() を使うと音が急に切れてノイズになるため、
                    # 前の音をフェードアウト、次の音をフェードインさせます。

                    # 1. 前の音があれば、フェードアウトさせる（急停止しない）
                    if current_note_index != -1:
                        sounds[current_note_index].fadeout(FADE_OUT_MS)
                    
                    # 2. 次の音があれば、フェードインさせて鳴らす
                    if note_index_to_play != -1:
                        # play(fade_ms=...) で滑らかに音を立ち上げる
                        # loops=-1 でループ再生
                        sounds[note_index_to_play].play(loops=-1, fade_ms=FADE_IN_MS)
                    
                    current_note_index = note_index_to_play
                
                # Unityへデータを送信
                try:
                    # Unityには加工前の生の距離(current_distance)を送ることで、Unity側での柔軟な利用も可能にします
                    message = f"{features['A(t)']},{current_distance},{current_airflow},{scaled_total_value}"
                    udp_client.sendto(message.encode('utf-8'), (UDP_HOST, UDP_PORT))
                except Exception as e:
                    print(f"UDP送信エラー: {e}")
        
        # 顔が検出されなかった場合
        else:
            # もし音が鳴っている状態なら停止する
            if current_note_index != -1:
                # ★変更点: ここもフェードアウトで停止
                sounds[current_note_index].fadeout(FADE_OUT_MS)
                current_note_index = -1

        # PC上でプレビューウィンドウを表示します。
        cv2.imshow('Sensor Fusion Monitor', image)
        
        # 'q'キーが押されたらループ抜けます。
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
    pygame.mixer.quit() # Pygameを終了


# このスクリプトが直接実行された場合に`main`関数を呼び出します。
if __name__ == '__main__':
    main()