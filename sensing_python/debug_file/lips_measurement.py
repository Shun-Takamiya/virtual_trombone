# -*- coding: utf-8 -*-

# 唇の形状をリアルタイムで検出し、各種特徴量を計算・可視化するプログラム

# このプログラムは，PCのカメラから映像を取得し，GoogleのMediaPipeライブラリを使用して顔のランドマークを検出．
# 特に唇の形状に注目し，以下の特徴量をリアルタイムで計算して画面に表示．

# 1.  A(t): 開口部の面積
# 2.  C(t): 唇全体の面積（接触面積の代用）
# 3.  R(t): 開口部の縦横比


import cv2
import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh
import numpy as np
import math

# --- 設定値 ---
# このセクションの数値を変更することで，表示スタイルを簡単に調整が可能．

# 表示スタイル設定
# 内唇の輪郭線の設定
INNER_LIP_LINE_THICKNESS = 4                # 太さ
INNER_LIP_LINE_COLOR = (0, 255, 0)          # 色 (B, G, R) - 緑

# 外唇の輪郭線の設定
OUTER_LIP_LINE_THICKNESS = 2                # 太さ
OUTER_LIP_LINE_COLOR = (255, 0, 0)          # 色 (B, G, R) - 青

# 内唇のランドマーク（特徴点）の設定
INNER_LIP_POINT_RADIUS = 3                  # 半径
INNER_LIP_POINT_COLOR = (0, 0, 255)         # 色 (B, G, R) - 赤

# 外唇のランドマーク（特徴点）の設定
OUTER_LIP_POINT_RADIUS = 3                  # 半径
OUTER_LIP_POINT_COLOR = (0, 255, 255)       # 色 (B, G, R) - 黄色

# ■ テキスト表示設定
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX      # フォント
TEXT_SCALE = 0.8                            # 文字サイズ
TEXT_THICKNESS = 2                          # 文字の太さ
TEXT_COLOR_A_T = (0, 255, 0)                # A(t) の文字色
TEXT_COLOR_R_T = (0, 255, 255)              # R(t) の文字色
TEXT_COLOR_C_T = (255, 0, 0)                # C(t) の文字色

# --- グローバル定数 ---
# MediaPipeが定義するランドマークのインデックス番号
# 参考: https://github.com/tensorflow/tfjs-models/blob/master/face-landmarks-detection/src/mediapipe-facemesh/keypoints.ts
LIP_INNER_INDICES = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
LIP_OUTER_INDICES = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]


def get_lip_landmarks(face_landmarks, image_shape):

    # 顔のランドマーク情報から、唇の座標リストを取得する関数

    # Args:
        # face_landmarks: MediaPipeが検出した顔のランドマーク
        # image_shape (tuple): 画像の形状 (高さ, 幅, チャンネル数)

    # Returns:
        # tuple: 内唇の座標リスト, 外唇の座標リスト
    
    h, w, _ = image_shape
    inner_lip_coords = np.array([(face_landmarks.landmark[i].x * w, face_landmarks.landmark[i].y * h) for i in LIP_INNER_INDICES], dtype=np.int32)
    outer_lip_coords = np.array([(face_landmarks.landmark[i].x * w, face_landmarks.landmark[i].y * h) for i in LIP_OUTER_INDICES], dtype=np.int32)
    return inner_lip_coords, outer_lip_coords

def calculate_features(inner_lip_coords, outer_lip_coords):
    
    # 唇の座標リストから、各種特徴量を計算する関数

    # Args:
        # inner_lip_coords (np.array): 内唇の座標リスト
        # outer_lip_coords (np.array): 外唇の座標リスト

    # Returns:
        # dict: 計算された特徴量を含む辞書
    
    # 多角形の面積を計算する内部関数 (シューレース公式)
    def polygon_area(coords):
        if coords.shape[0] < 3: return 0
        x, y = coords[:, 0], coords[:, 1]
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    # 1. 開口部面積 A(t)
    opening_area = polygon_area(inner_lip_coords)

    # 2. 接触面積 C(t) (外唇の面積で代用)
    contact_area = polygon_area(outer_lip_coords)

    # 3. 縦横比 R(t)
    lip_ys, lip_xs = inner_lip_coords[:, 1], inner_lip_coords[:, 0]
    lip_height = lip_ys.max() - lip_ys.min()
    lip_width = lip_xs.max() - lip_xs.min()
    aspect_ratio = lip_height / lip_width if lip_width > 0 else 0

    return {
        "A(t)": opening_area,
        "C(t)": contact_area,
        "R(t)": aspect_ratio
    }

def draw_visualizations(image, inner_coords, outer_coords, features):
    
    # 画像上に検出結果と計算された特徴量を書き込む関数

    # Args:
        # image: 描画対象の画像
        # inner_coords (np.array): 内唇の座標リスト
        # outer_coords (np.array): 外唇の座標リスト
        # features (dict): 計算された特徴量
    
    # 輪郭線の描画
    cv2.polylines(image, [inner_coords], isClosed=True, color=INNER_LIP_LINE_COLOR, thickness=INNER_LIP_LINE_THICKNESS)
    cv2.polylines(image, [outer_coords], isClosed=True, color=OUTER_LIP_LINE_COLOR, thickness=OUTER_LIP_LINE_THICKNESS)
    
    # ランドマーク点の描画
    for point in inner_coords: cv2.circle(image, tuple(point), radius=INNER_LIP_POINT_RADIUS, color=INNER_LIP_POINT_COLOR, thickness=-1)
    for point in outer_coords: cv2.circle(image, tuple(point), radius=OUTER_LIP_POINT_RADIUS, color=OUTER_LIP_POINT_COLOR, thickness=-1)

    # 計算結果のテキスト描画
    cv2.putText(image, f"Opening Area A(t): {features['A(t)']:.2f}", (30, 50), TEXT_FONT, TEXT_SCALE, TEXT_COLOR_A_T, TEXT_THICKNESS)
    cv2.putText(image, f"Aspect Ratio R(t): {features['R(t)']:.2f}", (30, 80), TEXT_FONT, TEXT_SCALE, TEXT_COLOR_R_T, TEXT_THICKNESS)
    cv2.putText(image, f"Contact Area C(t): {features['C(t)']:.2f}", (30, 110), TEXT_FONT, TEXT_SCALE, TEXT_COLOR_C_T, TEXT_THICKNESS)


def main():
    # --- グローバル定数 ---
    # MediaPipe FaceMeshモデルの初期化
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

    # カメラを起動
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("エラー: カメラを起動できませんでした．")
        return

    # メインループ
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("フレームが取得できませんでした．")
            continue

        # 表示のために画像を左右反転
        image = cv2.flip(image, 1)

        # MediaPipeで処理するために色空間を変換
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)

        # 顔が検出された場合のみ描画処理を行う
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # 1. 唇の座標を取得
                inner_coords, outer_coords = get_lip_landmarks(face_landmarks, image.shape)
                
                # 2. 特徴量を計算
                features = calculate_features(inner_coords, outer_coords)
                
                # 3. 結果を可視化
                draw_visualizations(image, inner_coords, outer_coords, features)

        # 結果を表示
        cv2.imshow('Readable Lip Measurement', image)
        
        # 'q'キーが押されたらループを抜ける
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    # 後処理
    face_mesh.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()