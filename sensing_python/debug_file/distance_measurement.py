import serial
import time

# --- 設定 ---
# Arduinoが接続されているシリアルポート名

SERIAL_PORT = '/dev/tty.usbmodem1301'  # ★★★自分の環境に合わせて変更★★★

# Arduinoの`Serial.begin()`で設定したボーレート
BAUD_RATE = 9600

# --- ここからプログラム ---
try:
    # シリアルポートへの接続
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2) # 接続が安定するまで少し待つ
    print(f"{SERIAL_PORT}に接続しました。")

    # 何か固定の値 (例: 1.5)
    some_value = 1.5

    while True:
        # Arduinoから1行分のデータを受信する
        # `readline()`は末尾に改行コード(\n)を含むバイト列を返す
        line = ser.readline()

        # データが空でなければ処理を実行
        if line:
            try:
                # バイト列を文字列にデコードし、前後の空白や改行を削除
                distance_str = line.decode('utf-8').strip()

                # 文字列を浮動小数点数（float）に変換
                distance_cm = float(distance_str)

                # --- ここで距離の値を係数として計算に利用 ---
                # 例: ある値に、取得した距離(cm)を係数として掛け合わせる
                calculation_result = some_value * distance_cm

                # 結果を出力
                print(f"受信した距離: {distance_cm:.2f} cm | 計算結果: {calculation_result:.2f}")

            except ValueError:
                # 数値に変換できなかった場合はエラーメッセージを表示
                # (プログラム起動直後など、Arduinoから不完全なデータが送られてくることがあるため)
                print(f"エラー: 数値に変換できませんでした -> {line}")
            except Exception as e:
                print(f"予期せぬエラーが発生しました: {e}")

except serial.SerialException as e:
    print(f"エラー: シリアルポート {SERIAL_PORT} が見つからないか、開けません。")
    print("ArduinoがPCに接続されているか、ポート名が正しいか確認してください。")
except KeyboardInterrupt:
    print("プログラムを終了します。")
finally:
    if 'ser' in locals() and ser.is_open:
        ser.close()
        print("シリアルポートを閉じました。")