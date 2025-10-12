// ２つのセンサーライブラリをインポートします
#include <Wire.h> // I2C通信用 (FS3000)
#include <SparkFun_FS3000_Arduino_Library.h> // FS3000センサー用

// --- 距離センサー(HC-SR04)のピン設定 ---
const int echoPin = 2; // Echo Pin
const int trigPin = 3; // Trigger Pin

// Arduino UnoのI2Cピンは A4 (SDA) と A5 (SCL) に固定されています。

// --- 風量センサー(FS3000)のオブジェクトを作成 ---
FS3000 fs3000;

void setup() {
  // シリアル通信を開始 (Python側とボーレートを合わせます)
  Serial.begin(9600);

  // --- 距離センサーのピンモード設定 ---
  pinMode(echoPin, INPUT);
  pinMode(trigPin, OUTPUT);

  // --- 風量センサーの初期化 ---
  Wire.begin(); // I2C通信を開始
  if (fs3000.begin() == false) {
    // センサーが見つからない場合はエラーメッセージを出し、停止します
    Serial.println("FS3000 sensor not detected. Check wiring.");
    while (1); // ここで処理を停止
  }
}

void loop() {
  // --- 1. 距離センサーの値を測定 (単位: cm) ---
  // トリガーピンを操作して超音波を発信
  digitalWrite(trigPin, LOW); 
  delayMicroseconds(2); 
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);
  
  // エコーピンで反射波の時間を測定
  double duration = pulseIn(echoPin, HIGH, 25000); // タイムアウトを25msに設定
  
  double distance_cm;
  if (duration > 0) {
    // 時間を距離(cm)に変換 (音速 約343m/s で計算)
    distance_cm = (duration / 2.0) * 0.0343;
  } else {
    distance_cm = 0; // 測定失敗時 (タイムアウトなど) は0を返す
  }

  // --- 2. 風量センサーの値を測定 (単位: m/s) ---
  float airFlow_mps = fs3000.readMetersPerSecond();

  // --- 3. データをカンマ区切りでPythonへ送信 ---
  Serial.print(distance_cm);   // 1つ目のデータ: 距離 (cm)
  Serial.print(",");           // 区切り文字のカンマ
  Serial.println(airFlow_mps); // 2つ目のデータ: 風速 (m/s)

  // 0.1秒待機
  delay(100);
}