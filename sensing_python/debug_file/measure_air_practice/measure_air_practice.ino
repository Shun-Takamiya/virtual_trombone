#include <Wire.h> // I2C通信のための標準ライブラリ
#include <SparkFun_FS3000_Arduino_Library.h> // FS3000センサ用のライブラリ

// センサ用のオブジェクトを作成
FS3000 fs3000;

// Arduino UnoのI2Cピンは A4 (SDA) と A5 (SCL) に固定されています。

void setup() {
  // シリアル通信の開始 (結果はシリアルモニターに表示します)
  Serial.begin(9600);
  while (!Serial); // シリアルポートが開くまで待機 (Leonardoなど一部のボードで必要)

  Serial.println("FS3000 Air Velocity Sensor with Arduino Uno");

  // I2C通信の開始 (Unoではピン指定は不要です)
  Wire.begin();
  Wire.setClock(400000); // I2Cの速度を400kHzに設定 (オプション)

  // センサの接続を確認
  if (fs3000.begin() == false) {
    Serial.println("Sensor not detected. Please check wiring and I2C address.");
    while(1); // 応答がない場合はここで停止
  }
  
  Serial.println("Sensor connected successfully.");
  Serial.println("------------------------------------");
}

void loop() {
  // センサからデータを読み取る
  // readMetersPerSecond()はFS3000-1005用です
  float metersPerSecond = fs3000.readMetersPerSecond();
  
  // 単位を変換
  float kmPerHour = metersPerSecond * 3.6;
  uint16_t rawData = fs3000.readRaw();

  // --- シリアルモニターに値を表示 ---
  Serial.print("Raw: ");
  Serial.print(rawData);
  
  Serial.print("\t"); // タブで区切って見やすくする
  
  Serial.print("Wind Speed: ");
  Serial.print(metersPerSecond, 2); // 小数点以下2桁で表示
  Serial.print(" m/s");
  
  Serial.print("\t"); // タブで区切る
  
  Serial.print(kmPerHour, 2); // 小数点以下2桁で表示
  Serial.println(" km/h");
  
  // 1秒待機
  delay(1000);
}