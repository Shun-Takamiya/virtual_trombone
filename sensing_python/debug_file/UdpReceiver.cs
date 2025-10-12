// ValueController.cs

using UnityEngine;
using System;
using System.Text;
using System.Net;
using System.Net.Sockets;
using System.Threading;

public class ValueController : MonoBehaviour
{
    // --- Public Variables (Inspectorで調整可能) ---
    [Header("ネットワーク設定")]
    [Tooltip("Python側と合わせるポート番号")]
    public int port = 50007;

    [Header("オブジェクトの動きに関する設定")]
    [Tooltip("オブジェクトが移動する最大の距離")]
    public float moveDistance = 10.0f;
    [Tooltip("受信する合計値の最大値（この値で正規化される）")]
    public float valueMax = 1500.0f;
    [Tooltip("動きを滑らかにするための係数（大きいほど速く追従）")]
    public float smoothSpeed = 5.0f;

    // --- Private Variables ---
    private UdpClient client; // UDP通信を行うためのクライアント
    private Thread receiveThread; // 受信処理をバックグラウンドで行うためのスレッド
    private Vector3 initialPosition; // オブジェクトの初期位置を保存
    private Vector3 targetPosition; // 計算された目標位置を保存
    
    // --- 他のスクリプトから参照するための公開プロパティ ---
    // 最新の受信データを格納します。
    public float LipArea { get; private set; }
    public float Distance { get; private set; }
    public float Airflow { get; private set; }
    public float TotalValue { get; private set; }

    // --- Unity Event Functions ---

    // ゲーム開始時に一度だけ呼ばれる関数
    void Start()
    {
        // このスクリプトがアタッチされているオブジェクトの初期位置を保存
        initialPosition = transform.position;
        targetPosition = initialPosition;

        // UDPクライアントを初期化し、別スレッドで受信を開始します。
        // これをしないと、受信待機でUnityのメイン処理（描画など）が止まってしまいます。
        client = new UdpClient(port);
        receiveThread = new Thread(new ThreadStart(ReceiveData));
        receiveThread.IsBackground = true; // アプリ終了時にスレッドも終了させる
        receiveThread.Start();
        Debug.Log("UDP受信を開始しました Port: " + port);
    }

    // 毎フレーム呼ばれる関数
    void Update()
    {
        // TotalValueを0から1の範囲に正規化（ノーマライズ）します。
        // これにより、値の大小に関わらず、移動量を一定の割合で扱えるようになります。
        // Clamp01は、値を強制的に0.0fから1.0fの間に収める関数です。
        float normalizedValue = Mathf.Clamp01(TotalValue / valueMax);

        // 正規化された値に基づいて、X軸方向の目標位置を計算します。
        targetPosition.x = initialPosition.x + normalizedValue * moveDistance;

        // 現在位置から目標位置へ滑らかに移動させます（Lerp: 線形補間）。
        transform.position = Vector3.Lerp(transform.position, targetPosition, Time.deltaTime * smoothSpeed);
    }

    // アプリケーションが終了する時に呼ばれる関数
    void OnApplicationQuit()
    {
        // アプリ終了時にスレッドとクライアントを確実に閉じて、リソースを解放します。
        if (receiveThread != null && receiveThread.IsAlive) receiveThread.Abort();
        if (client != null) client.Close();
    }
    
    // --- Private Methods ---

    // バックグラウンドスレッドで実行されるデータ受信専門の関数
    private void ReceiveData()
    {
        // 任意のIPアドレスからデータを受信できるように設定します。
        IPEndPoint anyIP = new IPEndPoint(IPAddress.Any, 0);

        // 無限ループでデータを受信し続けます。
        while (true)
        {
            try
            {
                // データを受信するまでここで処理が待機（ブロック）されます。
                byte[] data = client.Receive(ref anyIP);
                // 受信したバイトデータをUTF-8形式の文字列に変換します。
                string text = Encoding.UTF8.GetString(data);
                // 文字列をカンマで分割します。
                string[] values = text.Split(',');

                // Pythonから送られてくるデータは4つなので、要素数を確認します。
                if (values.Length == 4)
                {
                    // それぞれの値をfloat型に変換し、プロパティに格納します。
                    LipArea = float.Parse(values[0]);
                    Distance = float.Parse(values[1]);
                    Airflow = float.Parse(values[2]);
                    TotalValue = float.Parse(values[3]);
                }
            }
            catch (Exception err)
            {
                // エラーが発生した場合はUnityのコンソールに表示します。
                Debug.LogError(err.ToString());
            }
        }
    }
}