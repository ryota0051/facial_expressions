import json
import logging
from paho.mqtt import client as mqtt
logger = logging.getLogger(__name__)

class Publisher():
    '''mqttのパブリッシュを管理するクラス
    '''
    def __init__(self, setting_file_path: str) -> None:
        with open(setting_file_path, 'r') as f:
            mqtt_setting = json.load(f)

        # クライアント生成
        broker_info = mqtt_setting['borker_info']
        host = broker_info['host']
        port = int(broker_info['port'])
        keepalive = int(broker_info['keepalive'])
        self.mqtt_client = create_client(host, port, keepalive)

        self.topics = mqtt_setting['topics']

    def publish(self, msg: str, topic_type: str) -> None:
        '''メッセージをパブリッシュするメソッド

        Parameter
        ---------
        msg: 送信するメッセージ
        topic_type: self.topicsのkey
        '''
        self.mqtt_client.publish(self.topics[topic_type], msg)


def create_client(
        host: str,
        port: int = 1883,
        keepalive: int = 60) -> mqtt.Client:
    '''パブリッシュ用のクライアントを作成し、返す関数

    Parameter
    ---------
    host: ブローカのipアドレス
    port: ブローカのポート番号
    keepalive: ブローカとの接続確認処理実行時間間隔(sec)

    Returns
    ---------
    ブローカに接続したクライアント
    '''
    mqtt_client = mqtt.Client()
    # コールバック関数設定
    mqtt_client.on_connect = on_connect
    mqtt_client.on_disconnect = on_disconnect

    # ブローカに接続
    mqtt_client.connect(host, port, keepalive)

    # 通信開始
    mqtt_client.loop_start()
    return mqtt_client


def on_connect(client, userdata, flag, rc):
    logger.info('ブローカとの接続しました。 Result code: {}'.format(rc))

def on_disconnect(client, userdata, flag, rc):
    if rc != 0:
        logger.warning('予期しないブローカとの切断を確認')
