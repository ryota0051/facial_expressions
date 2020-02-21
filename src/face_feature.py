import os
from typing import Dict, Tuple, List
import json
import time

import tensorflow as tf
import numpy as np

from type_def import BOUNDARY_BOX_TYPE, PERSONAL_INFO_TYPE

class FaceFeatureExtractor():
    def __init__(self, base_model_path: str, nationality_model_path: str, label_path: str) -> None:
        '''必要なファイルを読み込むインスタンスメソッド
        Parameter
        ----------
        base_model_path: mobilenetV2の畳み込み部分のみのモデルパス
        nationality_model_path: 国籍モデルのパス
        label_path: 各モデルが出力する数字が表す文字列を格納したファイルのパス
            ファイル内容の例:
            {
                "gender": {
                    0: "female",
                    1: "male"
                },
                ...
            }
        '''
        self.base_model = self.__load_model(base_model_path)
        self.nationality_model = self.__load_model(nationality_model_path)
        self.labels = self.__load_labels(label_path)

    def get_personal_data_from_faces(self, img_batch: np.array, rect_list: BOUNDARY_BOX_TYPE) -> PERSONAL_INFO_TYPE:
        '''顔画像データから性別、年齢、人種を判別するメソッド
        Parameter
        ----------
        img_batch: バッチ画像
        rect_list: 顔座標

        Returns
        ----------
        例:
        {
            (x1, y1, w1, h1): {"age": "20代", "gender": "male", "race": "Asian"},
            (x2, y2, w2, h2): {"age": "30代", "gender": "male", "race": "Black"},
            ...
        }
        '''
        features = self.get_feature_batch(img_batch)
        features = features.reshape(len(features), -1)

        # 国籍判定
        nationality_list = self.predict_facial_expression(features, self.nationality_model)

        results = {}
        for rect, nationality in zip(rect_list, nationality_list):
            results[rect] = {}
            results[rect]['nationality'] = self.labels['nationality'][str(nationality)]
        return results

    def get_feature_batch(self, img_batch: np.array) -> np.array:
        '''ベースとなるモバイルネットからバッチ画像ごとに特徴量を抽出するメソッド

        Parameter
        ---------
        img_batch: バッチ画像

        Returns
        ---------
        モバイルネットが出力する特徴量
        '''
        assert isinstance(img_batch, np.ndarray)
        assert img_batch.ndim == 4
        x = tf.keras.applications.mobilenet_v2.preprocess_input(img_batch)
        features = self.base_model.predict(x)
        return features

    def predict_facial_expression(
            self,
            features: np.array,
            model: '学習済み予測部分モデル') -> List[int]:
        '''指定モデルにおける顔の属性を予測するメソッド

        Parameter
        ---------
        features: modelに入力する特徴量
        model: 属性予測モデル(kerasのクラスラベルを返すメソッドである
            predict_classesを用いているので、別のフレームワークを使う場合は、
            classなどでラッパーする。)

        Returns
        ---------
        要素として、予測結果の数値ラベルをもつリスト
        '''
        return model.predict_classes(features).tolist()

    def __load_labels(self, label_path: str) -> Dict[str, Dict[str, str]]:
        '''json形式で記述されたファイルからone-hot-vectorが表す文字列辞書を取得するメソッド
        Parameter
        ----------
        label_path: 各モデルが出力するラベルが表す文字列辞書が記述されたjsonファイルのパス

        Returns
        ----------
        one-hot-vectorが表す文字列辞書
        例:
            {
              "gender":
              {
                "0": "female",
                "1": "male"
              },
              "age": {
                "0": "10代",
                "1": "20代",
                "2": "30代",
                "3": "40代",
                "4": "50代"
              },
              "race":
              {
                "0": "Asian",
                "1": "Black",
                "2": "Indian",
                "3": "others",
                "4": "White"
              }
            }
        '''
        self.__check_file_exists(label_path)
        with open(label_path, 'r') as f:
            labels = json.load(f)
        return labels

    def __load_model(self, model_path:str) -> 'kerasのmodel':
        '''kerasモデルを読み込むメソッド
        Parameter
        ----------
        model_path: 読み込みモデルパス

        Returns
        ----------
        tf.keras.models.load_modelの返り値
        '''
        self.__check_file_exists(model_path)
        return tf.keras.models.load_model(model_path)

    def __check_file_exists(self, file_path: str) -> None:
        '''ファイルが存在するかを確かめるメソッド(ファイルが存在しない場合は、例外を出力する。)
        Parameter
        ----------
        file_path: 存在を確かめるファイル
        '''
        if not os.path.exists(file_path):
            raise FileNotFoundError('[{}]が存在しません。'.format(file_path))
