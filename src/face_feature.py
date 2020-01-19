import os
from typing import Dict, Tuple
import json
import time
import threading

import tensorflow as tf
import numpy as np

from type_def import BOUNDARY_BOX_TYPE, PERSONAL_INFO_TYPE

graph = tf.get_default_graph()
sess = tf.Session()

class FaceFeatureExtractor():
    def __init__(self, base_model_path: str, age_model_path: str, gender_model_path: str, race_model_path: str, emotion_model_path: str, one_hot_vector_dict_path: str) -> None:
        '''必要なファイルを読み込むインスタンスメソッド
        Parameter
        ----------
        base_model_path: mobilenetV2の畳み込み部分のみのモデルパス
        age_model_path: 年齢推定モデルのパス
        gender_model_path: 性別推定モデルのパス
        race_model_path: 人種推定モデルのパス
        one_hot_vector_dict_path: 各モデルが出力するone-hot-vectorが表す文字列を格納したファイルのパス
            ファイル内容の例:
            {
                "gender": {
                    0: "female",
                    1: "male"
                },
                ...
            }
        '''
        self.base_model = self.load_model(base_model_path)
        self.age_model = self.load_model(age_model_path)
        self.gender_model = self.load_model(gender_model_path)
        self.race_model = self.load_model(race_model_path)
        self.emotion_model = self.load_model(emotion_model_path)
        self.one_hot_vector_dict = self.load_one_hot_vector_dict(one_hot_vector_dict_path)

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
        assert isinstance(img_batch, np.ndarray)
        assert img_batch.ndim == 4
        x = tf.keras.applications.mobilenet_v2.preprocess_input(img_batch)
        start = time.time()
        feature = self.base_model.predict(x).reshape(len(x), -1)

        # 性別推定
        gender_list = self.gender_model.predict_classes(feature).tolist()
        # 年齢推定
        age_list = self.age_model.predict_classes(feature).tolist()
        # 人種推定
        race_list = self.race_model.predict_classes(feature).tolist()
        # 感情推定
        emotion_list = self.emotion_model.predict_classes(feature).tolist()

        results = {}
        for rect, gender, age, race, emotion in zip(rect_list, gender_list, age_list, race_list, emotion_list):
            results[rect] = {}
            results[rect]['gender'] = self.one_hot_vector_dict['gender'][str(gender[0])]
            results[rect]['age'] = self.one_hot_vector_dict['age'][str(age)]
            results[rect]['race'] = self.one_hot_vector_dict['race'][str(race)]
            results[rect]['emotion'] = self.one_hot_vector_dict['emotion'][str(emotion)]
        return results

    def load_one_hot_vector_dict(self, one_hot_vector_dict_path: str) -> Dict[str, Dict[str, str]]:
        '''json形式で記述されたファイルからone-hot-vectorが表す文字列辞書を取得するメソッド
        Parameter
        ----------
        one_hot_vector_dict_path: one-hot-vectorが表す文字列辞書が記述されたjsonファイルのパス

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
        self.check_file_exists(one_hot_vector_dict_path)
        with open(one_hot_vector_dict_path, 'r') as f:
            one_hot_vector_dict = json.load(f)
        return one_hot_vector_dict

    def load_model(self, model_path:str) -> 'kerasのmodel':
        '''kerasモデルを読み込むメソッド
        Parameter
        ----------
        model_path: 読み込みモデルパス

        Returns
        ----------
        tf.keras.models.load_modelの返り値
        '''
        self.check_file_exists(model_path)
        return tf.keras.models.load_model(model_path)

    def check_file_exists(self, file_path: str) -> None:
        '''ファイルが存在するかを確かめるメソッド(ファイルが存在しない場合は、例外を出力する。)
        Parameter
        ----------
        file_path: 存在を確かめるファイル
        '''
        if not os.path.exists(file_path):
            raise FileNotFoundError(f'[{file_path}]が存在しません。')

class FineTunedFaceFeatureExtractor(FaceFeatureExtractor):
    def __init__(self, age_model_path: str, gender_model_path: str, race_model_path: str, one_hot_vector_dict_path: str) -> None:
        '''必要なファイルを読み込むインスタンスメソッド
        Parameter
        ----------
        age_model_path: 年齢推定モデルのパス
        gender_model_path: 性別推定モデルのパス
        race_model_path: 人種推定モデルのパス
        one_hot_vector_dict_path: 各モデルが出力するone-hot-vectorが表す文字列を格納したファイルのパス
            ファイル内容の例:
            {
                "gender": {
                    0: "female",
                    1: "male"
                },
                ...
            }
        '''
        self.age_model = self.load_model(age_model_path)
        self.gender_model = self.load_model(gender_model_path)
        self.race_model = self.load_model(race_model_path)
        self.one_hot_vector_dict = self.load_one_hot_vector_dict(one_hot_vector_dict_path)
        self.prediction_results = {
            'gender': None,
            'race': None,
            'age': None,
        }

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
        x = tf.keras.applications.mobilenet_v2.preprocess_input(img_batch)
        gender_th = threading.Thread(target=self.__get_prediction_result, args=(x, self.gender_model, 'gender'))
        age_th = threading.Thread(target=self.__get_prediction_result, args=(x, self.age_model, 'age'))
        race_th = threading.Thread(target=self.__get_prediction_result, args=(x, self.race_model, 'race'))
        th_list = [gender_th, age_th, race_th]
        for i, th in enumerate(th_list):
            th.start()
            th.join()

        results = {}
        print(self.prediction_results)
        for rect, gender, age, race in zip(rect_list,
                                           self.prediction_results['gender'],
                                           self.prediction_results['age'],
                                           self.prediction_results['race']):
            results[rect] = {}
            results[rect]['gender'] = self.one_hot_vector_dict['gender'][str(gender[0])]
            results[rect]['age'] = self.one_hot_vector_dict['age'][str(age)]
            results[rect]['race'] = self.one_hot_vector_dict['race'][str(race)]
        self.__reset_prediction_results()
        return results

    def load_model(self, model_path:str) -> 'kerasのmodel':
        '''kerasモデルを読み込むメソッド
        Parameter
        ----------
        model_path: 読み込みモデルパス

        Returns
        ----------
        tf.keras.models.load_modelの返り値
        '''
        self.check_file_exists(model_path)
        tf.keras.backend.set_session(sess)
        with graph.as_default():
            model = tf.keras.models.load_model(model_path)
        model._make_predict_function()
        return model

    def __get_prediction_result(self, img_patch: np.array, model, prediction_target_name: str) -> None:
        '''モデルから予測結果を取得し、指定したself.prediction_resultsのkeyに結果を代入するメソッド
        '''
        global sess
        global graph
        with graph.as_default():
            tf.keras.backend.set_session(sess)
            predict_result = model.predict_classes(img_patch).tolist()
        self.prediction_results[prediction_target_name] = predict_result

    def __reset_prediction_results(self) -> Dict[str, None]:
        self.prediction_results = {
            'gender': None,
            'race': None,
            'age': None,
            'emotion': None
        }
