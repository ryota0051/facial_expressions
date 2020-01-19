import logging
import os
from typing import Tuple, List, Union, Dict
import time
import pickle
logger = logging.getLogger()

import cv2
import numpy as np

from type_def import BOUNDARY_BOX_TYPE, PERSONAL_INFO_TYPE
from face_feature import FaceFeatureExtractor

FACE_CASCADE_PATH = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_alt.xml')

def capture_img(path_dict: Dict[str, str], debug: bool = True, shape: Tuple[int, int] = (150, 150)) -> None:
    '''カメラで撮影した顔から個人の属性を抽出する関数
    Parameter
    ----------
    path_dict: 以下の形式の辞書
        {
            "base_model_path": mobilenetV2の畳み込み部分のみのモデルパス
            "age_model_path": 年齢推定モデルのパス
            "gender_model_path": 性別推定モデルのパス
            "race_model_path": 人種推定モデルのパス
            "one_hot_vector_dict_path": 各モデルが出力するone-hot-vectorが表す文字列を格納したファイルのパス
        }
    debug: デバッグ用の画像を表示するか否か
    shape: 推論時に使用する画像サイズ
    '''
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    feature_extractor = __get_feature_extractor(path_dict)
    timer = {
        'face_detect': [],
        'feature': []
    }

    try:
        while cap.isOpened():
            all_start_time = time.time()
            results = None
            img, faces = __detect_face(cap, face_cascade)
            if len(faces) != 0:
                clipped_img_list = []
                rect_list = []
                for (x, y, w, h) in faces:
                    clipped_img = __get_clipped_img(img, x, y, w, h, shape)
                    clipped_img_list.append(clipped_img)
                    rect_list.append((x, y, w, h))
                results = feature_extractor.get_personal_data_from_faces(np.array(clipped_img_list), rect_list)
                logger.debug(results)
            else:
                logger.debug('no faces')
            elapsed_time = time.time() - all_start_time
            fps = f'fps={(1 / elapsed_time):.2f}'
            if debug:
                if results is not None:
                    __write_personal_info2img(img, results)
                cv2.putText(img, fps, (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.waitKey(1)
                cv2.imshow('test', img)
    except KeyboardInterrupt:
        logger.info('end')

def __get_feature_extractor(path_dict: Dict[str, str]):
    '''特徴抽出クラスをインスタンス化して返す関数
    Parameter
    ----------
    path_dict: 以下の形式の辞書
        {
            base_model_path: mobilenetV2の畳み込み部分のみのモデルパス
            age_model_path: 年齢推定モデルのパス
            gender_model_path: 性別推定モデルのパス
            race_model_path: 人種推定モデルのパス
            one_hot_vector_dict_path: 各モデルが出力するone-hot-vectorが表す文字列を格納したファイルのパス
        }

    Returns
    ----------
    '''
    base_model_path = path_dict['base_model_path']
    age_model_path = path_dict['age_model_path']
    gender_model_path = path_dict['gender_model_path']
    race_model_path = path_dict['race_model_path']
    emotion_model_path = path_dict['emotion_model_path']
    one_hot_vector_dict_path = path_dict['one_hot_vector_dict_path']
    feature_extractor = FaceFeatureExtractor(base_model_path, age_model_path, gender_model_path, race_model_path, emotion_model_path, one_hot_vector_dict_path)
    return feature_extractor

def __detect_face(cap: cv2.VideoCapture, cascade_classifier: cv2.CascadeClassifier) \
    -> Tuple[np.array, Union[Tuple[None], BOUNDARY_BOX_TYPE]]:
    '''カメラから画像を取得し、顔部分の座標を取得する関数
    Parameter
    ----------
    cap: カメラから画像を取得するクラス
    cascade_classifier: 顔検出器

    Returns
    ----------
    1. 取得画像
    2. 以下の構造の顔座標データ
        顔が検出された場合: ((顔1の座標x, 顔1の座標y, 顔1の横幅w, 顔1の縦幅), (顔2の座標x, 顔2の座標y, 顔2の横幅w, 顔2の縦幅), ...)
        顔が検出されない場合: ()
    '''
    _, img = cap.read()
    faces = cascade_classifier.detectMultiScale(img)
    return img, faces

def __get_clipped_img(base_img: np.array, x: int, y: int, w: int, h: int, shape: Tuple[int, int]) -> np.array:
    '''撮影画像から画像を切り抜き、推論に必要なサイズに変更する関数
    Parameter
    ----------
    base_img: 撮影画像
    x: 撮影画像の左上のx座標
    y: 撮影画像の左上のy座標
    w: 切り取り対象の幅
    h: 切り取り対象の高さ
    shape: 推論に必要なサイズ(x, y)
    '''
    clipped_img = base_img[y:y+h, x:x+w]
    clipped_img = cv2.resize(clipped_img, shape)
    clipped_img = cv2.cvtColor(clipped_img, cv2.COLOR_BGR2RGB)
    return clipped_img

def __write_personal_info2img(base_img: np.array, personal_info_list: PERSONAL_INFO_TYPE) -> None:
    '''撮影した画像に性別、年齢、人種情報を書き込む関数
    Parameter
    ----------
    base_img: 書き込み対象画像
    personal_info_list: 以下の形式の辞書
        {
            (x1, y1, w1, h1): {"age": "20代", "gender": "male", "race": "Asian"},
            (x2, y2, w2, h2): {"age": "30代", "gender": "male", "race": "Black"},
            ...
        }
    '''
    for (x, y, w, h) in personal_info_list:
        personal_info = personal_info_list[(x, y, w, h)]
        age_txt = f'age: {personal_info["age"]}'
        gender_txt = f'gender: {personal_info["gender"]}'
        race_txt = f'race: {personal_info["race"]}'
        emotion_txt = f'emotion: {personal_info["emotion"]}'
        cv2.putText(base_img, emotion_txt, (x, y-70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(base_img, age_txt, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(base_img, gender_txt, (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(base_img, race_txt, (x, y-50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.rectangle(base_img, (x, y), (x+w, y+h), (0, 0, 255), thickness=2)

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    path_dict = {
        'base_model_path': '../models/base_model.h5',
        'age_model_path': '../models/age_model.h5',
        'gender_model_path': '../models/gender_model.h5',
        'race_model_path': '../models/race_model_with_up_sampling.h5',
        'emotion_model_path': '../models/emotion.h5',
        'one_hot_vector_dict_path': '../models/one_hot_vector_dict.json',
    }
    capture_img(path_dict)
