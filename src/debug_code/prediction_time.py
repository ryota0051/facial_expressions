import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np

import time
import pprint

def measure_mobile_net_speed_changing_input_image(iter_num=11, alpha=1.0):
    shape_list = [
        (96, 96, 3),
        (128, 128, 3),
        (160, 160, 3),
        (192, 192, 3),
        (224, 224, 3)
    ]
    time_results = {shape: [] for shape in shape_list}
    fps_results = {shape: [] for shape in shape_list}
    for shape in shape_list:
        initialized = False
        for i in range(iter_num):
            if not initialized:
                model = tf.keras.applications.MobileNetV2(
                    input_shape=shape,
                    include_top=False,
                    alpha=alpha,
                    weights='imagenet')
                img = image.load_img('../../debug_data/elephant.jpg',
                    target_size=(shape[0], shape[1]))
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)
                initialized = True
            start = time.time()
            predict = model.predict(x)
            elapsed_time = time.time() - start
            time_results[shape].append(elapsed_time)
            fps_results[shape].append(1/elapsed_time)
        print(f'{shape}終了')
    time_results = {shape: np.mean(time_results[shape][1:]) for shape in time_results}
    fps_results = {shape: np.mean(fps_results[shape][1:]) for shape in fps_results}
    return time_results, fps_results

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('alpha', type=float)
    args = parser.parse_args()
    time_results, fps_results = measure_mobile_net_speed_changing_input_image(alpha=args.alpha)
    pprint.pprint(time_results)
    pprint.pprint(fps_results)
