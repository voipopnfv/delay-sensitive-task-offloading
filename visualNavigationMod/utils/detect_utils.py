#!/usr/bin/python3.6
import os
import sys
import time
import argparse
import multiprocessing
import cv2
import numpy as np
import tensorflow as tf

detect_thres = 0.7

def tensorflow_shut_up():
    """
    Make Tensorflow less verbose
    """
    try:
        # noinspection PyPackageRequirements
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        # Monkey patching deprecation utils to shut it up! Maybe good idea to disable this once after upgrade
        # noinspection PyUnusedLocal
        def deprecated(date, instructions, warn_once=True):
            def deprecated_wrapper(func):
                return func
            return deprecated_wrapper

        from tensorflow.python.util import deprecation
        deprecation.deprecated = deprecated

    except ImportError:
        print("Import ERROR")
        pass

def cv2_show_image(img, time=5, rgb_flag=True):

    if rgb_flag:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    cv2.imshow('', img)
    cv2.waitKey(time)

    return

def cal_org_coordinates(boxes, lengths):
    
    y_len, x_len = lengths[:2]
    x_min, x_max = np.around(x_len * boxes[1::2]).astype(int)
    y_min, y_max = np.around(y_len * boxes[0::2]).astype(int)
 
    return [y_min, y_max, x_min, x_max]


def detect_objects(image_np, sess, detection_graph):
    global overlap_index, total_frame
    global fps_video    

    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Each box represents a part of the image where a particular object was detected.
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Actual detection.
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

    boxes = np.squeeze(boxes)
    scores = np.squeeze(scores)
    classes = np.squeeze(classes).astype(int)

    target_index = np.where(scores < detect_thres)[0]
    target_index = np.where(classes[:target_index[0]] == 1)[0]
    # print("Target index: {}".format(target_index))

    box_images = []
    for i in target_index:
        y_min, y_max, x_min, x_max = cal_org_coordinates(boxes[i], image_np.shape)
        box_images.append([(y_min, y_max, x_min, x_max), image_np[y_min:y_max, x_min:x_max, :]])

    return np.array(box_images)


def worker(input_ls):
    # Load a (frozen) Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile("model/frozen_inference_graph.pb", 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(graph=detection_graph, config=config)
    
    output_ls = []
    for indx, frame in enumerate(input_ls):
        st = time.time()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output_ls.append([(coord, cv2.cvtColor(img, cv2.COLOR_RGB2BGR)) for coord, img in detect_objects(frame_rgb, sess, detection_graph)])
        et = time.time()
        print(et - st)
        
    sess.close()
    return output_ls

def detect_img(img_ls):

    logger = multiprocessing.log_to_stderr()
    logger.setLevel(multiprocessing.SUBDEBUG)

    rst_ls = worker(img_ls)

    cv2.destroyAllWindows()
    return rst_ls

if __name__ == "__main__":
    # tensorflow_shut_up()
    img = cv2.imread("TestImage.jpg")
    print("Start Detection")
    Output_ls = detect_img([img])
    print("Finished Detection")
    Count = 0
    for i in Output_ls[0]:
        coord, Img = i
        # print(coord)
        # cv2_show_image(Img)
        cv2.imwrite("%d.jpg" % Count, Img)
        Count += 1