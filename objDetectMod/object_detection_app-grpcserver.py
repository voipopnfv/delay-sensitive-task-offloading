import os
import cv2
import sys
import time
import argparse
import multiprocessing
import numpy as np
import tensorflow as tf
from PIL import Image


from utils.app_utils import FPS, WebcamVideoStream
from utils.functions import output_xml_and_img, find_overlap, get_boxes_distance_matrix
from multiprocessing import Queue, Pool
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

import grpc
from concurrent import futures
import service.objDetectMod_pb2 as objDetectMod_pb2
import service.objDetectMod_pb2_grpc as objDetectMod_pb2_grpc
# import xml.etree.ElementTree as ET

import random

############### by Ren-Jie ########################
# overlap_index = 0
# current_frame_count = 0
# fps_video = 0
# total_frame = 0  
############## by Ren-Jie #########################


# parser = argparse.ArgumentParser()
# parser.add_argument('-src', '--source', dest='video_source', type=int,
#                     default=0, help='Device index of the camera.')
# parser.add_argument('-wd', '--width', dest='width', type=int,
#                     default=1080, help='Width of the frames in the video stream.')
# parser.add_argument('-ht', '--height', dest='height', type=int,
#                     default=1920, help='Height of the frames in the video stream.')
# parser.add_argument('-num-w', '--num-workers', dest='num_workers', type=int,
#                     default=1, help='Number of workers.')
# parser.add_argument('-q-size', '--queue-size', dest='queue_size', type=int,
#                     default=5, help='Size of the queue.')

# parser.add_argument('--display_type', default='video', choices=['camera', 'video', 'live'] ,help='Using camera or video')
# parser.add_argument('--model_path', help='Path of model.pb file.')
# parser.add_argument('--label_path', default='object-detection.pbtxt', help='Path of .pbtxt file.')
# parser.add_argument('--num_classes', type=int, help='Number of classes')
# parser.add_argument('--video_path', help='Path of video to display if using video.')
# parser.add_argument('--overlap_output_dir', help='Dir of overlap output.')
# parser.add_argument('--overlap_output', default='no', choices=['yes', 'no'], help='True or False for Finding Overlap')
# parser.add_argument('--overlap_ratio', type=float, help='setting Overlap_ratio')
# parser.add_argument('--detected_output_dir', help='Dir of detected_output.')
# parser.add_argument('--detected_output', default='no', choices=['yes', 'no'], help='True or False for Outputing detected Object\'s xml and image')
# parser.add_argument('--distance_output', action='store_true', default=False, help='Find boxes distances')
# parser.add_argument('--output_labeled_video_path', default=None, help='Path of output labeled video.')
# parser.add_argument('--gpudev', default='0', help='GPU device number')
# args = parser.parse_args()

# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpudev
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
task = 0

# Loading label map

#print(label_map)
#print(categories)
#print(category_index)
'''
item {
  name: "car"
  id: 1
}
item {
  name: "bus_stop"
  id: 2
}
item {
  name: "bus_grid"
  id: 3
}
item {
  name: "bus_station"
  id: 4
}
[{'name': 'car', 'id': 1}, {'name': 'bus_stop', 'id': 2}, {'name': 'bus_grid', 'id': 3}, {'name': 'bus_station', 'id': 4}]
{1: {'name': 'car', 'id': 1}, 2: {'name': 'bus_stop', 'id': 2}, 3: {'name': 'bus_grid', 'id': 3}, 4: {'name': 'bus_station', 'id': 4}}
'''
#exit()

# class testclass():
class objDetectModserviceServicer(objDetectMod_pb2_grpc.objDetectModserviceServicer):
    def __init__(self):
        ## Args
        self.video_source = 0
        self.width = 1080
        self.height = 1920
        self.num_workers = 1
        self.queue_size = 5
        self.display_type = "video"
        self.model_path = "model/frozen_inference_graph.pb" ## required
        self.label_path = "label_class.pbtxt" ## required
        self.num_classes = 5 ## required
        self.video_path = "Video/onesec.mp4" ## required
        self.overlap_output_dir = "overlap" ## required
        self.overlap_output = "no" ## required
        self.overlap_ratio = None
        self.detected_output_dir = "output" ## required
        self.detected_output = "no" ## required
        self.distance_output = False
        self.output_labeled_video_path = "output/onesec.mp4" ## required

        self.category_index = None

        self.input_q = None
        self.output_q = None
        self.pool = None

        self.fps_video = 0
        self.total_frame = 0
        self.overlap_index = 0
        self.current_frame_count = 0

    def setup(self):
        self.model_path = "model/frozen_inference_graph.pb"
        self.label_path = "label_class.pbtxt"
        self.num_classes = 5
        self.video_path = "Video/onesec.mp4"
        self.overlap_output_dir = "overlap"
        self.overlap_output = "no"
        self.detected_output = "yes"
        self.detected_output_dir = "output"
        self.output_labeled_video_path = "output/onesec.mp4"
        # self.gpudev = "0"

    # def init(self):
    def init(self, request, context):

        # if request.gpudev : self.gpudev = request.gpudev
        if request.label_path : self.label_path = request.label_path
        if request.num_classes : self.num_classes = request.num_classes
        if request.queue_size : self.queue_size = request.queue_size

        # os.environ["CUDA_VISIBLE_DEVICES"] = self.gpudev

        label_map = label_map_util.load_labelmap(self.label_path)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=self.num_classes, use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)

        logger = multiprocessing.log_to_stderr()
        logger.setLevel(multiprocessing.SUBDEBUG)

        self.input_q = Queue(maxsize=self.queue_size)
        self.output_q = Queue(maxsize=self.queue_size)
        self.pool = Pool(self.num_workers, self.worker, (self.input_q, self.output_q))
        return objDetectMod_pb2.initOutput(status="ok")

    def inference(self, request, context):
        global task
        task += 1
        meantime = []
        maxtask = []
        if request.model_path : self.model_path = request.model_path
        if request.video_path : self.video_path = request.video_path
        if request.overlap_output_dir : self.overlap_output_dir = request.overlap_output_dir
        if request.overlap_output : self.overlap_output = request.overlap_output
        if request.detected_output : self.detected_output = request.detected_output
        if request.detected_output_dir : self.detected_output_dir = request.detected_output_dir
        if request.output_labeled_video_path : self.output_labeled_video_path = request.output_labeled_video_path

        self.fps_video = 0
        self.total_frame = 0
        self.overlap_index = 0
        self.current_frame_count = 0
        if self.display_type == 'camera':
            video_capture = WebcamVideoStream(src=self.video_source, width=self.width, height=self.height).start()  
        elif self.display_type == 'video':
            video_capture = cv2.VideoCapture(self.video_path)
            ################# by Ren-Jie ##############################
            self.fps_video = video_capture.get(cv2.CAP_PROP_FPS)
            self.total_frame = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
            ###############################################
        elif self.display_type == 'live':
            #video_capture = cv2.VideoCapture("rtsp://140.112.28.40:8554/h264VideoViaFIFO")
            video_capture = cv2.VideoCapture('rtsp://root:chunyuchen7968@140.112.41.154:554/live.sdp')
            #video_capture = cv2.VideoCapture("rtsp://192.168.2.10:8554/h264VideoViaFIFO")

        if self.output_labeled_video_path:
            output_labeled_video_path = os.path.abspath(self.output_labeled_video_path)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            v_writer = cv2.VideoWriter(output_labeled_video_path, fourcc, self.fps_video, (width, height))
        
        # self.pool = Pool(self.num_workers, self.worker, (self.input_q, self.output_q))

        fps = FPS().start()
        print ('Start~~~~~~~~~~~~~~~~')
        
        while True:  
            if self.display_type == 'camera':
                frame = video_capture.read()
            elif self.display_type == 'video':
                ret, frame = video_capture.read()
            elif self.display_type == 'live':
                ret, frame = video_capture.read()
            if not ret:
                break

            self.input_q.put(frame)

            t = time.time()

            output_rgb = cv2.cvtColor(self.output_q.get(), cv2.COLOR_RGB2BGR)
            print('[INFO] elapsed time: {:.3f} task_num: {:d}'.format(time.time() - t, task))
            meantime.append(time.time() - t) ## in second
            maxtask.append(task)
            # cv2.namedWindow('Video', 0)
            # cv2.resizeWindow('Video', 960 , 540)
            # cv2.imshow('Video', output_rgb)
            #im2 = Image.fromarray(output_rgb,'RGB')
            #im2.show()
            fps.update()

            if self.output_labeled_video_path:
                v_writer.write(output_rgb)

        fps.stop()

        print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
        print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))
        print('[INFO] elapseTimePerFrame: {:.3f}s maxTask: {:d}'.format(sum(meantime) / float(len(meantime)), max(maxtask)))

        if self.output_labeled_video_path:
            v_writer.release()

        # self.pool.terminate()

        # cv2.destroyAllWindows()
        task -= 1
        return objDetectMod_pb2.inferenceOutput(status="ok", score=random.random())

    def worker(self, input_q, output_q):
        # Load a (frozen) Tensorflow model into memory.
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
            config = tf.ConfigProto(gpu_options=gpu_options)
            config.gpu_options.allow_growth = True
            sess = tf.Session(graph=detection_graph, config=config)

        fps = FPS().start()
        while True:
            fps.update()
            frame = input_q.get()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output_q.put(self.detect_objects(frame_rgb, sess, detection_graph))

        fps.stop()
        sess.close()

    def detect_objects(self, image_np, sess, detection_graph):
        # overlap_index, 
        # total_frame
        # fps_video    


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
        
        #wait = input("PRESS ENTER TO CONTINUE.")
        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            self.category_index,
            use_normalized_coordinates=True,
            line_thickness=8)
        

        ###### by Ren-Jie ##########################
        # global current_frame_count
        self.current_frame_count = self.current_frame_count + 1

        if self.detected_output == 'yes':
            #output one detected object's xml and image per second
            if self.current_frame_count % (2* self.fps_video) == 0:
                output_xml_and_img(np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes), image_np, self.category_index, self.detected_output_dir)
        
        if self.overlap_output == 'yes':
            #find overlap between the two objects (one per second)
            if self.current_frame_count % self.fps_video == 0:
                self.overlap_index, fps_video_temp = find_overlap(boxes, scores, classes, self.current_frame_count, image_np, self.overlap_ratio, overlap_index, self.fps_video, self.overlap_output_dir, self.total_frame)

        if self.distance_output:
            if self.current_frame_count % self.fps_video == 0:
                print('{}th frame'.format(self.current_frame_count))
                distance_matrix = get_boxes_distance_matrix(boxes, scores, classes, image_np,
                                                            self.current_frame_count, displacement=True)
                print('distance_matrix ={}'.format(distance_matrix))
        return image_np


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    objDetectMod_pb2_grpc.add_objDetectModserviceServicer_to_server(objDetectModserviceServicer(), server)
    server.add_insecure_port('0.0.0.0:50052')
    server.start()
    server.wait_for_termination()

serve()

# test = testclass()
# test.setup()
# test.init()
# test.inference()
