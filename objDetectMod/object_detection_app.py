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
import xml.etree.ElementTree as ET

############### by Ren-Jie ########################
overlap_index = 0
current_frame_count = 0
fps_video = 0
total_frame = 0
############## by Ren-Jie #########################


parser = argparse.ArgumentParser()
parser.add_argument('-src', '--source', dest='video_source', type=int,
                    default=0, help='Device index of the camera.')
parser.add_argument('-wd', '--width', dest='width', type=int,
                    default=1080, help='Width of the frames in the video stream.')
parser.add_argument('-ht', '--height', dest='height', type=int,
                    default=1920, help='Height of the frames in the video stream.')
parser.add_argument('-num-w', '--num-workers', dest='num_workers', type=int,
                    default=1, help='Number of workers.')
parser.add_argument('-q-size', '--queue-size', dest='queue_size', type=int,
                    default=5, help='Size of the queue.')

parser.add_argument('--display_type', default='video', choices=['camera', 'video', 'live'] ,help='Using camera or video')
parser.add_argument('--model_path', help='Path of model.pb file.')
parser.add_argument('--label_path', default='object-detection.pbtxt', help='Path of .pbtxt file.')
parser.add_argument('--num_classes', type=int, help='Number of classes')
parser.add_argument('--video_path', help='Path of video to display if using video.')
parser.add_argument('--overlap_output_dir', help='Dir of overlap output.')
parser.add_argument('--overlap_output', default='no', choices=['yes', 'no'], help='True or False for Finding Overlap')
parser.add_argument('--overlap_ratio', type=float, help='setting Overlap_ratio')
parser.add_argument('--detected_output_dir', help='Dir of detected_output.')
parser.add_argument('--detected_output', default='no', choices=['yes', 'no'], help='True or False for Outputing detected Object\'s xml and image')
parser.add_argument('--distance_output', action='store_true', default=False, help='Find boxes distances')
parser.add_argument('--output_labeled_video_path', default=None, help='Path of output labeled video.')
parser.add_argument('--gpudev', default='0', help='GPU device number')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpudev

# Loading label map
label_map = label_map_util.load_labelmap(args.label_path)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=args.num_classes, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
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
    
    #wait = input("PRESS ENTER TO CONTINUE.")
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8)
    

    ###### by Ren-Jie ##########################
    global current_frame_count
    current_frame_count = current_frame_count + 1

    if args.detected_output == 'yes':
        #output one detected object's xml and image per second 
        if current_frame_count % (2* fps_video) == 0:
            output_xml_and_img(np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes), image_np, category_index, args.detected_output_dir)
    
    if args.overlap_output == 'yes':
        #find overlap between the two objects (one per second)
        if current_frame_count % fps_video == 0:
            overlap_index, fps_video_temp = find_overlap(boxes, scores, classes, current_frame_count, image_np, args.overlap_ratio, overlap_index, fps_video, args.overlap_output_dir, total_frame)

    if args.distance_output:
        if current_frame_count % fps_video == 0:
            print('{}th frame'.format(current_frame_count))
            distance_matrix = get_boxes_distance_matrix(boxes, scores, classes, image_np,
                                                        current_frame_count, displacement=True)
            print('distance_matrix ={}'.format(distance_matrix))

    return image_np


def worker(input_q, output_q):
    # Load a (frozen) Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(args.model_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = False
        # sess = tf.Session(config=config)

        sess = tf.Session(graph=detection_graph, config=config)

    fps = FPS().start()
    while True:
        fps.update()
        frame = input_q.get()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output_q.put(detect_objects(frame_rgb, sess, detection_graph))

    fps.stop()
    sess.close()


if __name__ == '__main__':

    logger = multiprocessing.log_to_stderr()
    logger.setLevel(multiprocessing.SUBDEBUG)

    #input_q = Queue(maxsize=args.queue_size)
    #output_q = Queue(maxsize=args.queue_size)
    #pool = Pool(args.num_workers, worker, (input_q, output_q))

    if args.display_type == 'camera':
        video_capture = WebcamVideoStream(src=args.video_source, width=args.width, height=args.height).start()  
    elif args.display_type == 'video':
        video_capture = cv2.VideoCapture(args.video_path)
        ################# by Ren-Jie ##############################
        fps_video = video_capture.get(cv2.CAP_PROP_FPS)
        total_frame = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
        ###############################################
    elif args.display_type == 'live':
        #video_capture = cv2.VideoCapture("rtsp://140.112.28.40:8554/h264VideoViaFIFO")
        video_capture = cv2.VideoCapture('rtsp://root:chunyuchen7968@140.112.41.154:554/live.sdp')
        #video_capture = cv2.VideoCapture("rtsp://192.168.2.10:8554/h264VideoViaFIFO")

    if args.output_labeled_video_path:
        output_labeled_video_path = os.path.abspath(args.output_labeled_video_path)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        v_writer = cv2.VideoWriter(output_labeled_video_path, fourcc, fps_video, (width, height))
        
    fps = FPS().start()
    
    input_q = Queue(maxsize=args.queue_size)
    output_q = Queue(maxsize=args.queue_size)
    pool = Pool(args.num_workers, worker, (input_q, output_q))
    
    print ('Start~~~~~~~~~~~~~~~~')
    
    while True:  
        if args.display_type == 'camera':
            frame = video_capture.read()
        elif args.display_type == 'video':
            ret, frame = video_capture.read()
        elif args.display_type == 'live':
            ret, frame = video_capture.read()
        if not ret:
            break

        input_q.put(frame)

        t = time.time()

        output_rgb = cv2.cvtColor(output_q.get(), cv2.COLOR_RGB2BGR)
        print('[INFO] elapsed time: {:.2f}'.format(time.time() - t))
        # cv2.namedWindow('Video', 0)
        # cv2.resizeWindow('Video', 960 , 540)
        # cv2.imshow('Video', output_rgb)
        #im2 = Image.fromarray(output_rgb,'RGB')
        #im2.show()
        fps.update()

        if args.output_labeled_video_path:
            v_writer.write(output_rgb)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    fps.stop()

    print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
    print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))

    if args.output_labeled_video_path:
        v_writer.release()

    pool.terminate()
    cv2.destroyAllWindows()
