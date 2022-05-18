# coding: utf-8

# # Signboard Selection

# In[1]:


import os
import sys
import cv2
import math
import time
import scipy
import scipy.spatial
import numpy as np
# import tensorflow.compat.v1 as tf
import tensorflow as tf
import matplotlib.pyplot as plt
import grpc
from concurrent import futures

from utils.detect_utils_gpu import *
import service.visualNavigationMod_pb2 as visualNavigationMod_pb2
import service.visualNavigationMod_pb2_grpc as visualNavigationMod_pb2_grpc

import random

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
task = 0
# tf.disable_v2_behavior()

# In[2]:


class timer:
    def __init__(self):
        self.st = time.time()
        self.et = time.time()
    
    def tick(self):
        self.st = time.time()
    
    def tock(self):
        self.et = time.time()
    
    def get_time(self):
        return (self.et - self.st) * 1000


# ### Load Image

# In[3]:


def load_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def show_image(img, title=None, figsize=None):
    plt.figure(figsize=figsize)
    if title is not None:
        plt.title(title)
    plt.imshow(img)
    plt.show()
    
def save_image(img, path):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)


# ### Video Context

# In[4]:


class VideoContext:
    def __init__(self, path, verbose=1):
        self.path = path
        self.verbose = verbose
        self.timer = timer()
        self.setup()
        
    def setup(self):
        self.timer.tick()
        self.cap = cv2.VideoCapture(self.path)
        self.fps = int(np.round(self.cap.get(cv2.CAP_PROP_FPS)))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                           int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.cur_frame = -1
        self.timer.tock()
        if self.verbose: print("INFO: Video context setup complete, time: %dms" % (self.timer.get_time()))
        
    def get_next_frame(self):
        if self.cur_frame >= self.total_frames-1:
            if self.verbose: print("ERROR: Reached the last frame, please reset with set_frame(0)")
            return None
        ret, frame = self.cap.read()
        if ret == False:
            if self.verbose: print("ERROR: Video Capture Read Failed")
            return None
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.cur_frame += 1
        return frame
    
    def set_frame(self, index):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        self.cur_frame = index-1
        
    def release(self):
        self.cap.release()


# ### Object Detection Model

# In[5]:


class ObjectDetectionModel:
    def __init__(self, path, verbose=1):
        self.path = path
        self.verbose = verbose
        self.timer = timer()
        self.setup()

    def setup(self):
        self.timer.tick()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction=0.5
        self.graph = tf.Graph()
        with self.graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(self.path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            self.sess = tf.Session(graph=self.graph, config=config)
        
        # Warmup
        test = np.zeros((1280, 720, 3))
        self.detect(test)
        
        self.timer.tock()
        if self.verbose: print("INFO: Object detection model setup complete, time: %dms" % (self.timer.get_time()))

    def detect(self, input_frame):
        # return value: (coord, crop_img)
        # ((y_min, y_max, x_min, x_max), image_np[y_min:y_max, x_min:x_max, :])
        bb_img_data = detect_objects(input_frame, self.sess, self.graph)
        return bb_img_data


# ### Siamese Model

# In[6]:


class SiameseModel:
    def __init__(self, path, verbose=1):
        self.path = path
        self.verbose = verbose
        self.timer = timer()
        self.setup()

    def setup(self):
        self.timer.tick()
        trained_checkpoint_prefix = self.path
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        # Restore from checkpoint
        loader = tf.train.import_meta_graph(trained_checkpoint_prefix + '.meta')
        loader.restore(sess, trained_checkpoint_prefix)
        graph = tf.get_default_graph()
        self.sess = sess
        self.X1 = graph.get_tensor_by_name("X1:0")
        self.T = graph.get_tensor_by_name("Training:0")
        self.Feature = graph.get_tensor_by_name("GAP:0")
        
        # Warmup
        test = np.zeros((128, 112, 112, 3))
        self.get_features(test)
        
        self.timer.tock()
        if self.verbose: print("INFO: Siamese model setup complete, time: %dms" % (self.timer.get_time()))
        
    def preprocess_imgs(self, imgs):
        resize_imgs = []
        for img in imgs:
            resize_img = cv2.resize(img, (112, 112))
            resize_imgs.append(resize_img)
        resize_imgs = np.array(resize_imgs)
        return resize_imgs
    
    def get_features(self, imgs, max_len=128):
        imgs = self.preprocess_imgs(imgs)
        output = []
        for i in range(0, len(imgs), max_len):
            feed_dict = {self.X1: imgs[i:min(i+max_len, len(imgs))],
                         self.T: False}
            feats = self.sess.run(self.Feature, feed_dict=feed_dict)
            output.append(feats)
        output = np.concatenate(output, axis=0)
        return output


# ### Visual Navigation App (Multi-Signboard Version)

# In[7]:


class VisualNavigationApp:
    def __init__(self, target_images, con, obj_model, siamese, threshold1, threshold2, 
                 valid_color, valid_bg_color=(255, 255, 255), invalid_color=None, invalid_bg_color=(255, 255, 255), 
                 bb_thickness=1, text_color=(255, 255, 255), text_font=cv2.FONT_HERSHEY_SIMPLEX, text_scale=0.5, 
                 text_thickness=1, verbose=1):
        self.target_images = target_images
        self.video_context = con
        self.object_detection_model = obj_model
        self.siamese_model = siamese
        self.threshold1 = threshold1
        self.threshold2 = threshold2
        self.valid_color = valid_color
        self.valid_bg_color = valid_bg_color
        self.invalid_color = invalid_color
        self.invalid_bg_color = invalid_bg_color
        self.text_color = text_color
        self.text_font = text_font
        self.text_scale = text_scale
        self.bb_thickness = bb_thickness
        self.text_thickness = text_thickness
        self.verbose = verbose
        self.score_track = []
        self.timer = timer()
        
    def setup_and_run(self):
        self.setup()
        self.run()
        
    def setup(self):
        self.timer.tick()
        self.bb_frames = []
        self.target_num = len(self.target_images)
        self.target_features = self.siamese_model.get_features(self.target_images)
        self.video_context.set_frame(0)
        self.kernel = self.discrete_gaussian_kernel(self.target_num)
        self.score_track = []
        self.timer.tock()
        if self.verbose: print("INFO: Visual navigation app setup complete, time: %dms" % (self.timer.get_time()))
            
    def discrete_gaussian_kernel(self, n, t=10):
        ns = np.arange(1, n+1)
        kernel = math.exp(-t) * scipy.special.iv(ns, t)
        kernel /= sum(kernel)
        return kernel
        
    def get_distances(self, features):
        distances = scipy.spatial.distance.cdist(features, self.target_features)
        return distances
    
    def get_match(self, distances):
        matches = []
        for i in range(len(distances[0])):
            match_id = np.argmin(distances[:, i])
            backtrack = np.argmin(distances[match_id])
            if (backtrack == i) and (distances[match_id, i] < self.threshold1):
                matches.append((match_id, i))
        return matches
    
    def get_match_score(self, matches, distances):
        if not matches:
            return self.threshold1
        scores = []
        for i in matches:
            scores.append(distances[i])
        scores += [self.threshold1 for i in range(self.target_num - len(matches))]
        scores = np.sort(scores)
        score = np.sum(scores * self.kernel)
        return score
        
    def draw_bounding_box(self, bb_frame, coord, fg_color, bg_color, bb_thickness, 
                          text, text_color, text_font, text_scale, text_thickness):
        y_min, y_max, x_min, x_max = coord
        (text_width, text_height) = cv2.getTextSize(text, text_font, fontScale=text_scale, thickness=text_thickness)[0]
        
        # Draw bounding box
        cv2.rectangle(bb_frame, (x_min, y_min), (x_max, y_max), fg_color, bb_thickness)
        
        # Draw text background & put text
        if text:
            cv2.rectangle(bb_frame, (x_min, y_max-text_height-2), (x_min+text_width, y_max), bg_color, cv2.FILLED)
            cv2.putText(bb_frame, text, (x_min, y_max-2), text_font, fontScale=text_scale, 
                        color=text_color, thickness=text_thickness)
            
    def put_match_image(self, bb_frame, coord, img, color, thickness, target_size=50):
        h, w, _ = img.shape
        y_min, y_max, x_min, x_max = coord
        target_h = min(target_size, y_max-y_min)
        target_w = min(target_size, x_max-x_min)
        scale = min(target_h/h, target_w/w)
        
        # Resize image & draw an outline
        nh, nw = int(h * scale), int(w * scale)
        img = cv2.resize(img, (nw, nh))
        cv2.rectangle(img, (0, 0), (nw-thickness, nh-thickness), color, thickness)
        
        # WOW! Nested try except! BEAUTIFUL
        try: # put image under bottom left corner
            bb_frame[y_max:y_max+nh, x_min:x_min+nw] = img
        except: 
            try: # put image under top left corner
                bb_frame[y_min:y_min+nh, x_min:x_min+nw] = img
            except:
                try: # put image under bottom right corner
                    bb_frame[y_max:y_max+nh, x_max:x_max+nw] = img
                except: 
                    try: # put image under top right corner
                        bb_frame[y_min:y_min+nh, x_max:x_max+nw] = img
                    except:
                        pass
                    
    def run(self, beta=0.0):
        global task
        task += 1
        maxtask = 0
        score = self.threshold1
        frame = self.video_context.get_next_frame()
        while frame is not None:
            maxtask = max(maxtask, task)
            self.timer.tick()
            bb_frame = frame.copy()
            
            # Detect signboards from frame
            bb_imgs = self.object_detection_model.detect(frame)
                
            if len(bb_imgs) > 0:
                coords = bb_imgs[:, 0]
                imgs = bb_imgs[:, 1]

                # Extract features
                features = self.siamese_model.get_features(imgs)

                # Compare signboards
                distances = self.get_distances(features)
                
                # Match signboards
                match = self.get_match(distances)
                cur_score = self.get_match_score(match, distances)
                self.score_track.append(cur_score)
                score = score * beta + cur_score * (1.0 - beta)
                
                valid_ids = np.array([i[0] for i in match])
                valid_match = np.array([i[1] for i in match])
                valid_len = len(valid_ids)
                if len(valid_ids): valid_coords = coords[valid_ids]
                valid_distances = [distances[i] for i in match]
                
                invalid_ids = []
                for i in range(len(imgs)): 
                    if i not in valid_ids:
                        invalid_ids.append(i)
                invalid_ids = np.array(invalid_ids)
                invalid_len = len(invalid_ids)
                if len(invalid_ids): invalid_coords = coords[invalid_ids]
                
                # Draw bounding boxes
                for i in range(valid_len):
                    text = "dis = %.5f" % (valid_distances[i])
                    self.draw_bounding_box(bb_frame, valid_coords[i], self.valid_color, self.valid_bg_color, 
                                           self.bb_thickness, text, self.text_color, self.text_font, 
                                           self.text_scale, self.text_thickness)
                    self.put_match_image(bb_frame, valid_coords[i], self.target_images[valid_match[i]], 
                                         self.valid_color, self.bb_thickness)

                if self.invalid_color is not None:
                    for i in range(invalid_len):
                        text = "" # "dis = %.5f" % (invalid_distances[i])
                        self.draw_bounding_box(bb_frame, invalid_coords[i], self.invalid_color, self.invalid_bg_color, 
                                               self.bb_thickness, text, self.text_color, self.text_font, 
                                               self.text_scale, self.text_thickness)
            else: # no signboards found
                score = score * beta + self.threshold1 * (1.0 - beta)
                
            # Temporary ui
            cv2.rectangle(bb_frame, (0, 0), (250, 40), self.invalid_bg_color, cv2.FILLED)
            cv2.putText(bb_frame, "score: %.5f" % score, (2, 30), self.text_font, fontScale=1, 
                        color=self.text_color, thickness=2)
            if score < self.threshold2:
                cv2.putText(bb_frame, "-TURN-", (250, 430), self.text_font, fontScale=5, 
                            color=(255, 0, 0), thickness=10)
            
            # Collect frames
            self.bb_frames.append(bb_frame)
            
            self.timer.tock()
            if self.verbose: print("INFO: Frame %d done, %d signboards detected, time: %dms" % (self.video_context.cur_frame, len(bb_imgs), self.timer.get_time()))

            # Get next frame
            frame = self.video_context.get_next_frame()
        task -= 1
        return maxtask

    def store_video(self, path):
        assert path[-4:] == ".mp4", "Only supports .mp4 currently"
        if self.verbose: print("INFO: Saving video to %s" % (path))
        self.timer.tick()
        
        fps = self.video_context.fps
        frame_size = self.video_context.frame_size
        out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'MP4V'), fps, frame_size)
        # out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'XVID'), fps, frame_size)
        
        for frame in self.bb_frames:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame)
        
        out.release()
        self.timer.tock()
        if self.verbose: print("INFO: Saving video done, time: %dms" % (self.timer.get_time()))


## grpc server

class visualNavigationModserviceServicer(visualNavigationMod_pb2_grpc.visualNavigationModserviceServicer):

    def __init__(self):
        self.preloadVideo = "./input/Cut2-030-U3_01_20190817_012500.mp4"
        self.preloadVideoSec = 0.9
        self.intersection_imgs = None
        self.inferenceVideo = "./input/Cut2-089-FY_01_20190825_082000.mp4"
        self.outputVideo = "./output/test.mp4"
        self.object_detection_model_path = "model/frozen_inference_graph.pb"
        self.siamese_model_path = "model/Siamese_tracking.ckpt"


    def init(self, request, context):

        # object_detection_model_path = "model/frozen_inference_graph.pb"
        # siamese_model_path = "model/Siamese_tracking.ckpt"
        object_detection_model_path = request.objDetectModel if request.objDetectModel else self.object_detection_model_path
        siamese_model_path = request.siameseModel if request.siameseModel else self.siamese_model_path

        self.object_detection_model = ObjectDetectionModel(object_detection_model_path, verbose=1)
        self.siamese_model = SiameseModel(siamese_model_path, verbose=1)

        video_path = request.input if request.input else self.preloadVideo
        intersection_sec = request.sec if request.sec else self.preloadVideoSec
        video_context = VideoContext(video_path)

        intersection_frame_num = int(intersection_sec * video_context.fps)
        print("Selected frame: %d" % intersection_frame_num)
        video_context.set_frame(intersection_frame_num)
        intersection_frame = video_context.get_next_frame()

        detect_out = self.object_detection_model.detect(intersection_frame)
        self.intersection_imgs = [i[1] for i in detect_out]
        return visualNavigationMod_pb2.initOutput(status = "ok")


        # """Missing associated documentation comment in .proto file."""
        # context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        # context.set_details('Method not implemented!')
        # raise NotImplementedError('Method not implemented!')

    def inference(self, request, context):
        video_path = request.input if request.input else self.inferenceVideo
        video_context = VideoContext(video_path)
        output_video_path = request.output if request.output else self.outputVideo

        # Run app
        threshold1 = 0.05
        threshold2 = 0.035
        valid_color = (0, 255, 0)
        valid_bg_color = (255, 255, 255)
        invalid_color = (0, 0, 0)
        invalid_bg_color = (255, 255, 255)
        bb_thickness = 2
        text_color = (0, 0, 0)
        text_font = cv2.FONT_HERSHEY_SIMPLEX
        text_scale = 0.5
        text_thickness = 1
        beta = 0.9

        app = VisualNavigationApp(self.intersection_imgs, video_context, self.object_detection_model, self.siamese_model, 
                                threshold1, threshold2, valid_color, valid_bg_color, invalid_color, invalid_bg_color, 
                                bb_thickness, text_color, text_font, text_scale, text_thickness, verbose=1)
        app.setup()
        app_timer = timer()
        app_timer.tick()
        maxtask = app.run(beta=beta) # beta is the smoothing term
        app_timer.tock()
        print("INFO: Total run time: %dms" % (app_timer.get_time()))
        print("[INFO] elapseTimePerFrame: %.0fms maxTask: %d" % (app_timer.get_time()/30.0, maxtask))
        app.store_video(output_video_path)

        track = app.score_track
        smooth_track = []
        score = threshold1
        for i in track:
            score = score * beta + i * (1.0-beta)
            smooth_track.append(score)

        # return smooth_track
        return visualNavigationMod_pb2.inferenceOutput(status="ok",score=random.random())

        # """Missing associated documentation comment in .proto file."""
        # context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        # context.set_details('Method not implemented!')
        # raise NotImplementedError('Method not implemented!')

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    visualNavigationMod_pb2_grpc.add_visualNavigationModserviceServicer_to_server(visualNavigationModserviceServicer(), server)
    server.add_insecure_port('0.0.0.0:50051')
    server.start()
    server.wait_for_termination()

serve()