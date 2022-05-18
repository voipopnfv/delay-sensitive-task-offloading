#!/bin/sh
# python object_detection_app-grpcserver.py \
python object_detection_app.py \
--model_path model/frozen_inference_graph.pb \
--label_path label_class.pbtxt \
--num_classes 5 \
--video_path Video/onesec.mp4 \
--overlap_output_dir overlap/ \
--overlap_output 'no' \
--detected_output 'yes' \
--detected_output_dir output \
--output_labeled_video_path output/onesec.mp4 \
--gpudev 0


# --video_path Video/293-FM_01_20181115_073900.mp4 \
# --output_labeled_video_path output/output.mp4 \