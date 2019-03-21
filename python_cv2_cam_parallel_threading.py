from threading import Thread#, Lock
import cv2
import os
import time
import numpy as np

from utils import DataClass, FPS, Detection, display_in_thread, detect_in_thread, fetch_in_thread
       
       
'''
#fn_video = "data/20190125_125054.mp4"
#fn_video = "C:/Users/kevin/Videos/sunny_moon/20190125_125054.mp4"
#fn_video = "data/test.mp4"
'''
fn_video = 0

th_confidence = 0.5
th_nms_iou = 0.3

# load the COCO class labels our YOLO model was trained on
dir_yolo = "data"
labelsPath = os.path.sep.join([dir_yolo, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")
 
# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")


class_data = DataClass()
thread_fetch  = Thread(target = fetch_in_thread, args = (class_data, fn_video, 8))
thread_fetch.start()
thread_detect  = Thread(target = detect_in_thread, args = (class_data, dir_yolo, th_confidence, th_nms_iou, LABELS))
thread_detect.start()

thread_display  = Thread(target = display_in_thread, args = (class_data, COLORS))
thread_display.start()


thread_fetch.join()
thread_detect.join()
thread_display.join()

