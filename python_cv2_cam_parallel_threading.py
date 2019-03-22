from threading import Thread#, Lock
import cv2
import os
import time
import numpy as np

from utils import DataClass, FPS, Detection, display_in_thread, detect_in_thread, fetch_in_thread, init_detection
       

fn_video, dir_yolo, th_confidence, th_nms_iou, COLORS, LABELS = init_detection()

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

