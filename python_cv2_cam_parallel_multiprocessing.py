from multiprocessing.managers import BaseManager
import multiprocessing
import cv2
import os
import numpy as np
from utils import DataClass, FPS, Detection, display_in_thread, detect_in_thread, fetch_in_thread

class MyManager(BaseManager): pass

def Manager():
    m = MyManager()
    m.start()
    return m
 

MyManager.register('DataClass', DataClass)


'''
#fn_video = "data/20190125_125054.mp4"
#fn_video = "C:/Users/kevin/Videos/sunny_moon/20190125_125054.mp4"
#fn_video = "data/test.mp4"
is_this_video = True
'''
fn_video = 0
dir_yolo = "data"

th_confidence = 0.5
th_nms_iou = 0.3

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([dir_yolo, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")
 
# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")


manager = Manager()
class_data = manager.DataClass()
n_cpu = multiprocessing.cpu_count()
pool = multiprocessing.Pool(multiprocessing.cpu_count())
pool.apply_async(func = display_in_thread, args = (class_data, COLORS))
pool.apply_async(func = detect_in_thread, args = (class_data, dir_yolo, th_confidence, th_nms_iou, LABELS))
pool.apply_async(func = fetch_in_thread, args = (class_data, fn_video, 8))
pool.close()
pool.join()



