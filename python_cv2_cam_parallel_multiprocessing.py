from multiprocessing.managers import BaseManager
import multiprocessing
import cv2
import os
import numpy as np
from utils import DataClass, FPS, Detection, display_in_thread, detect_in_thread, fetch_in_thread, init_detection

class MyManager(BaseManager): pass

def Manager():
    m = MyManager()
    m.start()
    return m
 
fn_video, dir_yolo, th_confidence, th_nms_iou, COLORS, LABELS = init_detection()

MyManager.register('DataClass', DataClass)
manager = Manager()
class_data = manager.DataClass()

pool = multiprocessing.Pool(multiprocessing.cpu_count())

pool.apply_async(func = display_in_thread, args = (class_data, COLORS, 'multiprocessing'))
pool.apply_async(func = detect_in_thread, args = (class_data, dir_yolo, th_confidence, th_nms_iou, LABELS))
pool.apply_async(func = fetch_in_thread, args = (class_data, fn_video, 8))

pool.close()
pool.join()



