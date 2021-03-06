import datetime
import cv2
import os
import numpy as np
import time
from threading import Lock

class DataClass(object):

    def __init__(self):
        self.end_of_capture = False
        self.li_det = []
        self.im_rgb = None
        #self.im_rgb = 0
        #self.im_rgb_copy = None
        #self.im_rgb_copy = 0
        self.fps_det = None
        self.fps_fetch = None
        '''        
        self.lock_rgb = Lock()
        #self.lock_bbox = Lock()
        self.lock_li_rgb = Lock()
        self.lock_li_det = Lock()
        self.lock_fps_det = Lock()
        self.lock_fps_fetch = Lock()
        '''

    def get_eoc(self):
        return self.end_of_capture

    def set_eoc(self, is_eoc):
        self.end_of_capture = is_eoc

    def set_rgb(self, im_rgb):
        #time.sleep(0.01)
        #self.lock_rgb.acquire()
        self.im_rgb = im_rgb
        #self.im_rgb += 1
        #print('self.im_rgb is set by : ', str_by) 
        #print('self.im_rgb is ', self.im_rgb, ' set by : ', str_by)
        '''
        if self.im_rgb is None:
            print('self.im_rgb is None by : ', str_by)
        else:
            print('self.im_rgb is NOT None by : ', str_by)
        '''

    def get_rgb(self):
        #time.sleep(0.01)
        #self.lock_rgb.acquire()
        #print('self.im_rgb is ', self.im_rgb, ' gotton from : ', str_from)
        '''
        if self.im_rgb is None:
            print('self.im_rgb is None from : ', str_from)
        else:
            print('self.im_rgb is NOT None from : ', str_from)
        '''     
        #self.im_rgb_copy = self.im_rgb
        #finally:
            #self.lock_rgb.release()
        return self.im_rgb

    def set_li_rgb(self, li_im_rgb):
        #with self.lock_li_rgb:
        #    self.li_im_rgb = li_im_rgb
            #print('self.im_rgb is set by : ', str_from)
        self.li_im_rgb = li_im_rgb

    def get_li_rgb(self):
        #with self.lock_li_rgb:
        #    return self.li_im_rgb
        return self.li_im_rgb

    def set_li_det(self, li_det):
        #with self.lock_li_det:
        #    self.li_det = li_det
        self.li_det = li_det

    def get_li_det(self):
        #with self.lock_li_det:
        #    return self.li_det
        return self.li_det

    def set_fps_det(self, fps_det):
        #with self.lock_fps_det:
        #    self.fps_det = fps_det
        self.fps_det = fps_det

    def get_fps_det(self):
        #with self.lock_fps_det:
        #    return self.fps_det
        return self.fps_det

    def set_fps_fetch(self, fps_fetch):
        #with self.lock_fps_fetch:
        #    self.fps_fetch = fps_fetch
        self.fps_fetch = fps_fetch
    def get_fps_fetch(self):
        #with self.lock_fps_fetch:
        #    return self.fps_fetch
        return self.fps_fetch




class FPS:
    def __init__(self):
        # store the start time, end time, and total number of frames
        # that were examined between the start and end intervals
        self._start = None
        self._end = None
        self._numFrames = 0
                                             
    def start(self):
        # start the timer
        self._start = datetime.datetime.now()
        return self
                                                  
    def update(self):
        # increment the total number of frames examined during the
        # start and end intervals
        self._numFrames += 1

    def _elapsed(self):
        # return the total number of seconds between the start and
        # end interval
        return (datetime.datetime.now() - self._start).total_seconds()
    
    def fps(self):
        # compute the (approximate) frames per second
        return self._numFrames / self._elapsed()



class Detection:
    def __init__(self, x, y, w, h, class_id, label, confidence):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.class_id = class_id
        self.label = label
        self.confidence = confidence


def display_in_thread(class_data_proxy, COLORS, prephix):

    fps_disp = FPS().start()
    is_huda = False
    while not class_data_proxy.get_eoc():
        
        im_rgb = class_data_proxy.get_rgb()
    
        if im_rgb is None:
            if is_huda:
                class_data_proxy.set_eoc(True)
                #print('im_rgb of display is NOT None')
            else:
                #print('First frame of display thread has not been arrived')
                continue
        is_huda = True
        #time.sleep(0.5);        continue
        hei, wid = im_rgb.shape[:2]
        #print('fps_disp._numFrames : ', fps_disp._numFrames)
        #print('hei : ', hei)
        #print('wid : ', wid)
        li_det = class_data_proxy.get_li_det()
        im_bgr = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2BGR)
        #for i, det in enumerate(li_det):
        for idx, det in enumerate(li_det):
            #print('idx : ', idx)
            x, y, w, h = det.x, det.y, det.w, det.h
            color = [int(c) for c in COLORS[det.class_id]]
            cv2.rectangle(im_bgr, (x, y), (x + w, y + h), color, 1)
            text = "{}: {:.4f}".format(det.label, det.confidence)
            cv2.putText(im_bgr, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        #print('AAA det')
        fps_det = class_data_proxy.get_fps_det()
        if fps_det:
            text = "fps det : {:.1f}".format(fps_det)
            #print("fps det in display thread : {:.1f}".format(fps_det))
            cv2.putText(im_bgr, text, (int(wid * 0.35), hei - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 1)
        
        fps_fetch = class_data_proxy.get_fps_fetch()
        if prephix is not None:
            cv2.putText(im_bgr, prephix, (int(wid * 0.35), 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1)
        if fps_fetch is not None:
            text = "fps fetch : {:.1f}".format(fps_fetch)
            #print("fps fetch in display thread : {:.1f}".format(fps_fetch))
            cv2.putText(im_bgr, text, (int(wid * 0.35), hei - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1)

        fps_disp.update();
        text = "fps disp : {:.1f}".format(fps_disp.fps())
        cv2.putText(im_bgr, text, (int(wid * 0.35), hei - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)
        cv2.imshow('im_bgr', im_bgr)
        k = cv2.waitKey(1) & 0xFF
        if k == 27: # esc key
            cv2.destroyAllWindows()
            class_data_proxy.set_eoc(True)
        #elif k = ord('s'): # 's' key
            #cv2.imwrite('lenagray.png',img)
            #cv2.destroyAllWindow()
        #print('fps_display : ', fps_disp.fps())
    print("class_data.end_of_capture is True : display_in_thread") 
    #return class_data_proxy

def detect_in_thread(class_data_proxy, dir_yolo, th_confidence, th_nms_iou, LABELS):

    # derive the paths to the YOLO weights and model configuration
    #weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
    weightsPath = os.path.sep.join([dir_yolo, "yolov3.weights"])
###########################################################################################  
    #   download yolov3.weights into 'data' folder
    #wget https://pjreddie.com/media/files/yolov3.weights
###########################################################################################
    #configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])
    configPath = os.path.sep.join([dir_yolo, "yolov3.cfg"])
 
    # load our YOLO object detector trained on COCO dataset (80 classes)
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    print("[INFO] finish loading YOLO from disk...")
    ln = net.getLayerNames()
    print("[INFO] finish getLayerLames() ..")
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    print("[INFO] ln finished ..."); #exit()
    fps_det = FPS().start()
    print('class_data.end_of_capture of detect in thread : ', class_data_proxy.get_eoc())#; exit()
    
    is_huda = False
    while not class_data_proxy.get_eoc():
        im_rgb = class_data_proxy.get_rgb()
        #print("im_rgb : ", im_rgb)
        if im_rgb is None:
            if is_huda:
                class_data_proxy.set_eoc()
                print('class_data.end_of_capture of detect in thread is True'); #exit()
            continue
        is_huda = True
        #time.sleep(0.5);        continue
        #print('is_huda : ', is_huda); #exit()
        blob = cv2.dnn.blobFromImage(im_rgb, 1 / 255.0, (416, 416), swapRB=False, crop=False)
        #print('fps_det._numFrames : ', fps_det._numFrames); #exit()
        H, W = im_rgb.shape[:2]
        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln)
        end = time.time()

        # show timing information on YOLO
        print("[INFO] YOLO took {:.6f} seconds".format(end - start))

        # initialize our lists of detected bounding boxes, confidences, and
        # class IDs, respectively
        boxes = []
        confidences = []
        classIDs = []
        
        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability) of
                # the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                                                 
                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                #if confidence > args["confidence"]:
                if confidence > th_confidence:
                    # scale the bounding box coordinates back relative to the
                    # size of the image, keeping in mind that YOLO actually
                    # returns the center (x, y)-coordinates of the bounding
                    # box followed by the boxes' width and height
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    # use the center (x, y)-coordinates to derive the top and
                    # and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    # update our list of bounding box coordinates, confidences,
                    # and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        # apply non-maxima suppression to suppress weak, overlapping bounding
        # boxes
        #idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, th_confidence, th_nms_iou)


        li_det = []
        # ensure at least one detection exists
        if len(idxs) > 0:
        # loop over the indexes we are keeping
            for i in idxs.flatten():
                # extract the bounding box coordinates
                x, y, w, h = (boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3])
                class_id = classIDs[i]
                label = LABELS[class_id]
                confidence = confidences[i]
                det = Detection(x, y, w, h, class_id, label, confidence)
                li_det.append(det)
                # draw a bounding box rectangle and label on the image
                #color = [int(c) for c in COLORS[classIDs[i]]]
                #cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                #text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                #cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        class_data_proxy.set_li_det(li_det)
        fps_det.update();   class_data_proxy.set_fps_det(fps_det.fps())
        #print('fps_det : ', fps_det.fps())
    print("class_data.end_of_capture is True : detect_in_thread") 
    #return class_data_proxy



def fetch_in_thread(class_data_proxy, fn_video_or_cam, len_li_rgb):
    li_rgb = []
    is_huda = False
    print("fn_video_or_cam : ", fn_video_or_cam)
    kapture = cv2.VideoCapture(fn_video_or_cam)
    #kapture = cv2.VideoCapture(1)
    class_data_proxy.set_eoc(not kapture.isOpened())
    #if is_video:
    #    n_frame_total = int(kapture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_fetch = FPS().start()
    while not class_data_proxy.get_eoc():
        ret, im_bgr = kapture.read()
        if ret:
            #print('im_bgr is retrived in fetch thread');
            is_huda = True
            #cv2.imshow("temp", im_bgr); cv2.waitKey(10000)
            im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
            class_data_proxy.set_rgb(im_rgb)
            li_rgb.append(im_rgb)
            if len(li_rgb) >= len_li_rgb:
                class_data_proxy.set_li_rgb(li_rgb)
                del li_rgb[:]
        else:
            if is_huda:
                class_data_proxy.set_eoc(True) 
        #if is_video:
        #    idx_frame = int(kapture.get(cv2.CAP_PROP_POS_FRAMES))
        #    if idx_frame >= n_frame_total - 1:
        #        class_data_proxy.end_of_capture = True

        #print('fps_fetch._numFrames : ', fps_fetch._numFrames)
        fps_fetch.update();   class_data_proxy.set_fps_fetch(fps_fetch.fps())
        #print('fps_fetch : ', fps_fetch.fps())
   
        #time.sleep(0.5)

    print("class_data.end_of_capture is True : fetch_in_thread") 
    #return class_data_proxy


def init_detection():
    

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
    return fn_video, dir_yolo, th_confidence, th_nms_iou, COLORS, LABELS


