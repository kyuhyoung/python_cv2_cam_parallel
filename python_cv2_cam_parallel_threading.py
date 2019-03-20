from threading import Thread, Lock
import cv2
import os
import time
import numpy as np

#g_end_of_capture = False

# import the necessary packages
import datetime
 
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


class DataClass:
    def __init__(self):
        self.end_of_capture = False
        self.li_det = []
        self.im_rgb = None
        self.fps_det = None
        self.fps_fetch = None
        #'''        
        self.lock_rgb = Lock()
        #self.lock_bbox = Lock()
        self.lock_li_rgb = Lock()
        self.lock_li_det = Lock()
        self.lock_fps_det = Lock()
        self.lock_fps_fetch = Lock()
        #'''


    def set_rgb(self, im_rgb):
        with self.lock_rgb:
           self.im_rgb = im_rgb
        #self.im_rgb = im_rgb
        return
    def get_rgb(self):
        with self.lock_rgb:
           return self.im_rgb
        #return self.im_rgb

    def set_li_rgb(self, li_im_rgb):
        with self.lock_li_rgb:
            self.li_im_rgb = li_im_rgb
        #self.li_im_rgb = li_im_rgb
    def get_li_rgb(self):
        with self.lock_li_rgb:
            return self.li_im_rgb
        #return self.li_im_rgb

    def set_li_det(self, li_det):
        with self.lock_li_det:
            self.li_det = li_det
        #self.li_det = li_det
    def get_li_det(self):
        with self.lock_li_det:
            return self.li_det
        #return self.li_det

    def set_fps_det(self, fps_det):
        with self.lock_fps_det:
            self.fps_det = fps_det
        #self.fps_det = fps_det
    def get_fps_det(self):
        with self.lock_fps_det:
            return self.fps_det
        #return self.fps_det

    def set_fps_fetch(self, fps_fetch):
        with self.lock_fps_fetch:
            self.fps_fetch = fps_fetch
        #self.fps_fetch = fps_fetch
    def get_fps_fetch(self):
        with self.lock_fps_fetch:
            return self.fps_fetch
        #return self.fps_fetch

def display_in_thread(class_data, COLORS):

    fps_disp = FPS().start()
    is_huda = False
    while not class_data.end_of_capture:

        im_rgb = class_data.get_rgb()
    
        if im_rgb is None:
            if is_huda:
                class_data.end_of_capture = True
                print('No more image for display')
                print('No more image for display')
                print('No more image for display')
                print('No more image for display')
                print('No more image for display')
                print('im_rgb of display is NOT None')
            else:
                #print('First frame of display thread has not been arrived')
                continue
        is_huda = True
        hei, wid = im_rgb.shape[:2]
        #print('fps_disp._numFrames : ', fps_disp._numFrames)
        #print('hei : ', hei)
        #print('wid : ', wid)
        li_det = class_data.get_li_det()
        im_bgr = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2BGR)
        #for i, det in enumerate(li_det):
        for idx, det in enumerate(li_det):
            #print('idx : ', idx)
            x, y, w, h = det.x, det.y, det.w, det.h
            color = [int(c) for c in COLORS[det.class_id]]
            cv2.rectangle(im_bgr, (x, y), (x + w, y + h), color, 1)
            text = "{}: {:.4f}".format(det.label, det.confidence)
            cv2.putText(im_bgr, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        fps_det = class_data.get_fps_det()
        if fps_det:
            text = "fps det : {:.1f}".format(fps_det)
            #print("fps det in display thread : {:.1f}".format(fps_det))
            cv2.putText(im_bgr, text, (int(wid * 0.4), hei - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 1)
        
        fps_fetch = class_data.get_fps_fetch()
        if fps_fetch is not None:
            text = "fps fetch : {:.1f}".format(fps_fetch)
            #print("fps fetch in display thread : {:.1f}".format(fps_fetch))
            cv2.putText(im_bgr, text, (int(wid * 0.4), hei - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1)

        fps_disp.update();
        text = "fps disp : {:.1f}".format(fps_disp.fps())
        cv2.putText(im_bgr, text, (int(wid * 0.4), hei - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)
        cv2.imshow('im_bgr', im_bgr)
        k = cv2.waitKey(1) & 0xFF
        if k == 27: # esc key
            cv2.destroyAllWindows()
            class_data.end_of_capture = True
        #elif k = ord('s'): # 's' key
            #cv2.imwrite('lenagray.png',img)
            #cv2.destroyAllWindow()
        #print('fps_display : ', fps_disp.fps())
    print("class_data.end_of_capture is True : display_in_thread") 

def detect_in_thread(class_data, dir_yolo, th_confidence, th_nms_iou, LABEL):

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
    print('class_data.end_of_capture of detect in thread : ', class_data.end_of_capture); #exit()
    is_huda = False
    while not class_data.end_of_capture:
        im_rgb = class_data.get_rgb()
        #print("im_rgb : ", im_rgb)
        if im_rgb is None:
            if is_huda:
                class_data.end_of_capture = True
                print('class_data.end_of_capture of detect in thread is True'); #exit()
            continue
        is_huda = True
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
        class_data.set_li_det(li_det)
        fps_det.update();   class_data.set_fps_det(fps_det.fps())
        #print('fps_det : ', fps_det.fps())
    print("class_data.end_of_capture is True : detect_in_thread") 
        
def fetch_in_thread(class_data, fn_video_or_cam, len_li_rgb, is_video):
    li_rgb = []
    is_huda = False
    print("fn_video_or_cam : ", fn_video_or_cam)
    kapture = cv2.VideoCapture(fn_video_or_cam)
    #kapture = cv2.VideoCapture(1)
    class_data.end_of_capture = not kapture.isOpened()
    if is_video:
        n_frame_total = int(kapture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_fetch = FPS().start()
    while not class_data.end_of_capture:
        #print('class_data.end_of_capture is True : fetch thread');
        ret, im_bgr = kapture.read()
        if ret:
            is_huda = True
            im_rgb = class_data.set_rgb(cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB))
            li_rgb.append(im_rgb)
            if len(li_rgb) >= len_li_rgb:
                class_data.set_li_rgb(li_rgb)
                del li_rgb[:]
        else:
            if is_huda:
                class_data.end_of_capture = True 
        if is_video:
            idx_frame = int(kapture.get(cv2.CAP_PROP_POS_FRAMES))
            if idx_frame >= n_frame_total - 1:
                class_data.end_of_capture = True

        #print('fps_fetch._numFrames : ', fps_fetch._numFrames)
        fps_fetch.update();   class_data.set_fps_fetch(fps_fetch.fps())
        #print('fps_fetch : ', fps_fetch.fps())
   
    print("class_data.end_of_capture is True : fetch_in_thread") 




'''
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
    help="path to input video")
    ap.add_argument("-o", "--output", required=True,
        help="path to output video")
        ap.add_argument("-y", "--yolo", required=True,
            help="base path to YOLO directory")
            ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
                ap.add_argument("-t", "--threshold", type=float, default=0.3,
                    help="threshold when applyong non-maxima suppression")
                    args = vars(ap.parse_args())

'''
        
#def main():
#fn_video = "C:/Users/kevin/Videos/sunny_moon/20190125_125054.mp4"
dir_yolo = "data"
'''
#fn_video = "data/20190125_125054.mp4"
#fn_video = "data/test.mp4"
is_this_video = True
'''
fn_video = 0
is_this_video = False

th_confidence = 0.5
th_nms_iou = 0.3

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([dir_yolo, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")
 
# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")


class_data = DataClass()
thread_fetch  = Thread(target = fetch_in_thread, args = (class_data, fn_video, 8, is_this_video))
thread_fetch.start()
thread_detect  = Thread(target = detect_in_thread, args = (class_data, dir_yolo, th_confidence, th_nms_iou, LABELS))
thread_detect.start()

thread_display  = Thread(target = display_in_thread, args = (class_data, COLORS))
thread_display.start()
#display_in_thread(class_data, COLORS)


'''
fps_disp = FPS().start()
is_huda = False
while not class_data.end_of_capture:

    im_rgb = class_data.get_rgb()
    
    if im_rgb is None:
        if is_huda:
            class_data.end_of_capture = True
            print('im_rgb of display is NOT None')
        else:
            #print('First frame of display thread has not been arrived')
            continue
    is_huda = True
    hei, wid = im_rgb.shape[:2]
    print('fps_disp._numFrames : ', fps_disp._numFrames)
    #print('hei : ', hei)
    #print('wid : ', wid)
    li_det = class_data.get_li_det()
    im_bgr = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2BGR)
    #for i, det in enumerate(li_det):
    for idx, det in enumerate(li_det):
        print('idx : ', idx)
        x, y, w, h = det.x, det.y, det.w, det.h
        color = [int(c) for c in COLORS[det.class_id]]
        cv2.rectangle(im_bgr, (x, y), (x + w, y + h), color, 2)
        text = "{}: {:.4f}".format(det.label, det.confidence)
        cv2.putText(im_bgr, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    fps_det = class_data.get_fps_det()
    if fps_det:
        text = "fps det : {:.1f}".format(fps_det)
        print("fps det in display thread : {:.1f}".format(fps_det))
        cv2.putText(im_bgr, text, (int(wid * 0.4), hei - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
    fps_fetch = class_data.get_fps_fetch()
    if fps_fetch is not None:
        text = "fps fetch : {:.1f}".format(fps_fetch)
        print("fps fetch in display thread : {:.1f}".format(fps_fetch))
        cv2.putText(im_bgr, text, (int(wid * 0.4), hei - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    fps_disp.update();
    text = "fps disp : {:.1f}".format(fps_disp.fps())
    cv2.putText(im_bgr, text, (int(wid * 0.4), hei - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow('im_bgr', im_bgr)
    k = cv2.waitKey(1) & 0xFF
    if k == 27: # esc key
        cv2.destroyAllWindow()
        class_data.end_of_capture = True
        print('class_data.end_of_capture is True')
    print('fps_display : ', fps_disp.fps())
print("class_data.end_of_capture is True : display_in_thread") 
'''




thread_fetch.join()
thread_detect.join()
thread_display.join()
'''
if __main__ == "__main__":
    main()
'''

