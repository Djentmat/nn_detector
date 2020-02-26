import time
from threading import Thread

import cv2

from yolo_func import YOLOFunc


class YOLO(YOLOFunc):
    def __init__(self):
        super().__init__()
        self.net = YOLOFunc()
        self.net.cfg_path = 'D:\_activity\_programming\_parking_slot_detection\_source\_nn_configs\_yolo_cfg/yolov3.cfg'
        self.net.weights_path = 'D:\_activity\_programming\_parking_slot_detection\_source\_nn_configs\_yolo_cfg/yolov3.weights'
        # self.net.target = cv2.dnn.DNN_TARGET_OPENCL
        self.net.yolo_config()

    def detect_cars(self, image):
        self.net.get_blob(image)
        result = self.net.yolo_detect()
        boxes, confidenses, coordinates = self.net.data_of_detection(
            image, result)
        print('%s bounding_boxes found' % len(boxes))


if __name__ == '__main__':

    start_time = time.time()
    yolo1 = YOLO()
    print("Yolo1 configuration for %s seconds" % (time.time() - start_time))

    start_time = time.time()
    yolo2 = YOLO()
    print("Yolo2 configuration for %s seconds" % (time.time() - start_time))


    def read_and_detect():
        success = True
        count = 0
        vidcap = cv2.VideoCapture(
            'D:\_activity\_programming\_parking_slot_detection\_source\_video/parking_test1.mp4')
        while success:
            start_timer = time.time()
            success, image = vidcap.read()
            count += 1
            yolo1.detect_cars(image)
            print("Yolo1 total detect time: %s seconds" % (
                    time.time() - start_timer))


    def read_and_detect2():
        success = True
        count = 0
        vidcap = cv2.VideoCapture(
            'D:\_activity\_programming\_parking_slot_detection\_source\_video/parking_test1.mp4')
        while success:
            start_timer = time.time()
            success, image = vidcap.read()
            count += 1
            yolo2.detect_cars(image)
            print("Yolo2 total detect time: %s seconds" % (
                    time.time() - start_timer))


    thread1 = Thread(target=read_and_detect)
    thread2 = Thread(target=read_and_detect2)
    thread1.start()
    thread2.start()
