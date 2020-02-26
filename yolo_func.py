import cv2
import numpy as np
import time


class YOLOFunc:

    def __init__(self):
        self.net = None
        self.weights_path = ''
        self.cfg_path = ''
        self.backend = cv2.dnn.DNN_BACKEND_OPENCV
        self.target = cv2.dnn.DNN_TARGET_CPU
        self.layers = None
        self.mode = (608, 608)
        self.score_threshhold = 0.5
        self.nms_threshhold = 0.4

    def get_net(self):
        self.net = cv2.dnn.readNet(self.weights_path, self.cfg_path)

    def set_backend(self):
        self.net.setPreferableBackend(self.backend)

    def set_target(self):
        self.net.setPreferableTarget(self.target)

    def get_yolo_layers(self):
        layer_names = self.net.getLayerNames()
        self.layers = [layer_names[i[0] - 1]
                       for i in self.net.getUnconnectedOutLayers()]

    def yolo_config(self):
        self.get_net()
        self.set_backend()
        self.set_target()
        self.get_yolo_layers()

    def get_blob(self, image_to_blob):
        start_time = time.time()
        blob = cv2.dnn.blobFromImage(image_to_blob,
                                     0.00392,
                                     self.mode,
                                     (0, 0, 0),
                                     True,
                                     crop=False)
        self.net.setInput(blob)
        print("\nImage blobbed for %s seconds" % (time.time() - start_time))

    def yolo_detect(self):
        start_time = time.time()
        result_of_detect = self.net.forward(self.layers)
        print("Objects detected for %s seconds" % (time.time() - start_time))
        return result_of_detect

    def data_of_detection(self, image_for_detecting, result_of_detecting):
        start_time = time.time()
        height, width, channels = image_for_detecting.shape
        confidences = []
        bounding_boxes = []
        boxes_filtered = []
        confidences_filtered = []
        coordinates_of_center_bb = []
        coordinates_filtered = []

        for out in result_of_detecting:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                if x < 0:
                    x = 0
                if y < 0:
                    y = 0
                bounding_boxes.append([x, y, w, h])
                coordinates_of_center_bb.append([center_x, center_y])
                confidences.append(float(confidence))

        indexes = cv2.dnn.NMSBoxes(bounding_boxes, confidences,
                                   self.score_threshhold, self.nms_threshhold)

        for i in range(len(bounding_boxes)):
            if i in indexes:
                boxes_filtered.append(bounding_boxes[i])
                confidences_filtered.append(confidences[i])
                coordinates_filtered.append(coordinates_of_center_bb[i])
        print("Data received for %s seconds" % (time.time() - start_time))
        return boxes_filtered, confidences_filtered, coordinates_filtered

    @staticmethod
    def draw_result(image_to_draw, bounding_boxes, confidences=None):
        if isinstance(bounding_boxes[0], list):
            for i in range(len(bounding_boxes)):
                x, y, w, h = bounding_boxes[i]
                # center_x = int((x+x+w)/2)
                # center_y = int((y+y+h)/2)
                cv2.putText(image_to_draw,
                            str(np.round(confidences[i], 2)),
                            (x, y + h),
                            cv2.FONT_HERSHEY_COMPLEX,
                            1,
                            (0, 255, 0),
                            1,
                            cv2.FILLED, )
                if confidences:
                    cv2.rectangle(image_to_draw,
                                  (x, y),
                                  (x + w, y + h),
                                  (0, 0, 255),
                                  1)
            cv2.imshow('image', image_to_draw)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                pass
        else:
            x = bounding_boxes[0]
            y = bounding_boxes[1]
            w = bounding_boxes[2]
            h = bounding_boxes[3]
            # center_x = int((x+x+w)/2)
            # center_y = int((y+y+h)/2)
            if confidences:
                cv2.putText(image_to_draw,
                            str(np.round(confidences, 2)),
                            (x, y + h),
                            cv2.FONT_HERSHEY_COMPLEX,
                            1,
                            (0, 255, 0),
                            1,
                            cv2.FILLED, )
            cv2.rectangle(image_to_draw,
                          (x, y),
                          (x + w, y + h),
                          (0, 0, 255),
                          1)
        cv2.imshow('image', image_to_draw)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            pass

    def transform_coords(self, raw_bb):
        x_left_top = raw_bb[0] - int(raw_bb[2] / 2)
        y_left_top = raw_bb[1] - int(raw_bb[3] / 2)
        x_right_bottom = raw_bb[0] + int(raw_bb[2] / 2)
        y_right_bottom = raw_bb[1] + int(raw_bb[3] / 2)
        return [x_left_top, y_left_top, x_right_bottom, y_right_bottom]

    def bb_intersection_over_union(self, box1, box2):

        boxA = self.transform_coords(box1)
        boxB = self.transform_coords(box2)

        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        # return the intersection over union value
        return iou


if __name__ == '__main__':
    image = cv2.imread(
        'D:\_parking_slot_detection\_source\_images\_my_timelapse/frame256.png')
    frame_ROI = [0, 544,
                 608, 544 + 608]
    image = image[frame_ROI[1]:frame_ROI[3],
            frame_ROI[0]:frame_ROI[2]]
    yolo_test = YOLOFunc()
    yolo_test.cfg_path = 'D:\_parking_slot_detection\_source\_nn_configs\_yolo_cfg\yolov3.cfg'
    yolo_test.weights_path = 'D:\_parking_slot_detection\_source\_nn_configs\_yolo_cfg\yolov3.weights'
    yolo_test.yolo_config()
    yolo_test.get_blob(image)
    result = yolo_test.yolo_detect()
    boxes, confidenses, coordinates = yolo_test.data_of_detection(image,
                                                                  result)
    yolo_test.draw_result(image, boxes, confidenses)
