from yolo_func import YOLOFunc
import glob
import os
import cv2


class YOLO(YOLOFunc):
    def __init__(self):
        super().__init__()
        path = 'D:/_activity/_programming/_parking_slot_detection/_source' \
               '/_nn_configs/_yolo_cfg/ '
        self.net = YOLOFunc()
        self.net.cfg_path = path + 'yolov3.cfg'
        self.net.weights_path = path + 'yolov3.weights'
        # self.net.target = cv2.dnn.DNN_TARGET_OPENCL
        self.net.yolo_config()


if __name__ == '__main__':
    subfolders = [f.path for f in os.scandir("_dataset") if f.is_dir()]
    print(subfolders)
    yolo_test = YOLO()
    for fld in range(len(subfolders)):
        img_list = glob.glob(subfolders[fld] + '/*.jpg')
        if not img_list:
            break
        for img in range(len(img_list)):
            image = cv2.imread(img_list[img])
            height, width, channels = image.shape
            ground_truth_bb = []
            try:
                with open(img_list[img].replace('.jpg', '.txt')) as f:
                    strings = f.readlines()
                    for i in range(len(strings)):
                        data = strings[i].split(' ')
                        occupied = data[0]
                        center_x = int(float(data[1]) * width)
                        center_y = int(float(data[2]) * height)
                        size_w = int(float(data[3]) * width)
                        size_h = int(float(data[4]) * height)
                        # print(occupied, center_x, center_y, size_w, size_h)
                        ground_truth_bb.append([int(center_x - size_w / 2),
                                                int(center_y - size_h / 2),
                                                size_w, size_h])
                        cv2.rectangle(image,
                                      (center_x - int(size_w / 2),
                                       center_y - int(size_h / 2)),
                                      (center_x + int(size_w / 2),
                                       center_y + int(size_h / 2)),
                                      (0, 255, 0), 1)
                    print('Number of ground-truth objects: %s' % str(
                        len(ground_truth_bb)))
                    cv2.imshow('1', image)
                    cv2.waitKey(0)
                    yolo_test = YOLOFunc()
                    yolo_test.cfg_path = 'D:/_activity/_programming' \
                                         '/_parking_slot_detection/_source' \
                                         '/_nn_configs/_yolo_cfg/yolov3.cfg '
                    yolo_test.weights_path = 'D:/_activity/_programming' \
                                             '/_parking_slot_detection' \
                                             '/_source/_nn_configs/_yolo_cfg' \
                                             '/yolov3.weights'
                    yolo_test.yolo_config()
                    yolo_test.get_blob(image)
                    result = yolo_test.yolo_detect()

                    boxes, confidenses, \
                    coordinates = yolo_test.data_of_detection(image,
                                                              result)

                    print('Number of predicted objects: %s' % str(len(boxes)))
                    # yolo_test.draw_result(image.copy(), boxes, confidenses)
                    iou_list = []
                    iou_confidences = []
                    for i in range(len(boxes)):
                        for j in range(len(ground_truth_bb)):

                            iou = yolo_test.bb_intersection_over_union(
                                boxes[i],
                                ground_truth_bb[j])
                            if iou > 0.6:
                                # DEBUG
                                # print(iou)
                                # print(boxes[i])
                                # print(ground_truth_bb[j])
                                # yolo_test.draw_result(image.copy(), boxes[i])
                                # yolo_test.draw_result(image.copy(),
                                #                       ground_truth_bb[j])
                                iou_list.append(boxes[i])
                                iou_confidences.append(iou)
                    print('Number of iou objects: %s' % str(len(iou_list)))
                    yolo_test.draw_result(image, iou_list, iou_confidences)
            except Exception as err:
                print(str(err))
