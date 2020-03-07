# -*- coding: utf-8 -*-
"""
@author: plakseev
"""

from yolo_func import YOLOFunc
import glob
import os
import cv2
import numpy as np

IMAGE_DEBUG = True
TEXT_DEBUG = True
IMAGE_EXT = 'jpg'

cfg_path = 'D:/_projects/_parking_slot_detection/_source' \
            '/_nn_configs/_yolo_cfg/yolov3.cfg'
weights_path = 'D:/_projects/_parking_slot_detection' \
               '/_source/_nn_configs/_yolo_cfg/yolov3.weights'

iou_threshhold = 0.5

gt_boxes_color = (0, 255, 0)
pred_bb_color = (0, 255, 0)
text_color = (0, 0, 255)

# =============================================================================
subfolders = [f.path for f in os.scandir("_dataset") if f.is_dir()]

if TEXT_DEBUG:
    print(subfolders)

if np.size(subfolders) > 0:
    for fld in range(len(subfolders)):
        dir_TP = []
        dir_FP = []
        dir_FN = []
        dir_precision = []
        dir_recall = []
        dir_f1_score = []
        img_list = glob.glob(subfolders[fld] + '/*.' + IMAGE_EXT)
        if np.size(img_list) == 0:
            print('Directory %s is empty!' % subfolders[fld])
            break

        for img in range(len(img_list)):
            TP = []
            FP = []
            FN = []
            precision = []
            recall = []
            f1_score = []

            image = cv2.imread(img_list[img])
            height, width, channels = image.shape
            ground_truth_bb = []
            try:
                with open(img_list[img].replace('.' + IMAGE_EXT,
                                                '.txt')) as f:
                    strings = f.readlines()
                    for i in range(len(strings)):
                        data = strings[i].split(' ')
                        occupied = data[0]
                        center_x = int(float(data[1]) * width)
                        center_y = int(float(data[2]) * height)
                        size_w = int(float(data[3]) * width)
                        size_h = int(float(data[4]) * height)
                        # DEBUG
                        # print(occupied, center_x, center_y, size_w, size_h)
                        ground_truth_bb.append([int(center_x - size_w / 2),
                                                int(center_y - size_h / 2),
                                                size_w, size_h])
                        if IMAGE_DEBUG:
                            cv2.rectangle(image,
                                          (center_x - int(size_w / 2),
                                           center_y - int(size_h / 2)),
                                          (center_x + int(size_w / 2),
                                           center_y + int(size_h / 2)),
                                          gt_boxes_color, 1)
                    if IMAGE_DEBUG:
                        cv2.imshow('1', image)
                        cv2.waitKey(0)
                    if TEXT_DEBUG:
                        print('Number of ground-truth objects: %s' % str(
                              len(ground_truth_bb)))

                    yolo_test = YOLOFunc()
                    yolo_test.cfg_path = cfg_path
                    yolo_test.weights_path = weights_path
                    yolo_test.yolo_config()
                    yolo_test.get_blob(image)
                    result = yolo_test.yolo_detect()

                    boxes, confidenses, coordinates = \
                        yolo_test.data_of_detection(image, result)
                    if TEXT_DEBUG:
                        print('Number of predicted objects: %s' % str(len(
                                boxes)))
                    if IMAGE_DEBUG:
                        yolo_test.draw_result(image.copy(), boxes,
                                              confidenses,
                                              bb_color=pred_bb_color,
                                              text_color=text_color)

                    iou_list = []
                    iou_confidences = []
                    for i in range(len(boxes)):
                        for j in range(len(ground_truth_bb)):

                            iou = yolo_test.bb_intersection_over_union(
                                    boxes[i],
                                    ground_truth_bb[j])
                            if iou > iou_threshhold:
                                # DEBUG
                                # print(iou)
                                # print(boxes[i])
                                # print(ground_truth_bb[j])
                                # yolo_test.draw_result(image.copy(), boxes[i])
                                # yolo_test.draw_result(image.copy(),
                                #                       ground_truth_bb[j])
                                iou_list.append(boxes[i])
                                iou_confidences.append(iou)
                                TP.append(boxes[i])
                    TP = len(iou_list)
                    FP = len(boxes) - len(iou_list)
                    FN = len(ground_truth_bb) - len(iou_list)
                    precision = TP / (TP + FP)
                    recall = TP / (TP + FN)
                    f1_score = 2 * ((precision * recall) / (precision +
                                                            recall))
                    if TEXT_DEBUG:
                        print('Number of iou objects: %s' % str(len(iou_list)))
                        print('TP = %s' % str(TP))
                        print('FP = %s' % str(FP))
                        print('FN = %s' % str(FN))
                        print('Precision = %s' % str(precision))
                        print('Recall = %s' % str(recall))
                        print('F1 score = %s' % str(f1_score))
                    dir_TP.append(TP)
                    dir_FP.append(FP)
                    dir_FN.append(FN)
                    dir_precision.append(precision)
                    dir_recall.append(recall)
                    dir_f1_score.append(f1_score)
                    yolo_test.draw_result(image, iou_list, iou_confidences,
                                          bb_color=pred_bb_color,
                                          text_color=text_color)
            except Exception as err:
                print(str(err) + '\n')

        subfolder_name = subfolders[fld].split('\\')
        with open(subfolders[fld] + '/_' + subfolder_name[1] +
                  '_result.txt', 'w') as f:
            f.write(f'total_tp = {str(np.mean(dir_TP))}' + '\n')
            f.write(f'total_fp = {str(np.mean(dir_FP))}' + '\n')
            f.write(f'total_fn = {str(np.mean(dir_FN))}' + '\n')
            f.write(f'total_precision = {str(np.mean(dir_precision))}' + '\n')
            f.write(f'total_recall = {str(np.mean(dir_recall))}' + '\n')
            f.write(f'total_f1_score = {str(np.mean(dir_f1_score))}' + '\n')
else:
    print('Error: No directories found in /_dataset/!')
