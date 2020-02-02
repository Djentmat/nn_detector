import numpy as np
import cv2
import mrcnn.config
import mrcnn.utils
from mrcnn.model import MaskRCNN


class MaskRCNNConfig(mrcnn.config.Config):
    NAME = "coco_pretrained_model_config"
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    NUM_CLASSES = 1 + 80
    DETECTION_MIN_CONFIDENCE = 0.6


def get_cars_bb_from_result(results):
    results_ext = results[0]
    boxes = results_ext['rois']
    class_ids = results_ext['class_ids']
    cars_bounding_boxes = []
    for i, box in enumerate(boxes):
        if class_ids[i] in [3, 8, 6]:
            cars_bounding_boxes.append(box)
    return np.array(cars_bounding_boxes)


def get_cars_masks_from_result(results):
    from PIL import Image
    results_ext = results[0]
    boxes = results_ext['rois']
    class_ids = results_ext['class_ids']
    scores = results_ext['scores']
    masks = results_ext['masks']
    im = Image.fromarray(masks[:, :, 0])
    im.show()
    print(len(boxes))
    print(len(class_ids))
    print(len(scores))
    print(len(masks))
    cars_bounding_boxes = []
    for i, box in enumerate(boxes):
        if class_ids[i] in [3, 8, 6]:
            cars_bounding_boxes.append(box)
    return np.array(cars_bounding_boxes)


class MRCNNFunc(MaskRCNNConfig):
    def __init__(self):
        super().__init__()
        self.cp_width = 1
        self. cp_color = (0, 255, 0)
        self.cp_size = 2
        self.cp_on = True
        self.bb_width = 1
        self.bb_color = (0, 0, 255)
        self.log_dir = '../_nn_detector/logs/'
        self.h5weights_path = "mask_rcnn_coco.h5"

    def get_mrcnn(self):
        model = MaskRCNN(mode="inference",
                         model_dir=self.log_dir,
                         config=MaskRCNNConfig())
        model.load_weights(self.h5weights_path, by_name=True)
        return model

    def detect_by_mrcnn(self, image):
        assert isinstance(image, object)
        net_init = self.get_mrcnn()
        results = net_init.detect([image], verbose=0)
        return results

    # TODO Write a func for detecting different objects, not for cars only
    # TODO Write a timer for operations
    # TODO Create a mask paint for images

    def draw_bb(self, image, boxes):
        for i in range(len(boxes)):
            y1, x1, y2, x2 = boxes[i]
            center_x = (int((x2 - x1) / 2) + x1)
            center_y = (int((y2 - y1) / 2) + y1)

            if self.cp_on:
                cv2.circle(image,
                           (center_x, center_y),
                           self.cp_size,
                           self.cp_color,
                           self.cp_width)

            cv2.rectangle(image,
                          (x1, y1),
                          (x2, y2),
                          self.bb_color, self.bb_width)
        return image

    #
    # def draw_masks(self, image_to_draw, masks):
    #     from PIL import Image
    #     image_out = image_to_draw.copy()
    #     for mask in masks:
    #         image_out = cv2.drawContours(image_out, mask)
    #     cv2.imshow('1', image_out)
    #     cv2.waitKey(0)
    #     blended = Image.blend(image_to_draw, image_out, alpha=0.5)
    #     blended.show()

if __name__ == '__main__':
    test = MRCNNFunc()
    test.h5weights_path = 'D:\_parking_slot_detection\_source\_nn_configs\_mrcnn_cfg\mask_rcnn_coco.h5'
    test.DETECTION_MIN_CONFIDENCE = 0.6
    net = test.get_mrcnn()
    print(net)
    im = cv2.imread('D:\_parking_slot_detection\_source\_datasets\CNRPARKIT\CNR-EXT_FULL_IMAGE_1000x750\FULL_IMAGE_1000x750\OVERCAST\/2015-11-16\camera1\/2015-11-16_1240.jpg')
    # im = cv2.resize(im)
    result = test.detect_by_mrcnn(im)
    bounding_boxes = get_cars_bb_from_result(result)
    masks = get_cars_masks_from_result(result)
    test.cp_on = False
    im = test.draw_bb(im, bounding_boxes)
    # test.draw_masks(im, masks)
    cv2.imshow('1', im)
    cv2.waitKey(0)
