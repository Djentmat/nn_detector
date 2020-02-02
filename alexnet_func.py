import cv2
import numpy as np

class AlexNetFunc():
    def __init__(self):
        self.deploy_path = '../_nn_detector/_nn_configs/_alexnet/'
        self.model_path = '../_nn_detector/_nn_configs/_alexnet/'
        self.deploy_name = 'deploy.prototxt'
        self.model_name = 'snapshot_iter_870.caffemodel'

    def get_alexnet(self):
        net = cv2.dnn.readNetFromCaffe(self.deploy_path + self.deploy_name,
                                       self.model_path + self.model_name)
        return net

    def detect_by_alexnet(self, image):
        net = self.get_alexnet()
        blob = cv2.dnn.blobFromImage(image,
                                     scalefactor=0.00392, size=(224, 224), mean=(0, 0, 0))
        net.setInput(blob)
        detections = net.forward()
        return detections

if __name__ == '__main__':
    test = AlexNetFunc()
    net = test.get_alexnet()
    print(net)
    im = cv2.imread('D:\_parking_slot_detection\_source\_datasets\CNRPARKIT\CNR-EXT-Patches-150x150\PATCHES\OVERCAST\/2015-11-20\camera1\O_2015-11-20_07.40_C01_229.jpg')
    result = test.detect_by_alexnet(im)
    print(np.round(result, 2))