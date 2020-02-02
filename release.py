from yolo_func import YOLOFunc


class YOLO(YOLOFunc):
    def __init__(self):
        super().__init__()
        self.net = YOLOFunc()
        self.net.cfg_path = 'yolov3.cfg'
        self.net.weights_path = 'yolov3.weights'
        # self.net.target = cv2.dnn.DNN_TARGET_OPENCL
        self.net.yolo_config()


if __name__ == '__main__':
    yolo_test = YOLO()

