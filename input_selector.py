import cv2
import pafy
import glob


class InputSelector():
    def __init__(self):
        self.image_path = '../_nn_detector/_test_images/'
        self.video_path = '../_nn_detector/_test_videos/'
        self.url = 'https://www.youtube.com/watch?v=3M4BR5s03K8'

    def select_image_input(self, im_type):
        files_list = glob.glob(self.image_path + '*.' + im_type)
        images_list = []
        for i in range(len(files_list)):
            files_list[i] = files_list[i].replace('\\', '//')
            images_list.append(cv2.imread(files_list[i]))
        return images_list

    def select_video_input(self, file_name):
        cap = cv2.VideoCapture(self.video_path + file_name)
        return cap

    def select_pafy_input(self, url):
        if not url:
            url = self.url
        vPafy = pafy.new(url)
        play = vPafy.getbest()
        return play

if __name__ == '__main__':

    test = InputSelector()
    test.image_path = 'D:\_parking_slot_detection\_source\_images\_my_timelapse/'
    test.video_path = 'D:\_parking_slot_detection\_source\_video/'

# paf = test.select_pafy_input('')
# cap = test.select_video_input('parking_test1.mp4')
# print(cap, paf)

# images = test.select_image_input('png')
#
# print(len(images))
# for image_to_draw in images:
#     cv2.imshow('1', image_to_draw)
#     cv2.waitKey(0)
