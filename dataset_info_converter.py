# -*- coding: utf-8 -*-
import xml.etree.ElementTree as ET
import os
import cv2
import glob


def xml_to_txt(xml_path):
    # xml_path = 'D:\_activity\_programming\_parking_slot_detection\_source\_datasets\CNRPARKIT\PKLot\PKLot\PKLot\PUCPR\Cloudy/2012-09-12/2012-09-12_07_07_55.xml'
    file_name = os.path.split(xml_path)[1]
    image = cv2.imread(xml_path[:-3] + 'jpg')
    
    height, width, channels = image.shape
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    line_for_rec = []
    center_x = 0
    center_y = 0
    size_h = 0
    size_w = 0
    
    with open(xml_path[:-3] + 'txt', 'w') as file:
        for i, child in enumerate(root):
            print(i, str(child.attrib))
            try:
                occupied = str(int(child.attrib['occupied']))
                for occupied_child in child:
                    if occupied_child.tag == 'rotatedRect':
                        for rect_child in occupied_child:
                            if rect_child.tag == 'center':
                                center_x = int(rect_child.attrib['x'])
                                center_y = int(rect_child.attrib['y'])
                            if rect_child.tag == 'size':
                                size_w = int(rect_child.attrib['w'])
                                size_h = int(rect_child.attrib['h'])
                                # cv2.rectangle(image,
                                #               (center_x-int(size_w/2), center_y-int(size_h/2)),
                                #               (center_x + int(size_w/2), center_y + int(size_h/2)), (0, 0, 255), 1)
                                # cv2.imshow('1', image)
                                # cv2.waitKey(0)
                file.write(occupied + ' ' + \
                               str(center_x/width) + ' ' + \
                               str(center_y/height) + ' ' + \
                               str(size_w/width) + ' ' + \
                               str(size_h/height) + '\r')
            except:
                pass

                        # line_for_rec.append([occupied,
                        #                      center_x/width,
                        #                      center_y/height,
                        #                      size_w/width, size_h/height])

if __name__ == '__main__':
    from pathlib import Path
    for filename in Path('D:\_activity\_programming\_parking_slot_detection\_source\_datasets\CNRPARKIT\PKLot\PKLot\PKLot').rglob('*.xml'):
        print('Loaded %s' % os.path.split(filename)[1])
        xml_to_txt(str(filename))