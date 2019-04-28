import cvlib as cv
from cvlib.object_detection import draw_bbox
import cv2
import numpy as np
import modules.globals as g
import os


class Object:
    def __init__(self):
        g.log.debug('Initialized Object Detection')

    # runs yolov3 object detection
    def detect(self, f,ext, args):
        fi = f+ext
        fo = f+'-object'+ext
        print("Reading {}".format(fi))
        image = cv2.imread(fi)
        bbox, label, conf = cv.detect_common_objects(image)

        if not args['delete']:
            out = draw_bbox(image, bbox, label, conf)
            print("Writing {}".format(fo))
            cv2.imwrite(fo, out)

        detections = []

        for l, c, b in zip(label, conf, bbox):
            c = "{:.2f}%".format(c * 100)
            obj = {
                'type': l,
                'confidence': c,
                'box': b
            }
            detections.append(obj)

        if args['delete']:
            os.remove(fi)
        return detections
