import cvlib as cv
import cv2
import numpy as np
import modules.globals as g
import os


class Face:
    def __init__(self):
        g.log.debug('Initialized Face')

    def detect(self, f,ext, args):
        fi = f+ext
        fo = f+'-face'+ext
        g.log.debug("Reading {}".format(fi))
        image = cv2.imread(fi)
        faces, conf = cv.detect_face(image)

        detections = []
        for f, c in zip(faces, conf):
            c = "{:.2f}%".format(c * 100)

            (startX, startY) = f[0], f[1]
            (endX, endY) = f[2], f[3]
            cv2.rectangle(image, (startX, startY),
                          (endX, endY), (0, 255, 0), 2)
            rect = [int(startX), int(startY), int(endX), int(endY)]

            obj = {
                'type': 'face',
                'confidence': c,
                'box': rect
            }

            if args['gender']:
                face_crop = np.copy(image[startY:endY, startX:endX])
                (gender_label_arr, gender_confidence_arr) = cv.detect_gender(face_crop)
                idx = np.argmax(gender_confidence_arr)
                gender_label = gender_label_arr[idx]
                gender_confidence = "{:.2f}%".format(
                    gender_confidence_arr[idx] * 100)
                obj['gender'] = gender_label
                obj['gender_confidence'] = gender_confidence
                Y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.putText(image, gender_label, (startX, Y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            detections.append(obj)

        if not args['delete']:
            g.log.debug("Writing {}".format(fo))
            cv2.imwrite(fo, image)

        if args['delete']:
            os.remove(fi)

        return detections
