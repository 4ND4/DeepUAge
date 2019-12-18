import os
from imutils import face_utils
import numpy as np
import dlib
import cv2
from align import align_face_dlib
import argparse
import config
import math

SHAPE_PREDICTOR = os.path.expanduser(config.shape_predictor_path)


def face_remap(shape_input):
    remapped_image = cv2.convexHull(shape_input)
    return remapped_image


parser = argparse.ArgumentParser(description='DCA face cropping tool')
parser.add_argument("--f", default='test.jpg', type=str, help="path of image")
parser.add_argument("--o", default='output.jpg', type=str, help="output path of cropped image")

args = parser.parse_args()


"""
MAIN CODE STARTS HERE
"""
# load the input image, resize it, and convert it to greyscale

file_path = args.f
output_path = args.o

if not os.path.exists(file_path):
    print('file does not exist')
    exit(0)
else:
    image = cv2.imread(file_path)
    out_face = np.zeros_like(image)

    # initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(SHAPE_PREDICTOR)

    # detect faces in greyscale

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 1)

    if len(rects) == 0:
        print('no face detected')

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        """
       Determine the facial landmarks for the face region, then convert the facial landmark (x, y)-coordinates to a NumPy array
       """
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        aligned_face = align_face_dlib(image, shape)
        aligned_gray = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2GRAY)
        aligned_rects = detector(aligned_gray, 1)

        for (j, aligned_rect) in enumerate(aligned_rects):
            aligned_shape = predictor(aligned_gray, aligned_rect)
            aligned_shape = face_utils.shape_to_np(aligned_shape)

            # initialize mask array
            feature_mask = np.zeros((aligned_face.shape[0], aligned_face.shape[1]))

            # we extract the face

            # add points to aligned_shape

            p1 = aligned_shape[15]
            p2 = aligned_shape[34]

            distance_1 = math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))

            # print(distance_1)

            p1 = aligned_shape[3]
            p2 = aligned_shape[34]

            distance_2 = math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))

            # print(distance_2)

            avg_distance = (distance_1 + distance_2) / 2

            w = h = int(avg_distance * 2)

            x = aligned_shape[34][0] - int(avg_distance)
            y = aligned_shape[34][1] - h

            vertex1 = (x, y)

            vertex2 = (x + w, y + h)

            color = (255, 0, 0)

            center_point_x = aligned_shape[34][0]
            center_point_y = aligned_shape[34][1] - int(avg_distance)

            # define polygon sides (ICOSAGON)

            N = 20

            # rotate points also with angle used in align

            p = np.zeros((11, 2))
            for k in range(0, N):
                if k >= 11:
                    break

                p[k][0] = int(center_point_x - int(avg_distance) * math.cos(k * 2 * math.pi / N))
                p[k][1] = int(center_point_y - int(avg_distance) * math.sin(k * 2 * math.pi / N))

                p[k][0] = p[k][0] if p[k][0] > 0 else 0
                p[k][1] = p[k][1] if p[k][1] > 0 else 0

            m = 0
            for l in range(18, 27+1):
                aligned_shape[l][0] = p[m][0]
                aligned_shape[l][1] = p[m][1]
                m += 1

            counter = 0
            for item in aligned_shape:

                aligned_shape[counter][0] = aligned_shape[counter][0] if aligned_shape[counter][0] > 0 else 0
                aligned_shape[counter][1] = aligned_shape[counter][1] if aligned_shape[counter][1] > 0 else 0

                counter += 1

            remapped_shape = face_remap(aligned_shape)

            c = remapped_shape[0:27]

            cv2.fillConvexPoly(feature_mask, c, 1)

            extLeft = tuple(c[c[:, :, 0].argmin()][0])
            extRight = tuple(c[c[:, :, 0].argmax()][0])
            extTop = tuple(c[c[:, :, 1].argmin()][0])
            extBot = tuple(c[c[:, :, 1].argmax()][0])

            feature_mask = feature_mask.astype(np.bool)
            out_face[feature_mask] = aligned_face[feature_mask]

            crop = out_face[extTop[1]:extBot[1], extLeft[0]:extRight[0]]
            cv2.imwrite(output_path, crop)

            print('cropped', file_path)
