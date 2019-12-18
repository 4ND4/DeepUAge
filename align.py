from collections import OrderedDict
import cv2
import numpy as np

FACIAL_LANDMARKS_IDXS = OrderedDict([
    ("mouth", (48, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 36)),
    ("jaw", (0, 17))
])


def align_face_dlib(image, shape):
    # extract the left and right eye (x, y)-coordinates

    (lStart, lEnd) = FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = FACIAL_LANDMARKS_IDXS["right_eye"]
    leftEyePts = shape[lStart:lEnd]
    rightEyePts = shape[rStart:rEnd]

    left_eye_center = leftEyePts.mean(axis=0).astype("int")
    right_eye_center = rightEyePts.mean(axis=0).astype("int")

    # compute the angle between the eye centroids

    if left_eye_center is not None and right_eye_center is not None:
        dY = right_eye_center[1] - left_eye_center[1]
        dX = right_eye_center[0] - left_eye_center[0]
        angle = np.degrees(np.arctan2(dY, dX)) - 180

        # compute center (x, y)-coordinates (i.e., the median point)
        # between the two eyes in the input image
        eyes_center = ((left_eye_center[0] + right_eye_center[0]) // 2,
                       (left_eye_center[1] + right_eye_center[1]) // 2)

        # grab the rotation matrix for rotating and scaling the face
        M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)

        if image is not None:
            result = cv2.warpAffine(image, M, image.shape[1::-1], flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))

            return result
    else:
        print('error with eye center values')
