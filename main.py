import math

import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import time as ts
import copy
from scipy.cluster.hierarchy import linkage, fcluster
from bisect import bisect_left

MAX_DIST = 25
classes = ['Fist', 'L', 'Ok', 'Palm', 'Peace']  # 'Nothing']
# classes = [
#     'A',
#     'B',
#     'C',
#     'D',
#     'E',
#     'F',
#     'G',
#     'H',
#     'I',
#     'J',
#     'L',
#     'O',
#     'P',
#     'Q',
#     'R',
#     'T',
#     'U',
#     'V',
#     'W',
#     'X',
#     'Y',
#     'Z',
#     'nothing'
# ]
#classes = ['A', 'B', 'C', 'D', 'G', 'H', 'I', 'L', 'V', 'Y']
# classes = ['Thump down', 'Palm (H)', 'L', 'Fist (H)', 'Fist (V)', 'Thumb Up', 'Index', 'Ok', 'Palm (V)', 'C']

cap = cv2.VideoCapture(0)


def get_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def get_thresh(img, blur=27, threshold=50):
    gray = get_gray(img)
    gray = cv2.GaussianBlur(gray, (blur, blur), 0)
    ret, threshold_image = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return threshold_image


def remove_background(frame, fgbg, learningRate=0):
    fgmask = fgbg.apply(frame, learningRate=learningRate)
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res


def load_model(path):
    return tf.keras.models.load_model(path + 'saved_models/' + 'VGG_cross_validated.h5',
                                      custom_objects={'KerasLayer': hub.KerasLayer})


def predict_thresh(thresh, model):
    target = np.stack((thresh,) * 3, axis=-1)
    target = cv2.resize(target, (224, 224))
    target = target.reshape(1, 224, 224, 3)
    prediction = model.predict(target)
    return classes[int(np.argmax(prediction))], prediction[0][np.argmax(prediction)]


def predict_gray(gray, model):
    target = gray
    target = cv2.resize(target, (120, 320))
    target = target.reshape(1, 120, 320, 1)
    prediction = model.predict(target)
    return classes[int(np.argmax(prediction))], prediction[0][np.argmax(prediction)]


def pt_dist(p1, p2):
    return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2


def pred(p1, p2):
    return pt_dist(p1, p2) <= MAX_DIST


def group_points(points, max_dst):
    if len(points) < 2:
        return []
    # points = np.float32(points).reshape((points.shape[0], points.shape[2]))

    data = linkage(points,
                   method='complete',
                   metric='euclidean'
                   )
    clusters = fcluster(data, max_dst, criterion='distance')

    groups = [[] for _ in range(len(clusters))]

    for idx, i in enumerate(clusters):
        groups[i - 1].append(tuple(points[idx]))

    res = []

    for gr in groups:
        if len(gr) > 0:
            mid = (sum([x[0] for x in gr]) / len(gr), sum([y[1] for y in gr]) / len(gr))
            pos = bisect_left(sorted(gr), mid)
            res.append(gr[pos])

    return res


def filter_vertices_byangle(points, contour, max_deg):
    res = []
    if points is not None and cv2.contourArea(contour) >= 5000:
        for i in range(len(points)):
            s, e, f, d = points[i]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])
            a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 180 / math.pi
            if angle < max_deg:
                res.append(far)
    return res


def get_contuors(frame, thresh):
    thresh_contours = copy.deepcopy(thresh)
    no_back_drawing = np.zeros(frame.shape, np.uint8)
    drawing = frame

    contours, hierarchy = cv2.findContours(thresh_contours, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    length = len(contours)
    max_area = -1

    if length > 0:
        ci = 0
        for i in range(length):  # find the biggest contour (according to area)
            temp = contours[i]
            area = cv2.contourArea(temp)
            if area > max_area:
                max_area = area
                ci = i

        max_contour = contours[ci]

        hull = cv2.convexHull(max_contour, returnPoints=False)

        hull_p = np.array([max_contour[i] for i in hull])
        hull_p = hull_p.reshape(hull_p.shape[0], hull_p.shape[3])

        points = group_points(hull_p, MAX_DIST)

        points = np.array(points).astype('float32')

        defects = cv2.convexityDefects(max_contour, hull)

        cv2.drawContours(no_back_drawing, [max_contour], 0, (0, 255, 0), 1)
        cv2.drawContours(no_back_drawing, [hull_p], 0, (0, 0, 255), 1)

        cv2.drawContours(drawing, [max_contour], 0, (0, 255, 0), 1)
        cv2.drawContours(drawing, [hull_p], 0, (0, 0, 255), 1)

        for pt in points:
            cv2.circle(no_back_drawing, tuple(pt), 8, (255, 127, 0), 3)
            cv2.circle(drawing, tuple(pt), 8, (255, 127, 0), 3)


        if defects is not None:
            defects = defects.reshape(defects.shape[0], defects.shape[2])
        else:
            defects = np.array([])

        def_p = []
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i]
            far = tuple(max_contour[f][0])
            def_p.append(far)

        def_p = filter_vertices_byangle(defects, max_contour, 60)
        def_p = group_points(def_p, MAX_DIST / 3 * 2)

        for pt in def_p:
            cv2.circle(no_back_drawing, tuple(pt), 5, (0, 45, 200), 2)
            cv2.circle(drawing, tuple(pt), 5, (0, 45, 200), 2)

    return no_back_drawing, drawing


def main():
    # model = load_model('/home/stepan/Documents/')
    model = load_model('')

    bgSubThreshold = 90
    cap_region_x_begin = 0.6  # start point/total width
    cap_region_y_end = 0.4  # start point/total width
    fgbg = None
    start = 0
    end = 0
    prev_prediction = 'Nothing'

    while True:
        # Capture frame-by-frame

        ret, frame = cap.read()

        x1, y1 = 0, 0
        x2, y2 = int(cap_region_x_begin * frame.shape[0]), int(cap_region_y_end * frame.shape[1])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        frame = cv2.bilateralFilter(frame, 5, 50, 100)
        frame = cv2.flip(frame, 1)

        # Display the resulting frame
        cv2.imshow('frame', frame)

        # gray = get_gray(frame)
        # thresh = get_thresh(frame)
        if fgbg is not None:
            subtracted = remove_background(frame, fgbg)
            tmp_sub = cv2.resize(subtracted[:y2, -x2:], (425, 400), interpolation=cv2.INTER_AREA)
            cv2.imshow('subtracted', tmp_sub)

            thresh = get_thresh(subtracted)

            res = thresh[:y2, -x2:]
            # gray = get_gray(frame)
            # res = gray[:y2, -x2:]

            if end - start >= 1:
                prediction, score = predict_thresh(res, model)
                start = ts.time()

                cv2.putText(res, "Prediction: {}".format(prediction), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255))
                prev_prediction = prediction
            else:
                ...
                cv2.putText(res, "Prediction: {}".format(prev_prediction), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255))

            # cv2.putText(thresh, "Score: ({}%)".format(score), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
            #             (255, 255, 255))

            tmp_res = cv2.resize(res, (425, 400), interpolation=cv2.INTER_AREA)
            cv2.imshow('thresh', tmp_res)

            # get the contours
            thresh = get_thresh(subtracted)
            thresh = thresh[:y2, -x2:]
            tmp_frame = frame[:y2, -x2:]
            no_back_drawing, drawning = get_contuors(tmp_frame, thresh)

            res_no_back = cv2.resize(no_back_drawing[:y2, :x2], (425, 400), interpolation=cv2.INTER_AREA)
            res_back = cv2.resize(drawning[:y2, :x2], (425, 400), interpolation=cv2.INTER_AREA)
            cv2.imshow('convex', res_no_back)
            cv2.imshow('real', res_back)

            end = ts.time()

        k = cv2.waitKey(10)

        if k == ord('q'):
            break

        elif k == ord('b'):
            global fgbd
            fgbg = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
