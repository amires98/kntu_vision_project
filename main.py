import numpy as np
import cv2
from matplotlib import pyplot as plt
import torch

from Cnn import *

with torch.no_grad():
    cap = cv2.VideoCapture('data\input.mp4')
    pt1 = np.asarray([(139, 167), (638, 108), (1142, 116), (872, 778)]).astype(np.float32)
    pt3 = np.array([(162, 151), (524, 2), (886, 151), (524, 697)]).astype(np.float32)
    # pt2 = np.array([(16.4, 13.9), (52.5, 0), (88.6, 13.9), (52.5, 68)]).astype(np.float32)

    H = cv2.getPerspectiveTransform(pt1, pt3)
    bsb1 = cv2.bgsegm.createBackgroundSubtractorMOG()
    bsb2 = cv2.createBackgroundSubtractorKNN()

    while True:
        background = cv2.imread('data/2D_field.png')
        ret, frame = cap.read()
        if not ret:
            break
        resized = cv2.resize(frame, (int(frame.shape[0] // 1.5), int(frame.shape[1] // 1.5)))
        cv2.imshow('original resized frame', resized)
        mask = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # fr = cv2.adaptiveThreshold(mask, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 10)

        mask = bsb1.apply(mask)

        k1 = np.array([[0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0],
                       [0, 1, 1, 1, 0],
                       [0, 1, 1, 1, 0],
                       [0, 1, 1, 1, 0],
                       [0, 1, 1, 1, 0],
                       [0, 0, 0, 0, 0]], np.uint8)

        k2 = np.array([[0, 0, 0, 0, 0],
                       [0, 0, 1, 0, 0],
                       [0, 1, 1, 1, 0],
                       [0, 1, 1, 1, 0],
                       [0, 1, 1, 1, 0],
                       [0, 0, 1, 0, 0],
                       [0, 0, 0, 0, 0]], np.uint8)

        k3 = np.array([[0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 1, 1, 0, 0, 0],
                       [0, 0, 1, 1, 1, 1, 0],
                       [0, 1, 1, 1, 1, 1, 0],
                       [0, 1, 1, 1, 1, 1, 0],
                       [0, 0, 0, 1, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0]], np.uint8)
        k4 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

        kernel1 = np.ones((3, 3), np.uint8)
        kernel2 = np.ones((2, 3), np.uint8)

        mask = cv2.GaussianBlur(mask, (5, 5), 1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k1, iterations=1)
        row = int(mask.shape[0] // 6)
        col = int(mask.shape[1] // 3)
        # mask[0:row, 0:col] = cv2.dilate(mask[0:row, 0:col], k2, iterations=1)
        # patch = cv2.dilate(patch, k3, iterations=10)
        mask[0:row, 0:col] = cv2.morphologyEx(mask[0:row, 0:col], cv2.MORPH_CLOSE, k1, iterations=3)
        # patch = cv2.dilate(patch, k2, iterations=5)
        # patch = cv2.erode(patch, k3, iterations=20)
        # mask = cv2.GaussianBlur(mask, (5, 5), 1)

        # contours, hier = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        # cv2.drawContours(mask, contours, -1, (0, 255, 255), 3)

        # ret2, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        resized_one = cv2.resize(mask, (int(mask.shape[0] // 1.5), int(mask.shape[1] // 1.5)))
        cv2.imshow('last frame', resized_one)

        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if nb_components == 1 or nb_components == 0:
            continue

        tf_centroids = np.array([centroids], dtype=np.float32)
        # beta = tf_centroids[:,1]//960*20
        # tf_centroids[:,1]+=1-beta
        tf_centroids = cv2.perspectiveTransform(tf_centroids, H)
        tf_centroids = np.squeeze(tf_centroids)
        data_batch = []
        indexes = []

        my_model = load_model()

        for i in range(1, nb_components):
            left, top, w, h, area = stats[i]
            if area < 100:
                continue

            # alpha = tf_centroids[i, 1] / 960
            # if area < 1100 * alpha:
            #     continue
            if tf_centroids[i, 1] < 5 or tf_centroids[i, 1] > 688:
                continue

            alpha = tf_centroids[i, 1] / 960
            if alpha < 1 / 3:
                alpha *= 3 / 5
            elif alpha < 1 / 5:
                alpha *= 1 / 5

            # if tf_centroids[i,0]< 48/100*1050 and tf_centroids[i,1]< 4/10*700:
            #     alpha*=2/5

            if area < 1100 * alpha:
                continue

            # tl = (left, top)
            # br = (left + w, top + h)
            # mask = cv2.rectangle(mask, tl, br, 255, -1)
            ff = frame[top:top + h, left:left + w, ::].copy()
            ff = cv2.cvtColor(ff, cv2.COLOR_BGR2RGB)
            data_batch.append(ff)
            indexes.append(i)

        labels = my_prediction(data_batch, my_model)

        for i in range(len(indexes)):
            arr = labels[i].numpy()
            arr = np.argmax(arr, axis=1)
            color_tup = (0, 0, 255)
            if arr == 1:
                color_tup = (255, 0, 0)
            elif arr == 2:
                color_tup = (0, 255, 0)

            tup = (int(tf_centroids[indexes[i], 0]), int(tf_centroids[indexes[i], 1]))
            background = cv2.circle(background, tup, 5,
                                    color_tup, -1)

        cv2.imshow('output', background)
        p = cv2.waitKey(1)
        if p & 0xFF == ord('q'):
            break
