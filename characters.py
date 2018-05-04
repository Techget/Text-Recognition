import cv2
import numpy as np
from copy import deepcopy
from lib import calc_bbox

DEBUG = True


def extract_characters_bbox(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    height, width = img.shape[0:2]

    #mser = cv2.MSER_create()
    #coodinates, bboxes = mser.detectRegions(gray_img)

    edges_img = cv2.Canny(gray_img, 10, 100)
    _, contours, _ = cv2.findContours(edges_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = calc_bbox(contours, width, height)


    if DEBUG:
        cv2.imshow('char_img', edges_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        debug_img = deepcopy(img)
        for bbox in bboxes:
            x, y, w, h = bbox
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 1)

        cv2.imshow('char_debug_img', debug_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # TODO: Have to handle disconnected characters i.e. i
    res = []


def extract_characters(img):
    extract_characters_bbox(img)
