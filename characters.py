import cv2
import numpy as np
from copy import deepcopy
from lib import calc_bbox, X, Y, WIDTH, HEIGHT, BboxImg


DEBUG = True


def extract_characters_bbox(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    height, width = img.shape[0:2]

    #mser = cv2.MSER_create()
    #coodinates, bboxes = mser.detectRegions(gray_img)

    #edges_img = cv2.Canny(gray_img, 10, 100)
    edges_img = gray_img

    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 10))
    #char_img = cv2.morphologyEx(edges_img, cv2.MORPH_CLOSE, kernel)
    char_img = cv2.threshold(edges_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    char_img = 255 - char_img

    _, contours, _ = cv2.findContours(char_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    bboxes = calc_bbox(contours, width, height)

    if DEBUG:
        #cv2.imwrite('char_img.png', char_img)
        cv2.imshow('char_img', char_img)
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
    # TODO: Remove boxes inside of a larger box
    return bboxes


def extract_characters(img):
    '''DEPECRETATED, use fan's implementation'''
    bboxes = extract_characters_bbox(img)

    res = []
    for bbox in bboxes:
        bbox_img = img[bbox[Y]:bbox[Y]+bbox[HEIGHT], bbox[X]:bbox[X]+bbox[WIDTH]]
        res.append(BboxImg(bbox, bbox_img))

    res.sort(key=lambda k: k.bbox[X], reverse=False)
    return [i.img for i in res]
