import cv2
import numpy as np
from collections import namedtuple

X = 0
Y = 1
WIDTH = 2
HEIGHT = 3

BboxImg = namedtuple('BboxImg', ['bbox', 'img'])


def none_border(bbox, to_modify_x, to_modify_y):
    bbox_h = bbox[HEIGHT]
    bbox_w = bbox[WIDTH]
    return (bbox_w, bbox_h)

def percent_inc_border(bbox, to_modify_x, to_modify_y):
    bbox_h = int(bbox[HEIGHT] * to_modify_y)
    bbox_w = int(bbox[WIDTH] * to_modify_x)
    return (bbox_w, bbox_h)


def add_inc_border(bbox, to_modify_x, to_modify_y):
    bbox_h = int(bbox[HEIGHT] + to_modify_y)
    bbox_w = int(bbox[WIDTH] + to_modify_x)
    return (bbox_w, bbox_h)


def calc_bbox(contours, img_width, img_height, change_border_func=none_border, to_modify_x=0, to_modify_y=0):
    # TODO: instead of using a multipler for width, inc by half of character width
    # or quarter of height?
    bboxes = []
    for c in contours:
        bbox = cv2.boundingRect(c)

        # make the bbox slightly bigger
        bbox_w, bbox_h = change_border_func(bbox, to_modify_x, to_modify_y)
        bbox_w_dif = bbox_w - bbox[WIDTH]
        bbox_h_dif = bbox_h - bbox[HEIGHT]
        bbox_x = int(bbox[X] - bbox_w_dif / 2)
        bbox_y = int(bbox[Y] - bbox_h_dif / 2)

        if bbox_x < 0:
            bbox_x = 0
        if bbox_y < 0:
            bbox_y = 0

        if bbox_x + bbox_w > img_width:
            bbox_w = img_width - bbox_x
        if bbox_y + bbox_h > img_height:
            bbox_h = img_height - bbox_y

        bboxes.append((bbox_x, bbox_y, bbox_w, bbox_h))
        #bboxes.append(bbox)

    return bboxes
