import cv2
import numpy as np
from copy import deepcopy
from lib import calc_bbox, X, Y, WIDTH, HEIGHT, BboxImg, add_inc_border, percent_inc_border

DEBUG = False
# CHARACTER_SIZE_THRESHOLD = 15

def mean(list, item_func):
    sum = 0
    for i in list:
        val = item_func(i)
        sum += val

    res = sum / len(list)
    return res


def estimate_avg_char_size(img):
    'Returns the mean height of characters in img'
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    height, width = img.shape[0:2]

    delta = 5
    min_area = 55
    max_area = 14500
    max_variation = 0.3
    min_diversity = .2
    max_evolution = 200
    area_threshold = 1.01
    min_margin = 0.005
    edge_blur_size = 5

    mser = cv2.MSER_create(
        _delta = delta,
        _min_area = min_area,
        _max_area = max_area,
        _max_variation = max_variation,
        _min_diversity = min_diversity,
        _max_evolution = max_evolution,
        _area_threshold = area_threshold,
        _min_margin = min_margin,
        _edge_blur_size = edge_blur_size
    )

    coodinates, bboxes = mser.detectRegions(gray_img)

    if DEBUG:
        debug_img = deepcopy(img)
        for bbox in bboxes:
            x, y, w, h = bbox
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.imshow('estimate_char_size_debug_img', debug_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    m = mean(bboxes, lambda k: k[HEIGHT])
    return int(m)


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

    if DEBUG:
        cv2.imwrite('test.png', char_img)

    _, contours, _ = cv2.findContours(char_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    bboxes = calc_bbox(contours, width, height, percent_inc_border, 1.02, 1.1)

    # Remove boxes inside of a larger box
    # Have to handle disconnected characters i.e. i
    to_remove = []
    for i in range(0, len(bboxes)):
        for j in range(0, len(bboxes)):
            if i == j:
                break
            if (bboxes[i][X] >= bboxes[j][X] and bboxes[i][X] + bboxes[i][WIDTH] <= bboxes[j][X] + bboxes[j][WIDTH] and
            bboxes[i][Y] >= bboxes[j][Y] and bboxes[i][Y] + bboxes[i][HEIGHT] <=  bboxes[j][Y] + bboxes[j][HEIGHT]):
                to_remove.append(bboxes[j])

            elif ((bboxes[i][X] >= bboxes[j][X] and bboxes[j][X] + bboxes[j][WIDTH] >= bboxes[i][X] + bboxes[i][WIDTH])):
                # combine them
                to_remove.append(bboxes[i])
                new_width = bboxes[j][WIDTH]
                new_y = 0
                new_height = height
                # build new tuple and replace
                new_bbox = (bboxes[j][X], new_y, new_width, new_height)
                bboxes[j] = new_bbox

    bboxes = [x for x in bboxes if x not in to_remove]

    heights = [x[HEIGHT] for x in bboxes]
    avg_heights = sum(heights)/float(len(heights))
    threshold = avg_heights * 0.7
    bboxes = list(filter(lambda x: x[HEIGHT] > threshold, bboxes))

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

    return bboxes


def extract_characters(img):
    bboxes = extract_characters_bbox(img)

    res = []
    for bbox in bboxes:
        bbox_img = img[bbox[Y]:bbox[Y]+bbox[HEIGHT], bbox[X]:bbox[X]+bbox[WIDTH]]
        res.append(BboxImg(bbox, bbox_img))

    res.sort(key=lambda k: k.bbox[X], reverse=False)

    if DEBUG:
        for r in res:
            cv2.imshow('char_img', r.img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    return [i.img for i in res]
