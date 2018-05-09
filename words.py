import cv2
import numpy as np
from copy import deepcopy
from collections import namedtuple
from lib import calc_bbox, X, Y, WIDTH, HEIGHT, BboxImg, percent_inc_border, add_inc_border
from characters import estimate_avg_char_size
import PIL

DEBUG = True


def _extract_lines(img):
    ''' Returns a list of images of each line '''
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Do edge detection using canny edge detection
    height, width = img.shape[0:2]
    edges_img = cv2.Canny(gray_img, 10, 100)

    if DEBUG:
        cv2.imshow('edges_img', edges_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Close the image to form lines
    m = estimate_avg_char_size(img)
    # make the kernel size dynamic according to the mean character size
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (m*5, int(m/3)))
    line_img = cv2.morphologyEx(edges_img, cv2.MORPH_CLOSE, rect_kernel)

    if DEBUG:
        cv2.imshow('line_img', line_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # otsu's method
    line_img = cv2.threshold(line_img, 0, 255, cv2.THRESH_BINARY)[1]

    # find the bounding boxes
    _, contours, _ = cv2.findContours(line_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = calc_bbox(contours, width, height, add_inc_border, m/4, m/4)

    if DEBUG:
        debug_img = deepcopy(img)
        for bbox in bboxes:
            x, y, w, h = bbox
            cv2.rectangle(debug_img, (x,y), (x+w,y+h), (0,255,0), 1)

        cv2.imshow('line_debug_img', debug_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # split into images
    res = []
    for bbox in bboxes:
        bbox_img = img[bbox[Y]:bbox[Y]+bbox[HEIGHT], bbox[X]:bbox[X]+bbox[WIDTH]]
        res.append(BboxImg(bbox, bbox_img))

    # sort the lines from decending order y
    res.sort(key=lambda k: k.bbox[Y], reverse=False)

    return [i.img for i in res]


def _extract_words_line(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    height, width = img.shape[0:2]
    edges_img = cv2.Canny(gray_img, 10, 100)

    # dynamically size the kernel according to character size in line
    m = estimate_avg_char_size(img)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(m/2.5), int(m/3)))
    word_img = cv2.morphologyEx(edges_img, cv2.MORPH_CLOSE, kernel)

    if DEBUG:
        cv2.imshow('word_img', word_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    word_img = cv2.threshold(word_img, 0, 255, cv2.THRESH_BINARY)[1]

    _, contours, _ = cv2.findContours(word_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = calc_bbox(contours, width, height, add_inc_border, m/4, m/4)

    if DEBUG:
        debug_img = deepcopy(img)

        for bbox in bboxes:
            x, y, w, h = bbox
            cv2.rectangle(debug_img, (x,y), (x+w,y+h), (0,255,0), 1)

        cv2.imshow('word_debug_img', debug_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    res = []
    for bbox in bboxes:
        bbox_img = img[bbox[Y]:bbox[Y]+bbox[HEIGHT], bbox[X]:bbox[X]+bbox[WIDTH]]
        res.append(BboxImg(bbox, bbox_img))

    res.sort(key=lambda k: k.bbox[X], reverse=False)

    return [i.img for i in res]


def extract_words(img):
    # Extract the lines and for each line, extract the words
    lines = _extract_lines(img)

    # convert the words into individual images
    words = []
    for l in lines:
        line_words = _extract_words_line(l)
        words.append(line_words)

    # return a list where it holds a list that contains a line of words
    return words


def extract_regions(img):
    '''
    Extracts text regions from a image that has no text
    i.g. has used SWT to filter non text
    '''
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    height, width = img.shape[0:2]

    edges_img = cv2.Canny(gray_img, 10, 100)

    m = estimate_avg_char_size(img)

    # make kernel dynamically sized (use image size?)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5*m, 5*m))
    region_img = cv2.dilate(edges_img, kernel, iterations=1)

    if DEBUG:
        cv2.imshow('region_img', region_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    _, contours, _ = cv2.findContours(region_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = calc_bbox(contours, width, height, add_inc_border, m, m)

    if DEBUG:
        debug_img = deepcopy(img)

        for bbox in bboxes:
            x, y, w, h = bbox
            cv2.rectangle(debug_img, (x,y), (x+w,y+h), (0,255,0), 1)

        cv2.imshow('region_debug_img', debug_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    res = []
    for bbox in bboxes:
        bbox_img = img[bbox[Y]:bbox[Y]+bbox[HEIGHT], bbox[X]:bbox[X]+bbox[WIDTH]]
        res.append(BboxImg(bbox, bbox_img))

    res.sort(key=lambda k: k.bbox[X], reverse=False)
    return [i.img for i in res]
