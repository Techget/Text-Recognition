import pytesseract
import argparse
import cv2
from PIL import Image
import os
import editdistance
import string

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image', type=str)
    parser.add_argument('gtText', type=str)
    args = parser.parse_args()

    with open(args.gtText, 'r') as gtfile:
        gt_data=gtfile.read().replace('\n', '')

    text = pytesseract.image_to_string(Image.open(args.image))

    # remove newline
    text = text.replace('\n', ' ').replace('\r', '')
    # remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    
    print(text)
    print(editdistance.eval(text, gt_data))

