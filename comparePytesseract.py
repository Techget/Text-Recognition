import pytesseract
import argparse
import cv2
from PIL import Image
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image', type=str)
    args = parser.parse_args()
    text = pytesseract.image_to_string(Image.open(args.image))

    print(text)


