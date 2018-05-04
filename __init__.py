import cv2
import numpy as np
import argparse
from words import extract_words
from characters import extract_characters

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image', type=str)

    args = parser.parse_args()

    to_extract_img = cv2.imread(args.image, 1)

    cv2.imshow('img', to_extract_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    lines = extract_words(to_extract_img)

    for words in lines:
        for word_img in words:
            characters = extract_characters(word_img)
            for char in characters:
                # TODO classify using CNN
                pass
