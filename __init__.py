import cv2
import numpy as np
import argparse
from words import extract_words, extract_regions
from characters import extract_characters, estimate_avg_char_size
from PIL import Image
#import pillowfight as pf

DEBUG = True

def PIL_to_cv_img(PIL_img):
    CV_img = np.array(PIL_img)
    return CV_img


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image', type=str)

    args = parser.parse_args()

    # Need to run SWT algorithm to get rid of non text
    #PIL_img = Image.open(args.image)
    #PIL_no_text_img = pf.swt(PIL_img, output_type=pf.SWT_OUTPUT_ORIGINAL_BOXES)
    #to_extract_img = PIL_to_cv_img(PIL_no_text_img)
    to_extract_img = cv2.imread(args.image, 1)

    if DEBUG:
        cv2.imshow('img', to_extract_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    regions = extract_regions(to_extract_img)

    for r in regions:
        lines = extract_words(r)

        for words in lines:
            for word_img in words:
                characters = extract_characters(word_img)
                for char in characters:
                    # TODO add padding if necessary
                    # TODO resize character img to for CNN
                    # TODO classify using CNN
                    pass
