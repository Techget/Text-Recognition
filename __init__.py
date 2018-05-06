import cv2
import numpy as np
import argparse
from words import extract_words, extract_regions
from characters import extract_characters
import PIL
import pillowfight as pf


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image', type=str)

    args = parser.parse_args()

    # Need to run SWT algorithm to get rid of non text
    '''
    PIL_img = PIL.Image.open(args.image)
    PIL_no_text_img = pf.swt(img_in, output_type=pf.SWT_OUTPUT_ORIGINAL_BOXES)
    PIL_no_text_img.save('test.png')
    '''

    to_extract_img = cv2.imread(args.image, 1)

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
                    # TODO classify using CNN
                    pass
