import cv2
import numpy as np
import argparse
from words import extract_words, extract_regions
from characters import extract_characters, estimate_avg_char_size
from PIL import Image
import pillowfight as pf
from CNN.ocr_deep import ConvolutionNN
from autocorrect import spell


DEBUG = True

def PIL_to_cv_img(PIL_img):
    CV_img = np.array(PIL_img)
    return CV_img


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image', type=str)

    args = parser.parse_args()

    # preprocess, GaussianBlur and segmentation
    img = cv2.imread(args.image)
    blurred_img = cv2.GaussianBlur(img, (5, 5), 0)
    cv2.imwrite('blurred'+args.image, blurred_img)

    # Need to run SWT algorithm to get rid of non text
    # with blurred image, it will neglect trivia part, but it will also degradate performance
    # PIL_img = Image.open('blurred'+args.image) 
    PIL_img = Image.open(args.image)
    PIL_no_text_img = pf.swt(PIL_img, output_type=pf.SWT_OUTPUT_ORIGINAL_BOXES)
    to_extract_img = PIL_to_cv_img(PIL_no_text_img)
    # to_extract_img = cv2.imread(args.image, 1)

    if DEBUG:
        cv2.imshow('img', to_extract_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    CNN_model = ConvolutionNN()

    region_imgs, region_coords = extract_regions(to_extract_img)
    characters = []

    for i in range(len(region_imgs)):
        lines = extract_words(region_imgs[i])
        region_text_block = ''

        for words in lines:
            for word_img in words:
                character_imgs = extract_characters(word_img)
                for char_img in character_imgs:
                    # TODO add padding if necessary
                    # TODO resize character img to for CNN
                    # TODO classify using CNN
                    pad_word_image= cv2.copyMakeBorder(char_img,2,2,5,5,cv2.BORDER_CONSTANT,value=[255,255,255])
                    if DEBUG:
                        cv2.imshow('padded', pad_word_image)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                    resize_char_img = np.array(cv2.resize(pad_word_image, (28, 28), interpolation=cv2.INTER_CUBIC))
                    # cv2.imwrite('tempCharImg.png', resize_char_img)
                    gray_image = cv2.cvtColor(resize_char_img, cv2.COLOR_BGR2GRAY)

                    constrained_value_img = 1 - np.array(gray_image, dtype=np.float32) / 255

                    ravel_char_img = constrained_value_img.ravel()
                    prediction = CNN_model.predict(ravel_char_img)
                    temp = CNN_model.test_data.id2char[np.argmax(prediction) + 1]
                    # print(temp)
                    characters.append(temp)

                corresponding_word = ''.join(map(str, characters))
                corresponding_word = spell(corresponding_word)
                print(corresponding_word)
                region_text_block += ' '+corresponding_word
                characters = []

        with open('result.txt', 'a+') as f:
            print("{}: {}".format(region_coords[i], region_text_block),file=f)

        cv2.rectangle(img,
            (region_coords[i][0], region_coords[i][1]),
            (region_coords[i][0]+region_coords[i][2],region_coords[i][1]+region_coords[i][3]),
            (255,0,0),3)

    cv2.imshow('text block found', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('textBlcok'+args.image, img)