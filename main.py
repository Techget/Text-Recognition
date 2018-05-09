from __future__ import print_function
import ccvwrapper
import numpy as np
from skimage import draw
from skimage.io import imread, imshow, imsave
from matplotlib import pyplot as plt
import sys
import cv2
from CNN.ocr_deep import ConvolutionNN
# from characters import extract_characters
from words import extract_words, extract_regions
from characters import extract_characters, estimate_avg_char_size


def rectangle_perimeter(r0, c0, width, height, shape=None, clip=False):
    rr, cc = [r0, r0 + width, r0 + width, r0], [c0, c0, c0 + height, c0 + height]

    return draw.polygon_perimeter(rr, cc, shape=shape, clip=clip)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        image_name = sys.argv[1]
    else:
        image_name = "img_with_border.jpg"
    
    # SWT get words in the image
    bytesx = open(image_name, "rb").read()
    swt_result_raw = ccvwrapper.swt(bytesx, len(bytesx), 1024, 1360)
    swt_result = np.reshape(swt_result_raw, (len(swt_result_raw) / 4, 4))
    print(swt_result)

    # load trained CNN for character recognition
    CNN_model = ConvolutionNN()

    # extract words, recognize each character and group them as a word
    image = imread(image_name, as_grey=False)
    j = 0
    for x, y, width, height in swt_result:
        for i in xrange(0, 3): # just to make lines thicker
            rr, cc = rectangle_perimeter(y + i, x + i, height, width, shape=image.shape, clip=True)
            image[rr, cc] = (255, 0, 0)
        
        subimage = image[y:y+height, x:x+width]

        # pad_word_image= cv2.copyMakeBorder(subimage,10,10,10,10,cv2.BORDER_CONSTANT,value=[255,255,255])
        characters = []

        lines = extract_words(subimage)
        for words in lines:
            for word_img in words:
                character_imgs = extract_characters(word_img)
                
                for char_img in character_imgs:
                    resize_char_img = np.array(cv2.resize(char_img, (28, 28), interpolation=cv2.INTER_CUBIC))
                    gray_image = cv2.cvtColor(resize_char_img, cv2.COLOR_BGR2GRAY)
                    ravel_char_img = gray_image.ravel()
                    prediction = CNN_model.predict(ravel_char_img)
                    temp = CNN_model.test_data.id2char[np.argmax(prediction) + 1]
                    # print(temp)
                    characters.append(temp)

        corresponding_text = ''.join(map(str, characters))

        cv2.putText(image, corresponding_text, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0),2)
        # imsave('result'+str(j)+'.jpg', subimage)
        j+=1

    imshow(image)
    imsave("result.jpg", image)
    plt.show()

