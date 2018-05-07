from __future__ import print_function
import ccvwrapper
import numpy as np
from skimage import draw
from skimage.io import imread, imshow, imsave
from matplotlib import pyplot as plt
import sys
import cv2

def rectangle_perimeter(r0, c0, width, height, shape=None, clip=False):
    rr, cc = [r0, r0 + width, r0 + width, r0], [c0, c0, c0 + height, c0 + height]

    return draw.polygon_perimeter(rr, cc, shape=shape, clip=clip)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        image_name = sys.argv[1]
    else:
        image_name = "img_with_border.jpg"
    bytesx = open(image_name, "rb").read()
    swt_result_raw = ccvwrapper.swt(bytesx, len(bytesx), 1024, 1360)
    swt_result = np.reshape(swt_result_raw, (len(swt_result_raw) / 4, 4))
    print(swt_result)

    image = imread(image_name, as_grey=False)
    j = 0
    for x, y, width, height in swt_result:
        for i in xrange(0, 3): # just to make lines thicker
            rr, cc = rectangle_perimeter(y + i, x + i, height, width, shape=image.shape, clip=True)
            image[rr, cc] = (255, 0, 0)
        
        subimage = image[y:y+height, x:x+width]
        character_imgs = extract_characters(subimage)
        characters = []
        for char_img in character_imgs:
            # TODO classify using CNN
            pass

        corresponding_text = ''.join(map(str, mylist))

        cv2.putText(image, corresponding_text, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0),2)
        # imsave('result'+str(j)+'.jpg', subimage)
        j+=1

    imshow(image)
    imsave("result.jpg", image)
    plt.show()

