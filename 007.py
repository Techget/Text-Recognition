import cv2

strFormula="1!((x+1)*(x+2))" # '!' means a character is not allowed in file name
img = cv2.imread("was.png")
imgGray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
ret, imgThresh = cv2.threshold(imgGray, 127, 255, 0)

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
if int(major_ver)  < 3 :
    contours , hierarchy  = cv2.findContours(imgThresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
else :
    image, contours , _   = cv2.findContours(imgThresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#:if

lstBoundingBoxes = []
for cnt in contours:  lstBoundingBoxes.append(cv2.boundingRect(cnt))
lstBoundingBoxes.sort()

charNo=0
for item in lstBoundingBoxes[1:]: # skip first element ('bounding box' == entire image)
    charNo += 1
    fName = "charAtPosNo-" + str(charNo).zfill(2) + "_is_[ " + strFormula[charNo-1] + " ]"+ ".png"; 
    x,y,w,h = item
    cv2.imwrite(fName, img[y:y+h, x:x+w])
