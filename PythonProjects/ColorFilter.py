__author__ = 'Charlie'
#An attempt at implementing colors to setup the emotion when visualizing the iamge
#Motivation: https://en.wikipedia.org/wiki/Contrasting_and_categorization_of_emotions#Plutchik.27s_wheel_of_emotions
import numpy as np, cv2 as cv

def imageRead(filename):
    readFile = open(filename, 'rb')
    img_str = readFile.read()
    readFile.close()
    img = np.array(bytearray(img_str), np.uint8).reshape((480, 480, 3))  # shape height, width, 3
    cv.cvtColor(img, cv.COLOR_BGR2RGB, img)
    return img

def showImage(title, image):
    cv.namedWindow(title, cv.WINDOW_AUTOSIZE)
    cv.imshow(title, image)
    cv.waitKey(0)

class ColorFilter:
    def __init__(self):
        pass

    image = np.array([])
    targetColor = np.zeros((1,3), dtype = np.uint16)
    colorDict = {}

    def set_image(self, input_image):
        self.image = np.copy(inputImage)


    def set_target_color(self, color):
        self.targetColor = cv.cvtColor(color, cv.COLOR_BGR2HSV)
        for iii in range(0,256):
            transformColor = np.array([[[ self.targetColor[0,0,0], iii*100/255,self.targetColor[0,0,2] ]]], dtype=np.uint8)
            self.colorDict[iii] = cv.cvtColor(transformColor, cv.COLOR_HSV2RGB)

    def filter(self):
        for iii in range(0, self.image.shape[0]):
            for jjj in range(0, self.image.shape[1]):
                data = self.image[iii, jjj]
                data[0] = self.colorDict[data[0]][0,0,0]
                data[1] = self.colorDict[data[1]][0,0,1]
                data[2] = self.colorDict[data[2]][0,0,2]

        return self.image


inputImage = cv.imread('Image1.jpg')
colorFilter = ColorFilter()
colorFilter.set_image(inputImage)
colorFilter.set_target_color(np.array([[[0, 0, 224]]], dtype= np.uint8))
output = (0.3*inputImage + 0.7*colorFilter.filter()).astype(np.uint8)
showImage('Output', output)
cv.imwrite("results.jpg", output)