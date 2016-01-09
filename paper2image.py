#!/usr/bin/env python2.7
import sys
from optparse import OptionParser

try:
    import numpy as np
except:
    sys.exit("Please install numpy")

try:
    import cv2
except:
    sys.exit("Please install OpenCV")

# Parser
parser = OptionParser()
parser.add_option("-i", "--image", type="string", dest="image",
                  help="Path of image which will be scanned")
parser.add_option("-c", "--colorless", action="store_true", dest="colorless",
                  default=False, help="only having black and white")

(options, args) = parser.parse_args()
image = options.image
if image is None:
    sys.exit("No image given")
colorless = bool(options.colorless)


def detectEdges(edged):
    """ gettings the coordinates to the conturs"""
    (conturs, _) = cv2.findContours(
        edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    conturs = sorted(conturs, key=cv2.contourArea, reverse=True)[:5]
    return conturs


def readImage(image):
    """ reads the image """
    return cv2.imread(image)


def imageDetail(img):
    """ get edges and gray scale"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)
    return gray, edged


def getPolygon(conturs):
    """ returns the polygon of the paper """
    for line in conturs:
        # approximate the contour
        peri = cv2.arcLength(line, True)
        approx = cv2.approxPolyDP(line, 0.02 * peri, True)
        if len(approx) == 4:
            # print(approx)
            return approx
    else:
        sys.exit("no paper found")


def blackAndWhite(warped):
    """ making the image 'black or white' """
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 9)


def displayOutput(img, body, warped):
    """ display windows """
    cv2.drawContours(img, [body], -1, (0, 255, 0), 2)
    cv2.imshow("Paper", img)
    cv2.imshow("Outline", warped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def calculatePicture(image):
    """ call all the functions to get result"""
    img = readImage(image)
    gray, edged = imageDetail(img)
    conturs = detectEdges(edged)
    body = getPolygon(conturs)
    polygon = np.float32(body)
    # TODO here the intelligent part for detecting the x and y of vertices
    x = int(img.shape[1])
    y = int(img.shape[0])
    target = np.float32([[x, 0], [0, 0], [0, y], [x, y]])
    # TODO here sorting the tuples or finding out why the order is like that
    # print(polygon)
    # print(target)
    M = cv2.getPerspectiveTransform(polygon, target)
    # print(M)
    warped = cv2.warpPerspective(img, M, (x, y))
    if colorless:
        warped = blackAndWhite(warped)
    output = image.split(".")[0] + "-image.jpeg"
    cv2.imwrite(output, warped)
    displayOutput(img, body, warped)


if __name__ == '__main__':
    calculatePicture(image)
