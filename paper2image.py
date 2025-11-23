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
                  help="Path of image which will be scanned", metavar="FILE")
parser.add_option("-g", "--grayscale", action="store_true", dest="grayscale",
                  default=False, help="only having black and white")
parser.add_option("-b", "--blur", action="store_true", dest="blur",
                  default=False, help="gausian-blur the output")
parser.add_option("-v", "--verbose", action="store_true", dest="verbose",
                  default=False, help="showing the results")

(options, args) = parser.parse_args()
image = options.image
if image is None:
    sys.exit("No image given")
grayscale = bool(options.grayscale)
blur = bool(options.blur)
verbose = bool(options.verbose)


def detectEdges(edged):
    """ gettings the coordinates to the conturs"""
    (_, conturs, _) = cv2.findContours(
        edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    conturs = sorted(conturs, key=cv2.contourArea, reverse=True)[:5]
    return conturs


def readImage(image):
    """ reads the image """
    try:
        img = cv2.imread(image)
    except:
        sys.exit("No image found")
    return img


def imageDetail(img):
    """ get edges and gray scale"""
    if blur:
        img = cv2.GaussianBlur(img, (7, 7), 0)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
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
            return approx.reshape(4, 2)
    else:
        sys.exit("no paper found")


def order_points(points):
    """ get the correct order of the points"""
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, and
    # the bottom-right point will have the largest sum
    s = points.sum(axis=1)
    rect[0] = points[np.argmin(s)]
    rect[2] = points[np.argmax(s)]

    # the difference between the points,
    # the top-right point will have the smallest difference,
    # and the bottom-left will have the largest difference
    diff = np.diff(points, axis=1)
    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]

    return rect


def getNewSize(rect):
    """ get the size of the new image """
    (tl, tr, br, bl) = rect

    # max((br - bl), (tr - tl)) = x
    widthB = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthT = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthB), int(widthT))

    # max((tr - br), (tl - bl)) =  y
    heightR = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightL = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightR), int(heightL))
    return maxWidth, maxHeight


def blackAndWhite(warped):
    """ making the image 'black or white' """
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 9)


def displayOutput(img, points, warped):
    """ display windows """
    cv2.drawContours(img, [points], -1, (0, 255, 0), 2)
    cv2.imshow("Paper", img)
    cv2.imshow("Outline", warped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def calculatePicture(image):
    """ call all the functions to get result"""
    img = readImage(image)
    gray, edged = imageDetail(img)
    conturs = detectEdges(edged)
    points = getPolygon(conturs)
    rect = order_points(points)
    maxWidth, maxHeight = getNewSize(rect)
    polygon = np.float32(rect)
    # print(polygon)
    target = np.float32([[0, 0],
                         [maxWidth - 1, 0],
                         [maxWidth - 1, maxHeight - 1],
                         [0, maxHeight - 1]])
    # print(target)
    M = cv2.getPerspectiveTransform(polygon, target)
    # print(M)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
    if grayscale:
        warped = blackAndWhite(warped)
    output = image.split(".")[0] + "-image.jpeg"
    # output = image.split(".")[0] + "-image.png"
    cv2.imwrite(output, warped)
    if verbose:
        displayOutput(img, points, warped)


if __name__ == '__main__':
    calculatePicture(image)
