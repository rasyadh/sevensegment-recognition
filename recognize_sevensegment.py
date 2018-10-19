import os
from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import cv2
import numpy as np

# Define the dictionary of digit 7-segment
DIGITS_LOOKUP = {
    #(0, 1, 2, 3, 4, 5, 6)
    (1, 1, 1, 0, 1, 1, 1): 0,
    (0, 0, 1, 0, 0, 1, 0): 1,
    (1, 0, 1, 1, 1, 0, 1): 2,
    (1, 0, 1, 1, 0, 1, 1): 3,
    (0, 1, 1, 1, 0, 1, 0): 4,
    (1, 1, 0, 1, 0, 1, 1): 5,
    (1, 1, 0, 1, 1, 1, 1): 6,
    (0, 1, 0, 1, 1, 1, 1): 6,
    (1, 1, 1, 0, 0, 1, 0): 7,
    (1, 0, 1, 0, 0, 1, 0): 7,
    (1, 1, 1, 1, 1, 1, 1): 8,
    (1, 1, 1, 1, 0, 1, 1): 9,
    (1, 1, 1, 1, 0, 1, 0): 9
}

def load_image(path):    
    # load the image
    image = cv2.imread(path)
    # print('image shape: {}'.format(image.shape))
    return image

def preprocessing(image, height=100, kernel=(5, 5)):
    # pre-processing the image
    # resizing, grayscaling, blurring, and edge map
    image = imutils.resize(image, height=height)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, kernel, 0)
    # print('image shape resized: {}'.format(image.shape))
    return image, blurred

def get_threshold(blurred):
    # threshold image
    threshold = cv2.threshold(
        blurred, 0, 255, 
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )[1]
    # apply morphological operator
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (1, 5)
    )
    threshold = cv2.morphologyEx(
        threshold, 
        cv2.MORPH_OPEN, 
        kernel
    )
    # print('kernel: {}'.format(kernel))
    return threshold

def get_adaptive_threshold(blurred):
    # apply CLAHE
    # (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(6, 6))
    blurred = clahe.apply(blurred)
    # apply adaptive threshold
    threshold = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        127, 10)

    # get structuring element
    # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
    # perform morphological transformation
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
    return threshold

def find_digits(threshold, image):
    #find contours in the threshold image
    cnts = cv2.findContours(
        threshold.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    digitsCnts = []
    # print('cnts: {}'.format(cnts))

    # loop over the digit area candidates
    for c in cnts:
        # bounding boxx
        (x, y, w, h) = cv2.boundingRect(c)
        print('x, y, w, h: {}'.format((x, y, w, h)))

        # get the digit of the contours
        if w >= 10 and (h >= 60 and h <= 100):
            digitsCnts.append(c)
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 1)
        '''
        if h >= 70 and h <= 90:
            if w <= 20:
                digitsCnts.append(c)
                cv2.rectangle(
                    image, (x, y), (x+w, y+h), (0, 255, 0), 1
                )
            elif w >= 30 and w <= 60:
                digitsCnts.append(c)
                cv2.rectangle(
                    image, (x, y), (x+w, y+h), (0, 255, 0), 1
                )
        '''
    print('digits contours detected : {}'.format(len(digitsCnts)))
    
    # if digits not found throw assertion error
    # assert len(digitsCnts) > 0, "Failed to find digit's position"

    # sort the contours from left-to-right
    if digitsCnts:
        digitsCnts = contours.sort_contours(
            digitsCnts,
            method="left-to-right"
        )[0]
    return digitsCnts

def recognize_digits(image, threshold, digitsCnts):
    digits = []
    # loop over each of the digits
    for c in digitsCnts:
        # extract the digit ROI
        (x, y, w, h) = cv2.boundingRect(c)
        # get region of interest (roi) from threshold image
        roi = threshold[y : y+h, x : x+w]
        # print('roi: {}'.format(roi))
        
        if w <= 30:
            on = [0, 0, 1, 0, 0, 1, 0]
        else:
            # compute the width and height of each the 7 segments
            (roiH, roiW) = roi.shape
            (dW, dH) = (int(roiW * 0.25), int(roiH * 0.15))
            dHC = int(roiH * 0.05)
            print('roi shape: {}'.format(roi.shape))
            print('dw, dh: {}'.format((dW, dH)))
            print('dHC: {}'.format(dHC))

            # define the set of 7 segments
            segments = [
                ((0, 0), (w, dH)),          # top
                ((0, 0), (dW, h // 2)),     # top-left
                ((w - dW, 0), (w, h // 2)), # top-right
                ((0, (h // 2) - dHC), (w, (h // 2) + dHC)),  # center
                ((0, h // 2), (dW, h)),     # bottom-left
                ((w - dW, h // 2), (w, h)), # bottom-right
                ((0, h -dH), (w, h))        # bottom
            ]
            # prepare the key of digit
            on = [0] * len(segments)

            # loop over the segments
            for (i, ((xA, yA), (xB, yB))) in enumerate(segments):
                # extract the segment ROI
                segROI = roi[yA:yB, xA:xB]
                # count the total of thresholded pixel in the segment
                total = cv2.countNonZero(segROI)
                area = (xB - xA) * (yB - yA)

                # if total number of non-zero pixel is greater 
                # than 50% of the area, mark the segment as on
                if total / float(area) > 0.4:
                    on[i] = 1

        # lookup the digit
        if tuple(on) in DIGITS_LOOKUP.keys():
            digit = DIGITS_LOOKUP[tuple(on)]
            digits.append(digit)
            # draw roi on the image
            cv2.rectangle(
                image, (x, y), (x+w, y+h), (0, 255, 0), 1
            )
            cv2.putText(
                image, str(digit), (x, y+15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 0), 1
            )
    return digits

def test_datatest(img_path):
    cwd = os.getcwd()
    success = []
    for img in os.listdir(os.path.join(cwd, img_path)):
        main(os.path.join(cwd, img_path, img))
        
def main(img_name):
    image = load_image(img_name)
    image, blurred = preprocessing(image, kernel=(7, 7))
    # threshold = get_threshold(blurred)
    threshold = get_adaptive_threshold(blurred)
    digitsCnts = find_digits(threshold, image)
    digits = recognize_digits(image, threshold, digitsCnts)

    print('Digits : {}'.format(digits))
    cv2.imshow('threshold', threshold)
    cv2.imshow('result', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    image_test_path = 'test'
    test_datatest(image_test_path)

    # image = load_image('test/6A68Z.jpg')
    # image, blurred = preprocessing(image, kernel=(7, 7))
    # #threshold = get_threshold(blurred)
    # threshold = get_adaptive_threshold(blurred)
    # digitsCnts = find_digits(threshold, image)
    # digits = recognize_digits(image, threshold, digitsCnts)

    # # print('Digits : {}'.format(digits))
    # cv2.imshow('threshold', threshold)
    # cv2.imshow('result', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()