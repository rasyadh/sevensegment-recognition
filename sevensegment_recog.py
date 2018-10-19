import os
import cv2
import numpy as np

DIGITS_LOOKUP = {
    (1, 1, 1, 1, 1, 1, 0): 0,
    (1, 1, 0, 0, 0, 0, 0): 1,
    (1, 0, 1, 1, 0, 1, 1): 2,
    (1, 1, 1, 0, 0, 1, 1): 3,
    (1, 1, 0, 0, 1, 0, 1): 4,
    (0, 1, 1, 0, 1, 1, 1): 5,
    (0, 1, 1, 1, 1, 1, 1): 6,
    (1, 1, 0, 0, 0, 1, 0): 7,
    (1, 1, 1, 1, 1, 1, 1): 8,
    (1, 1, 1, 0, 1, 1, 1): 9
}

H_W_RATIO = 1.9
THRESHOLD = 35
arc_tan_theta = 6.0

def load_image(path):
    # load image as grayscale
    gray = cv2.imread(path, 0)
    # get width and height of the image
    height, width = gray.shape
    # apply gaussian blur to gray image
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    # show gray and blurred image
    cv2.imshow('gray', gray)
    cv2.imshow('blurred', blurred)
    return blurred, gray

def preprocess(image, threshold, kernel_size=(5, 5)):
    # apply CLAHE 
    # (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(6, 6))
    image = clahe.apply(image)
    cv2.imshow('clahe', image)
    # apply adaptive threshold 
    # to get black and white image
    dst = cv2.adaptiveThreshold(
        image, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        127, threshold)
    cv2.imshow('adaptive threshold', dst)
    # get structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, kernel_size)
    # perform morphological transformation
    dst = cv2.morphologyEx(dst, cv2.MORPH_CLOSE, kernel)
    cv2.imshow('morph close', dst)
    dst = cv2.morphologyEx(dst, cv2.MORPH_OPEN, kernel)
    cv2.imshow('morph open', dst)
    return dst

def helper_extract(one_d_array, threshold=20):
    res = []
    flag = 0
    temp = 0

    for i in range(len(one_d_array)):
        if one_d_array[i] < 12 * 255:
            if flag > threshold:
                start = i - flag
                end = i
                temp = end
                if end - start > 20:
                    res.append((start, end))
            flag = 0
        else:
            flag += 1
    else:
        if flag > threshold:
            start = temp
            end = len(one_d_array)
            if end - start > 50:
                res.append((start, end))
    return res

def find_digits_positions(image, reserved_threshold=20):
    digits_positions = []
    # calculate horizontal position
    img_array = np.sum(image, axis=0)
    print('img array horizontal: {}'.format(img_array))
    horizontal_position = helper_extract(img_array, threshold=reserved_threshold)
    print('horizontal position: {}'.format(horizontal_position))
    print()
    # calculate vertical position
    img_array = np.sum(image, axis=1)
    print('img array vertical: {}'.format(img_array))
    vertical_position = helper_extract(img_array, threshold=reserved_threshold * 4)
    print('vertical position: {}'.format(vertical_position))
    print()
    # make vertical position has only one element
    if len(vertical_position) > 1:
        vertical_position = [
            (vertical_position[0][0], vertical_position[len(vertical_position) - 1][1])
        ]
    print('after vertical position change to one element')
    print('horizontal position : {}'.format(horizontal_position))
    print('vertical position : {}'.format(vertical_position))
    # create coordinat from horizontal and vertical positions
    for horizontal in horizontal_position:
        for vertical in vertical_position:
            digits_positions.append(list(zip(horizontal, vertical)))
    # check if digits's found
    # if not throw assertion error
    # assert len(digits_positions) > 0, "Failed to find digits's positions"
    return digits_positions

def recognize_digits_line_method(digits_positions, output_img, input_img):
    digits = []
    for coordinat in digits_positions:
        x0, y0 = coordinat[0]
        x1, y1 = coordinat[1]
        roi = input_img[y0:y1, x0:x1]
        h, w = roi.shape
        suppose_W = max(1, int(h / H_W_RATIO))

        if x1 - x0 < 25 and cv2.countNonZero(roi) / ((y1 - y0) * (x1 - x0)) < 0.2:
            continue

        if w < suppose_W / 2:
            x0 = max(x0 + w - suppose_W, 0)
            roi = input_img[y0:y1, x0:x1]
            w = roi.shape[1]

        center_y = h // 2
        quater_y_1 = h // 4
        quater_y_3 = quater_y_1 * 3
        center_x = w // 2
        line_width = 5  # line's width

        width = (max(int(w * 0.15), 1) + max(int(h * 0.15), 1)) // 2
        small_delta = int(h / arc_tan_theta) // 4
        segments = [
            ((w - 2 * width, quater_y_1 - line_width), (w, quater_y_1 + line_width)),
            ((w - 2 * width, quater_y_3 - line_width), (w, quater_y_3 + line_width)),
            ((center_x - line_width - small_delta, h - 2 * width), (center_x - small_delta + line_width, h)),
            ((0, quater_y_3 - line_width), (2 * width, quater_y_3 + line_width)),
            ((0, quater_y_1 - line_width), (2 * width, quater_y_1 + line_width)),
            ((center_x - line_width, 0), (center_x + line_width, 2 * width)),
            ((center_x - line_width, center_y - line_width), (center_x + line_width, center_y + line_width)),
        ]
        on = [0] * len(segments)

        for (i, ((xa, ya), (xb, yb))) in enumerate(segments):
            seg_roi = roi[ya:yb, xa:xb]
            total = cv2.countNonZero(seg_roi)
            area = (xb - xa) * (yb - ya) * 0.9

            if total / float(area) > 0.25:
                on[i] = 1
        
        if tuple(on) in DIGITS_LOOKUP.keys():
            digit = DIGITS_LOOKUP[tuple(on)]
        else:
            digit = ''
        digits.append(digit)

        # detect dot in seven segment
        '''
        if cv2.countNonZero(roi[h - int(3 * width / 4):h, w - int(3 * width / 4):w]) / (9. / 16 * width * width) > 0.65:
            digits.append('.')
            cv2.rectangle(output_img,
                          (x0 + w - int(3 * width / 4), y0 + h - int(3 * width / 4)),
                          (x1, y1), (0, 128, 0), 2)
            cv2.putText(output_img, 'dot',
                        (x0 + w - int(3 * width / 4), y0 + h - int(3 * width / 4) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 128, 0), 2)
        '''

        cv2.rectangle(output_img, (x0, y0), (x1, y1), (0, 128, 0), 2)
        cv2.putText(output_img, str(digit), (x0 + 3, y0 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 128, 0), 2)
    return digits

def test_dataset(img_path):
    cwd = os.getcwd()
    success = []
    for img in os.listdir(os.path.join(cwd, img_path)):
        main(os.path.join(cwd, img_path, img))

def main(img_name):
    blurred, gray = load_image(img_name)
    print(blurred.shape)
    output = blurred
    dst = preprocess(blurred, THRESHOLD)
    digits_positions = find_digits_positions(dst)
    print('digits position : {}'.format(digits_positions))
    digits = recognize_digits_line_method(digits_positions, output, dst)
    print('digits : {}'.format(digits))
    cv2.imshow('result', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    image_test_path = 'test'
    test_dataset(image_test_path)

    # blurred, gray = load_image('test/5A155.png')
    # print(blurred.shape)
    # output = blurred
    # dst = preprocess(blurred, THRESHOLD)
    # digits_positions = find_digits_positions(dst)
    # print('digits position : {}'.format(digits_positions))
    # digits = recognize_digits_line_method(digits_positions, output, dst)
    # print('digits : {}'.format(digits))
    # cv2.imshow('result', output)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()