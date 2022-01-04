import numpy as np
from scipy.signal import convolve2d
import cv2.cv2 as cv2


def print_IDs():
    print("305237257+312162027\n")


def calculate_sobel_edge_detection(img):
    img = np.array(img, dtype=float)

    sobel_matrix_x_direction = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])
    x_edges = convolve2d(img, sobel_matrix_x_direction, mode='same')

    sobel_matrix_y_direction = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ])
    y_edges = convolve2d(img, sobel_matrix_y_direction, mode='same')

    return np.sqrt(np.add(np.square(x_edges), np.square(y_edges)))


def calculate_canny_image(img, low_thresh, high_thresh, blur_radius, std):
    img = np.array(img, dtype=np.uint8)
    img = cv2.GaussianBlur(img, blur_radius, std)
    edges = cv2.Canny(img, threshold1=low_thresh, threshold2=high_thresh)

    return edges


def find_circles_using_hough(img, canny_high_thresh, canny_low_thresh, min_dist_between_circles, blur_radius, std):
    img = np.array(img, dtype=np.uint8)
    img = cv2.GaussianBlur(img, blur_radius, std)
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, min_dist_between_circles,
                               param1=canny_high_thresh,
                               param2=canny_low_thresh,
                               minRadius=0,
                               maxRadius=0
                               )
    circles = np.uint16(np.around(circles))
    return circles


def find_lines_using_hough(img, canny_params, line_thresh):
    img = np.array(img, dtype=float)
    canny_img = calculate_canny_image(img, canny_params['low_thresh'], canny_params['high_thresh'],
                                      canny_params['blur_radius'], canny_params['std'])

    lines = cv2.HoughLines(canny_img, 1, np.pi / 180, line_thresh)

    return lines
