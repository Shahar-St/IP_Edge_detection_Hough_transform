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

    return np.sqrt(np.square(x_edges) + np.square(y_edges))


def calculate_canny_image(img, low_thresh, high_thresh, sobel_mat_size):
    img = np.array(img, dtype=np.uint8)
    edges = cv2.Canny(img, threshold1=low_thresh, threshold2=high_thresh, apertureSize=sobel_mat_size, L2gradient=True)
    return edges


def find_circles_using_hough(img, canny_high_threshold, canny_low_threshold, min_dist_between_circles):
    img = np.array(img, dtype=np.uint8)
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, min_dist_between_circles,
                               param1=canny_high_threshold,
                               param2=canny_low_threshold,
                               minRadius=0,
                               maxRadius=0
                               )
    circles = np.uint16(np.around(circles))
    return circles


def find_lines_using_hough(img, canny_params):
    img = np.array(img, dtype=float)
    canny_img = calculate_canny_image(img, canny_params['low_thresh'], canny_params['high_thresh'],
                                      canny_params['sobel_mat_size'])

    # todo investigate more on the params
    #  https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html
    lines = cv2.HoughLines(canny_img, 1, np.pi / 180, 200)

    return lines
