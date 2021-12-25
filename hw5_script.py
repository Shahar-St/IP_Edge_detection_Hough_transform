import matplotlib.pyplot as plt
import cv2.cv2 as cv2

from hw5_functions import *


def main():
    print("----------------------------------------------\n")
    print_IDs()

    print("-----------------------Task 1----------------------\n")
    balls1_img = cv2.cvtColor(cv2.imread(r'images/balls1.tif'), cv2.COLOR_BGR2GRAY)

    balls1_sobel_edges = calculate_sobel_edge_detection(balls1_img)
    threshold = 90
    _, bw_img = cv2.threshold(balls1_sobel_edges, threshold, 255, cv2.THRESH_BINARY)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.title('Original')
    plt.imshow(balls1_img, cmap='gray', vmin=0, vmax=255)
    plt.subplot(1, 2, 2)
    plt.title('Sobel Image')
    plt.imshow(bw_img, cmap='gray', vmin=0, vmax=255)

    print("-----------------------Task 2----------------------\n")

    print("-----------------------Task 3----------------------\n")

    print("-----------------------Task 4----------------------\n")

    print("-----------------------Task 5----------------------\n")

    plt.show()


if __name__ == '__main__':
    main()
