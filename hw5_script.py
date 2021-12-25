import matplotlib.pyplot as plt

from hw5_functions import *


def main():
    print("----------------------------------------------\n")
    print_IDs()

    print("-----------------------Task 1 - Sobel Edge Detection----------------------\n")
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

    print("-----------------------Task 2 - Canny Edge Detection----------------------\n")
    coins1_img = cv2.cvtColor(cv2.imread(r'images/coins1.tif'), cv2.COLOR_BGR2GRAY)
    # todo play w the params
    sobel_mat_size = 5
    low_thresh, high_thresh = 3000, 4000
    canny_img = calculate_canny_image(coins1_img, low_thresh, high_thresh, sobel_mat_size)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.title('Original')
    plt.imshow(coins1_img, cmap='gray', vmin=0, vmax=255)
    plt.subplot(1, 2, 2)
    plt.title('Canny Image')
    plt.imshow(canny_img, cmap='gray', vmin=0, vmax=255)

    print("-----------------Task 3 - Canny Edge Detection – Challenge----------------\n")

    print("--------------------Task 4 - Hough Transform – Circles--------------------\n")

    print("---------------------Task 5 - Hough Transform – Lines---------------------\n")

    plt.show()


if __name__ == '__main__':
    main()
