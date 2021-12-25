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
    threshold = 90
    _, bw_img = cv2.threshold(canny_img, threshold, 255, cv2.THRESH_BINARY)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.title('Original')
    plt.imshow(coins1_img, cmap='gray', vmin=0, vmax=255)
    plt.subplot(1, 2, 2)
    plt.title('Canny Image')
    plt.imshow(bw_img, cmap='gray', vmin=0, vmax=255)

    print("-----------------Task 3 - Canny Edge Detection – Challenge----------------\n")
    balls1_img = cv2.cvtColor(cv2.imread(r'images/balls1.tif'), cv2.COLOR_BGR2GRAY)
    # todo play w the params
    sobel_mat_size = 5  # can be 3 to 7
    low_thresh, high_thresh = 3000, 4000
    canny_img = calculate_canny_image(balls1_img, low_thresh, high_thresh, sobel_mat_size)
    threshold = 90
    _, bw_img = cv2.threshold(canny_img, threshold, 255, cv2.THRESH_BINARY)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.title('Original')
    plt.imshow(balls1_img, cmap='gray', vmin=0, vmax=255)
    plt.subplot(1, 2, 2)
    plt.title('Canny Image')
    plt.imshow(bw_img, cmap='gray', vmin=0, vmax=255)

    print("--------------------Task 4 - Hough Transform – Circles--------------------\n")
    coins3_img = cv2.cvtColor(cv2.imread(r'images/coins3.tif'), cv2.COLOR_BGR2GRAY)
    # todo very close but need to play w the params
    canny_high_threshold = 500
    canny_low_threshold = 60
    min_dist_between_circles = 10
    circles = find_circles_using_hough(coins3_img, canny_high_threshold,
                                       canny_low_threshold, min_dist_between_circles)

    # draw circles
    img_with_circles = np.copy(coins3_img)
    for circle in circles[0, :]:
        # draw the outer circle
        cv2.circle(img_with_circles, (circle[0], circle[1]), circle[2], (0, 255, 0), 2)
        # draw the center of the circle
        cv2.circle(img_with_circles, (circle[0], circle[1]), 2, (0, 0, 255), 3)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.title('Original')
    plt.imshow(coins3_img, cmap='gray', vmin=0, vmax=255)
    plt.subplot(1, 2, 2)
    plt.title('Hough Circles')
    plt.imshow(img_with_circles, cmap='gray', vmin=0, vmax=255)

    print("---------------------Task 5 - Hough Transform – Lines---------------------\n")
    box1_name = 'boxOfChocolates1.tif'
    box1_img = cv2.cvtColor(cv2.imread(fr'images/{box1_name}'), cv2.COLOR_BGR2GRAY)
    canny_params = {
        'low_thresh': 50,  # todo play
        'high_thresh': 20,
        'sobel_mat_size': 3
    }
    found_lines_1 = find_lines_using_hough(box1_img, canny_params)

    box2_name = 'boxOfChocolates2.tif'
    box2_img = cv2.cvtColor(cv2.imread(fr'images/{box2_name}'), cv2.COLOR_BGR2GRAY)
    canny_params = {
        'low_thresh': 50,  # todo play
        'high_thresh': 20,
        'sobel_mat_size': 3
    }
    found_lines_2 = find_lines_using_hough(box2_img, canny_params)

    box3_name = 'boxOfChocolates2rot.tif'
    box3_img = cv2.cvtColor(cv2.imread(fr'images/{box3_name}'), cv2.COLOR_BGR2GRAY)
    canny_params = {
        'low_thresh': 100,  # todo play
        'high_thresh': 20,
        'sobel_mat_size': 3
    }
    found_lines_3 = find_lines_using_hough(box3_img, canny_params)

    hough_lines_arr = (
        (box1_name, box1_img, found_lines_1), (box2_name, box2_img, found_lines_2),
        (box3_name, box3_img, found_lines_3))
    for hough_lines in hough_lines_arr:
        file__name = hough_lines[0]
        original = hough_lines[1]
        lines = hough_lines[2]
        lines_on_image = np.copy(original)
        # todo - this line is generally not needed and only here because we dont find lines in 2 and 3
        if lines is not None and len(lines) > 0:
            for rho, theta in lines[0]:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * a)
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * a)
                cv2.line(lines_on_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

        plt.figure()
        plt.subplot(1, 2, 1)
        plt.title('Original')
        plt.imshow(original, cmap='gray', vmin=0, vmax=255)
        plt.subplot(1, 2, 2)
        plt.title(f'Hough Lines - {file__name}')
        plt.imshow(lines_on_image, cmap='gray', vmin=0, vmax=255)

    plt.show()


if __name__ == '__main__':
    main()
