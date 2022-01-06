import math

import matplotlib.pyplot as plt

from hw5_functions import *


def main():
    print("----------------------------------------------\n")
    print_IDs()

    print("-----------------------Task 1 - Sobel Edge Detection----------------------\n")
    balls1_img = cv2.cvtColor(cv2.imread(r'images/balls1.tif'), cv2.COLOR_BGR2GRAY)

    balls1_sobel_edges = calculate_sobel_edge_detection(balls1_img)
    threshold = 98
    _, bw_img = cv2.threshold(balls1_sobel_edges, threshold, 255, cv2.THRESH_BINARY)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.title('Original')
    plt.imshow(balls1_img, cmap='gray', vmin=0, vmax=255)
    plt.subplot(1, 2, 2)
    plt.title('section1: Sobel Image')
    plt.imshow(bw_img, cmap='gray', vmin=0, vmax=255)

    print("-----------------------Task 2 - Canny Edge Detection----------------------\n")
    coins1_img = cv2.cvtColor(cv2.imread(r'images/coins1.tif'), cv2.COLOR_BGR2GRAY)

    low_thresh, high_thresh = 50, 169
    blur_radius = (11, 11)
    std = 2.4
    canny_img = calculate_canny_image(coins1_img, low_thresh, high_thresh, blur_radius, std)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.title('Original')
    plt.imshow(coins1_img, cmap='gray', vmin=0, vmax=255)
    plt.subplot(1, 2, 2)
    plt.title('section2: Canny Image')
    plt.imshow(canny_img, cmap='gray', vmin=0, vmax=255)

    print("-----------------Task 3 - Canny Edge Detection – Challenge----------------\n")
    balls1_img = cv2.cvtColor(cv2.imread(r'images/balls1.tif'), cv2.COLOR_BGR2GRAY)

    low_thresh, high_thresh = 70, 150
    blur_radius = (1, 1)
    std = 1
    canny_img = calculate_canny_image(balls1_img, low_thresh, high_thresh, blur_radius, std)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.title('Original')
    plt.imshow(balls1_img, cmap='gray', vmin=0, vmax=255)
    plt.subplot(1, 2, 2)
    plt.title('section3: Canny Image')
    plt.imshow(canny_img, cmap='gray', vmin=0, vmax=255)

    print("--------------------Task 4 - Hough Transform – Circles--------------------\n")
    coins3_img = cv2.cvtColor(cv2.imread(r'images/coins3.tif'), cv2.COLOR_BGR2GRAY)

    canny_high_threshold = 100
    canny_low_threshold = 50
    min_dist_between_circles = 15
    blur_radius = (11, 11)
    std = 3
    circles = find_circles_using_hough(coins3_img, canny_high_threshold,
                                       canny_low_threshold, min_dist_between_circles, blur_radius, std)

    # draw circles
    img_with_circles = np.copy(coins3_img)
    for circle in circles[0, :]:
        cv2.circle(img_with_circles, (circle[0], circle[1]), circle[2], (0, 255, 0), 2)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.title('Original')
    plt.imshow(coins3_img, cmap='gray', vmin=0, vmax=255)
    plt.subplot(1, 2, 2)
    plt.title('section4: Hough Circles')
    plt.imshow(img_with_circles, cmap='gray', vmin=0, vmax=255)

    print("---------------------Task 5 - Hough Transform – Lines---------------------\n")
    box1_name = 'boxOfChocolates1.tif'
    box1_img = cv2.cvtColor(cv2.imread(fr'images/{box1_name}'), cv2.COLOR_BGR2GRAY)
    canny_params = {
        'low_thresh': 0,
        'high_thresh': 100,
        'blur_radius': (7, 7),
        'std': 3,
    }
    line_thresh = 134
    found_lines_1 = find_lines_using_hough(box1_img, canny_params, line_thresh)

    box2_name = 'boxOfChocolates2.tif'
    box2_img = cv2.cvtColor(cv2.imread(fr'images/{box2_name}'), cv2.COLOR_BGR2GRAY)
    canny_params = {
        'low_thresh': 35,
        'high_thresh': 35,
        'blur_radius': (11, 11),
        'std': 1,
    }
    line_thresh = 92
    found_lines_2 = find_lines_using_hough(box2_img, canny_params, line_thresh)

    box3_name = 'boxOfChocolates2rot.tif'
    box3_img = cv2.cvtColor(cv2.imread(fr'images/{box3_name}'), cv2.COLOR_BGR2GRAY)
    canny_params = {
        'low_thresh': 20,
        'high_thresh': 20,
        'blur_radius': (7, 7),
        'std': 2,
    }
    line_thresh = 90
    found_lines_3 = find_lines_using_hough(box3_img, canny_params, line_thresh)

    hough_lines_arr = (
        (box1_name, box1_img, found_lines_1, 14), (box2_name, box2_img, found_lines_2, 12),
        (box3_name, box3_img, found_lines_3, 12))

    img_num = 1
    for hough_lines in hough_lines_arr:
        file__name = hough_lines[0]
        original = hough_lines[1]
        lines = hough_lines[2]
        lines_on_image = np.copy(original)

        # remove duplicated lines
        thresh_rho = hough_lines[3]
        if img_num == 2:
            thresh_theta = 1
        else:
            thresh_theta = 0

        lines_to_draw = [lines[0]]
        for i in range(len(lines)):
            append_flag = True
            for j in range(len(lines_to_draw)):
                if abs(lines_to_draw[j][0][0] - lines[i][0][0]) < thresh_rho or lines[i][0][1] < thresh_theta:
                    append_flag = False

            if append_flag:
                lines_to_draw.append(lines[i])

        # draw lines
        for line in lines_to_draw:
            rho = line[0][0]
            theta = line[0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * a))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * a))
            cv2.line(lines_on_image, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)


        plt.figure()
        plt.subplot(1, 2, 1)
        plt.title('Original')
        plt.imshow(original, vmin=0, vmax=255)
        plt.subplot(1, 2, 2)
        plt.title(f'section5: Hough Lines - {file__name}')
        plt.imshow(lines_on_image, vmin=0, vmax=255)

        img_num += 1

    plt.show()


if __name__ == '__main__':
    main()
