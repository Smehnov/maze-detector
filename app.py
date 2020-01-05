import cv2
import numpy as np
import argparse


def find_line_len(dot1, dot2):
    return np.sqrt((dot1[0] - dot2[0]) ** 2 + (dot1[1] - dot2[1]) ** 2)


def find_dots_for_rect(dots, img_w, img_h):
    dot1 = dots[0]  # left-up has min dist to (0;0)
    dot2 = dots[0]  # right-down has min dist to (img_w;img_h)
    dot3 = dots[0]
    dot4 = dots[0]

    for dot in dots:
        if find_line_len([0, 0], dot) < find_line_len([0, 0], dot1):
            dot1 = dot.copy()
        if find_line_len([img_w, img_h], dot) < find_line_len([img_w, img_h], dot3):
            dot3 = dot.copy()
        if find_line_len([img_w, 0], dot) < find_line_len([img_w, 0], dot2):
            dot2 = dot.copy()
        if find_line_len([0, img_h], dot) < find_line_len([0, img_h], dot4):
            dot4 = dot.copy()
    return np.array([dot1, dot2, dot3, dot4])


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    max_w = max(int(find_line_len(tr, tl)), int(find_line_len(br, bl)))
    max_h = max(int(find_line_len(tr, br)), int(find_line_len(tl, bl)))

    dst = np.array([
        [0, 0],
        [max_w - 1, 0],
        [max_w - 1, max_h - 1],
        [0, max_h - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (max_w, max_h))
    return warped


def get_background_color(img):
    box = img[1:10, 1:10, :]
    box[:, :] = box.mean(axis=0).mean(axis=0)
    box = cv2.cvtColor(box, cv2.COLOR_BGR2HSV)
    box[:, :, 1:3] = 0
    return np.array([box[0][0][0] - 5, 0, 0])


def remove_background(img):
    img = cv2.bilateralFilter(img, 16, 16, 16)
    lower = get_background_color(img)
    upper = np.array([255, 255, 255])
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.bitwise_not(mask)
    mask = cv2.erode(mask, np.ones((4, 4)), iterations=1)
    new_img = cv2.bitwise_and(img, img, mask=mask)
    return new_img, mask


def cut_photo(link):
    img = cv2.imread(link)
    new_img, mask = remove_background(img)
    contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    dots = []
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        for dot in approx:
            dots.append(dot[0])
    new_img = four_point_transform(new_img, find_dots_for_rect(dots, np.shape(img)[1], np.shape(img)[0]))
    return new_img


def get_bin_maze(img):
    bil = cv2.bilateralFilter(img, 10, 16, 16)
    box = bil[1:10, 1:10, :]
    box[:, :] = box.mean(axis=0).mean(axis=0)
    box = cv2.cvtColor(box, cv2.COLOR_BGR2HSV)
    color = box[0][0]
    lower = np.array([color[0] - 10, color[1] - 10, color[2] - 10])
    upper = np.array([color[0] + 10, color[1] + 10, color[2] + 10])
    hsv = cv2.cvtColor(bil, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.bitwise_not(mask)
    return mask


def resize_to(img, w=17, h=17):
    maze_img = get_bin_maze(img)
    small_maze = cv2.resize(maze_img, (w, h), interpolation=cv2.INTER_BITS)
    return small_maze


def print_matrix(mat):
    for i in mat:
        for j in i:
            print(j, " ", end='')
        print()


parser = argparse.ArgumentParser()
parser.add_argument("image_path")
args = parser.parse_args()
print(args.image_path)

img = cut_photo(args.image_path)
maze_img = get_bin_maze(img)
small_maze = resize_to(img)

small_maze_matrix = 1 - np.array(np.where(small_maze < 128, 0, 1))
print_matrix(small_maze_matrix)
