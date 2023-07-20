import os, sys

import matplotlib.pyplot as plt
import numpy as np
import cv2

target_board_values = [
        255, 250, 
         11,  12,  13, 
        181, 182, 183, 
         41,  42,  43, 
        131, 132, 133, 
         61,  62,  63, 
        101, 102, 103,
        151, 152, 153,
         21,  22,  23,
        171, 172, 173,
         31,  32,  33,
        191, 192, 193,
         71,  72,  73,
        161, 162, 163,
         81,  82,  83,
        111, 112, 113,
        141, 142, 143,
         91,  92,  93,
        121, 122, 123,
         51,  52,  53,
        201, 202, 203    
    ]
board_scores = [(0, score, 0) for score in target_board_values]


def find_image_type(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    mask_red1 = cv2.inRange(hsv_image, lower_red, upper_red)
    mask_red2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    mask_red = mask_red1 + mask_red2
    mask_red = 1 * (mask_red > 0)
    red_pixel_diffs = []

    # Type1:
    num_red = mask_red[620:670,220:270].flatten().sum()
    p_type1 = num_red / mask_red[620:670,220:270].flatten().shape[0]
    red_pixel_diffs.append(p_type1)

    # Type2:
    num_red = mask_red[65:102, 641:679].flatten().sum()
    p_type2 = num_red / mask_red[65:102, 641:679].flatten().shape[0]
    red_pixel_diffs.append(p_type2)
    
    #Type3: 
    num_red = mask_red[625:665, 634:673].flatten().sum()
    p_type3 = num_red / mask_red[625:665, 634:673].flatten().shape[0]
    red_pixel_diffs.append(p_type3)    
    return 1 + np.argmax(np.array(red_pixel_diffs))


def get_first_last_frame_difference(video_path):
    video_capture = cv2.VideoCapture(video_path)
    ret, first_frame = video_capture.read()
    while ret:
        ret, frame = video_capture.read()
        if ret:
            last_frame = frame
    video_capture.release()

    first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    last_gray = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)
    frame_difference = cv2.absdiff(first_gray, last_gray)
    _, mask = cv2.threshold(frame_difference, 50, 255, cv2.THRESH_BINARY)
    return first_frame, last_frame, mask


def decode_score(score):
    if score == 255:
        return "b50"
    if score == 250:
        return "b25" 
    
    multipliers = {1: "s", 2: "d", 3: "t"}
    multiplier = multipliers[score % 10]
    return f"{multiplier}{score//10}"


def find_location(tip, mask):
    if len(mask.shape) > 2:
        mask = mask[:, :, 1]
    margin = 4
    y, x = tip
    sample = mask[x - margin : x + margin, y - margin : y + margin].flatten()
    unique_elements, element_counts = np.unique(sample, return_counts=True)

    most_common_index = np.argmax(element_counts)
    score =  unique_elements[most_common_index]
    return decode_score(score)


def find_leftmost_point(contour):
    leftmost_point = None
    contour = contour.squeeze()
    min_x = np.min(contour[:, 0])
    if leftmost_point is None or min_x < leftmost_point[0]:
        leftmost_point = (min_x, contour[contour[:, 0].argmin(), 1])
    return leftmost_point


def find_score(frame_difference, type):
    img = frame_difference.copy()
    mask = cv2.imread(f"task3_type{type}mask.png")    
    segments = np.isin(mask, board_scores).all(axis=-1)
    img[~segments] = 0
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = np.concatenate(contours, axis=0)
    tip = find_leftmost_point(contours)
    score = find_location(tip, mask)
    return score


def solve_task_3(video_path):
    first_frame, last_frame, frame_difference = get_first_last_frame_difference(video_path)
    type = find_image_type(first_frame)
    score = find_score(frame_difference, type)
    return score

if __name__ == "__main__":
    video_path = sys.argv[1]
    print(video_path)
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Path '{video_path}' could not be found")
    score = solve_task_3(video_path)
    print(score)