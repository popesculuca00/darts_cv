import os, sys, glob

import cv2
import numpy as np
from sklearn.cluster import KMeans


def compute_new_objects(current_image, background):
    current_gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.GaussianBlur(current_gray, (5, 5), 0)
    background = cv2.GaussianBlur(background, (5, 5), 0)
    diff = cv2.absdiff(current_gray, background)
    _, mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel=np.ones((11, 11), dtype=np.uint8))
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours, hierarchy


def get_median_pixels(img_dir):
    images = glob.glob(f"{img_dir}/*.jpg")
    image_arrays = [cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2GRAY) for image in images]
    pixel_medians = np.median(image_arrays, axis=0).astype(np.uint8)
    return pixel_medians


def find_flags(path):
    image = cv2.imread(path)
    hsv_image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2HSV)
    
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])

    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([130, 255, 255])

    red_mask = cv2.inRange(hsv_image, lower_red, upper_red)
    blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
    british_flag_mask = cv2.bitwise_or(red_mask, blue_mask)

    kernel_size = 9
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    british_flag_mask = cv2.dilate(british_flag_mask, kernel, iterations=3)
    british_flag_mask = cv2.erode(british_flag_mask, kernel, iterations=3)

    contours, _ = cv2.findContours(british_flag_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    arrow_flags = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if len(arrow_flags)>2 or area<19_000:
            continue
        arrow_flags.append(cv2.boundingRect(contour))
    return arrow_flags


def get_contour_centroid(contour):
    eps = 1e-10
    moments = cv2.moments(contour)
    centroid_x = int(moments["m10"] / (moments["m00"] + eps))
    centroid_y = int(moments["m01"] / (moments["m00"] + eps))
    return [centroid_x, centroid_y]


def cluster_contours(contours, n_clusters):
    centroids = np.array([get_contour_centroid(contour) for contour in contours])
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(centroids)
    labels = kmeans.labels_
    clustered_contours = {}

    for label in range(n_clusters):
        indices = np.where(labels == label)[0]
        clustered_contours[label] = [contours[idx] for idx in indices]
    return clustered_contours


def remove_points_inside_boxes(contours, bound_rects):
    filtered_contours = []
    for contour in contours:
        points_inside = []
        for point in contour:
            is_inside = False
            for rect in bound_rects:
                x, y, w, h = rect
                if x <= point[0][0] <= x + w and y <= point[0][1] <= y + h:
                    is_inside = True
                    break
            if not is_inside:
                points_inside.append(point)
        if len(points_inside) > 0:
            filtered_contours.append(np.array(points_inside))
    return filtered_contours


def draw_circle_around_leftmost(contours, image):
    output_image = image.copy()
    for contour_cluster in contours:
        leftmost_point = tuple(contour_cluster[contour_cluster[:, :, 0].argmin()][0])
        cv2.circle(output_image, leftmost_point, 10, (0, 255, 0), 2)
    return output_image


def find_leftmost_point(contour):
    leftmost_point = None
    contour = contour.squeeze()
    min_x = np.min(contour[:, 0])
    if leftmost_point is None or min_x < leftmost_point[0]:
        leftmost_point = (min_x, contour[contour[:, 0].argmin(), 1])
    return leftmost_point


def find_location(tip):
    mask = cv2.imread("mask_task1.png")[:, :, 1]
    mask = np.clip(mask - 100, 0, 255)
    margin = 4
    y, x = tip
    
    sample = mask[x - margin : x + margin, y - margin : y + margin].flatten()
    unique_elements, element_counts = np.unique(sample, return_counts=True)

    most_common_index = np.argmax(element_counts)
    return unique_elements[most_common_index]


if __name__ == "__main__":
    
    path = sys.argv[1]
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path '{path}' could not be found")

    median_img = get_median_pixels("train/Task1")    
    
    img = cv2.imread(path)
    contours, hierarchy = compute_new_objects(img, median_img)

    flags = find_flags(path)
    n_clusters = len(flags)


    flags = [ (x-40, y-40, w+40, h+40) for (x,y,w,h) in flags ]

    contours = remove_points_inside_boxes(contours, flags)

    clustered_contours = cluster_contours(contours, n_clusters)
    scores = []

    output_image = np.zeros_like(img, dtype=np.uint8)
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # Define different colors for clusters
    for i, label in enumerate(clustered_contours):


        big_contour =  np.concatenate(clustered_contours[label], axis=0)
        tip = find_leftmost_point(big_contour)
        scores.append(find_location(tip))
        cv2.circle(img, tip, 20, (0, 255, 0), 10)   
    
    print(sorted(scores))
    