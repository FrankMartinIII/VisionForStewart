import cv2
import numpy as np

def get_ellipse_perimeter(ellipse):
    center, axes, angle = ellipse
    major_axis = max(axes)
    minor_axis = min(axes)
    if minor_axis == 0:
        return -99999
    semi_maj = major_axis / 2
    semi_min = minor_axis / 2
    ellipse_perimeter = np.pi * (3 * (semi_maj+semi_min) - (np.sqrt((3 * semi_maj + semi_min) * (semi_maj + 3 * semi_min))))
    return ellipse_perimeter

def fit_ellipse(contours, min_length=5):
    ellipses = []
    contour_lens = []
    for contour in contours:
        if len(contour) >= min_length:
            ellipse = cv2.fitEllipse(contour)
            ellipses.append(ellipse)
            contour_len = cv2.arcLength(contour, True)
            contour_lens.append(contour_len)
    return ellipses, contour_lens

def find_best_circles2(contours, min_perimeter=0, min_ellipse_perimeter=0, min_elliptical_aspect_ratio=1, max_elliptical_aspect_ratio=3):
    #Function to find circles that best meet the required characteristics
    initial_cond_met = []
    ellipse_contours = []
    ellipse_in_ARBounds = []
    for contour in contours:
        #First eliminate really small contours
        perimeter = cv2.arcLength(contour, True)
        if perimeter > min_perimeter:
            initial_cond_met.append(contour)
    good_ones = initial_cond_met
    if len(initial_cond_met) != 0:
        ellipse_contours, contour_lengths = fit_ellipse(initial_cond_met)
        for ellipse in ellipse_contours:
            center, axes, angle = ellipse
            major_axis = max(axes)
            minor_axis = min(axes)
            if minor_axis != 0:
                #Anything with an axis of 0 is already a degenerate case
                ellipse_perimeter = get_ellipse_perimeter(ellipse)
                aspect_ratio = major_axis / minor_axis
                if (ellipse_perimeter >= min_ellipse_perimeter) and (min_elliptical_aspect_ratio <= aspect_ratio <= max_elliptical_aspect_ratio):
                    ellipse_in_ARBounds.append(ellipse)
    return good_ones, ellipse_in_ARBounds

def find_largest_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    return largest

def sort_ellipses_by_size(ellipse_list):
    def ellipse_area(ellipse):
        _, axes, _ = ellipse
        return np.pi * axes[0] * axes[1]  # major * minor

    return sorted(ellipse_list, key=ellipse_area, reverse=True)

def display_ellipses(ellipses, img):
    k = 0
    for es in ellipses:
        center, axes, angle = es
        major_axis = max(axes)
        minor_axis = min(axes)

        center, axes, angle = es
        major_axis = max(axes)
        minor_axis = min(axes)
        if minor_axis == 0:
            continue
        semi_maj = major_axis / 2
        semi_min = minor_axis / 2
        ellipse_perimeter = get_ellipse_perimeter(es)
        print(k, " perimeter: ", ellipse_perimeter)
        aspect_ratio = major_axis / minor_axis
        k+=1
        if aspect_ratio > 3:
            #Very elliptical will be green
            cv2.ellipse(img, es, (0, 255, 0), 2)
        else:
            cv2.ellipse(img, es, (0, 0, 255), 2)


def get_ellipse_mask(image_shape, ellipse):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    cv2.ellipse(mask, ellipse, 255, -1)  # filled white ellipse
    return mask