import cv2
import numpy as np

def is_circle_like(contour, min_area=20):
    area = cv2.contourArea(contour)
    if area < min_area:
        return False
    perimeter = cv2.arcLength(contour, True)
    if perimeter < 200:
        return False
    circularity = 4 * np.pi * (area / (perimeter * perimeter))
    print("Area: ", area)
    print("Perimeter: ", perimeter)
    print("Circularity: ", circularity)
    return 0 < circularity < 1000

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

def find_best_circles(contours, min_perimeter=150, min_elliptical_aspect_ratio=1, max_elliptical_aspect_ratio=3):
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
    if len(initial_cond_met) is not 0:
        ellipse_contours, contour_lenghts = fit_ellipse(initial_cond_met)
        for ellipse in ellipse_contours:
            center, axes, angle = ellipse
            major_axis = max(axes)
            minor_axis = min(axes)
            aspect_ratio = major_axis / minor_axis
            if min_elliptical_aspect_ratio <= aspect_ratio <= max_elliptical_aspect_ratio:
                ellipse_in_ARBounds.append(ellipse)

    return good_ones, ellipse_in_ARBounds
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


#Read image
img = cv2.imread('skinAngle.jpg', cv2.IMREAD_COLOR)

img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
height = img.shape[0]
width = img.shape[1]
print("Image size", width, " x ", height)
greyimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#blur using 3x3 kernel
#blurred = cv2.blur(greyimg, (3,3))
blurred = cv2.GaussianBlur(greyimg, (9,9), 2)
cv2.imshow("Blurred", blurred)

kernel = np.ones((8,8),np.uint8)

#Another try with opening instead
# Perform closing to remove hair and blur the image
closing = cv2.morphologyEx(img,cv2.MORPH_OPEN, kernel, iterations = 2)
blurred = cv2.GaussianBlur(closing, (9,9), 2)
#blurred = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
cv2.imshow("Blurred2", blurred)

edges = cv2.Canny(blurred,0,40, apertureSize=3, L2gradient=True)



cv2.imshow("edge img", edges)

contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# Make a copy of the original image to draw contours
contour_img = img.copy()

#circle_contours = [c for c in contours if is_circle_like(c, 20)]
#print(circle_contours)
circle_countours = []
i = 0
for c in contours:
    print("countour ", i)
    if is_circle_like(c,1):
        circle_countours.append(c)
    i+=1

# Draw all contours in green with thickness 2
cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
cv2.drawContours(contour_img, circle_countours, -1, (255, 0, 0), 2)

for i, contour in enumerate(contours):
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        cv2.putText(contour_img, str(i), (cX, cY),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 100), 1)
        print(i)


for i, contour in enumerate(circle_countours):
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        cv2.putText(contour_img, str(i), (cX, cY),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 200), 1)
        print(i)
        is_circle_like(contour,20)
# Show the result
cv2.imshow("Contours", contour_img)
        

cv2.waitKey()




circ_img = img.copy()

for i, contour in enumerate(circle_countours):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    if perimeter == 0:
        continue

    circularity = 4 * np.pi * (area / (perimeter * perimeter))

    # Draw all contours for reference
    color = (0, 255, 0) if 0.0001 < circularity < 1.2 else (100, 100, 255)  # green = likely circle
    cv2.drawContours(circ_img, [contour], -1, color, 2)

    # Compute center to draw text
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        label = f"{i}: {circularity:.2f}"
        cv2.putText(circ_img, label, (cX - 20, cY),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        

# Show final image
cv2.imshow("Circularity Visualization", circ_img)
cv2.waitKey(0)

#Trying ellipse fitting instead
ellipse_img = img.copy()
ellipses, arc_lens = fit_ellipse(contours)
upperBound = 2.5
lowerBound = .85
for e in ellipses:
    (center, axes, angle) = e
    major, minor = axes
    aspect_ratio = 0

    if major != 0:
        aspect_ratio = minor / major
    else:
        aspect_ratio = 0

    if lowerBound < aspect_ratio < upperBound:
        #Ellipses within bounds are green 
        cv2.ellipse(ellipse_img, e, (0, 255, 0), 2)
    elif aspect_ratio < lowerBound:
        #Ellipses less than bound are yellow
        #Very few come out yellow and those that do are tiny, so we do not need these
        cv2.ellipse(ellipse_img, e, (0, 255, 255), 2)
    elif upperBound < aspect_ratio:
        #Ellipses greater than upperbound will be purple
        cv2.ellipse(ellipse_img, e, (255, 0, 255), 2)
    else:
        cv2.ellipse(ellipse_img, e, (255, 45, 0), 2)

cv2.imshow("Ellipse img", ellipse_img)
cv2.waitKey(0)


#Ellipse fitting but checking how well the contour actually follows the ellipse
contour_follow = img.copy()

upperBound = 1.2
lowerBound = 0.2
for i in range(len(ellipses)):
    e = ellipses[i]
    (center, axes, angle) = e
    major, minor = axes
    aspect_ratio = 0

    if major != 0:
        aspect_ratio = minor / major
    else:
        aspect_ratio = 0

    ellipse_len = get_ellipse_perimeter(e)
    contour_len = arc_lens[i]
    ratio = ellipse_len / contour_len

    if lowerBound < ratio < upperBound:
        #Ellipses within bounds are green 
        cv2.ellipse(contour_follow, e, (0, 255, 0), 2)
    elif ratio < lowerBound:
        #Ellipses less than bound are yellow
        #Very few come out yellow and those that do are tiny, so we do not need these
        cv2.ellipse(contour_follow, e, (0, 255, 255), 2)
    elif upperBound < ratio:
        #Ellipses greater than upperbound will be purple
        cv2.ellipse(contour_follow, e, (255, 0, 255), 2)
    else:
        cv2.ellipse(contour_follow, e, (255, 45, 0), 2)

cv2.imshow("Contour follow img", contour_follow)
cv2.waitKey(0)

#Now trying a new method
detection_img = img.copy()
ellipse_detection_img = img.copy()
detected, detectedEllipse = find_best_circles2(contours, min_perimeter=100, min_ellipse_perimeter=100)
for ds in detected:
    cv2.drawContours(detection_img, [ds], -1, (0, 255, 0), 2)

k = 0
for es in detectedEllipse:
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
        cv2.ellipse(ellipse_detection_img, es, (0, 255, 0), 2)
    else:
        cv2.ellipse(ellipse_detection_img, es, (0, 0, 255), 2)

cv2.imshow("Detection image", detection_img)
cv2.imshow("Detection Ellipse image", ellipse_detection_img)
cv2.waitKey(0)


cv2.destroyAllWindows()