import cv2
import numpy as np
from os import walk

k_class = 0
labels = []

files_all = next(walk('images/'), (None, None, []))[2]

def starts_img(w: str):
    return w.startswith("img")

def starts_mask(w: str):
    return w.startswith("mask")

img_files = list(filter(starts_img, files_all))
mask_files = list(filter(starts_mask, files_all))

files = list(zip(img_files, mask_files))

print(files)

MIN_AREA = 30  # minimum blob area

for (img_f, mask_f) in files:
    img_path = f'images/{img_f}'
    mask_path = f'images/{mask_f}'

    frame = cv2.imread(mask_path)
    if frame is None:
        continue

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_white = np.array([0, 0, 240], dtype=np.uint8)
    upper_white = np.array([255, 5, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_white, upper_white)

    # remvoe stray white pixels
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # locate blobs of white pixels
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        continue
    large_contours = [c for c in contours if cv2.contourArea(c) >= MIN_AREA] # dont allow for very small blovbs to affect the label

    if len(large_contours) == 0:
        continue

    # every blob into one :)
    all_points = np.vstack(large_contours)

    reconstructed_mask = np.zeros_like(mask)

    cv2.drawContours(reconstructed_mask, large_contours, -1, 255, thickness=cv2.FILLED)

    # cv2.imshow("reconstructed_mask", reconstructed_mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    x, y, w, h = cv2.boundingRect(all_points)

    img_h, img_w = frame.shape[:2]

    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    width = w / img_w
    height = h / img_h

    label_line = f"{k_class} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
    labels.append(label_line)

    label_path = img_path.replace('images', 'labels').replace('.png', '.txt')

    with open(label_path, 'w') as f:
        f.write(label_line + '\n')

print(labels)
print(len(labels))