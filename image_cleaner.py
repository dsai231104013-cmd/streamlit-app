import cv2
import numpy as np

def clean_image(image_path, output_path):
    img = cv2.imread(image_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 🔥 Strong binary (solid shapes)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    # Invert (logo becomes white)
    thresh = 255 - thresh

    # 🔥 Fill gaps
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # 🔥 Remove noise
    thresh = cv2.medianBlur(thresh, 5)

    cv2.imwrite(output_path, thresh)
    return output_path