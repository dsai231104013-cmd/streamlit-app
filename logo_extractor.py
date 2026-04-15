import cv2
import numpy as np

def extract_logo(input_path, output_path):
    img = cv2.imread(input_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 🔥 Better threshold
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    # 🔥 Morphology (remove noise)
    kernel = np.ones((5,5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return input_path

    # 🔥 Largest object detection
    largest = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(largest)

    # Add padding (important)
    pad = 10
    x = max(0, x-pad)
    y = max(0, y-pad)
    w = min(img.shape[1]-x, w+pad*2)
    h = min(img.shape[0]-y, h+pad*2)

    cropped = img[y:y+h, x:x+w]

    cv2.imwrite(output_path, cropped)
    return output_path