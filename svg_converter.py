import cv2
import numpy as np
import svgwrite


def convert_to_svg(image_path, output_svg):
    img = cv2.imread(image_path, 0)

    # Strong binary
    _, thresh = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY)

    # Invert
    thresh = 255 - thresh

    # Find contours (HIGH DETAIL)
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    dwg = svgwrite.Drawing(output_svg)

    for cnt in contours:
        # 🔥 VERY SMOOTH
        epsilon = 0.002 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        path = "M "
        for point in approx:
            x, y = point[0]
            path += f"{x},{y} "

        path += "Z"

        dwg.add(dwg.path(d=path, fill="black"))

    dwg.save()

    return output_svg