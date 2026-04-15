import streamlit as st
import numpy as np
import cv2
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from io import BytesIO

st.title("💎 COLOR VECTOR LOGO EXPORT (FINAL FIXED VERSION)")

uploaded_files = st.file_uploader(
    "Upload Images",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True
)

def preprocess(image):

    img = np.array(image.convert("RGB"))

    img = cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    white_pixel = np.sum(thresh == 255)
    black_pixel = np.sum(thresh == 0)

    if white_pixel > black_pixel:
        thresh = cv2.bitwise_not(thresh)

    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    return img, thresh


def create_pdf(original_image, binary):

    img = original_image

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return None

    h, w = binary.shape

    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)

    page_w, page_h = A4

    scale_x = page_w / w
    scale_y = page_h / h

    for contour in contours:

        if cv2.contourArea(contour) < 200:
            continue

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)

        pixels = img[mask == 255]

        if len(pixels) == 0:
            continue

        r, g, b = np.median(pixels, axis=0)
        r, g, b = r/255, g/255, b/255

        pts = contour.squeeze()

        if len(pts.shape) != 2:
            continue

        path = c.beginPath()

        x0, y0 = pts[0]
        path.moveTo(x0 * scale_x, page_h - y0 * scale_y)

        for x, y in pts:
            path.lineTo(x * scale_x, page_h - y * scale_y)

        path.close()

        c.setFillColorRGB(r, g, b)
        c.drawPath(path, fill=1, stroke=0)

    c.save()
    buffer.seek(0)

    return buffer


if uploaded_files:

    for idx, uploaded_file in enumerate(uploaded_files):

        st.subheader(f"🖼 Processing {idx+1}")

        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)

        original_img, binary = preprocess(image)

        st.image(binary, caption="Binary Mask", use_container_width=True)

        pdf_buffer = create_pdf(original_img, binary)

        if pdf_buffer:

            st.success("💎 PERFECT COLOR VECTOR PDF READY")

            st.download_button(
                f"⬇ Download PDF {idx+1}",
                pdf_buffer,
                file_name=f"vector_{idx+1}.pdf",
                mime="application/pdf"
            )

        else:
            st.error("No object detected")