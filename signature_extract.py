import cv2
import numpy as np

def extract_signature_region(image_path):
    """Extract the dark ink region robustly (consistent crop even on same cheque)."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    # invert binary with mild threshold, keeps writing lines
    _, mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    # dilate to connect strokes, ensures same region each time
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return cv2.resize(gray, (300,120))

    # find bounding box around *all* contours (not only largest)
    x_min = min(cv2.boundingRect(c)[0] for c in contours)
    y_min = min(cv2.boundingRect(c)[1] for c in contours)
    x_max = max(cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] for c in contours)
    y_max = max(cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] for c in contours)
    roi = gray[y_min:y_max, x_min:x_max]

    # normalize output
    roi = cv2.resize(roi, (300,120))
    roi = cv2.equalizeHist(roi)
    return roi
