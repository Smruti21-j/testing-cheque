import cv2
import numpy as np

def verify_signatures(cheque_file, reference_file, threshold=70):
    """
    Compare cheque signature and reference signature using ORB feature matching.
    Returns result ("Signature Match"/"Signature Mismatch") and confidence score.
    """

    # Read uploaded images directly from Flask file object
    img1 = cv2.imdecode(np.frombuffer(cheque_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imdecode(np.frombuffer(reference_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)

    # Initialize ORB detector
    orb = cv2.ORB_create(nfeatures=1000)

    # Find keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        return "Unable to detect signature features", 0.0

    # Match descriptors using brute-force matcher (Hamming distance)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    if len(matches) == 0:
        return "Signature Mismatch", 0.0

    # Sort by distance (lower = better)
    matches = sorted(matches, key=lambda x: x.distance)

    # Good matches: distance < 60 is considered strong similarity
    good_matches = [m for m in matches if m.distance < 60]

    confidence = (len(good_matches) / len(matches)) * 100
    result = "Signature Match" if confidence >= threshold else "Signature Mismatch"

    return result, round(confidence, 2)
