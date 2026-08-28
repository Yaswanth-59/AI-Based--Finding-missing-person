import os
import cv2
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
CASCADE_PATH = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")


def _detector():
    detector = cv2.CascadeClassifier(CASCADE_PATH)
    if detector.empty():
        raise RuntimeError(f"Could not load face detector: {CASCADE_PATH}")
    return detector


def _preprocess_gray(image):
    """Convert RGB/BGR/grayscale input to a clean grayscale image."""
    if image is None:
        return None

    if image.ndim == 2:
        gray = image.copy()
    elif image.shape[2] == 4:
        rgb = image[:, :, :3]
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    else:
        # This project loads uploaded images with PIL as RGB.
        rgb = image[:, :, :3]
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return gray


def detect_all_faces(image, max_faces=5):
    """CPU-only Haar face detection with light preprocessing."""
    gray = _preprocess_gray(image)
    if gray is None:
        return []

    detector = _detector()
    h, w = gray.shape[:2]
    min_side = max(50, int(min(h, w) * 0.08))

    boxes = detector.detectMultiScale(
        gray,
        scaleFactor=1.08,
        minNeighbors=6,
        minSize=(min_side, min_side),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)[:max_faces]

    faces = []
    for (x, y, fw, fh) in boxes:
        faces.append({
            "landmarks": [x / w, y / h, (x + fw) / w, (y + fh) / h],
            "bbox": (int(x), int(y), int(x + fw), int(y + fh)),
            "area_ratio": float((fw * fh) / max(1, w * h)),
        })
    return faces


def face_crop_from_image(image, face_index=0):
    """Return a padded crop around a detected face."""
    faces = detect_all_faces(image, max_faces=10)
    if not faces:
        return None

    face_index = max(0, min(int(face_index), len(faces) - 1))
    x1, y1, x2, y2 = faces[face_index]["bbox"]
    h, w = image.shape[:2]

    pad_x = int((x2 - x1) * 0.22)
    pad_y = int((y2 - y1) * 0.28)
    x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    x2, y2 = min(w, x2 + pad_x), min(h, y2 + pad_y)

    crop = image[y1:y2, x1:x2]
    return crop if crop.size else None


def assess_face_quality(image, bbox):
    """Return a simple quality report for the detected face."""
    x1, y1, x2, y2 = bbox
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    crop = image[y1:y2, x1:x2]

    if crop.size == 0:
        return {"ok": False, "score": 0, "message": "Face crop is empty."}

    if crop.ndim == 3:
        gray = cv2.cvtColor(crop[:, :, :3], cv2.COLOR_RGB2GRAY)
    else:
        gray = crop

    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray))
    area_ratio = float((x2 - x1) * (y2 - y1) / max(1, w * h))

    # These are screening thresholds, not medical/identity guarantees.
    sharp_ok = sharpness >= 45.0
    bright_ok = 35.0 <= brightness <= 220.0
    size_ok = area_ratio >= 0.015
    ok = sharp_ok and bright_ok and size_ok

    score = 100.0
    if not sharp_ok:
        score -= 35.0
    if not bright_ok:
        score -= 25.0
    if not size_ok:
        score -= 30.0
    score = max(0.0, score)

    if not sharp_ok:
        message = "Face is blurry. Use a sharper photo."
    elif not bright_ok:
        message = "Lighting is poor. Use a brighter, evenly lit photo."
    elif not size_ok:
        message = "Face is too small. Upload a closer photo."
    else:
        message = "Good face quality."

    return {"ok": ok, "score": score, "message": message}
