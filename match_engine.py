import os
import cv2
import numpy as np
from sqlmodel import Session, select
from model.data_models import RegisteredCases

BASE = os.path.dirname(os.path.abspath(__file__))
UPLOAD = os.path.join(BASE, "static", "uploads")

# Screening thresholds. These should be validated with your own test dataset.
HIGH_MATCH = 80.0
REVIEW_MATCH = 60.0


def _case_photo(case_id):
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        p = os.path.join(UPLOAD, case_id + ext)
        if os.path.exists(p):
            return p
    return None


def _prepare_face(image, bbox=None, size=(160, 160)):
    if image is None:
        return None

    if image.ndim == 3:
        gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    if bbox:
        x1, y1, x2, y2 = map(int, bbox)
        fw, fh = x2 - x1, y2 - y1
        px, py = int(fw * 0.22), int(fh * 0.28)
        x1, y1 = max(0, x1 - px), max(0, y1 - py)
        x2, y2 = min(gray.shape[1], x2 + px), min(gray.shape[0], y2 + py)
        gray = gray[y1:y2, x1:x2]

    if gray.size == 0:
        return None

    gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    gray = cv2.equalizeHist(gray)
    return gray


def _variants(face):
    """Create small illumination-normalized variants for more stable matching."""
    variants = [face]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    variants.append(clahe.apply(face))
    return variants


def _lbph_available():
    return hasattr(cv2, "face") and hasattr(cv2.face, "LBPHFaceRecognizer_create")


def _lbph_distance(reference_face, query_face):
    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=2, neighbors=16, grid_x=8, grid_y=8
    )
    recognizer.train([reference_face], np.array([1], dtype=np.int32))
    _, distance = recognizer.predict(query_face)
    return float(distance)


def _normalized_correlation(a, b):
    a = a.astype(np.float32).reshape(-1)
    b = b.astype(np.float32).reshape(-1)
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-8
    corr = float(np.dot(a, b) / denom)
    return max(-1.0, min(1.0, corr))


def _score_from_lbph(distance):
    # This is a screening similarity score, NOT a calibrated probability.
    return max(0.0, min(100.0, 100.0 * np.exp(-distance / 55.0)))


def _score_from_pixel(distance):
    return max(0.0, min(100.0, 100.0 * np.exp(-distance * 3.2)))


def _classification(score):
    if score >= HIGH_MATCH:
        return "High Match", "good", "Strong candidate — verify manually."
    if score >= REVIEW_MATCH:
        return "Possible Match", "review", "Needs manual review."
    return "Low Match", "missing", "Low similarity; do not treat as a match."


def _compare_pair(reference_face, query_face):
    ref_variants = _variants(reference_face)
    query_variants = _variants(query_face)

    if _lbph_available():
        distances = []
        for ref in ref_variants:
            for query in query_variants:
                distances.append(_lbph_distance(ref, query))
        # Median reduces the effect of one poor illumination variant.
        distance = float(np.median(distances))
        lbph_score = _score_from_lbph(distance)
        corr = _normalized_correlation(ref_variants[0], query_variants[0])
        correlation_score = ((corr + 1.0) / 2.0) * 100.0
        # LBPH is primary; correlation is only a small stabilizer.
        similarity = (lbph_score * 0.85) + (correlation_score * 0.15)
        return float(similarity), distance, "LBPH + normalized correlation", "LBPH distance"

    # Fallback if opencv-contrib is unavailable.
    a = reference_face.astype(np.float32).reshape(-1)
    b = query_face.astype(np.float32).reshape(-1)
    a = (a - a.mean()) / (a.std() + 1e-6)
    b = (b - b.mean()) / (b.std() + 1e-6)
    cosine = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
    cosine = max(-1.0, min(1.0, cosine))
    distance = 1.0 - cosine
    return _score_from_pixel(distance), distance, "Normalized feature fallback", "Feature distance"


def compare_uploaded_photo(query_image, detector, max_results=5):
    """
    Compare one uploaded found-person photo against unresolved registered cases.

    Returns ranked screening results. Scores are similarity scores and should
    not be interpreted as calibrated identity probabilities.
    """
    from ai_utils import detect_all_faces, assess_face_quality

    query_faces = detect_all_faces(query_image, max_faces=1)
    if not query_faces:
        return [], "No face detected in the uploaded photo. Use a clear front-facing photo."

    quality = assess_face_quality(query_image, query_faces[0]["bbox"])
    if not quality["ok"]:
        return [], f"Image quality is too low: {quality['message']}"

    q = _prepare_face(query_image, query_faces[0]["bbox"])
    if q is None:
        return [], "Could not prepare the detected face."

    with Session(detector) as db:
        regs = db.exec(
            select(RegisteredCases).where(RegisteredCases.status == "NF")
        ).all()

    results = []
    for case in regs:
        photo_path = _case_photo(case.id)
        if not photo_path:
            continue
        try:
            raw = cv2.imread(photo_path)
            if raw is None:
                continue
            rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
            faces = detect_all_faces(rgb, max_faces=1)
            if not faces:
                continue

            ref_quality = assess_face_quality(rgb, faces[0]["bbox"])
            if not ref_quality["ok"]:
                continue

            ref = _prepare_face(rgb, faces[0]["bbox"])
            if ref is None:
                continue

            similarity, distance, method, distance_label = _compare_pair(ref, q)
            label, css_class, recommendation = _classification(similarity)

            results.append({
                "registered": case,
                "similarity": similarity,
                "distance": distance,
                "distance_label": distance_label,
                "method": method,
                "match_label": label,
                "match_class": css_class,
                "recommendation": recommendation,
                "photo": os.path.basename(photo_path),
                "quality_score": quality["score"],
            })
        except Exception:
            # One broken case image must not stop the complete search.
            continue

    results.sort(key=lambda x: x["similarity"], reverse=True)

    # Add ranking and relative margin so the UI can show how decisive the top result is.
    for i, result in enumerate(results):
        result["rank"] = i + 1
        if i == 0 and len(results) > 1:
            result["top_margin"] = result["similarity"] - results[1]["similarity"]
        else:
            result["top_margin"] = 0.0

    return results[:max_results], None


# Backward-compatible wrapper for older code paths.
def find_matches(engine, threshold=3.0):
    return []
