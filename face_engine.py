"""
face_engine.py
----------------
Face detection + recognition using OpenCV's built-in tools only:
  - Haar cascade classifier for face DETECTION (finding face regions)
  - LBPH (Local Binary Patterns Histograms) for face RECOGNITION
    (identifying whose face it is)

No dlib / face_recognition / deep learning model download is required,
which keeps setup simple (`pip install opencv-contrib-python` is enough)
at the cost of being less accurate than a modern deep-learning-based
recognizer under difficult lighting/angle conditions. See the README
for how you'd swap in a stronger model.
"""

import os
import json
import cv2
import numpy as np

FACE_SIZE = (200, 200)
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
CONFIDENCE_THRESHOLD = 75  # LBPH distance: LOWER is a better match; above this = "unknown"


def get_face_detector():
    return cv2.CascadeClassifier(CASCADE_PATH)


def detect_faces(image_bgr, detector=None):
    """
    Detect faces in a BGR image (as loaded by cv2.imread / decoded from upload).

    Returns:
        list[tuple]: [(x, y, w, h), ...] bounding boxes, largest first.
    """
    detector = detector or get_face_detector()
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = detector.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    return [tuple(int(v) for v in f) for f in faces]


def crop_and_normalize_face(image_bgr, box):
    """Crop a detected face region and normalize it for training/prediction."""
    x, y, w, h = box
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    face = gray[y:y + h, x:x + w]
    face = cv2.resize(face, FACE_SIZE)
    face = cv2.equalizeHist(face)
    return face


def load_labels(labels_path):
    if os.path.exists(labels_path):
        with open(labels_path, "r") as f:
            return json.load(f)
    return {}


def save_labels(labels_path, labels):
    with open(labels_path, "w") as f:
        json.dump(labels, f, indent=2)


def train_recognizer(dataset_dir, trainer_path, labels_path):
    """
    Train an LBPH face recognizer from a dataset directory structured as:

        dataset_dir/
          <person_id>__<name>/
            img1.jpg
            img2.jpg
            ...

    Saves the trained model to trainer_path and an id->name mapping to
    labels_path.

    Returns:
        dict: {"people_trained": int, "images_used": int}
    """
    detector = get_face_detector()
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    faces, ids = [], []
    labels = {}

    person_dirs = sorted(
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d)) and "__" in d
    )

    for person_dir in person_dirs:
        person_id_str, name = person_dir.split("__", 1)
        person_id = int(person_id_str)
        labels[str(person_id)] = name

        folder = os.path.join(dataset_dir, person_dir)
        for filename in os.listdir(folder):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            path = os.path.join(folder, filename)
            image = cv2.imread(path)
            if image is None:
                continue

            detected = detect_faces(image, detector)
            if not detected:
                continue

            face = crop_and_normalize_face(image, detected[0])
            faces.append(face)
            ids.append(person_id)

    if not faces:
        raise ValueError("No usable face images found in the dataset to train on.")

    recognizer.train(faces, np.array(ids))
    recognizer.save(trainer_path)
    save_labels(labels_path, labels)

    return {"people_trained": len(person_dirs), "images_used": len(faces)}


def load_recognizer(trainer_path):
    if not os.path.exists(trainer_path):
        return None
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(trainer_path)
    return recognizer


def recognize_faces(image_bgr, recognizer, labels, detector=None):
    """
    Detect and identify all faces in an image.

    Returns:
        list[dict]: [{"box": (x,y,w,h), "person_id": int|None,
                       "name": str, "confidence": float, "recognized": bool}, ...]
    """
    detector = detector or get_face_detector()
    boxes = detect_faces(image_bgr, detector)
    results = []

    for box in boxes:
        face = crop_and_normalize_face(image_bgr, box)

        if recognizer is None:
            results.append({
                "box": box, "person_id": None, "name": "Unknown",
                "confidence": None, "recognized": False,
            })
            continue

        person_id, distance = recognizer.predict(face)
        recognized = distance < CONFIDENCE_THRESHOLD
        name = labels.get(str(person_id), "Unknown") if recognized else "Unknown"

        results.append({
            "box": box,
            "person_id": person_id if recognized else None,
            "name": name,
            "confidence": round(float(distance), 1),
            "recognized": recognized,
        })

    return results
