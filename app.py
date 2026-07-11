"""
app.py
-------
Flask web application for the AI Attendance System.

Routes:
  GET  /              -> dashboard (counts, quick links)
  GET  /register        -> webcam-based enrollment form
  POST /register         -> save captured face images, train recognizer
  GET  /people             -> list / delete registered people
  POST /people/<id>/delete  -> remove a person and retrain
  GET  /checkin              -> webcam check-in page
  POST /checkin/scan           -> recognize faces in a captured frame, mark attendance
  GET  /records                  -> attendance log viewer

Run locally with:
    python app.py
Then open http://127.0.0.1:5000

IMPORTANT: webcam access requires either http://localhost or https://
in the browser — most browsers block camera access on plain HTTP for
non-localhost origins.
"""

import os
import re
import base64
from datetime import datetime
import numpy as np
import cv2
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify

import people as people_store
from face_engine import (
    train_recognizer, load_recognizer, load_labels, recognize_faces,
    get_face_detector, detect_faces,
)
import attendance as attendance_store

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINER_PATH = os.path.join(BASE_DIR, "trainer", "trainer.yml")
LABELS_PATH = os.path.join(BASE_DIR, "trainer", "labels.json")
MIN_IMAGES_PER_PERSON = 4

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key-change-me")

_detector = get_face_detector()


def decode_data_url(data_url):
    """Decode a 'data:image/jpeg;base64,...' string into a BGR OpenCV image."""
    match = re.match(r"data:image/\w+;base64,(.*)", data_url)
    b64 = match.group(1) if match else data_url
    binary = base64.b64decode(b64)
    arr = np.frombuffer(binary, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def get_recognizer_and_labels():
    recognizer = load_recognizer(TRAINER_PATH)
    labels = load_labels(LABELS_PATH)
    return recognizer, labels


@app.route("/", methods=["GET"])
def index():
    people = people_store.list_people()
    today_attendance = attendance_store.get_attendance()
    model_ready = os.path.exists(TRAINER_PATH)
    return render_template(
        "index.html",
        num_people=len(people),
        num_present_today=len(today_attendance),
        model_ready=model_ready,
    )


@app.route("/register", methods=["GET"])
def register():
    return render_template("register.html", min_images=MIN_IMAGES_PER_PERSON)


@app.route("/register", methods=["POST"])
def register_submit():
    payload = request.get_json(silent=True) or {}
    name = (payload.get("name") or "").strip()
    images = payload.get("images") or []

    if not name:
        return jsonify({"error": "Please enter a name."}), 400
    if len(images) < MIN_IMAGES_PER_PERSON:
        return jsonify({"error": f"Please capture at least {MIN_IMAGES_PER_PERSON} photos."}), 400

    person_id, folder = people_store.create_person_folder(name)

    saved = 0
    for i, data_url in enumerate(images):
        try:
            frame = decode_data_url(data_url)
        except Exception:
            continue
        if frame is None:
            continue
        boxes = detect_faces(frame, _detector)
        if not boxes:
            continue  # skip frames with no detectable face
        path = os.path.join(folder, f"img{i}.jpg")
        cv2.imwrite(path, frame)
        saved += 1

    if saved < MIN_IMAGES_PER_PERSON:
        people_store.delete_person(person_id)
        return jsonify({
            "error": "Couldn't detect a face in enough of those photos. "
                     "Make sure your face is well-lit and centered, then try again."
        }), 400

    try:
        stats = train_recognizer(
            people_store.DATASET_DIR, TRAINER_PATH, LABELS_PATH
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    return jsonify({
        "success": True,
        "person_id": person_id,
        "name": name,
        "images_saved": saved,
        "training_stats": stats,
    })


@app.route("/people", methods=["GET"])
def people_list():
    people = people_store.list_people()
    return render_template("people.html", people=people)


@app.route("/people/<int:person_id>/delete", methods=["POST"])
def people_delete(person_id):
    people_store.delete_person(person_id)
    remaining = people_store.list_people()

    if remaining:
        try:
            train_recognizer(people_store.DATASET_DIR, TRAINER_PATH, LABELS_PATH)
        except ValueError:
            if os.path.exists(TRAINER_PATH):
                os.remove(TRAINER_PATH)
    else:
        if os.path.exists(TRAINER_PATH):
            os.remove(TRAINER_PATH)
        if os.path.exists(LABELS_PATH):
            os.remove(LABELS_PATH)

    flash("Person removed and the recognizer was retrained.")
    return redirect(url_for("people_list"))


@app.route("/checkin", methods=["GET"])
def checkin():
    model_ready = os.path.exists(TRAINER_PATH)
    if not model_ready:
        flash("No one is registered yet. Register at least one person before checking in.")
        return redirect(url_for("register"))
    return render_template("checkin.html")


@app.route("/checkin/scan", methods=["POST"])
def checkin_scan():
    payload = request.get_json(silent=True) or {}
    data_url = payload.get("image")

    if not data_url:
        return jsonify({"error": "No image received."}), 400

    frame = decode_data_url(data_url)
    if frame is None:
        return jsonify({"error": "Could not decode image."}), 400

    recognizer, labels = get_recognizer_and_labels()
    results = recognize_faces(frame, recognizer, labels, _detector)

    marked = []
    for r in results:
        if r["recognized"]:
            outcome = attendance_store.mark_present(r["person_id"], r["name"], r["confidence"])
            marked.append({**r, **outcome})
        else:
            marked.append({**r, "marked": False, "already_marked": False, "time": None})

    return jsonify({"faces": marked})


@app.route("/records", methods=["GET"])
def records():
    date_str = request.args.get("date")
    rows = attendance_store.get_attendance(date_str)
    available_dates = attendance_store.list_available_dates()
    selected_date = date_str or datetime.now().strftime("%Y-%m-%d")
    return render_template(
        "records.html", rows=rows, dates=available_dates, selected_date=selected_date
    )


if __name__ == "__main__":
    app.run(debug=True)
