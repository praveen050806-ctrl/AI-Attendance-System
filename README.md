# AI Attendance System

A Flask web app that marks attendance using face recognition from a
webcam. Built entirely on OpenCV (Haar cascades for detection + LBPH for
recognition) — no dlib, no deep-learning model download, no external API.

## What it does

1. **Register** — enter a name, capture a few webcam photos, and the app
   trains a face recognizer on the spot.
2. **Check In** — scan your face and get marked present instantly, with
   duplicate check-ins for the same day automatically prevented.
3. **People** — view everyone enrolled, with photo counts, and remove
   someone (automatically retrains the model).
4. **Records** — browse attendance logs by date.

## Project structure

```
ai-attendance-system/
├── app.py               # Flask routes
├── face_engine.py         # Haar cascade detection + LBPH recognition
├── people.py                # Registered-person dataset management
├── attendance.py              # Daily CSV attendance logs + de-dup
├── requirements.txt
├── templates/
│   ├── index.html            # Dashboard
│   ├── register.html          # Webcam enrollment
│   ├── checkin.html             # Webcam check-in/scan
│   ├── people.html                # Manage registered people
│   └── records.html                 # Attendance log viewer
├── static/style.css
├── dataset/                # Enrolled face photos, per person (gitignored)
├── trainer/                  # Trained model + labels (gitignored)
├── attendance/                 # Daily CSV logs (gitignored)
└── README.md
```

## Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/<your-username>/ai-attendance-system.git
   cd ai-attendance-system
   ```

2. **Create a virtual environment (recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app**

   ```bash
   python app.py
   ```

5. Open **http://127.0.0.1:5000**. Register at least one person, then use
   Check In to mark attendance.

> **Camera access note:** browsers only allow webcam access on `localhost`
> or over HTTPS. Running locally with `python app.py` and visiting
> `http://127.0.0.1:5000` works fine; if you deploy this publicly, you'll
> need HTTPS for the camera to work.

## How it works

- **Detection**: OpenCV's Haar cascade (`haarcascade_frontalface_default.xml`,
  bundled with OpenCV) finds face bounding boxes in a frame.
- **Recognition**: each detected face is cropped, resized to 200×200,
  histogram-equalized, and fed to an **LBPH** (Local Binary Patterns
  Histograms) recognizer — a classic, dependency-light face recognition
  algorithm built into `opencv-contrib-python`.
- **Enrollment**: `people.py` stores each person's training photos in
  `dataset/<id>__<name>/`. Registering (or removing) someone retrains the
  LBPH model on all currently-enrolled photos.
- **Attendance**: `attendance.py` writes one CSV per day
  (`attendance/attendance_YYYY-MM-DD.csv`) and checks for an existing row
  before marking someone present again the same day.
- A match is only accepted if the LBPH distance is below a confidence
  threshold (`CONFIDENCE_THRESHOLD` in `face_engine.py`); otherwise the
  face is reported as "Unknown" rather than guessing.

## Accuracy & limitations (read before relying on this)

- **LBPH is a classical, lightweight algorithm** — it's noticeably less
  accurate than modern deep-learning face recognition, especially with
  varied lighting, angles, glasses, or facial hair changes. It's meant as
  an educational/demo-grade system, not a production access-control system.
- **Tune `CONFIDENCE_THRESHOLD`** in `face_engine.py` for your environment;
  lower = stricter matching (fewer false accepts, more false rejects).
- **More enrollment photos help.** Capture varied angles/expressions/lighting
  during registration for a more robust model.
- **This handles biometric data.** Face photos and derived models are
  sensitive personal data. If you deploy this for real people, get their
  informed consent, disclose retention/deletion policies, and check what
  biometric-privacy laws apply in your jurisdiction (e.g. GDPR, BIPA).

## Extending this project

- Swap LBPH for a deep-learning embedding model (e.g. `face_recognition`/dlib,
  or a FaceNet/ArcFace ONNX model) for meaningfully better accuracy.
- Add liveness detection (blink/challenge) to prevent photo spoofing.
- Move from CSV logs to a real database and add an admin login.
- Add email/Slack notifications on check-in, or CSV export for payroll systems.
- Support multiple cameras/checkpoints in one deployment.

## License

MIT — feel free to use this project as a portfolio piece or a starting point
for something bigger.
