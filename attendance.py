"""
attendance.py
---------------
Handles reading/writing daily attendance CSV logs. One CSV file per
day is created under the attendance/ directory, e.g. attendance_2026-07-11.csv.
Each person is only marked present once per day, at their first
recognized check-in.
"""

import os
import csv
from datetime import datetime

ATTENDANCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "attendance")
os.makedirs(ATTENDANCE_DIR, exist_ok=True)

FIELDNAMES = ["person_id", "name", "time", "confidence"]


def _file_for_date(date_str):
    return os.path.join(ATTENDANCE_DIR, f"attendance_{date_str}.csv")


def _read_rows(date_str):
    path = _file_for_date(date_str)
    if not os.path.exists(path):
        return []
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def mark_present(person_id, name, confidence, when=None):
    """
    Mark a person present for today (or a given datetime). If they've
    already been marked present today, this is a no-op.

    Returns:
        dict: {"marked": bool, "already_marked": bool, "time": str}
    """
    when = when or datetime.now()
    date_str = when.strftime("%Y-%m-%d")
    time_str = when.strftime("%H:%M:%S")

    rows = _read_rows(date_str)
    already = any(str(r["person_id"]) == str(person_id) for r in rows)

    if already:
        existing = next(r for r in rows if str(r["person_id"]) == str(person_id))
        return {"marked": False, "already_marked": True, "time": existing["time"]}

    path = _file_for_date(date_str)
    file_exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "person_id": person_id,
            "name": name,
            "time": time_str,
            "confidence": confidence,
        })

    return {"marked": True, "already_marked": False, "time": time_str}


def get_attendance(date_str=None):
    """Return the list of attendance rows for a given date (default today)."""
    date_str = date_str or datetime.now().strftime("%Y-%m-%d")
    return _read_rows(date_str)


def list_available_dates():
    """Return sorted list of dates (YYYY-MM-DD) that have an attendance log."""
    dates = []
    for filename in os.listdir(ATTENDANCE_DIR):
        if filename.startswith("attendance_") and filename.endswith(".csv"):
            dates.append(filename[len("attendance_"):-len(".csv")])
    return sorted(dates, reverse=True)
