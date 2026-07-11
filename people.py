"""
people.py
-----------
Helpers for managing registered people. Each registered person has a
folder under dataset/ named "<id>__<name>" containing their training
face images. The folder name IS the source of truth — there's no
separate database, keeping the project dependency-free.
"""

import os
import re
import shutil

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
os.makedirs(DATASET_DIR, exist_ok=True)


def _safe_name(name):
    name = name.strip()
    name = re.sub(r"[^\w\s-]", "", name)
    return name[:60] if name else "Unnamed"


def list_people():
    """Return [{"id": int, "name": str, "num_images": int}, ...] sorted by id."""
    people = []
    for entry in os.listdir(DATASET_DIR):
        full = os.path.join(DATASET_DIR, entry)
        if not os.path.isdir(full) or "__" not in entry:
            continue
        id_str, name = entry.split("__", 1)
        try:
            person_id = int(id_str)
        except ValueError:
            continue
        num_images = len([
            f for f in os.listdir(full)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])
        people.append({"id": person_id, "name": name, "num_images": num_images})

    return sorted(people, key=lambda p: p["id"])


def next_person_id():
    people = list_people()
    return (max((p["id"] for p in people), default=0)) + 1


def create_person_folder(name):
    """Create (or return existing) folder for a new person. Returns (id, folder_path)."""
    person_id = next_person_id()
    safe = _safe_name(name)
    folder = os.path.join(DATASET_DIR, f"{person_id}__{safe}")
    os.makedirs(folder, exist_ok=True)
    return person_id, folder


def delete_person(person_id):
    """Remove a person's dataset folder entirely."""
    for entry in os.listdir(DATASET_DIR):
        if entry.startswith(f"{person_id}__"):
            shutil.rmtree(os.path.join(DATASET_DIR, entry))
            return True
    return False
