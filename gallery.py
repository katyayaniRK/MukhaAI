"""Image-database (multi-person) search built on top of engine.py.

The video is read ONCE: faces are detected and tracked as before. Every track is then
compared with every person in the database, so adding more people costs very little
extra time (FaceNet results are cached per track and shared between people).
"""
import os
from dataclasses import dataclass, fields

import cv2
import numpy as np

import engine

IMAGE_EXT = (".jpg", ".jpeg", ".png")


# ---------------------------------------------------------------- data
@dataclass
class Person:
    name: str
    paths: list                    # photo files of this person
    ref: object                    # engine.Reference
    thumb: object                  # small BGR picture for the UI and the report


@dataclass
class Match(engine.Candidate):
    """A Candidate that also says WHICH person of the database it was compared with."""
    person: str = ""
    uid: int = 0                   # unique across the whole result (used for decisions)
    closest: bool = True           # False if another database person is closer to this same face track


@dataclass
class GalleryResult:
    video_path: str
    fps: float
    frames_processed: int
    n_detections: int
    n_tracks: int
    facenet_used: bool
    people: list                   # list[Person]
    matches: list                  # list[Match], grouped by person, ranked inside each person

    def for_person(self, name):
        return [m for m in self.matches if m.person == name]


# ---------------------------------------------------------------- database helpers
def unique_name(name, existing):
    """'Ravi' -> 'Ravi (2)' if 'Ravi' is already in the database."""
    if name not in existing:
        return name
    n = 2
    while f"{name} ({n})" in existing:
        n += 1
    return f"{name} ({n})"


def scan_folder(folder):
    """Turn a folder into {person name: [photo paths]}.

    * every SUB-folder is one person (all photos inside are that person)
    * every loose image directly in the folder is one person, named after the file
    """
    spec = {}
    for entry in sorted(os.listdir(folder)):
        full = os.path.join(folder, entry)
        if os.path.isdir(full):
            imgs = [os.path.join(full, f) for f in sorted(os.listdir(full))
                    if f.lower().endswith(IMAGE_EXT)]
            if imgs:
                spec[entry] = imgs
        elif entry.lower().endswith(IMAGE_EXT):
            spec[os.path.splitext(entry)[0]] = [full]
    return spec


def _small(img, side=200):
    h, w = img.shape[:2]
    s = side / float(max(h, w))
    if s >= 1.0:
        return img.copy()
    return cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)


def build_people(spec, models, on_progress=None, should_cancel=None):
    """spec: {name: [photo paths]}.  Returns (people, names_with_no_face)."""
    people, skipped = [], []
    total = len(spec)
    for i, (name, paths) in enumerate(spec.items(), 1):
        if should_cancel and should_cancel():
            raise engine.SearchCancelled()
        if on_progress:
            on_progress(i, total, name)
        try:
            ref = engine.build_reference(paths, models)
        except ValueError:                       # no face in any photo of this person
            skipped.append(name)
            continue
        thumb = None
        for p in paths:
            img = engine.read_image(p)
            if img is not None:
                thumb = _small(img)
                break
        people.append(Person(name, list(paths), ref, thumb))
    return people, skipped


# ---------------------------------------------------------------- search
def _attach_facenet(rows, tracks, ref, models, cache, top_n=20, n_crops=3):
    """Same idea as engine.attach_facenet_to_candidates, but the per-track FaceNet
    embedding is computed once and reused for every person in the database."""
    if models.facenet is None or ref.fn is None:
        return
    for c in rows[:top_n]:
        if c.track_id not in cache:
            t = tracks[c.track_id]
            idx = np.linspace(0, len(t) - 1, min(n_crops, len(t))).astype(int)
            fns = [models.facenet(t[i].crop) for i in idx]
            cache[c.track_id] = engine.l2norm(np.mean(fns, axis=0))
        c.fn_track = float(np.dot(cache[c.track_id], ref.fn))
        c.strong = bool(c.arc_track < engine.ARCFACE_THRESHOLD
                        and c.fn_track >= engine.FACENET_THRESHOLD)


def search_gallery(video_path, people, top_k=3, frame_skip=5, min_det_score=0.5,
                   min_face_size=15, min_track_len=3, min_candidate_n=3,
                   merge_fragments=True, models=None, progress=None, should_cancel=None):
    models = models or engine.get_models()

    dets, fps, processed = engine.extract_detections(
        video_path, models, frame_skip, min_det_score, min_face_size, progress, should_cancel)

    if progress:
        progress(0.92, "Following faces across frames")
    tracks = engine.build_tracks(dets, frame_skip, min_track_len)
    if merge_fragments:
        if progress:
            progress(0.94, "Joining fragments of the same person")
        tracks = engine.merge_tracks(tracks, fps)

    fn_cache = {}
    scored = {}                    # person name -> rows
    best_arc = {}                  # track id -> smallest ArcFace distance over all people
    n = len(people)
    for i, p in enumerate(people):
        if should_cancel and should_cancel():
            raise engine.SearchCancelled()
        if progress:
            progress(0.95 + 0.05 * i / max(1, n), f"Comparing with {p.name} ({i + 1} of {n})")
        rows = engine.score_tracks(tracks, p.ref, fps, min_candidate_n)
        _attach_facenet(rows, tracks, p.ref, models, fn_cache)
        scored[p.name] = rows
        for c in rows:
            best_arc[c.track_id] = min(best_arc.get(c.track_id, 9.0), c.arc_track)

    matches, uid = [], 0
    for p in people:
        for c in engine.shortlist(scored[p.name], top_k):
            uid += 1
            base = {f.name: getattr(c, f.name) for f in fields(c)}
            matches.append(Match(**base, person=p.name, uid=uid,
                                 closest=c.arc_track <= best_arc[c.track_id] + 1e-9))
    if progress:
        progress(1.0, "Done")
    return GalleryResult(video_path, fps, processed, len(dets), len(tracks),
                         models.facenet is not None, people, matches)

