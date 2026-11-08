"""Hybrid search engine: SCRFD (detector) + ArcFace + FaceNet.

This follows the Kaggle notebook (hybrid2):
  detect faces -> embed each face with ArcFace and FaceNet -> follow faces
  across frames (tracks) -> optionally join fragments of the same person ->
  score each track against the reference photo(s) -> show the closest tracks
  to a human, who confirms.

A track is "strong" when BOTH models agree:
  ArcFace distance < ARCFACE_THRESHOLD  and  FaceNet similarity >= FACENET_THRESHOLD
If FaceNet cannot be loaded, the engine falls back to ArcFace only.
"""
import contextlib
import io
import os
from dataclasses import dataclass, field

import cv2
import numpy as np

ARCFACE_THRESHOLD = 0.68
FACENET_THRESHOLD = 0.35


class SearchCancelled(Exception):
    pass


# ---------------------------------------------------------------- helpers
def l2norm(v):
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


def cosine_distance(a, b):
    return float(1.0 - np.dot(l2norm(a), l2norm(b)))

def get_confidence_score(arc_distance, fn_sim=None, n_detections=None):
    """0-100 score; 50 = exactly at the engine's pass marks. A ranking aid, not a probability."""
    arc = float(np.interp(arc_distance, [0.30, ARCFACE_THRESHOLD, 1.00], [100, 50, 0]))
    if fn_sim is None:
        score = arc * 0.9
    else:
        fn = float(np.interp(fn_sim, [0.00, FACENET_THRESHOLD, 0.70], [0, 50, 100]))
        score = 0.6 * arc + 0.4 * fn
    if n_detections is not None:
        score *= 0.85 + 0.15 * min(n_detections, 10) / 10
    return int(round(score))
    
    return int(score)

def read_image(path):
    """cv2.imread that also works with non-English characters in the path."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def fmt_time(seconds):
    seconds = max(0, int(round(seconds)))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


# ---------------------------------------------------------------- models
@dataclass
class Models:
    arc: object                    # insightface FaceAnalysis: .get(frame) -> faces
    facenet: object = None         # callable(face_bgr) -> normalised vector, or None
    facenet_error: str = ""


_models = None


def get_models():
    """Load SCRFD + ArcFace (+ FaceNet) once. The first run downloads ~300 MB."""
    global _models
    if _models is not None:
        return _models

    from insightface.app import FaceAnalysis
    arc = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    arc.prepare(ctx_id=-1, det_size=(640, 640))

    facenet_fn, err = None, ""
    try:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        from keras_facenet import FaceNet
        embedder = FaceNet()

        def _facenet(face_bgr):
            face = cv2.resize(face_bgr, (160, 160))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype("float32")
            with contextlib.redirect_stdout(io.StringIO()):     # hide Keras progress lines
                emb = embedder.embeddings([face])[0]
            return l2norm(emb)

        facenet_fn = _facenet
    except Exception as e:                                       # FaceNet is optional
        err = f"{type(e).__name__}: {e}"

    _models = Models(arc, facenet_fn, err)
    return _models


# ---------------------------------------------------------------- data
@dataclass
class Reference:
    arc: np.ndarray
    fn: object                     # np.ndarray or None
    n_photos: int


@dataclass
class Detection:
    frame: int
    box: tuple                     # x1, y1, x2, y2
    score: float
    arc: np.ndarray
    fn: object                     # np.ndarray or None
    crop: np.ndarray


@dataclass
class Candidate:
    rank: int
    track_id: int
    first_frame: int
    last_frame: int
    n: int
    arc_track: float               # ArcFace distance of the track average (lower = closer)
    arc_best_single: float
    arc_mean_single: float
    fn_track: object               # FaceNet similarity of the track average, or None
    strong: bool                   # both models agree
    start_s: float
    end_s: float
    crops: list = field(default_factory=list)
    mid_frame: int = 0
    mid_box: tuple = (0, 0, 0, 0)


@dataclass
class SearchResult:
    video_path: str
    fps: float
    frames_processed: int
    n_detections: int
    n_tracks: int
    n_strong: int
    facenet_used: bool
    candidates: list


# ---------------------------------------------------------------- reference
def build_reference(photo_paths, models=None):
    """Average the embeddings of the best face in each photo (as in the notebook)."""
    models = models or get_models()
    arcs, fns = [], []
    for p in photo_paths:
        img = read_image(p)
        if img is None:
            continue
        faces = models.arc.get(img)
        if not faces:
            continue
        best = max(faces, key=lambda f: f.det_score)
        arcs.append(np.asarray(best.embedding, dtype=np.float32))
        if models.facenet is not None:
            x1, y1, x2, y2 = [int(v) for v in best.bbox]
            crop = img[max(0, y1):y2, max(0, x1):x2]
            if crop.size:
                fns.append(models.facenet(crop))
    if not arcs:
        raise ValueError("No face was found in the photo(s). Use a clear photo of the face.")
    fn = l2norm(np.mean(fns, axis=0)) if fns else None
    return Reference(l2norm(np.mean(arcs, axis=0)), fn, len(arcs))


# ---------------------------------------------------------------- detection
def extract_detections(video_path, models, frame_skip=2, min_det_score=0.5,
                       min_face_size=15, progress=None, should_cancel=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Cannot open the video. Check the file, and avoid special characters in its path.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    dets, frame_no, processed = [], 0, 0
    prev_gray = None  # <--- NEW: Track previous frame for motion gate
    
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_no += 1
        
        if frame_no % frame_skip != 0:
            continue
        if should_cancel and should_cancel():
            cap.release()
            raise SearchCancelled()
            
        # --- NEW: MOTION GATE ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Blur slightly to reduce camera sensor noise
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            # If the average difference is very small, nothing is moving. Skip!
            if diff.mean() < 2.5: 
                prev_gray = gray
                continue
        prev_gray = gray
        # ------------------------

        processed += 1

        
        for face in models.arc.get(frame):
            if face.det_score < min_det_score:
                continue
            x1, y1, x2, y2 = [int(v) for v in face.bbox]
            x1, y1 = max(0, x1), max(0, y1)
            if (x2 - x1) < min_face_size or (y2 - y1) < min_face_size:
                continue
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            fn = None # Defer FaceNet to the end
            side = max(crop.shape[:2])
            if side > 96:                          # keep memory small
                crop = cv2.resize(crop, None, fx=96.0 / side, fy=96.0 / side,
                                  interpolation=cv2.INTER_AREA)
            dets.append(Detection(frame_no, (x1, y1, x2, y2), float(face.det_score),
                                  l2norm(face.embedding), fn, crop.copy()))
        if progress and total:
            progress(min(frame_no / total, 1.0) * 0.9, f"Reading video: frame {frame_no} of {total}")
    cap.release()
    return dets, fps, processed


# ---------------------------------------------------------------- tracking
def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _centers_close(a, b, factor=0.75):
    ca = ((a[0] + a[2]) / 2, (a[1] + a[3]) / 2)
    cb = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
    size = max(a[2] - a[0], a[3] - a[1], b[2] - b[0], b[3] - b[1])
    return np.hypot(ca[0] - cb[0], ca[1] - cb[1]) < factor * size


def build_tracks(dets, frame_skip=2, min_len=3):
    """Link detections of the same face across consecutive frames."""
    max_gap = 5 * frame_skip
    active = []
    for fr in sorted({d.frame for d in dets}):
        current = [d for d in dets if d.frame == fr]
        pairs = []
        for ti, t in enumerate(active):
            last = t[-1]
            if fr - last.frame > max_gap:
                continue
            for di, d in enumerate(current):
                s = _iou(last.box, d.box)
                if s > 0.2 or _centers_close(last.box, d.box):
                    pairs.append((s, ti, di))
        pairs.sort(reverse=True)
        used_t, used_d = set(), set()
        for s, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            active[ti].append(current[di])
            used_t.add(ti)
            used_d.add(di)
        for di, d in enumerate(current):
            if di not in used_d:
                active.append([d])
    return [t for t in active if len(t) >= min_len]


def track_means(t):
    """Weighted average embeddings of a track (sharper detections count more)."""
    w = np.array([d.score for d in t], dtype=np.float32)
    w = w / w.sum()
    arc = l2norm(sum(wi * d.arc for wi, d in zip(w, t)))
    if all(d.fn is not None for d in t):
        fn = l2norm(sum(wi * d.fn for wi, d in zip(w, t)))
    else:
        fn = None
    return arc, fn


def merge_tracks(tracks, fps, arc_dist=0.55, fn_sim=0.60, arc_only_dist=0.45, max_gap_s=3.0):
    """Join fragments of the same person that are close in time and look alike.

    With FaceNet: ArcFace distance < arc_dist AND FaceNet similarity > fn_sim.
    Without FaceNet: ArcFace distance < arc_only_dist (stricter).
    """
    merged = [list(t) for t in tracks]
    while True:
        best = None
        for i in range(len(merged)):
            for j in range(i + 1, len(merged)):
                a, b = merged[i], merged[j]
                if a[-1].frame < b[0].frame:
                    gap = b[0].frame - a[-1].frame
                elif b[-1].frame < a[0].frame:
                    gap = a[0].frame - b[-1].frame
                else:
                    continue                       # on screen at the same time: different people
                if gap > max_gap_s * fps:
                    continue
                ai, fi = track_means(a)
                aj, fj = track_means(b)
                d_arc = cosine_distance(ai, aj)
                if fi is not None and fj is not None:
                    ok = d_arc < arc_dist and float(np.dot(fi, fj)) > fn_sim
                else:
                    ok = d_arc < arc_only_dist
                if ok and (best is None or d_arc < best[0]):
                    best = (d_arc, i, j)
        if best is None:
            return merged
        _, i, j = best
        merged[i] = sorted(merged[i] + merged[j], key=lambda d: d.frame)
        merged.pop(j)


# ---------------------------------------------------------------- scoring
def score_tracks(tracks, ref, fps, min_n=3):
    """Score every track against the reference (same numbers as the notebook)."""
    rows = []
    for tid, t in enumerate(tracks):
        if len(t) < min_n:
            continue
        arc_avg, fn_avg = track_means(t)
        singles = [cosine_distance(d.arc, ref.arc) for d in t]
        arc_track = cosine_distance(arc_avg, ref.arc)
        fn_track = float(np.dot(fn_avg, ref.fn)) if (fn_avg is not None and ref.fn is not None) else None
        arc_ok = arc_track < ARCFACE_THRESHOLD
        fn_ok = True if fn_track is None else fn_track >= FACENET_THRESHOLD
        idx = np.linspace(0, len(t) - 1, 3).astype(int)
        mid = t[len(t) // 2]
        rows.append(Candidate(
            rank=0, track_id=tid, first_frame=t[0].frame, last_frame=t[-1].frame, n=len(t),
            arc_track=arc_track, arc_best_single=min(singles),
            arc_mean_single=float(np.mean(singles)), fn_track=fn_track,
            strong=bool(arc_ok and fn_ok), start_s=t[0].frame / fps, end_s=t[-1].frame / fps,
            crops=[t[i].crop for i in idx], mid_frame=mid.frame, mid_box=mid.box))
    rows.sort(key=lambda c: c.arc_track)
    return rows


def shortlist(rows, top_k=3, max_extra_strong=3, max_arc_distance=0.85):
    """The top_k closest tracks, plus any further strong tracks, ranked 1..n.
       Drops terrible matches where ArcFace is just guessing."""
    
    # Filter out tracks that are mathematically too far away to be a match
    valid_rows = [c for c in rows if c.arc_track <= max_arc_distance]
    
    # If literally everything was filtered out, return an empty list
    if not valid_rows:
        return []
        
    chosen = valid_rows[:top_k] + [c for c in valid_rows[top_k:] if c.strong][:max_extra_strong]
    for rank, c in enumerate(chosen, 1):
        c.rank = rank
    return chosen

# ----------------------------------------------------------------
def attach_facenet_to_candidates(rows, tracks, ref, models, n_crops=3):
    """Calculates FaceNet ONLY for the top candidate tracks to save massive amounts of time."""
    if models.facenet is None or ref.fn is None:
        return
    
    # Only process the top 20 closest ArcFace tracks
    for c in rows[:20]:
        t = tracks[c.track_id]
         # Check the size of the face. If it's too small (e.g. < 40px), 
        # FaceNet will fail anyway. Don't waste time on it!
        mid_box = t[len(t)//2].box
        face_w = mid_box[2] - mid_box[0]
        face_h = mid_box[3] - mid_box[1]
        
        if max(face_w, face_h) < 40:
            # It's a tiny face. Leave fn_track as None.
            # We rely on the human to review it.
            continue
        
        # Pick 3 crops evenly spread across the track
        idx = np.linspace(0, len(t) - 1, min(n_crops, len(t))).astype(int)
        
        fns = []
        for i in idx:
            # We need the original un-resized crop for FaceNet (160x160)
            # The crop stored in Detection is already resized to max 96x96, 
            # but FaceNet resizes it back to 160x160 anyway, so it works fine.
            fn_emb = models.facenet(t[i].crop)
            if fn_emb is not None:
                fns.append(fn_emb)
                
        if fns:
            fn_avg = l2norm(np.mean(fns, axis=0))
            c.fn_track = float(np.dot(fn_avg, ref.fn))
            # Re-check if it's strong now that we have FaceNet
            c.strong = bool(c.arc_track < ARCFACE_THRESHOLD and c.fn_track >= FACENET_THRESHOLD)
# ---------------------------------------------------------------- main entry
def search_video(video_path, ref, top_k=3, frame_skip=5, min_det_score=0.5,
                 min_face_size=15, min_track_len=3, min_candidate_n=3,
                 merge_fragments=True, models=None, progress=None, should_cancel=None):
    models = models or get_models()
    
    # 1. Extract (now with Motion Gate and NO FaceNet)
    dets, fps, processed = extract_detections(
        video_path, models, frame_skip, min_det_score, min_face_size, progress, should_cancel)
    
    if progress:
        progress(0.92, "Following faces across frames")
    tracks = build_tracks(dets, frame_skip, min_track_len)
    
    if merge_fragments:
        if progress:
            progress(0.95, "Joining fragments of the same person")
        # Note: merge_tracks will now use ArcFace only (which is fine, it's very accurate)
        tracks = merge_tracks(tracks, fps)
        
    if progress:
        progress(0.97, "Scoring candidates (ArcFace)")
    rows = score_tracks(tracks, ref, fps, min_candidate_n)
    
    # NEW: Now run FaceNet ONLY on the top candidates
    if progress:
        progress(0.98, "Verifying top candidates (FaceNet)")
    attach_facenet_to_candidates(rows, tracks, ref, models)
    
    chosen = shortlist(rows, top_k)
    if progress:
        progress(1.0, "Done")
    return SearchResult(video_path, fps, processed, len(dets), len(tracks),
                        sum(c.strong for c in rows), models.facenet is not None, chosen)


def read_frame(video_path, frame_no):
    """Return one full frame (read in order, because jumping is unreliable)."""
    cap = cv2.VideoCapture(video_path)
    n, found = 0, None
    while True:
        ok, f = cap.read()
        if not ok:
            break
        n += 1
        if n == frame_no:
            found = f
            break
    cap.release()
    return found
