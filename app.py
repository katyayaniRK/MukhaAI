"""Missing Person Search - Flask web version of the desktop (Tkinter) app.

Run:   python web_app.py
Open:  http://127.0.0.1:5000

Needs engine.py, gallery.py and gallery_report.py in this same folder
(the same modules the desktop app.py uses).

This is a single-operator tool: the database, video and results live in one
server-side session, just like the desktop window did. It listens on
127.0.0.1 only unless you set HOST (there is no login - do not expose it).
"""
import os
import re
import shutil
import tempfile
import threading
import time
import traceback
import uuid

import cv2
from flask import Flask, Response, abort, jsonify, render_template, request, send_file, send_from_directory
from werkzeug.utils import safe_join

import engine
import gallery
import gallery_report

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")

SPEEDS = {"fast": 8, "balanced": 5, "thorough": 2}
IMAGE_EXT = {".jpg", ".jpeg", ".png"}
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}

STEPS = [
    ("Loading face models",
     "Preparing the face detection and recognition models. The first run is slow."),
    ("Reading database photos",
     "Finding the face in each database photo and turning it into an embedding."),
    ("Scanning video frames",
     "Detecting faces in the video and turning each face into an embedding."),
    ("Following faces across frames",
     "Linking detections into tracks, one track for each person on screen."),
    ("Comparing with the database",
     "Measuring how close each tracked face is to every person in the database."),
]

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 ** 3      # long CCTV videos
app.config["MAX_FORM_PARTS"] = 100000                  # folders with many photos
app.config["MAX_FORM_MEMORY_SIZE"] = 16 * 1024 * 1024
app.config["TEMPLATES_AUTO_RELOAD"] = True


# ---------------------------------------------------------------- session
class Session:
    def __init__(self):
        self.lock = threading.RLock()
        self.cancel = threading.Event()
        self.worker = None
        self.people_spec = {}           # person name -> [photo paths]
        self.video_path = None
        self.video_name = None
        self.result = None              # gallery.GalleryResult
        self.skipped = []
        self.matches = {}               # str(uid) -> match
        self.person_index = {}          # person name -> index in result.people
        self.decisions = {}             # match.uid -> 'same' / 'not' / ''
        self.search_id = 0
        self.frame_cache = {}
        self.job = self.new_job()

    @staticmethod
    def new_job(state="idle"):
        return dict(state=state, step=0, fraction=None, status="", error="", started=time.time())

    def busy(self):
        return bool(self.worker and self.worker.is_alive())


S = Session()


# ---------------------------------------------------------------- helpers
def err(message, code=400):
    return jsonify(error=message), code


def clean_part(text):
    text = re.sub(r'[\x00-\x1f<>:"|?*]', "_", text).strip(" .")
    return text or "file"


def save_upload(f, dest_dir, rel=None):
    """Save an uploaded file under dest_dir, keeping a sanitised relative path."""
    raw = rel or f.filename or ""
    parts = [clean_part(p) for p in re.split(r"[\\/]+", raw) if p not in ("", ".", "..")]
    if not parts:
        return None
    path = os.path.join(dest_dir, *parts)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    f.save(path)
    return path


def jpeg(img, max_side=None, quality=88):
    if max_side:
        h, w = img.shape[:2]
        scale = max_side / float(max(h, w))
        if scale < 1:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        abort(500)
    resp = Response(buf.tobytes(), mimetype="image/jpeg")
    resp.headers["Cache-Control"] = "private, max-age=600"
    return resp


def photo_response(path):
    img = engine.read_image(path) if path else None
    if img is None:
        abort(404)
    return jpeg(img, max_side=1100)


def short_list(names, limit=6):
    names = list(names)
    if len(names) <= limit:
        return ", ".join(names)
    return ", ".join(names[:limit]) + f" and {len(names) - limit} more"


def database_payload():
    return [dict(name=n, n=len(p)) for n, p in S.people_spec.items()]


def job_payload():
    with S.lock:
        j = dict(S.job)
    j["elapsed"] = int(time.time() - j["started"]) if j["state"] == "running" else 0
    j["steps"] = [dict(title=t, desc=d) for t, d in STEPS]
    return j


# ---------------------------------------------------------------- pages
@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/state")
def api_state():
    with S.lock:
        return jsonify(people=database_payload(), video=S.video_name,
                       has_results=S.result is not None, search_id=S.search_id,
                       job=job_payload())


# ---------------------------------------------------------------- image database
@app.post("/api/people")
def api_add_people():
    """Add photos to the database.

    mode=each   every photo is a different person (named after the file)
    mode=one    all photos are one person (needs `name`)
    mode=folder a whole folder: one person per loose photo / per sub-folder
    """
    if S.busy():
        return err("A search is running. Wait for it to finish or cancel it first.", 409)
    mode = request.form.get("mode", "each")
    files = [f for f in request.files.getlist("files")
             if os.path.splitext(f.filename or "")[1].lower() in IMAGE_EXT]
    if not files:
        return err("No .jpg, .jpeg or .png photos were found.\n\n"
                   "Each photo directly in a folder becomes one person.\n"
                   "Each sub-folder becomes one person (all photos inside belong to them).")

    batch = os.path.join(UPLOAD_DIR, "db", uuid.uuid4().hex)
    added = []
    with S.lock:
        def add(name, paths):
            name = gallery.unique_name(name, S.people_spec)
            S.people_spec[name] = list(paths)
            added.append(name)

        if mode == "folder":
            rels = request.form.getlist("paths")          # one relative path per uploaded file
            all_files = request.files.getlist("files")
            if len(rels) != len(all_files):
                rels = [f.filename for f in all_files]
            first_parts = None
            for f, rel in zip(all_files, rels):
                if os.path.splitext(f.filename or "")[1].lower() not in IMAGE_EXT:
                    continue
                path = save_upload(f, batch, rel)
                if path and first_parts is None:
                    first_parts = os.path.relpath(path, batch).split(os.sep)
            root = os.path.join(batch, first_parts[0]) if first_parts and len(first_parts) > 1 else batch
            try:
                spec = gallery.scan_folder(root)
            except OSError as e:
                return err(f"Could not read the folder: {e}")
            if not spec:
                return err("No photos found in that folder.")
            for name, paths in spec.items():
                add(name, paths)
        else:
            saved = [(f, save_upload(f, batch, os.path.basename((f.filename or "").replace("\\", "/"))))
                     for f in files]
            saved = [(f, p) for f, p in saved if p]
            if mode == "one":
                name = (request.form.get("name") or "").strip()
                if not name:
                    return err("Enter a name for this person.")
                add(name, [p for _, p in saved])
            else:
                for f, p in saved:
                    add(os.path.splitext(os.path.basename(f.filename.replace("\\", "/")))[0], [p])
    return jsonify(added=added, people=database_payload())


@app.post("/api/people/remove")
def api_remove_people():
    if S.busy():
        return err("A search is running.", 409)
    names = (request.get_json(silent=True) or {}).get("names", [])
    with S.lock:
        for n in names:
            S.people_spec.pop(n, None)
    return jsonify(people=database_payload())


@app.post("/api/people/clear")
def api_clear_people():
    if S.busy():
        return err("A search is running.", 409)
    with S.lock:
        S.people_spec.clear()
    return jsonify(people=database_payload())


@app.get("/api/db/photo")
def api_db_photo():
    name, i = request.args.get("name", ""), request.args.get("i", 0, type=int)
    paths = S.people_spec.get(name) or []
    if not 0 <= i < len(paths):
        abort(404)
    return photo_response(paths[i])


# ---------------------------------------------------------------- video
@app.post("/api/video")
def api_video():
    if S.busy():
        return err("A search is running.", 409)
    f = request.files.get("file")
    if not f or os.path.splitext(f.filename or "")[1].lower() not in VIDEO_EXT:
        return err("Choose a .mp4, .avi, .mov or .mkv video.")
    folder = os.path.join(UPLOAD_DIR, "video", uuid.uuid4().hex)
    path = save_upload(f, folder, os.path.basename(f.filename.replace("\\", "/")))
    with S.lock:
        old = S.video_path
        S.video_path, S.video_name = path, os.path.basename(path)
    if old:
        shutil.rmtree(os.path.dirname(old), ignore_errors=True)
    return jsonify(video=S.video_name)


# ---------------------------------------------------------------- search
def _progress(text, fraction=None):
    """Same step logic as the desktop loading panel."""
    with S.lock:
        j = S.job
        step = j["step"]
        if fraction is None:
            low = text.lower()
            if low.startswith("loading"):
                step = 0
            elif low.startswith("reading database"):
                step = 1
        else:
            step = 2 if fraction < 0.92 else (3 if fraction < 0.95 else 4)
            j["fraction"] = fraction
        j["step"] = max(step, j["step"])
        j["status"] = text


def _run_search(spec, video, top_k, frame_skip):
    try:
        _progress("Loading the face models (the first run is slow)...")
        models = engine.get_models()
        people, skipped = gallery.build_people(
            spec, models,
            on_progress=lambda i, n, name: _progress(f"Reading database photos: {i} of {n} ({name})"),
            should_cancel=S.cancel.is_set)
        if not people:
            raise ValueError("No face was found in any database photo. Use clear photos of the face.")
        result = gallery.search_gallery(
            video, people, top_k=top_k, frame_skip=frame_skip, models=models,
            progress=lambda f, m: _progress(m, f),
            should_cancel=S.cancel.is_set)
        with S.lock:
            S.result, S.skipped = result, list(skipped)
            S.matches = {str(m.uid): m for m in result.matches}
            S.person_index = {p.name: i for i, p in enumerate(result.people)}
            S.decisions = {m.uid: "" for m in result.matches}
            S.frame_cache.clear()
            S.job.update(state="done", status="Done. Review the candidates.", fraction=1.0)
    except engine.SearchCancelled:
        with S.lock:
            S.job.update(state="cancelled", status="Search cancelled.")
    except Exception as e:
        traceback.print_exc()
        with S.lock:
            S.job.update(state="error", error=f"{type(e).__name__}: {e}", status="Search failed.")


@app.post("/api/search")
def api_search():
    if S.busy():
        return err("A search is already running.", 409)
    data = request.get_json(silent=True) or {}
    with S.lock:
        if not S.people_spec or not S.video_path:
            return err("Add photos and choose a video first.")
        top_k = max(1, min(10, int(data.get("top_k") or 3)))
        frame_skip = SPEEDS.get(data.get("speed"), 5)
        spec = {k: list(v) for k, v in S.people_spec.items()}
        S.cancel.clear()
        S.result, S.matches, S.decisions, S.skipped = None, {}, {}, []
        S.search_id += 1
        S.job = S.new_job("running")
        S.job["status"] = "Starting..."
        S.worker = threading.Thread(target=_run_search, args=(spec, S.video_path, top_k, frame_skip),
                                    daemon=True)
        S.worker.start()
    return jsonify(job=job_payload(), search_id=S.search_id)


@app.post("/api/cancel")
def api_cancel():
    S.cancel.set()
    return jsonify(ok=True)


@app.get("/api/status")
def api_status():
    return jsonify(dict(job_payload(), search_id=S.search_id))


# ---------------------------------------------------------------- results
def match_dict(m):
    face_w = m.mid_box[2] - m.mid_box[0]
    if m.strong:
        label = "STRONG MATCH"
    elif face_w < 40:
        label = "POSSIBLE - Face too small to verify"
    else:
        label = "POSSIBLE"
    conf = engine.get_confidence_score(m.arc_track, m.fn_track, m.n)
    return dict(
        uid=str(m.uid), rank=m.rank, person=m.person, person_idx=S.person_index.get(m.person),
        strong=bool(m.strong), label=label, confidence=conf,
        level="ok" if conf >= 65 else ("warn" if conf >= 40 else "bad"),
        start=engine.fmt_time(m.start_s), end=engine.fmt_time(m.end_s),
        start_s=m.start_s, end_s=m.end_s, n=m.n,
        first_frame=m.first_frame, last_frame=m.last_frame, mid_frame=m.mid_frame,
        arc=round(float(m.arc_track), 3), arc_ok=bool(m.arc_track < engine.ARCFACE_THRESHOLD),
        fn=None if m.fn_track is None else round(float(m.fn_track), 3),
        fn_ok=bool(m.fn_track is not None and m.fn_track >= engine.FACENET_THRESHOLD),
        closest=bool(m.closest), n_crops=len(m.crops),
        decision=S.decisions.get(m.uid, ""))


@app.get("/api/results")
def api_results():
    with S.lock:
        res = S.result
        if res is None:
            return err("No results yet.", 404)
        people, strong_people = [], []
        for idx, p in enumerate(res.people):
            ms = [match_dict(m) for m in res.for_person(p.name)]
            if any(m["strong"] for m in ms):
                strong_people.append(p.name)
            people.append(dict(idx=idx, name=p.name, has_thumb=p.thumb is not None,
                               n_photos=len(p.paths), matches=ms))
        mode = "ArcFace + FaceNet (hybrid)" if res.facenet_used else "ArcFace only (FaceNet not available)"
        head = [f"{os.path.basename(res.video_path)}: {res.n_detections} faces found, "
                f"{res.n_tracks} tracks. Compared with {len(res.people)} person(s). Mode: {mode}."]
        if strong_people:
            head.append(f"{len(strong_people)} person(s) with a STRONG match (both models agree): "
                        f"{short_list(strong_people)}.")
        elif res.matches:
            head.append("No person passed both models. The closest tracks are shown for human review.")
        else:
            head.append("No match found: no face in this video is similar enough to anyone in the database.")
        if S.skipped:
            head.append(f"No face found in the photo(s) of: {short_list(S.skipped)} (skipped).")
        return jsonify(head=head, people=people, n_tracks=res.n_tracks,
                       has_matches=bool(res.matches), search_id=S.search_id,
                       default_only_strong=len(res.people) > 1 and bool(strong_people))


@app.get("/api/results/person/<int:idx>/thumb.jpg")
def api_person_thumb(idx):
    res = S.result
    if res is None or not 0 <= idx < len(res.people) or res.people[idx].thumb is None:
        abort(404)
    return jpeg(res.people[idx].thumb)


@app.get("/api/results/person/<int:idx>/photo")
def api_person_photo(idx):
    res = S.result
    i = request.args.get("i", 0, type=int)
    if res is None or not 0 <= idx < len(res.people) or not 0 <= i < len(res.people[idx].paths):
        abort(404)
    return photo_response(res.people[idx].paths[i])


@app.get("/api/match/<uid>/crop/<int:i>.jpg")
def api_match_crop(uid, i):
    m = S.matches.get(uid)
    if m is None or not 0 <= i < len(m.crops):
        abort(404)
    return jpeg(m.crops[i])


@app.get("/api/match/<uid>/frame.jpg")
def api_match_frame(uid):
    m = S.matches.get(uid)
    if m is None or S.result is None:
        abort(404)
    with S.lock:
        cached = S.frame_cache.get(uid)
    if cached is None:
        frame = engine.read_frame(S.result.video_path, m.mid_frame)
        if frame is None:
            abort(404)
        x1, y1, x2, y2 = m.mid_box
        pad = 20
        cv2.rectangle(frame, (max(0, x1 - pad), max(0, y1 - pad)), (x2 + pad, y2 + pad), (0, 255, 0), 4)
        cached = frame
        with S.lock:
            S.frame_cache[uid] = cached
    return jpeg(cached, max_side=1600)


@app.post("/api/decision")
def api_decision():
    data = request.get_json(silent=True) or {}
    m = S.matches.get(str(data.get("uid")))
    value = data.get("value", "")
    if m is None or value not in ("same", "not", ""):
        return err("Unknown candidate.")
    with S.lock:
        S.decisions[m.uid] = value
    return jsonify(ok=True)


# ---------------------------------------------------------------- report
@app.post("/api/report")
def api_report():
    with S.lock:
        if S.result is None:
            return err("Run a search first.")
        dec = dict(S.decisions)
        if not gallery_report.reportable(S.result, dec):
            return err("No person is confirmed and no strong match is left.\n"
                       "Click 'This is the person' on a candidate, then save again.")
        try:
            folder = gallery_report.save_gallery_report(OUTPUT_DIR, S.result, dec)
        except Exception as e:
            traceback.print_exc()
            return err(f"Could not save the report: {type(e).__name__}: {e}", 500)
    rel = os.path.relpath(folder, OUTPUT_DIR).replace(os.sep, "/")
    return jsonify(folder=folder, url=f"/output/{rel}/report.html", zip=f"/api/report.zip?rel={rel}")


@app.get("/output/<path:filename>")
def serve_output(filename):
    return send_from_directory(OUTPUT_DIR, filename)


@app.get("/api/report.zip")
def report_zip():
    folder = safe_join(OUTPUT_DIR, request.args.get("rel", ""))
    if not folder or not os.path.isdir(folder) or os.path.abspath(folder) == os.path.abspath(OUTPUT_DIR):
        abort(404)
    base = os.path.join(tempfile.mkdtemp(), "report")
    path = shutil.make_archive(base, "zip", folder)
    return send_file(path, as_attachment=True, download_name=os.path.basename(folder.rstrip("/\\")) + ".zip")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "5000")),
            threaded=True, debug=False)
