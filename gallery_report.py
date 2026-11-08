"""Report for an image-database search: which person was found, at which time and frame.

Writes one folder containing
  report.html   - summary table + one section per person (reference photo, face crops,
                  time range, frame numbers, scores, annotated full frame)
  sightings.csv - the same facts in a spreadsheet-friendly form
  frames/ crops/ refs/ - the pictures used by the HTML
"""
import csv
import html
import os
import re
import time

import cv2
import numpy as np

import engine

CSS = """<style>
body{font-family:Arial,Helvetica,sans-serif;margin:24px;color:#222}
table{border-collapse:collapse;margin:8px 0 20px}
th,td{border:1px solid #ccc;padding:6px 10px;text-align:left;vertical-align:top}
th{background:#f0f0f0}
h2{margin-top:32px;border-bottom:2px solid #ddd;padding-bottom:6px}
.ref{height:90px;vertical-align:middle;margin-right:12px;border:1px solid #ccc}
.ok{color:#0a7a2f;font-weight:bold}.warn{color:#b36b00;font-weight:bold}
img.frame{width:460px;max-width:100%}
</style>"""


def _safe(text):
    return re.sub(r"[^\w\-]+", "_", text).strip("_")[:40] or "person"


def _write_jpg(path, img):
    """cv2.imwrite fails with non-English characters in the path on Windows; this does not."""
    ok, buf = cv2.imencode(".jpg", img)
    if ok:
        buf.tofile(path)


def reportable(result, decisions):
    """Sightings that go into the report: confirmed by the reviewer, or strong and not rejected."""
    rows = []
    for m in result.matches:
        d = decisions.get(m.uid, "")
        if d == "same":
            rows.append((m, "Confirmed by reviewer"))
        elif not d and m.strong:
            rows.append((m, "Strong match (not yet reviewed)"))
    return rows


def _save_frames(video_path, jobs):
    """jobs: {frame_no: [(box, label, out_path), ...]}.  One pass over the video."""
    if not jobs:
        return
    last = max(jobs)
    cap = cv2.VideoCapture(video_path)
    n = 0
    while n < last:
        if not cap.grab():
            break
        n += 1
        if n not in jobs:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            continue
        for box, label, path in jobs[n]:
            img = frame.copy()
            x1, y1, x2, y2 = box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)
            _write_jpg(path, img)
    cap.release()


def _crop_strip(crops, height=100):
    parts = []
    for c in crops:
        h, w = c.shape[:2]
        parts.append(cv2.resize(c, (max(1, int(w * height / float(h))), height)))
    return np.hstack(parts) if parts else None


def save_gallery_report(output_dir, result, decisions):
    rows = reportable(result, decisions)
    video_name = os.path.basename(result.video_path)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    folder = os.path.join(output_dir, f"search_{_safe(os.path.splitext(video_name)[0])}_{stamp}")
    for sub in ("frames", "crops", "refs"):
        os.makedirs(os.path.join(folder, sub), exist_ok=True)

    # reference thumbnails
    ref_file = {}
    for i, p in enumerate(result.people, 1):
        if p.thumb is not None:
            rel = f"refs/{i:03d}_{_safe(p.name)}.jpg"
            _write_jpg(os.path.join(folder, rel), p.thumb)
            ref_file[p.name] = rel

    # face crops and full frames (the video is read only once for all frames)
    jobs, files = {}, {}
    for m, _status in rows:
        tag = f"{m.uid:03d}_{_safe(m.person)}"
        frame_rel, crop_rel = f"frames/{tag}.jpg", f"crops/{tag}.jpg"
        strip = _crop_strip(m.crops)
        if strip is not None:
            _write_jpg(os.path.join(folder, crop_rel), strip)
        label = m.person.encode("ascii", "replace").decode()     # OpenCV cannot draw other scripts
        jobs.setdefault(m.mid_frame, []).append(
            (m.mid_box, label, os.path.join(folder, frame_rel)))
        files[m.uid] = (frame_rel, crop_rel)
    _save_frames(result.video_path, jobs)

    # CSV
    with open(os.path.join(folder, "sightings.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["person", "status", "start_time", "end_time", "first_frame", "last_frame",
                    "preview_frame", "detections", "arcface_distance", "facenet_similarity",
                    "frame_image", "video"])
        for m, status in rows:
            w.writerow([m.person, status, engine.fmt_time(m.start_s), engine.fmt_time(m.end_s),
                        m.first_frame, m.last_frame, m.mid_frame, m.n, f"{m.arc_track:.3f}",
                        "" if m.fn_track is None else f"{m.fn_track:.3f}",
                        files[m.uid][0], video_name])

    # HTML
    e = html.escape
    by_person = {}
    for m, status in rows:
        by_person.setdefault(m.person, []).append((m, status))

    out = ["<!DOCTYPE html><html><head><meta charset='utf-8'><title>Search report</title>", CSS,
           "</head><body>", "<h1>Image database search report</h1>",
           f"<p>Video: <b>{e(video_name)}</b><br>Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}"
           f"<br>People in the database: {len(result.people)}"
           f"<br>People found: {len(by_person)}</p>"]
    if not by_person:
        out.append("<p>No person was confirmed or matched strongly.</p>")
    else:
        out.append("<h2>Summary</h2><table><tr><th>Person</th><th>Sightings</th>"
                   "<th>First seen</th><th>Last seen</th><th>Frames</th></tr>")
        for name, items in by_person.items():
            ms = [m for m, _ in items]
            out.append(
                f"<tr><td>{e(name)}</td><td>{len(ms)}</td>"
                f"<td>{engine.fmt_time(min(m.start_s for m in ms))}</td>"
                f"<td>{engine.fmt_time(max(m.end_s for m in ms))}</td>"
                f"<td>{min(m.first_frame for m in ms)} - {max(m.last_frame for m in ms)}</td></tr>")
        out.append("</table>")
        for name, items in by_person.items():
            out.append("<h2>")
            if name in ref_file:
                out.append(f"<img class='ref' src='{ref_file[name]}' alt=''>")
            out.append(f"{e(name)}</h2><table><tr><th>Face crops</th><th>Details</th>"
                       "<th>Full frame</th></tr>")
            for m, status in items:
                frame_rel, crop_rel = files[m.uid]
                cls = "ok" if status.startswith("Confirmed") else "warn"
                fn = "n/a" if m.fn_track is None else f"{m.fn_track:.3f}"
                note = "" if m.closest else "<br><i>Another person in the database is closer to this face.</i>"
                out.append(
                    f"<tr><td><img src='{crop_rel}' alt=''></td>"
                    f"<td><span class='{cls}'>{e(status)}</span><br>"
                    f"Time {engine.fmt_time(m.start_s)} - {engine.fmt_time(m.end_s)}<br>"
                    f"Frames {m.first_frame} - {m.last_frame} (picture: frame {m.mid_frame})<br>"
                    f"{m.n} detections<br>ArcFace distance {m.arc_track:.3f}<br>"
                    f"FaceNet similarity {fn}{note}</td>"
                    f"<td><a href='{frame_rel}'><img class='frame' src='{frame_rel}' alt=''></a></td></tr>")
            out.append("</table>")
    out.append("</body></html>")
    with open(os.path.join(folder, "report.html"), "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    return folder