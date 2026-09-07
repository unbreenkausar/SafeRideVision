import time
import subprocess
import traceback
import logging
import gc
import re
import csv
import shutil
from pathlib import Path
from collections import deque, defaultdict
import os
import cv2
import numpy as np
from ultralytics import YOLO

# ------------------------------------------------------------------
# LOGGING SETUP — writes to BOTH the console and a log file, so if the
# Colab process/session dies silently we still have a file on disk
# (flushed after every line) showing exactly the last thing that
# happened before it died. Look at this file first when debugging a
# silent stop: /content/drive/MyDrive/output-Bike/Logs/run_debug.log
# ------------------------------------------------------------------
_LOG_DIR_FOR_LOGGING = "/content/drive/MyDrive/output-Bike/Logs"
os.makedirs(_LOG_DIR_FOR_LOGGING, exist_ok=True)
_log_file_path = os.path.join(_LOG_DIR_FOR_LOGGING, "run_debug.log")

logger = logging.getLogger("deepsortVideo")
logger.setLevel(logging.DEBUG)
logger.handlers.clear()

_file_handler = logging.FileHandler(_log_file_path, mode="a", encoding="utf-8")
_file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
logger.addHandler(_file_handler)

_console_handler = logging.StreamHandler()
_console_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
logger.addHandler(_console_handler)


def log_exception(context_msg):
    """Call this inside an `except Exception as e:` block. Prints the
    context message + the FULL traceback (not just str(e)) to both
    console and the log file, and flushes immediately so nothing is
    lost even if the process is killed right after."""
    logger.error(f"❌ EXCEPTION in {context_msg}")
    logger.error(traceback.format_exc())
    for h in logger.handlers:
        h.flush()
import os;
from torchvision import transforms
from blink_model import build_model, load_weights,ModelConfig
# --- Deep SORT (nwojke/deep_sort) imports ---
# Make sure `deep_sort` repo folder is in PYTHONPATH or installed as package
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort import preprocessing  # NMS on detections before tracker.update
from deep_sort.deep_sort.detection import Detection
from deep_sort.deep_sort.tracker import Tracker
from deep_sort.tools import generate_detections as gdet  # no longer used for the encoder now that OSNet replaces mars-small128; kept in case you want to fall back to it
import sys
sys.path.insert(0, "/content/deep-person-reid")  # cloned repo, not pip-installed globally
from torchreid.utils import FeatureExtractor  # pip install torchreid
import torch
from plate_detector import (
    detect_plates_in_bike_crop,
    read_plate_text_from_image,
    warm_up_ocr,
    warm_up_plate_detector,
)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
device ="cuda" if torch.cuda.is_available() else "cpu";
_orig_load = torch.load
torch.load = lambda *a, **k: _orig_load(*a, **{**k, 'weights_only': False})

# ----------------- User params -----------------
VIDEO_PATH = "/content/deepSort/Input";  # apna video path yahan
OUTPUT_PATH = "/content/drive/MyDrive/output-Bike" 
LOG_DIR=  "/content/drive/MyDrive/output-Bike/Logs"   # optional, processed video save karna ho
# Fallback root for a video's whole output folder (video + CSV + bikes/)
# when run_video() is called with output_path=None. See run_video() for the
# full layout:
#   <root>/<video_base>/annotatedvideo.mp4
#   <root>/<video_base>/<video_base>_bikes.csv
#   <root>/<video_base>/bikes/<track_id>/annotated_frame.jpg
#   <root>/<video_base>/bikes/<track_id>/plate_crop.jpg
PUBLIC_OUTPUT_DIR = os.path.join(OUTPUT_PATH, "public")
YOLO_MODEL = "yolov10m.pt"
# Bike-specific OSNet ReID weights (torchreid .pth.tar checkpoint), trained
# on your motorcycle re-identification dataset. Replaces the old TF
# mars-small128.pb pedestrian encoder — see build_osnet_encoder() below for
# the wrapper that makes it a drop-in replacement for gdet.create_box_encoder.
REID_WEIGHTS = "/content/drive/MyDrive/Motor-Bikes-Data/Replace-files/model.pth.tar-50"
OSNET_MODEL_NAME = "osnet_x1_0"          # matches build_model(name='osnet_x1_0', ...) in training
OSNET_INPUT_SIZE = (256, 256)            # (H, W) — matches ImageDataManager(height=256, width=256, ...)
                                          # NOTE: square, not the pedestrian-standard 256x128. Getting
                                          # this wrong silently distorts every crop at inference and
                                          # degrades the embeddings without throwing any error.
# Old combined mirror/indicator model — now used ONLY for the mirror class.
MIRROR_DETECTOR = "/content/deepSort/mirror-indicator-yolov10m/weights/bestM.pt"
# New model: yolov12m, trained on a single class ("indicator") only. Swap
# this path to wherever that weight file (e.g. best.pt) lives on disk.
INDICATOR_DETECTOR = "/content/deepSort/indicator-yolov12m/weights/best.pt"
Orientation_detector_path ='/content/drive/MyDrive/Motor-Bikes-Data/oreintation detector.pt'; 
# Tuned for mars-small128's pedestrian embedding space — OSNet bike features
# live in a different feature space, so re-tune this empirically once the
# new encoder is in (try a validation clip, sweep ~0.15-0.4, pick whichever
# minimizes ID switches without over-merging distinct bikes).
MAX_COSINE_DISTANCE = 0.2
NN_BUDGET = 100
MIN_CONFIDENCE = 0.3
NMS_MAX_OVERLAP = 1.0   # NMS applied to raw detections before they reach the tracker
N_INIT = 3
MAX_AGE = 30
DRAW_TRAILS = True
TRAIL_LEN = 20

# A bike's cropped bounding box must be at least this many pixels (w*h)
# before it's even considered as a candidate for that track's "best/largest
# image" (used later for plate OCR). Filters out the tiny/far/noisy boxes
# that would otherwise win "largest so far" on a track that never actually
# got close enough to read a plate from. Tune to your footage/resolution.
MIN_BIKE_AREA_FOR_BEST_IMAGE = 6000

# The plate DETECTOR (YOLO, from plate_detector.py) now runs every frame,
# for every live bike, inside the main loop below — same as the bike/
# mirror/indicator/orientation detectors already do. PLATE_CONF is that
# model's confidence threshold. OCR itself still does NOT run in the loop
# — see the post-loop OCR pass near the end of run_video().
PLATE_CONF = 0.3

# A detected plate box must be at least this many pixels (w*h) before it's
# considered a candidate for that bike's "best/largest plate seen so far"
# (the one whose crop gets saved as plate_crop.jpg). Same idea as
# MIN_BIKE_AREA_FOR_BEST_IMAGE, just tuned for a much smaller object —
# tune to your footage/resolution.
MIN_PLATE_AREA_FOR_SAVE = 400

# How many REAL (non carried-forward) detections a mirror/indicator slot
# needs before "this bike has both mirrors / both indicators" is allowed to
# become True. Keeps one noisy false-positive box from flipping that fact.
MIN_SLOT_HITS_FOR_BOTH = 3

# ----------------- Turn zones -----------------
# Define zero, one, or many "turn zones" — regions of the frame where, if a
# bike's box overlaps enough, it's considered to be taking a turn (in
# addition to the angle-based detector in track.py). Any mix of shapes is
# fine, and the list can be empty (pipeline then runs on angle-detection
# alone, same as before).
#
# Each zone is a dict:
#   {"type": "polygon",   "points": [(x1,y1), (x2,y2), (x3,y3), ...]}   # >=3 points, any shape
#   {"type": "rectangle", "points": [(x1,y1), (x2,y2)]}                  # top-left, bottom-right corners
#                                                                        # (a 4-point list also works — treated as a polygon)
#   {"type": "circle",    "center": (cx, cy), "radius": r}
#
# Example (uncomment / edit for your video):
# TURN_ZONES = [
#     {"type": "polygon",   "points": [(120, 300), (420, 300), (500, 520), (60, 520)]},
#     {"type": "rectangle", "points": [(900, 150), (1180, 420)]},
#     {"type": "circle",    "center": (1500, 650), "radius": 140},
# ]
TURN_ZONES = []

# % (0.0-1.0) of a bike's bounding-box area that must fall inside a turn
# zone before it counts as "taking the turn". Kept as its own variable so it
# can be tuned later without touching the detection logic itself.
TURN_ZONE_OVERLAP_THRESHOLD = 0.30

# Color/thickness the zones themselves are drawn with in the output video.
TURN_ZONE_DRAW_COLOR = (255, 255, 0)   # BGR — cyan-ish, distinct from bike/mirror/indicator boxes
TURN_ZONE_DRAW_THICKNESS = 2
# ------------------------------------------------
counter = 0;
os.makedirs(OUTPUT_PATH , exist_ok=True);
os.makedirs(LOG_DIR , exist_ok=True);


#----------------------------------------------------

cfg = ModelConfig(cnn_name="resnet18", seq_type="LSTM", num_layers=2, hidden_size=256, dropout=0.5)
blink_detector = build_model(cfg);
blink_detector = load_weights(blink_detector, "/content/drive/MyDrive/Motor-Bikes-Data/blink model weights/best_model.pth", device=device) 


# ImageNet normalization (resnet18 pretrained isi se train hua tha)
_normalize = transforms.Compose([
    transforms.ToTensor(),   # (H, W, C) numpy -> (C, H, W) tensor, scaled to [0, 1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
])

@torch.no_grad()
def predict_from_track(model, track, slot_idx, device="cpu"):
    """
    Track's stored 64x64 indicator images (for ONE indicator slot — left or
    right, kept separate) -> single blink prediction for that slot.
    Returns (predicted_class, confidence) or None if that slot has no
    stored crops yet.
    """
    frames = track.getIndicatorImages(slot_idx)
    if not frames:
        return None

    processed = []
    for frame in frames:
        frame = np.asarray(frame)
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)

        chw = _normalize(frame)          # (C, 64, 64), normalized
        hwc = chw.permute(1, 2, 0)       # (64, 64, C) -- model ye layout expect karta hai
        processed.append(hwc)

    seq = torch.stack(processed, dim=0).unsqueeze(0).to(device)   # (1, T, 64, 64, C)
    length = torch.tensor([seq.shape[1]], device=device)

    model.eval()
    logits = model(seq, length)
    probs = torch.softmax(logits, dim=1)
    pred = torch.argmax(probs, dim=1).item()
    conf = probs[0, pred].item()

    return pred, conf


def extract_indicator_crop(frame, indicator_box, size=(64, 64)):
    """
    Crop indicator from frame, resize to 64x64 and return numpy array.

    Parameters
    ----------
    frame : np.ndarray
        Original BGR image.
    indicator_box : list/tuple
        [confidence, x1, y1, x2, y2]
        or [x1, y1, x2, y2]
    size : tuple
        Output image size (default 64x64)

    Returns
    -------
    np.ndarray or None
        Shape: (64, 64, 3)
    """

    # Handle both formats
    if len(indicator_box) == 5:
        _, x1, y1, x2, y2 = indicator_box
    else:
        x1, y1, x2, y2 = indicator_box

    h, w = frame.shape[:2]

    # Clip coordinates
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(w, int(x2))
    y2 = min(h, int(y2))

    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    crop = cv2.resize(crop, size)
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)  # blink model expects RGB, frame is BGR (OpenCV)

    return crop.astype(np.uint8)


def build_osnet_encoder(model_path, model_name=OSNET_MODEL_NAME,
                         input_size=OSNET_INPUT_SIZE, device="cpu"):
    """
    Returns a callable `encoder(frame, boxes_tlwh) -> np.ndarray (N, feat_dim)`
    that matches the exact interface gdet.create_box_encoder(...) used to
    provide, so it drops straight into `features = encoder(frame, tlwhs)`
    with no other changes needed in run_video().

    Uses torchreid's FeatureExtractor, which handles resizing/normalization
    to match how the model was trained — just make sure `image_size` here
    matches what you trained OSNet at, or crops get distorted.
    """
    extractor = FeatureExtractor(
        model_name=model_name,
        model_path=model_path,
        image_size=input_size,   # (H, W)
        device=device,
    )

    def encoder(frame, boxes_tlwh):
        if len(boxes_tlwh) == 0:
            return np.zeros((0, 512), dtype=np.float32)  # osnet_x1_0 default feat dim

        h_frame, w_frame = frame.shape[:2]
        crops = []
        for (x, y, w, h) in boxes_tlwh:
            x1, y1 = max(0, int(x)), max(0, int(y))
            x2, y2 = min(w_frame, int(x + w)), min(h_frame, int(y + h))
            if x2 <= x1 or y2 <= y1:
                # degenerate box (can happen if a tlwh comes in clipped to
                # nothing) — feed a black patch so the batch stays aligned
                # with boxes_tlwh instead of silently dropping an index
                crop = np.zeros((input_size[0], input_size[1], 3), dtype=np.uint8)
            else:
                crop = frame[y1:y2, x1:x2]
            crops.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

        with torch.no_grad():
            feats = extractor(crops)                       # (N, feat_dim) torch tensor
            feats = torch.nn.functional.normalize(feats, p=2, dim=1)

        return feats.cpu().numpy().astype(np.float32)

    return encoder


# ----------------- Optical flow: removed -----------------
# RAFT (deep-learning optical flow) was pulled out entirely -- its
# correlation volume was OOM-ing the GPU at video-native resolution on top
# of YOLO/OSNet/the blink model already resident in VRAM. bike_flow_stats()
# below already treats flow=None as "no flow data" and returns all zeros,
# so run_video() just calls it with flow=None now -- setFlow()/CSV
# output/etc. downstream keep working unchanged, they just always get
# (0, 0, 0, 0) for the flow fields. If flow is needed again later, a
# lightweight classical method (e.g. cv2.calcOpticalFlowFarneback, CPU-only,
# no GPU memory pressure) is a much safer drop-in than another
# deep-learning flow-net on a shared T4.


def bike_flow_stats(flow, bbox):
    """
    Average optical-flow vector inside a single bike's bbox for this frame.
    Returns (dx, dy, magnitude, angle_deg) — all zero if flow is unavailable
    (first frame) or the box is degenerate.
    """
    if flow is None:
        return 0.0, 0.0, 0.0, 0.0

    h, w = flow.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0, 0.0, 0.0, 0.0

    patch = flow[y1:y2, x1:x2]
    dx = float(np.mean(patch[..., 0]))
    dy = float(np.mean(patch[..., 1]))
    magnitude = float(np.hypot(dx, dy))
    angle = float(np.degrees(np.arctan2(dy, dx)))
    return dx, dy, magnitude, angle


def xyxy_to_tlwh(box):
    # box = [x1, y1, x2, y2]
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    return [int(x1), int(y1), int(w), int(h)]


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Calculate intersection area
    inter_width = max(0, x2 - x1)
    inter_height = max(0, y2 - y1)
    inter_area = inter_width * inter_height
    
    # Calculate union area
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area
    
    iou = inter_area / union_area if union_area != 0 else 0
    return iou

def is_box_inside(box1, box2, method="center"):
    """
    Check if box2 is inside box1.

    Parameters:
        box1: list or tuple [x1, y1, x2, y2]  # parent box
        box2: list or tuple [x1, y1, x2, y2]  # child box
        method: "center" (default), "full", or "iou"
    
    Returns:
        True if box2 is considered inside box1, else False
    """

    x1, y1, x2, y2 = box1
    bx1, by1, bx2, by2 = box2

    if method == "center":
        # Check center point of box2
        cx = (bx1 + bx2) / 2
        cy = (by1 + by2) / 2
        return x1 <= cx <= x2 and y1 <= cy <= y2

    elif method == "full":
        # Check full containment
        return x1 <= bx1 and y1 <= by1 and x2 >= bx2 and y2 >= by2

    elif method == "iou":
        # Check IoU threshold (partial overlap)
        inter_x1 = max(x1, bx1)
        inter_y1 = max(y1, by1)
        inter_x2 = min(x2, bx2)
        inter_y2 = min(y2, by2)

        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (bx2 - bx1) * (by2 - by1)
        iou = inter_area / (area1 + area2 - inter_area) if (area1 + area2 - inter_area) > 0 else 0

        return iou > 0.1  # threshold adjustable

    else:
        raise ValueError("Method must be 'center', 'full', or 'iou'")

def _zone_polygon_points(zone):
    """Return an (N,2) int32 array of vertices for a polygon/rectangle zone.
    Rectangles given as 2 corner points get expanded to 4; anything else
    (>=3 points, arbitrary polygon) passes through as-is. Not used for
    circles, which are rasterized directly.
    """
    pts = zone["points"]
    if len(pts) == 2:
        (x1, y1), (x2, y2) = pts
        x1, x2 = sorted((float(x1), float(x2)))
        y1, y2 = sorted((float(y1), float(y2)))
        pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    return np.array(pts, dtype=np.int32)


def build_zone_mask(zones, width, height):
    """Rasterize every configured turn zone — any mix of polygon, rectangle,
    or circle, any count — into a single binary (H, W) mask where a pixel is
    1 if it falls inside ANY zone. Built once per video (zones are treated
    as static for the whole clip), so per-frame overlap checks below are
    just a cheap crop + count against this mask.

    Returns None if `zones` is empty, so callers can skip zone-based turn
    detection entirely when no zones are configured (the "no polygon" case).
    """
    if not zones:
        return None

    mask = np.zeros((height, width), dtype=np.uint8)
    for zone in zones:
        ztype = zone.get("type", "polygon").lower()
        if ztype == "circle":
            cx, cy = map(int, zone["center"])
            r = int(zone["radius"])
            cv2.circle(mask, (cx, cy), r, 1, thickness=-1)
        elif ztype in ("polygon", "rectangle"):
            pts_arr = _zone_polygon_points(zone)
            cv2.fillPoly(mask, [pts_arr], 1)
        else:
            raise ValueError(f"Unknown turn zone type: {ztype!r} (expected 'polygon', 'rectangle', or 'circle')")
    return mask


def zone_overlap_ratio(mask, bbox):
    """Fraction (0.0-1.0) of `bbox` [x1,y1,x2,y2] that falls inside ANY
    configured turn zone, per the mask from build_zone_mask(). Returns 0.0
    if there's no mask (no zones configured) or the box is degenerate/fully
    off-frame.
    """
    if mask is None:
        return 0.0

    h, w = mask.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0

    box_area = (x2 - x1) * (y2 - y1)
    inside = int(np.count_nonzero(mask[y1:y2, x1:x2]))
    return inside / box_area if box_area > 0 else 0.0


def draw_zones(frame, zones, color=TURN_ZONE_DRAW_COLOR, thickness=TURN_ZONE_DRAW_THICKNESS):
    """Draw every configured turn zone onto `frame` — any mix of polygon,
    rectangle, or circle, any count — so the output video shows exactly what
    bikes are being checked against. No-op if `zones` is empty.
    """
    for zone in zones:
        ztype = zone.get("type", "polygon").lower()
        if ztype == "circle":
            cx, cy = map(int, zone["center"])
            r = int(zone["radius"])
            cv2.circle(frame, (cx, cy), r, color, thickness)
        elif ztype in ("polygon", "rectangle"):
            pts_arr = _zone_polygon_points(zone).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts_arr], isClosed=True, color=color, thickness=thickness)
        else:
            raise ValueError(f"Unknown turn zone type: {ztype!r} (expected 'polygon', 'rectangle', or 'circle')")


def expand_box(x1,y1,x2,y2,  w, h, pad_ratio=0.30):
    bw = x2-x1
    bh = y2-y1

    pad_w = int(bw * pad_ratio)
    pad_h = int(bh * pad_ratio)

    x1 = max(0, x1-pad_w)
    y1 = max(0, y1-pad_h)
    x2 = min(w, x2+pad_w)
    y2 = min(h, y2+pad_h)

    return x1,y1,x2,y2


def getDetections(model, frame , detect , confidence = MIN_CONFIDENCE):
        try:
            results = model(frame , imgsz = 1280)
        except Exception:
            log_exception(f"YOLO inference failed for detect={detect}")
            return [], []

        detections = []
        class_name = []
        try:
            res = results[0]
            if hasattr(res, "boxes"):
                for box in res.boxes:
                    try:
                        cls_id= int(box.cls[0]) if hasattr(box.cls, "__getitem__") else int(box.cls)
                        name = res.names[cls_id]
                        print(f'ClassName : give: {detect} -> found {name}')
                        mx1, my1, mx2, my2 = box.xyxy[0].cpu().numpy()
                        mx1, my1, mx2, my2 = int(mx1), int(my1), int(mx2), int(my2)
                        conf = float(box.conf[0].cpu().numpy()) if hasattr(box.conf, "__getitem__") else float(box.conf)
                        if conf > confidence and name in detect:
                            detections.append([conf, mx1, my1, mx2, my2])
                            class_name.append(name)
                    except Exception:
                        log_exception(f"parsing one YOLO box for detect={detect}")
                        continue
        except Exception:
            log_exception(f"parsing YOLO results for detect={detect}")

        return detections , class_name


def active_tracks(tracker):
    """
    Tracks that are confirmed (passed n_init) AND were actually matched to a
    detection this frame (time_since_update == 0).

    Iterating tracker.tracks directly (as the original code did) also pulls
    in tentative tracks (unconfirmed, can vanish next frame) and stale/ghost
    tracks that Deep SORT is still coasting on Kalman prediction alone for up
    to max_age frames after last seeing them. Drawing/associating against
    those is what produces "duplicate" boxes on a single physical object
    when a fresh track spawns for it while the old ghost track is still
    being rendered.
    """
    return [t for t in tracker.tracks if t.is_confirmed() and t.time_since_update == 0]


def best_match_track(tracks, box, method="center"):
    """
    Return the single track whose box has the highest IoU with `box` among
    those where is_box_inside(...) also holds, or None.

    The original code assigned a mirror/indicator/orientation detection to
    EVERY track that satisfied is_box_inside(). When two tracked bikes
    overlap (common in traffic), that lets one detection get attached to
    more than one track -> duplicate annotations that look like ID mixups.
    Picking a single best match fixes that.
    """
    best_track, best_iou = None, 0.0
    for t in tracks:
        t_box = t.to_tlbr()
        if not is_box_inside(t_box, box, method=method):
            continue
        iou = compute_iou(t_box, box)
        if iou >= best_iou:
            best_iou = iou
            best_track = t
    return best_track


def _save_bike_snapshot(track, images_dir):
    """Write (or overwrite) this bike's PUBLIC annotated frame to disk
    RIGHT NOW, using whatever track.full_frame_image / track.full_frame_box
    currently hold:
        images_dir/<track_id>/annotated_frame.jpg   (full frame + box —
            this is the final artifact a human/frontend sees)

    Called every time update_best_image() reports a NEW best (bigger) crop
    — i.e. continuously, throughout video processing, not just once when
    the track ends — so the file keeps getting overwritten by whichever
    frame gave the biggest view of this bike so far. Also called once more
    from finalize_track() as a safety net.

    The bike's plate_crop.jpg is a SEPARATE, independent file — see
    _save_plate_snapshot() below, driven by its own "biggest plate box
    seen so far" tracking (best_plate_area_by_track in run_video()), not
    tied to whichever frame happened to have the biggest overall bike crop.

    Returns the annotated_frame.jpg path, or None if there was nothing to
    save.
    """
    if images_dir is None or track.best_image is None:
        return None
    try:
        bike_dir = os.path.join(images_dir, str(track.track_id))
        os.makedirs(bike_dir, exist_ok=True)

        if track.full_frame_image is not None:
            annotated = track.full_frame_image.copy()
            if track.full_frame_box is not None:
                x1, y1, x2, y2 = track.full_frame_box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated, f"ID:{track.track_id}", (x1, max(0, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            annotated_frame_path = os.path.join(bike_dir, "annotated_frame.jpg")
            cv2.imwrite(annotated_frame_path, annotated)
            return annotated_frame_path

        return None
    except Exception as e:
        print(f"Track {track.track_id}: failed to save snapshot image ({e}).")
        return None


def _save_plate_snapshot(track_id, plate_crop_rgb, images_dir):
    """Write (or overwrite) this bike's PUBLIC plate crop to disk RIGHT NOW:
        images_dir/<track_id>/plate_crop.jpg

    Called every frame the plate DETECTOR (running live, inside the main
    loop — see run_video()) finds a bigger/clearer plate box for this bike
    than any seen before that video. NO OCR happens here or anywhere in the
    frame loop — this only ever saves the located CROP. Reading the actual
    numbers off of it is the post-loop pass's job, at the very end of
    run_video(), once every frame has been processed — see
    read_plate_text_from_image() in plate_detector.py.

    Parameters
    ----------
    plate_crop_rgb : np.ndarray
        RGB image. detect_plates_in_bike_crop() converts its input to RGB
        internally before slicing the plate out of it, so what it returns
        is RGB, not the BGR order cv2 normally uses — flip it back here
        before cv2.imwrite, or the saved jpg's colors come out swapped.

    Returns the plate_crop.jpg path, or None if there was nothing to save.
    """
    if images_dir is None or plate_crop_rgb is None or plate_crop_rgb.size == 0:
        return None
    try:
        bike_dir = os.path.join(images_dir, str(track_id))
        os.makedirs(bike_dir, exist_ok=True)
        plate_crop_bgr = cv2.cvtColor(plate_crop_rgb, cv2.COLOR_RGB2BGR)
        plate_crop_path = os.path.join(bike_dir, "plate_crop.jpg")
        cv2.imwrite(plate_crop_path, plate_crop_bgr)
        return plate_crop_path
    except Exception as e:
        print(f"Track {track_id}: failed to save plate crop ({e}).")
        return None


def finalize_track(track, records, images_dir=None):
    """Called exactly once per bike, the moment its track is gone for good
    — either DeepSORT actually deleted it (lost for too many frames), or
    the video ended while it was still alive.

    Plate DETECTION already happened live, every frame this bike was on
    screen, in the main loop (see run_video()'s "plate DETECTOR (not OCR)"
    block) — so by the time this runs, images_dir/<track_id>/plate_crop.jpg
    already holds the biggest/clearest plate box ever found for this bike,
    if one ever was. OCR (reading the actual numbers off of it) is still
    deliberately NOT run here — that only happens once, at the very end of
    run_video(), after every detector model has been unloaded from CPU/GPU
    (see the post-loop OCR pass there), so PaddleOCR never has to compete
    with YOLO/OSNet/the blink model for CPU/GPU mid-video.

    So this function only:
      1. Re-saves this bike's PUBLIC annotated frame via
         _save_bike_snapshot() — images_dir/<track_id>/annotated_frame.jpg
         — as a safety net, in case the frame loop's own continuous saving
         never fired for this track.
      2. Appends one record — everything the final CSV row needs EXCEPT
         plate_number — to `records`, a list shared across the whole
         run_video() call. plate_number gets filled in by the post-loop
         OCR pass, which reads plate_crop.jpg straight off disk.

    Tracks that never produced a large-enough crop (see
    MIN_BIKE_AREA_FOR_BEST_IMAGE / Track.update_best_image) are skipped
    entirely and never get a record — a track that was always tiny was
    most likely a noisy/false detection, never a bike close enough to
    read a plate from, so it isn't worth a row.
    """
    if track.best_image is None:
        print(f"Track {track.track_id}: no large-enough crop ever captured — treating as noise, skipping.")
        return

    annotated_frame_path = _save_bike_snapshot(track, images_dir)
    if annotated_frame_path:
        print(f"Track {track.track_id}: snapshot image -> {annotated_frame_path}")

    records.append({
        "track_id": track.track_id,
        "mirror_seen_both": track.mirror_seen_both,
        "indicator_seen_both": track.indicator_seen_both,
        "ever_turned": track.ever_turned,
        "ever_signaled_while_turning": track.ever_signaled_while_turning,
    })

    print(f"Track {track.track_id} finalized (plate crop already captured live; OCR deferred to "
          f"end-of-video pass) — both_mirrors={track.mirror_seen_both}, "
          f"both_indicators={track.indicator_seen_both}, turned={track.ever_turned}, "
          f"signaled_while_turning={track.ever_signaled_while_turning}")


def draw_box(image, coords, label="Box", color=(0,255,0)):
    x1, y1, x2, y2 = map(int, coords)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def _safe_slug(name, fallback="video"):
    """Filesystem-safe base name (extension stripped), used for BOTH the
    public snapshot folder and the summary CSV filename — so the same
    original video name always maps to the same on-disk location, and
    re-uploading a video with that name overwrites its previous run
    instead of piling up next to it."""
    base = os.path.splitext(name)[0]
    base = re.sub(r"[^A-Za-z0-9_.-]+", "_", base).strip("._")
    return base or fallback


def run_video(video_path, output_path=None, display_name=None):
    """
    Parameters
    ----------
    video_path : str
        Actual file on disk to read frames from (can be a
        timestamp-prefixed filename so concurrent uploads never collide).
    output_path : str, optional
        Root folder this video's own output folder (named after
        video_base) gets created under. Everything this run produces —
        the annotated video, the summary CSV, and the per-bike bikes/
        folder — lives together inside <output_path>/<video_base>/.
    display_name : str, optional
        The video's ORIGINAL name (e.g. exactly what the user uploaded).
        Used to name that per-video output folder, so re-uploading a video
        with this same name OVERRIDES its previous output folder instead
        of creating a separate copy. Falls back to video_path's own
        filename when not given.

    Returns
    -------
    dict with:
        "output_video_path" : the final browser-playable .mp4 path
        "summary_csv_path"  : the per-bike summary CSV path (written once,
                               at the very end, after the post-loop OCR
                               pass reads each bike's plate_crop.jpg —
                               plate DETECTION itself already ran live,
                               every frame, during the main loop)
        "video_images_dir"  : the public per-bike "bikes" folder, one
                               subfolder per track_id
        "video_dir"         : the per-video root folder holding all of the
                               above (video, CSV, bikes/)
        "video_base"        : the sanitized base name used for video_dir
    """
    vidName = video_path.split("/")[-1]

    # Base name for this video's own output folder + files. Prefers the
    # ORIGINAL video name (display_name) so repeat uploads of the same
    # video overwrite the same location; falls back to the on-disk name
    # (e.g. when run_video() is called directly, see __main__ below).
    video_base = _safe_slug(display_name if display_name else vidName)

    # Everything this video produces now lives together under ONE folder:
    #
    #   <output_root>/<video_base>/
    #       annotatedvideo.mp4
    #       <video_base>_bikes.csv
    #       bikes/<track_id>/annotated_frame.jpg
    #       bikes/<track_id>/plate_crop.jpg
    #
    # `output_path` (the folder argument) is the root that <video_base>/
    # gets created under; PUBLIC_OUTPUT_DIR is only a fallback for when
    # run_video() is called without one (see __main__ below).
    #
    # If this video name was processed before, wipe its whole folder first
    # — an override, not a merge, so nothing stale (old bike folders, old
    # CSV, old video) from a previous run can ever linger into this one.
    output_root = output_path if output_path is not None else PUBLIC_OUTPUT_DIR
    video_dir = os.path.join(output_root, video_base)
    if os.path.isdir(video_dir):
        try:
            shutil.rmtree(video_dir)
        except Exception:
            log_exception(f"clearing previous output folder for video_base={video_base!r}")
    os.makedirs(video_dir, exist_ok=True)

    output_path = os.path.join(video_dir, "annotatedvideo.mp4")

    # The actual per-bike summary CSV is only written ONCE, at the very end
    # of this function, after the post-loop OCR pass has read every bike's
    # plate_crop.jpg — see near the bottom. No live per-row writes. Plate
    # DETECTION itself (locating + cropping the plate) already ran live,
    # every frame, in the main loop below.
    summary_csv_path = os.path.join(video_dir, video_base + "_bikes.csv")

    # Public per-bike snapshot folder — one subfolder per bike (named by
    # track_id), each holding that bike's annotated_frame.jpg (written
    # continuously during the loop) and plate_crop.jpg (written once, by
    # the post-loop pass, once a plate has actually been located).
    video_images_dir = os.path.join(video_dir, "bikes")
    os.makedirs(video_images_dir, exist_ok=True)

    # Every Track object we've ever seen this video, keyed by track_id. Kept
    # alive here even after DeepSORT drops a deleted track from
    # tracker.tracks, so we can still read its accumulated state
    # (best_image, mirror/indicator facts, turn facts) to finalize it.
    known_tracks = {}

    # Filled in by finalize_track() as each bike's track ends — one dict per
    # bike with everything the CSV needs EXCEPT plate_number, which is
    # resolved in the post-loop OCR pass at the end of this function.
    records = []

    # track_id -> area (pixels, w*h) of the biggest plate box the plate
    # DETECTOR has found for that bike so far, this video. Lives for the
    # same lifetime as known_tracks (a run_video()-wide dict, not reset per
    # frame) so a bike's plate_crop.jpg only ever gets overwritten by a
    # BIGGER/clearer plate box, never a smaller one — see the plate
    # DETECTOR block in the main loop below and _save_plate_snapshot().
    best_plate_area_by_track = {}

    yolo = YOLO(YOLO_MODEL).to(device)
    mirror_detector = YOLO(MIRROR_DETECTOR).to(device)
    indicator_detector = YOLO(INDICATOR_DETECTOR).to(device)
    orientation_detector = YOLO(Orientation_detector_path).to(device)

    # Plate DETECTOR (YOLO) — preloaded up front, right alongside the other
    # per-frame detectors above, since it now runs every frame inside the
    # main loop too (see the plate DETECTOR block below). This is just
    # another YOLO model — it's PaddleOCR (see warm_up_ocr(), called much
    # later) that's kept out of the loop, not this.
    try:
        warm_up_plate_detector()
    except Exception:
        log_exception("warm_up_plate_detector() before main loop")

    # Init DeepSORT
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", MAX_COSINE_DISTANCE, NN_BUDGET)
    tracker = Tracker(metric, max_iou_distance=0.7, max_age=MAX_AGE, n_init=N_INIT)

    encoder = build_osnet_encoder(REID_WEIGHTS, device=device)

    # NOTE: PaddleOCR is intentionally NOT warmed up here. It is only
    # loaded once, at the very end of this function, after every frame is
    # processed and every YOLO/OSNet/blink model (including the plate
    # detector) has been unloaded from CPU/GPU — see the post-loop OCR pass
    # near the bottom of run_video(). That is what actually avoids the
    # PaddlePaddle/PyTorch CUDA-context conflict that used to cause silent
    # hangs. The plate DETECTOR itself (warmed up just above) is a
    # different story — it's just another YOLO model, so it runs fine
    # every frame alongside the others.

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # OpenCV's mp4v encoder is reliable to write to, but the resulting file
    # uses a codec browsers can't decode (only desktop players like VLC/WMP
    # can). So we write to a temp file with mp4v as before, then re-encode
    # it to H.264 (browser-playable) as a final step below, and delete the
    # temp file. `output_path` stays the final, browser-compatible filename.
    raw_output_path = None
    writer = None
    if output_path:
        raw_output_path = output_path.replace(".mp4", "_raw_temp.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(raw_output_path, fourcc, fps, (width, height))

    # Built once — zones are static for the whole clip. None if TURN_ZONES
    # is empty, in which case zone_overlap_ratio() always returns 0.0 and
    # the pipeline runs on angle-based turn detection alone (unchanged
    # behavior for videos with no zones configured).
    zone_mask = build_zone_mask(TURN_ZONES, width, height)

    trails = {}
    frame_idx = 0
    t0 = time.time()

    while True:
        try:
            ret, frame = cap.read()
        except Exception:
            log_exception(f"cap.read() at frame_idx={frame_idx}")
            break
        if not ret:
            break
        frame_idx += 1
        # Heartbeat — flushed immediately. If the process dies with no
        # traceback, the LAST of these lines printed tells you exactly
        # which frame it died on.
        print(f"[FRAME {frame_idx}] --- start ---", flush=True)
        logger.debug(f"[FRAME {frame_idx}] start")

        flow = None  # optical flow removed (was RAFT) -- bike_flow_stats() returns zeros for this

        try:
            motorcycle_boxes ,_ = getDetections(yolo , frame , ['motorcycle']);
        except Exception:
            log_exception(f"getDetections(yolo/motorcycle) at frame_idx={frame_idx}")
            motorcycle_boxes = []

        detections = []
        try:
            if motorcycle_boxes:
                tlwhs = [xyxy_to_tlwh(p[1:]) for p in motorcycle_boxes]
                confidences = [p[0] for p in motorcycle_boxes]
                features = encoder(frame, tlwhs)

                # --- NMS on raw detections BEFORE they reach the tracker -------
                # Without this, two overlapping YOLO boxes on the same bike both
                # get fed to tracker.update() and Deep SORT can spawn a second
                # track for the same physical object -> duplicate IDs.
                boxes_arr = np.array(tlwhs, dtype=np.float32)
                scores_arr = np.array(confidences, dtype=np.float32)
                keep = preprocessing.non_max_suppression(boxes_arr, NMS_MAX_OVERLAP, scores_arr)

                for i in keep:
                    detections.append(Detection(tlwhs[i], confidences[i], features[i]))
        except Exception:
            log_exception(f"OSNet encoder / NMS at frame_idx={frame_idx}")
            detections = []

        try:
            tracker.predict()
            tracker.update(detections)
        except Exception:
            log_exception(f"tracker.predict()/update() at frame_idx={frame_idx}")
            continue

        # DeepSORT's Tracker.update() drops a track from tracker.tracks the
        # instant it's marked Deleted (lost for too many frames). Any
        # track_id that was in known_tracks last frame but is missing from
        # tracker.tracks now was JUST deleted -> this is the one moment we
        # finalize it (resolve its plate from the stored largest image,
        # write its one summary CSV row) and drop our own reference too.
        current_ids = {t.track_id for t in tracker.tracks}
        for tid in list(known_tracks.keys()):
            if tid not in current_ids:
                try:
                    finalize_track(known_tracks[tid], records, video_images_dir)
                except Exception:
                    log_exception(f"finalize_track(track_id={tid}) at frame_idx={frame_idx}")
                del known_tracks[tid]
        for t in tracker.tracks:
            known_tracks[t.track_id] = t

        try:
            mirrors,_  = getDetections(mirror_detector, frame , ['mirror']);
        except Exception:
            log_exception(f"getDetections(mirror_detector) at frame_idx={frame_idx}")
            mirrors = []
        try:
            indicators,_  = getDetections(indicator_detector, frame , ['indicator']);
        except Exception:
            log_exception(f"getDetections(indicator_detector) at frame_idx={frame_idx}")
            indicators = []
        try:
            orientation_boxes, orientation_classes = getDetections(
                orientation_detector,
                frame,
                ['front-back', 'side']
            )
        except Exception:
            log_exception(f"getDetections(orientation_detector) at frame_idx={frame_idx}")
            orientation_boxes, orientation_classes = [], []

        # NOTE: mirrors and indicators are no longer wiped every frame here.
        # Both now persist across missed frames on purpose (see
        # Track.update_mirrors() / Track.update_indicators()) — a mirror or
        # indicator YOLO fails to find this frame keeps its last-known box,
        # shifted forward by the bike's own motion, instead of going blank.

        # Only associate/draw against tracks that are confirmed AND were
        # actually matched to a detection this frame. Using tracker.tracks
        # directly pulls in tentative and stale/ghost tracks, which is the
        # other main source of "duplicate" boxes and mis-assigned
        # mirror/indicator/orientation detections.
        live_tracks = active_tracks(tracker)

        # Update trajectories / turn detection FIRST, so bike.is_turning
        # below reflects THIS frame rather than lagging by one frame (as in
        # the original ordering, where detect_turn() ran after is_turning
        # was already consumed for the indicator/blink check).
        turn_results = {}
        for bike in live_tracks:
            try:
                bike.setTrajectories(bike.to_tlbr())
                turn_results[bike.track_id] = bike.detect_turn()

                # Turn-zone check: works alongside the angle-based detector
                # above, not instead of it — either one flagging a turn is
                # enough. With TURN_ZONES empty, zone_mask is None and this is a
                # no-op (ratio 0.0, is_turning left exactly as detect_turn() set
                # it), so this is safe to leave on for every video.
                ratio = zone_overlap_ratio(zone_mask, bike.to_tlbr())
                bike.zone_overlap_ratio = ratio
                if ratio >= TURN_ZONE_OVERLAP_THRESHOLD:
                    bike.is_turning = True
            except Exception:
                log_exception(f"trajectory/turn update track_id={bike.track_id} at frame_idx={frame_idx}")

        for box, orientation in zip(orientation_boxes, orientation_classes):
            try:
                bbox = box[1:]  # x1, y1, x2, y2
                bike = best_match_track(live_tracks, bbox, method="iou")
                if bike is not None:
                    bike.setOrientation(orientation)
            except Exception:
                log_exception(f"orientation assignment at frame_idx={frame_idx}")

        # Group this frame's mirror detections by the track they matched,
        # for the same reason indicators are grouped below: update_mirrors()
        # needs to see both of a bike's mirror boxes together (0, 1, or 2)
        # to match them against the two persistent slots correctly.
        mirrors_by_track = defaultdict(list)
        for mirror in mirrors:
            try:
                bike = best_match_track(live_tracks, mirror[1:], method="center")
                if bike is not None:
                    mirrors_by_track[bike.track_id].append(mirror)
            except Exception:
                log_exception(f"mirror best_match_track at frame_idx={frame_idx}")

        for bike in live_tracks:
            try:
                dets = mirrors_by_track.get(bike.track_id, [])

                traj = bike.getAllTrajectoriesOfX()
                if len(traj) >= 2:
                    move_dx = traj[-1][0] - traj[-2][0]
                    move_dy = traj[-1][1] - traj[-2][1]
                else:
                    move_dx, move_dy = 0.0, 0.0

                bike.update_mirrors(dets, move_dx, move_dy,
                                     min_hits_for_both=MIN_SLOT_HITS_FOR_BOTH)
            except Exception:
                log_exception(f"update_mirrors track_id={bike.track_id} at frame_idx={frame_idx}")

        # Group this frame's indicator detections by the track they matched,
        # so both indicators on a bike (0, 1, or 2 boxes) are handed to
        # update_indicators() together — the nearest-neighbor slot matching
        # needs to see them as a set, not one at a time, or it can't tell
        # which detection is "the one that was already left" vs "right".
        indicators_by_track = defaultdict(list)
        for indicator in indicators:
            try:
                bike = best_match_track(live_tracks, indicator[1:], method="center")
                if bike is not None:
                    indicators_by_track[bike.track_id].append(indicator)
            except Exception:
                log_exception(f"indicator best_match_track at frame_idx={frame_idx}")

        for bike in live_tracks:
            try:
                dets = indicators_by_track.get(bike.track_id, [])

                # How far the bike itself moved this frame, so a slot that
                # misses detection this frame can still shift forward with the
                # bike instead of freezing or vanishing.
                traj = bike.getAllTrajectoriesOfX()
                if len(traj) >= 2:
                    move_dx = traj[-1][0] - traj[-2][0]
                    move_dy = traj[-1][1] - traj[-2][1]
                else:
                    move_dx, move_dy = 0.0, 0.0

                bike.update_indicators(dets, move_dx, move_dy,
                                        min_hits_for_both=MIN_SLOT_HITS_FOR_BOTH)
            except Exception:
                log_exception(f"update_indicators track_id={bike.track_id} at frame_idx={frame_idx}")
                continue

            if not bike.is_turning:
                continue

            any_signal = False
            for slot_idx, slot in enumerate(bike.indicator_slots):
                if slot["cord"] is None:
                    continue

                try:
                    croped = extract_indicator_crop(frame, slot["cord"])
                    if croped is not None:
                        # Each slot keeps its own crop history, so the blink
                        # model is trained/queried per-indicator, not on a mix
                        # of left+right frames.
                        bike.setIndicatorImage(slot_idx, croped)

                    result = predict_from_track(blink_detector, bike, slot_idx, device=device)
                    if result:
                        pred_class, confidence = result
                        slot["is_signal"] = True if pred_class == 0 else False
                        print(f"Track {bike.track_id} slot {slot_idx} prediction: {pred_class} ({confidence:.2%})", flush=True)
                except Exception:
                    log_exception(f"blink prediction track_id={bike.track_id} slot={slot_idx} at frame_idx={frame_idx}")
                any_signal = any_signal or slot["is_signal"]

            # Bike-level flag: on if EITHER indicator is judged to be
            # blinking (normal turning uses just one side; both only for
            # hazards, which still counts as "signaling").
            bike.is_signal = any_signal

        # Centralized color rule — applies to EVERY live track (not just
        # the ones that were turning), so straight bikes get green too.
        # Same rule regardless of whether is_turning came from the
        # angle-based detector, a turn-zone overlap, or both:
        #   not turning            -> green
        #   turning, no indicator  -> red
        #   turning, indicator ON  -> black
        for bike in live_tracks:
            try:
                bike.update_box_color()
                # Roll this frame's is_turning/is_signal into the sticky,
                # track-lifetime "ever turned" / "ever signaled while turning"
                # facts used in the final CSV summary.
                bike.mark_turn_frame()
            except Exception:
                log_exception(f"update_box_color/mark_turn_frame track_id={bike.track_id} at frame_idx={frame_idx}")

        # Keep each live bike's single largest-ever crop, and — the moment
        # a NEW bigger crop wins — overwrite that bike's annotated_frame.jpg
        # on disk right away. This runs continuously for as long as the
        # video is being processed, so the saved image only ever gets
        # replaced by a BIGGER one, never a smaller one. Small/far/blurry
        # boxes are ignored (see MIN_BIKE_AREA_FOR_BEST_IMAGE) so a noisy
        # tiny detection can never win "largest so far".
        #
        # Right alongside it: the plate DETECTOR (YOLO, from
        # plate_detector.py) runs on this SAME per-frame bike crop, every
        # frame, for every live bike — just another YOLO model, no
        # different from the bike/mirror/indicator/orientation detectors
        # already running in this loop. Every detected plate box is mapped
        # straight to the bike it belongs to (bike.track_id), and the
        # moment a BIGGER/clearer plate box is found for that track than
        # any seen before this video, its crop overwrites
        # images_dir/<track_id>/plate_crop.jpg right away (see
        # _save_plate_snapshot() and best_plate_area_by_track above).
        #
        # OCR (PaddleOCR — actually READING the numbers off the plate) is
        # DELIBERATELY NOT run here (or anywhere mid-video) — it only runs
        # once per bike, at the very end of run_video(), after every
        # detector/encoder model has been unloaded from CPU/GPU, reading
        # straight off the plate_crop.jpg files this block produces (see
        # the post-loop OCR pass near the end of run_video()).
        for bike in live_tracks:
            try:
                bx1, by1, bx2, by2 = map(int, bike.to_tlbr())
                bx1, by1 = max(0, bx1), max(0, by1)
                bx2, by2 = min(width, bx2), min(height, by2)
                if bx2 > bx1 and by2 > by1:
                    bike_crop = frame[by1:by2, bx1:bx2]

                    got_new_best = bike.update_best_image(
                        bike_crop,
                        min_area=MIN_BIKE_AREA_FOR_BEST_IMAGE,
                        full_frame=frame,
                        box=(bx1, by1, bx2, by2),
                    )
                    if got_new_best:
                        _save_bike_snapshot(bike, video_images_dir)

                    # ---- Plate DETECTOR only (use_ocr=False) — no PaddleOCR
                    # call happens inside this, just the YOLO plate-locator.
                    # offset=(bx1, by1) so every returned plate_box is already
                    # in FULL-FRAME coordinates (not just relative to
                    # bike_crop) — needed below to draw it directly on `frame`.
                    try:
                        plate_dets = detect_plates_in_bike_crop(
                            bike_crop, plate_conf=PLATE_CONF, use_ocr=False,
                            offset=(bx1, by1), debug=False
                        )
                    except Exception:
                        log_exception(f"plate detector track_id={bike.track_id} at frame_idx={frame_idx}")
                        plate_dets = []

                    # Draw EVERY plate box detected THIS frame on the
                    # annotated video — separate from, and unaffected by,
                    # the "keep only the biggest ever seen" save-best logic
                    # below (that logic still decides only what gets written
                    # to plate_crop.jpg on disk, not what gets drawn here).
                    for _pd in plate_dets:
                        try:
                            draw_box(frame, _pd["plate_box"],
                                     f'Plate {bike.track_id}', (0, 255, 255))
                        except Exception:
                            log_exception(f"drawing plate box track_id={bike.track_id} at frame_idx={frame_idx}")

                    if plate_dets:
                        # A bike normally has exactly one plate; if more than
                        # one box came back this frame (rare double-detection),
                        # keep only the largest one.
                        best_this_frame = max(
                            plate_dets,
                            key=lambda d: max(0, d["plate_box"][2] - d["plate_box"][0])
                                          * max(0, d["plate_box"][3] - d["plate_box"][1]),
                        )
                        px1, py1, px2, py2 = best_this_frame["plate_box"]
                        plate_area = max(0, px2 - px1) * max(0, py2 - py1)

                        if (plate_area >= MIN_PLATE_AREA_FOR_SAVE
                                and plate_area > best_plate_area_by_track.get(bike.track_id, 0)):
                            best_plate_area_by_track[bike.track_id] = plate_area
                            _save_plate_snapshot(
                                bike.track_id, best_this_frame["plate_crop"], video_images_dir
                            )
            except Exception:
                log_exception(f"update_best_image track_id={bike.track_id} at frame_idx={frame_idx}")

        print('================= Draw State ==============================')
        # No-op if TURN_ZONES is empty — safe to always call.
        draw_zones(frame, TURN_ZONES)

        for bikes in live_tracks:
            try:
                bbox = bikes.to_tlbr()
                strs = "N"

                if bikes.is_turning and bikes.is_signal:
                    strs = "TO"
                if bikes.is_turning and not bikes.is_signal:
                    strs = "TF"

                x1, y1, x2, y2 = map(int, bbox)
                flow_dx, flow_dy, flow_mag, flow_angle = bike_flow_stats(flow, bbox)
                bikes.setFlow((flow_dx, flow_dy, flow_mag, flow_angle))

                draw_box(frame, (x1, y1, x2, y2), f'bike: {bikes.track_id} - {strs}', bikes.box_color)
                for slot_idx, slot in enumerate(bikes.mirror_slots):
                    if slot["cord"] is None:
                        continue
                    draw_box(frame, slot["cord"][1:], f'Mirror-{slot_idx} {bikes.track_id}', (0, 255, 0))
                for slot_idx, slot in enumerate(bikes.indicator_slots):
                    if slot["cord"] is None:
                        continue
                    label = f'Ind-{slot_idx} {bikes.track_id}' + (' ON' if slot["is_signal"] else '')
                    slot_color = (0, 255, 0) if slot["is_signal"] else (0, 165, 255)
                    draw_box(frame, slot["cord"][1:], label, slot_color)
            except Exception:
                log_exception(f"drawing track_id={bikes.track_id} at frame_idx={frame_idx}")

        try:
            if writer:
                writer.write(frame)
        except Exception:
            log_exception(f"writer.write() at frame_idx={frame_idx}")
        #cv2.imshow("Video Tracking", frame)
        #if cv2.waitKey(1) & 0xFF == ord("q"):
        #    break

        print(f"[FRAME {frame_idx}] --- end ---", flush=True)

    cap.release()
    if writer:
        writer.release()
    #cv2.destroyAllWindows()

    # Video ended while these tracks were still alive — DeepSORT never got
    # the chance to delete them naturally, so finalize them here instead.
    # Without this, any bike still on-screen in the last frame would simply
    # never get a CSV row.
    for t in list(known_tracks.values()):
        try:
            finalize_track(t, records, video_images_dir)
        except Exception:
            log_exception(f"final finalize_track(track_id={t.track_id})")
    known_tracks.clear()

    # Re-encode the mp4v temp file to H.264 so it plays in browsers
    # (mp4v only plays in desktop players like VLC/WMP, not <video> tags).
    # -movflags +faststart moves metadata to the front of the file so it
    # can start streaming/playing before it's fully downloaded.
    if raw_output_path and os.path.exists(raw_output_path):
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", raw_output_path,
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart",
                    output_path,
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            os.remove(raw_output_path)
            print(f"Saved browser-compatible video: {output_path}")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            # ffmpeg missing or conversion failed -- keep the raw mp4v file
            # around (rename it back to the expected output_path) instead of
            # silently losing the processed video.
            print(f"WARNING: ffmpeg re-encode failed ({e}); "
                  f"keeping original mp4v file (plays in VLC/media players, not browsers).")
            os.replace(raw_output_path, output_path)

    # ------------------------------------------------------------------
    # Free every detector/encoder model from CPU/GPU memory BEFORE loading
    # PaddleOCR. This is the real fix for the old hangs: PaddlePaddle and
    # PyTorch both trying to hold a CUDA context (or fight over CPU
    # threads) in the same process at the same time is what caused them.
    # Trade-off: each run_video() call now reloads YOLO/OSNet from scratch
    # (a few seconds), but nothing is left resident on CPU/GPU by the time
    # OCR starts, whether this run used "cuda" or "cpu".
    # ------------------------------------------------------------------
    try:
        del yolo, mirror_detector, indicator_detector, orientation_detector, encoder
    except Exception:
        pass
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            log_exception("torch.cuda.empty_cache()/ipc_collect() after video loop")
    gc.collect()
    print(f"[{vidName}] All detector models unloaded, CPU/GPU freed "
          f"(cuda_available={torch.cuda.is_available()}).", flush=True)

    # ------------------------------------------------------------------
    # NOW (and only now) load PaddleOCR and run it exactly once per bike —
    # reading the numbers straight off images_dir/<track_id>/plate_crop.jpg,
    # which the plate DETECTOR already produced live, continuously, during
    # the frame loop above (see the plate DETECTOR block there, and
    # _save_plate_snapshot()). NO detection happens here — this pass only
    # ever loads an already-cropped plate image off disk and runs OCR on
    # it via read_plate_text_from_image(). This is the ONLY place OCR runs
    # for this video.
    # ------------------------------------------------------------------
    if records:
        try:
            warm_up_ocr()
        except Exception:
            log_exception("warm_up_ocr() in post-loop OCR pass")

        for rec in records:
            plate_text = ""
            bike_dir = os.path.join(video_images_dir, str(rec["track_id"]))
            plate_crop_path = os.path.join(bike_dir, "plate_crop.jpg")

            if os.path.exists(plate_crop_path):
                try:
                    plate_text = read_plate_text_from_image(plate_crop_path, debug=False)
                except Exception:
                    log_exception(f"post-loop OCR on track_id={rec['track_id']} ({plate_crop_path})")
            else:
                print(f"Track {rec['track_id']}: no plate_crop.jpg was ever produced during the "
                      f"frame loop — treating plate as unreadable.", flush=True)

            rec["plate_number"] = plate_text
            print(f"Track {rec['track_id']}: post-loop OCR -> plate={plate_text!r}", flush=True)

    # ------------------------------------------------------------------
    # Write the summary CSV exactly once, now that every bike's
    # plate_number is known (or confirmed unreadable). csv.writer is used
    # so a plate string (or anything else) containing a comma can never
    # corrupt the file the way the old manual ",".join(...) writing could.
    # ------------------------------------------------------------------
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow([
            "bike_id", "plate_number", "has_both_mirrors",
            "has_both_indicators", "took_turn", "indicator_on_while_turning",
        ])
        for rec in records:
            csv_writer.writerow([
                rec["track_id"],
                rec.get("plate_number", ""),
                rec["mirror_seen_both"],
                rec["indicator_seen_both"],
                rec["ever_turned"],
                rec["ever_signaled_while_turning"],
            ])
    print(f"[{vidName}] Summary CSV written -> {summary_csv_path} ({len(records)} bike(s)).", flush=True)

    return {
        "output_video_path": output_path,
        "summary_csv_path": summary_csv_path,
        "video_images_dir": video_images_dir,
        "video_dir": video_dir,
        "video_base": video_base,
    }


if __name__ == "__main__":

    for vid in os.listdir(VIDEO_PATH):
        print(f"===== Starting video: {vid} =====", flush=True)
        try:
            result = run_video(os.path.join(VIDEO_PATH, vid), OUTPUT_PATH, display_name=vid)
            print(f"Result: {result}", flush=True)
        except Exception:
            log_exception(f"run_video() top-level for video={vid}")
        print(f"===== Finished (or failed) video: {vid} =====", flush=True)

