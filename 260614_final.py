from pathlib import Path
import time

import cv2
import numpy as np
from ultralytics import YOLO

# Workflow
# Video frame -> YOLO detection -> ByteTrack -> ROI filtering -> TTC proxy calculation -> collision warning overlay
# ============================================================
# SETTINGS
# ============================================================
MODEL_PATH = "yolo26s.pt"   # e.g. yolo26s.pt, yolo26m.pt
SOURCE = "./dataset/IMG_5969_4.MOV"   # 0 = default webcam, 1 = second webcam, or set a video path string
SAVE_OUTPUT = False
OUTPUT_PATH = "demo3_final.mp4"
CONFIDENCE = 0.35  # Day 0.45 / Night 0.35
SHOW_FPS = True
WINDOW_NAME = "Real-time YOLO + ByteTrack + ROI TTC Warning"

# Mac Apple Silicon: "mps" / NVIDIA CUDA: 0 / CPU: "cpu"
DEVICE = "mps"

# ByteTrack is handled by Ultralytics. Keep tracking logic in the tracker config,
# not in extra custom re-link code, to avoid duplicate ID recovery logic.
TRACKER_CONFIG = "bytetrack_tuned.yaml"
# Detection classes follow the COCO class index convention used by YOLO.
CLASSES = [0, 1, 2, 5, 7, 9]      # COCO: person, bicycle, car, bus, truck, traffic light
TTC_CLASSES = [0, 1, 2, 5, 7]     # Exclude traffic light(9) from TTC calculation

# track_state stores only the minimal history required for ROI and TTC metrics.
# Lost-track and re-link logic is intentionally removed to avoid duplicating ByteTrack's track_buffer behavior.
TRACK_STATE_MAX_MISSED_FRAMES = 90

# ============================================================
# ROI + TTC SETTINGS
# ============================================================
DRAW_ROI = True
ROI_ALPHA = 0.18

# Ratio-based ROI coordinates. Adjust only these values for each video.
# Point format: (x_ratio, y_ratio), where 0.0 to 1.0 is scaled to the frame size.
# This keeps the ROI reusable across videos with different resolutions.

# Daytime demo ROI preset
# MAIN_ROI_RATIOS = [
#     (0.15, 0.90),
#     (0.44, 0.52),
#     (0.51, 0.52),
#     (0.78, 0.90),
# ]
#
# LEFT_ROI_RATIOS = [
#     (0.00, 0.90),
#     (0.15, 0.90),
#     (0.44, 0.52),
#     (0.28, 0.56),
#     (0.00, 0.68),
# ]
#
# RIGHT_ROI_RATIOS = [
#     (0.78, 0.90),
#     (1.00, 0.90),
#     (1.00, 0.70),
#     (0.72, 0.58),
#     (0.51, 0.52),
# ]

# Nighttime demo ROI preset
MAIN_ROI_RATIOS = [
    (0.19, 0.90),
    (0.45, 0.54),
    (0.49, 0.54),
    (0.74, 0.90),
]

LEFT_ROI_RATIOS = [
    (0.00, 0.90),
    (0.19, 0.90),
    (0.45, 0.54),
    (0.28, 0.60),
    (0.00, 0.73),
]

RIGHT_ROI_RATIOS = [
    (0.74, 0.90),
    (1.00, 0.90),
    (1.00, 0.75),
    (0.72, 0.62),
    (0.49, 0.54),
]

# TTC proxy thresholds based on bounding-box height growth.
# This is not real meter-based TTC; it is a monocular-camera risk heuristic.
# Larger and faster-growing boxes are treated as objects approaching the ego vehicle.
TTC_DANGER_SEC = 1.5
TTC_WARNING_SEC = 3.0
MIN_BBOX_GROWTH_PX_PER_SEC = 4.0
MIN_TTC_BBOX_HEIGHT_PX = 35
TTC_SMOOTHING_ALPHA = 0.35
WARNING_HOLD_SEC = 0.6

# Filters for reducing false warnings
MIN_TTC_BBOX_HEIGHT_RATIO = 0.06
MIN_TTC_FOOT_Y_RATIO = 0.55
MIN_APPROACH_FRAMES = 4

# ============================================================


def open_source(source):
    """Open a camera index or video path."""
    if isinstance(source, str):
        if source.isdigit():
            source = int(source)
        else:
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"Video source not found: {source}")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open source: {source}")
    return cap


def make_writer(output_path, fps, width, height):
    """Create a video writer for saving processed output."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(output_path, fourcc, fps, (width, height))


def make_polygon(width, height, ratio_points):
    """Convert ratio-based polygon points into pixel coordinates."""
    pts = []
    for rx, ry in ratio_points:
        x = int(np.clip(rx, 0.0, 1.0) * width)
        y = int(np.clip(ry, 0.0, 1.0) * height)
        pts.append([x, y])
    return np.array(pts, dtype=np.int32)


def build_rois(width, height):
    """Build lane-like ROI polygons for the current frame size."""
    return {
        "LEFT": make_polygon(width, height, LEFT_ROI_RATIOS),
        "MAIN": make_polygon(width, height, MAIN_ROI_RATIOS),
        "RIGHT": make_polygon(width, height, RIGHT_ROI_RATIOS),
    }


def classify_roi(foot_x, foot_y, rois):
    """Return which ROI contains the bbox bottom-center point."""
    # Check MAIN first so center-lane objects keep priority when ROIs overlap with LEFT/RIGHT.
    for roi_name in ["MAIN", "LEFT", "RIGHT"]:
        polygon = rois[roi_name]
        if cv2.pointPolygonTest(polygon, (float(foot_x), float(foot_y)), False) >= 0:
            return roi_name
    return "OUT"


def draw_roi_overlay(frame, rois):
    """Draw semi-transparent ROI polygons."""
    if not DRAW_ROI:
        return frame

    overlay = frame.copy()
    line_color = (255, 0, 0)  # BGR: blue

    for polygon in rois.values():
        cv2.fillPoly(overlay, [polygon], line_color)

    frame = cv2.addWeighted(overlay, ROI_ALPHA, frame, 1.0 - ROI_ALPHA, 0)

    for roi_name, polygon in rois.items():
        cv2.polylines(frame, [polygon], True, line_color, 2)
        x, y = tuple(polygon[0])
        cv2.putText(
            frame,
            roi_name,
            (x + 8, max(20, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            line_color,
            2,
            cv2.LINE_AA,
        )
    return frame


def init_track_state(track_state, track_id):
    """Create minimal per-track state used only for ROI/TTC metrics."""
    track_state[track_id] = {
        "last_bbox_h": None,
        "smooth_bbox_h": None,
        "smooth_dh_dt": 0.0,
        "last_ttc_ts": None,
        "ttc": None,
        "risk_level": "SAFE",
        "last_risk_ts": 0.0,
        "roi_name": "OUT",
        "last_seen_frame": 0,
        "approach_count": 0,
    }


def update_ttc_metrics(track_state, track_id, bbox_h, foot_y, frame_height, now_ts, roi_name, cls_id):
    """
    Calculate TTC proxy using bounding-box height expansion.

    TTC proxy = smoothed bbox height / smoothed bbox-height growth rate.
    This is not calibrated meter-based TTC. It is a monocular-camera collision-risk heuristic.
    """
    state = track_state.get(track_id)
    if state is None:
        return None, 0.0, "SAFE"

    state["roi_name"] = roi_name

    min_ttc_bbox_h = max(MIN_TTC_BBOX_HEIGHT_PX, int(frame_height * MIN_TTC_BBOX_HEIGHT_RATIO))
    min_ttc_foot_y = int(frame_height * MIN_TTC_FOOT_Y_RATIO)

    # TTC is evaluated only for relevant classes inside the MAIN ROI.
    # Small or high-positioned objects are ignored because they are less likely to be immediate collision risks.
    if (
            cls_id not in TTC_CLASSES
            or roi_name != "MAIN"
            or bbox_h < min_ttc_bbox_h
            or foot_y < min_ttc_foot_y
    ):
        state["last_bbox_h"] = bbox_h
        state["smooth_bbox_h"] = state.get("smooth_bbox_h", float(bbox_h))
        state["last_ttc_ts"] = now_ts
        state["ttc"] = None
        state["risk_level"] = "OUT_ROI" if roi_name == "OUT" else "SAFE"
        state["approach_count"] = 0
        return None, 0.0, state["risk_level"]

    prev_smooth_h = state.get("smooth_bbox_h")
    prev_ttc_ts = state.get("last_ttc_ts")

    if prev_smooth_h is None or prev_ttc_ts is None:
        state["last_bbox_h"] = bbox_h
        state["smooth_bbox_h"] = float(bbox_h)
        state["smooth_dh_dt"] = 0.0
        state["last_ttc_ts"] = now_ts
        state["ttc"] = None
        state["risk_level"] = "SAFE"
        return None, 0.0, "SAFE"

    dt = max(now_ts - prev_ttc_ts, 1e-6)
    alpha = TTC_SMOOTHING_ALPHA

    # Exponential smoothing reduces jitter from frame-to-frame bounding-box size changes.
    smooth_h = alpha * float(bbox_h) + (1.0 - alpha) * float(prev_smooth_h)
    raw_dh_dt = (smooth_h - float(prev_smooth_h)) / dt
    smooth_dh_dt = alpha * raw_dh_dt + (1.0 - alpha) * state.get("smooth_dh_dt", 0.0)

    # Count consecutive approaching frames before allowing WARNING/DANGER.
    # This prevents one-frame detection noise from triggering a warning.
    if smooth_dh_dt > MIN_BBOX_GROWTH_PX_PER_SEC:
        state["approach_count"] = state.get("approach_count", 0) + 1
    else:
        state["approach_count"] = 0

    if state["approach_count"] < MIN_APPROACH_FRAMES:
        ttc = None
        risk = "SAFE"
    else:
        ttc = smooth_h / smooth_dh_dt
        if ttc < TTC_DANGER_SEC:
            risk = "DANGER"
        elif ttc < TTC_WARNING_SEC:
            risk = "WARNING"
        else:
            risk = "SAFE"

    previous_risk = state.get("risk_level", "SAFE")
    last_risk_ts = state.get("last_risk_ts", 0.0)

    # Hold recent WARNING/DANGER briefly so the overlay does not flicker between frames.
    if risk in ["DANGER", "WARNING"]:
        state["last_risk_ts"] = now_ts
    elif previous_risk in ["DANGER", "WARNING"] and now_ts - last_risk_ts < WARNING_HOLD_SEC:
        risk = previous_risk

    state["last_bbox_h"] = bbox_h
    state["smooth_bbox_h"] = smooth_h
    state["smooth_dh_dt"] = smooth_dh_dt
    state["last_ttc_ts"] = now_ts
    state["ttc"] = ttc
    state["risk_level"] = risk

    return ttc, smooth_dh_dt, risk


def get_risk_color(risk):
    """Return BGR color for each risk level."""
    if risk == "DANGER":
        return (0, 0, 255)
    if risk == "WARNING":
        return (0, 165, 255)
    if risk == "OUT_ROI":
        return (160, 160, 160)
    return (0, 255, 0)


def draw_text_with_bg(frame, text, org, font_scale, text_color, bg_color, thickness=2, padding=6):
    """Draw readable text with a filled background rectangle."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = org

    cv2.rectangle(
        frame,
        (x - padding, y - th - padding),
        (x + tw + padding, y + baseline + padding),
        bg_color,
        -1,
    )
    cv2.putText(frame, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)


def prune_track_state(track_state, frame_idx):
    """Remove old metric states only. ByteTrack itself handles object memory/ID logic."""
    stale_ids = []
    for track_id, state in track_state.items():
        missed_frames = frame_idx - state.get("last_seen_frame", frame_idx)
        if missed_frames > TRACK_STATE_MAX_MISSED_FRAMES:
            stale_ids.append(track_id)

    for track_id in stale_ids:
        del track_state[track_id]


def draw_tracks(frame, result, track_state, names, frame_idx, frame_time_sec, rois):
    """Draw tracked boxes, ROI, TTC proxy, and collision warning."""
    height, width = frame.shape[:2]
    frame = draw_roi_overlay(frame, rois)

    boxes = result.boxes
    if boxes is None or boxes.xyxy is None or len(boxes.xyxy) == 0:
        prune_track_state(track_state, frame_idx)
        return frame

    ids = boxes.id
    if ids is None:
        prune_track_state(track_state, frame_idx)
        return frame

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy() if boxes.conf is not None else []
    clss = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else []
    track_ids = ids.int().cpu().tolist()

    now_ts = frame_time_sec
    most_dangerous = None

    for i, box in enumerate(xyxy):
        x1, y1, x2, y2 = map(int, box)
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))

        track_id = track_ids[i]
        conf = float(confs[i]) if len(confs) > i else 0.0
        cls_id = int(clss[i]) if len(clss) > i else -1
        cls_name = names.get(cls_id, str(cls_id))

        cx = (x1 + x2) / 2.0
        foot_x = int(cx)
        foot_y = int(y2)
        bbox_h = max(1, y2 - y1)
        roi_name = classify_roi(foot_x, foot_y, rois)

        if track_id not in track_state:
            init_track_state(track_state, track_id)

        ttc, dh_dt, risk = update_ttc_metrics(
            track_state,
            track_id,
            bbox_h,
            foot_y,
            height,
            now_ts,
            roi_name,
            cls_id,
        )

        track_state[track_id]["cls_id"] = cls_id
        track_state[track_id]["last_bbox"] = (x1, y1, x2, y2)
        track_state[track_id]["last_seen_frame"] = frame_idx

        color = get_risk_color(risk)
        thickness = 3 if risk in ["DANGER", "WARNING"] else 2

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        cv2.circle(frame, (foot_x, foot_y), 5, color, -1)

        ttc_text = "TTC:--" if ttc is None else f"TTC:{ttc:.1f}s"
        label = f"ID:{track_id} {cls_name} {conf:.2f} {roi_name} {ttc_text} {risk}"

        draw_text_with_bg(
            frame,
            label,
            (x1, max(25, y1 - 10)),
            0.55,
            (255, 255, 255),
            color,
            thickness=2,
            padding=4,
        )

        # Track the most urgent object so a single global warning banner can be displayed.
        if risk in ["DANGER", "WARNING"] and roi_name == "MAIN":
            risk_rank = 2 if risk == "DANGER" else 1
            ttc_for_sort = ttc if ttc is not None else 999.0
            score = (risk_rank, -ttc_for_sort, bbox_h)

            if most_dangerous is None or score > most_dangerous["score"]:
                most_dangerous = {
                    "score": score,
                    "risk": risk,
                    "ttc": ttc,
                }

    prune_track_state(track_state, frame_idx)

    if most_dangerous is not None:
        risk = most_dangerous["risk"]
        color = get_risk_color(risk)

        if most_dangerous["ttc"] is None:
            warning_text = f"WARNING: {risk} - VEHICLE CLOSE"
        else:
            warning_text = f"WARNING: {risk} - TTC {most_dangerous['ttc']:.1f}s"

        (tw, _), _ = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
        x = max(10, (width - tw) // 2)
        y = 55

        draw_text_with_bg(
            frame,
            warning_text,
            (x, y),
            1.0,
            (255, 255, 255),
            color,
            thickness=3,
            padding=8,
        )

    return frame


def main():
    print("Loading model...")
    model = YOLO(MODEL_PATH)

    print(f"Opening source: {SOURCE}")
    cap = open_source(SOURCE)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    fps_for_writer = source_fps if source_fps and source_fps > 1 else 30.0

    rois = build_rois(width, height)

    writer = None
    if SAVE_OUTPUT:
        writer = make_writer(OUTPUT_PATH, fps_for_writer, width, height)
        print(f"Saving output to: {OUTPUT_PATH}")

    track_state = {}
    frame_idx = 0

    # Stores per-frame processing times for average processing FPS calculation.
    processing_times = []

    print("Streaming started. Press 'q' to quit.")

    while True:
        frame_idx += 1

        ret, frame = cap.read()
        if not ret:
            print("No more frames or failed to read frame.")
            break

        # Start measuring core processing time.
        # Included range: YOLO + ByteTrack + ROI/TTC calculation + visualization drawing.
        proc_start = time.time()

        results = model.track(
            frame,
            conf=CONFIDENCE,
            device=DEVICE,
            classes=CLASSES,
            tracker=TRACKER_CONFIG,
            persist=True,
            verbose=False,
        )

        result = results[0]
        annotated_frame = frame.copy()

        # For video files, frame_idx / fps is more stable for the TTC proxy than wall-clock time.
        # For webcam input, this falls back to 30 FPS when the source FPS is unavailable.
        frame_time_sec = frame_idx / max(float(fps_for_writer), 1.0)

        annotated_frame = draw_tracks(
            annotated_frame,
            result,
            track_state,
            result.names,
            frame_idx,
            frame_time_sec,
            rois,
        )

        proc_end = time.time()
        proc_time = proc_end - proc_start
        processing_times.append(proc_time)

        # Current FPS is based on this frame only; average FPS is based on all measured frames.
        current_proc_fps = 1.0 / max(proc_time, 1e-6)
        avg_proc_time = sum(processing_times) / len(processing_times)
        avg_proc_fps = 1.0 / max(avg_proc_time, 1e-6)

        if SHOW_FPS:
            cv2.putText(
                annotated_frame,
                f"Proc FPS: {current_proc_fps:.1f}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                annotated_frame,
                f"Avg Proc FPS: {avg_proc_fps:.1f}",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow(WINDOW_NAME, annotated_frame)

        if writer is not None:
            writer.write(annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("Exit requested by user.")
            break

    cap.release()

    if writer is not None:
        writer.release()

    cv2.destroyAllWindows()

    # Final average FPS report
    if len(processing_times) > 0:
        total_processing_time = sum(processing_times)
        avg_processing_time = total_processing_time / len(processing_times)
        avg_processing_fps = 1.0 / max(avg_processing_time, 1e-6)

        print("============================================================")
        print("Processing Performance Summary")
        print("============================================================")
        print(f"Processed frames: {len(processing_times)}")
        print(f"Total measured processing time: {total_processing_time:.3f} sec")
        print(f"Average processing time per frame: {avg_processing_time:.4f} sec")
        print(f"Average processing FPS: {avg_processing_fps:.2f}")
        print("============================================================")

    print("Done.")


if __name__ == "__main__":
    main()