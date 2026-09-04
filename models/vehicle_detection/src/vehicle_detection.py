from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


SRC_DIR = Path(__file__).resolve().parent
MODULE_DIR = SRC_DIR.parent
MODEL_PATH = MODULE_DIR / "weights" / "yolov8n.pt"
OUTPUT_DIR = MODULE_DIR / "output"
VIDEO_DIR = MODULE_DIR / "video"

# ──────────────────────────────────────────────────────────────
# DEFAULT VIDEO  ← change this to quickly switch test videos
# Options in your 'video' folder: Cars1.mp4, Cars2.mp4
# ──────────────────────────────────────────────────────────────
DEFAULT_VIDEO = VIDEO_DIR / "Cars2.mp4"
DEFAULT_MODE  = "live"   # "live" = with preview window, "video" = save only

# COCO class IDs for vehicles:
# 1=bicycle, 2=car, 3=motorcycle, 5=bus, 7=truck
VEHICLE_CLASS_IDS = [1, 2, 3, 5, 7]

# Human-readable names for vehicle classes (COCO labels)
VEHICLE_NAMES = {
    1: "Bicycle",
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck",
}

# Color palette for each vehicle type (BGR format)
VEHICLE_COLORS = {
    1: (255, 200, 50),   # Bicycle  – light blue
    2: (50, 255, 50),    # Car      – green
    3: (0, 165, 255),    # Motorcycle – orange
    5: (0, 0, 255),      # Bus      – red
    7: (255, 0, 255),    # Truck    – magenta
}


def load_model() -> YOLO:
    """Load yolov8n.pt, downloading it automatically if it is missing."""
    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))
    print(f"Model loaded successfully from: {MODEL_PATH}")
    return model


def test_model() -> None:
    load_model()
    print("YOLO model download/load test passed.")


def draw_fancy_box(frame, x1, y1, x2, y2, color, label, confidence):
    """Draw a styled bounding box with rounded corners, label, and confidence."""
    overlay = frame.copy()
    
    # Draw filled rectangle with transparency
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
    
    # Draw border
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    # Corner accents (thicker, shorter lines at corners)
    corner_len = min(25, (x2 - x1) // 4, (y2 - y1) // 4)
    thickness = 3
    # Top-left
    cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, thickness)
    cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, thickness)
    # Top-right
    cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, thickness)
    cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, thickness)
    # Bottom-left
    cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, thickness)
    cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, thickness)
    # Bottom-right
    cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, thickness)
    cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, thickness)
    
    # Label background
    text = f"{label} {confidence:.0%}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, 1)
    label_y1 = max(y1 - th - 10, 0)
    label_y2 = y1
    cv2.rectangle(frame, (x1, label_y1), (x1 + tw + 10, label_y2), color, -1)
    cv2.putText(frame, text, (x1 + 5, label_y2 - 4), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)


def draw_dashboard(frame, vehicle_counts, total_vehicles, fps, frame_num, total_frames):
    """Draw a sleek HUD / dashboard overlay on the frame."""
    h, w = frame.shape[:2]
    
    # ── Top bar ──────────────────────────────────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 50), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # FPS on the right
    fps_text = f"FPS: {fps:.1f}"
    (fw, _), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.putText(frame, fps_text, (w - fw - 15, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
    
    # ── Bottom panel (vehicle counts) ────────────────────────
    panel_h = 90
    panel_y = h - panel_h
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, panel_y), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay2, 0.7, frame, 0.3, 0, frame)
    
    # Total vehicles count - large text
    total_text = f"TOTAL VEHICLES: {total_vehicles}"
    cv2.putText(frame, total_text, (15, panel_y + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Individual counts as small colored badges
    x_offset = 15
    badge_y = panel_y + 60
    for cls_id, name in VEHICLE_NAMES.items():
        count = vehicle_counts.get(cls_id, 0)
        color = VEHICLE_COLORS.get(cls_id, (200, 200, 200))
        badge_text = f"{name}: {count}"
        (bw, bh), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        # Badge background
        cv2.rectangle(frame, (x_offset - 4, badge_y - bh - 6),
                      (x_offset + bw + 8, badge_y + 6), color, -1)
        cv2.putText(frame, badge_text, (x_offset, badge_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        x_offset += bw + 20
    
    # ── Progress bar ─────────────────────────────────────────
    if total_frames > 0:
        progress = frame_num / total_frames
        bar_y = panel_y - 6
        cv2.rectangle(frame, (0, bar_y), (w, bar_y + 4), (40, 40, 40), -1)
        cv2.rectangle(frame, (0, bar_y), (int(w * progress), bar_y + 4), (0, 220, 255), -1)
    
    # ── Controls hint ────────────────────────────────────────
    hint = "Q: Quit | P: Pause | S: Screenshot"
    (hw, _), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.putText(frame, hint, (w - hw - 15, panel_y + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA)


def detect_image(model: YOLO, image_path: Path) -> None:
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    results = model.predict(
        source=str(image_path),
        classes=VEHICLE_CLASS_IDS,
        conf=0.25,
        save=False,
        verbose=False,
    )
    annotated_image = results[0].plot()
    output_path = OUTPUT_DIR / f"{image_path.stem}_detected{image_path.suffix}"
    cv2.imwrite(str(output_path), annotated_image)
    print(f"Annotated image saved to: {output_path}")


def detect_video_live(model: YOLO, video_path: Path) -> None:
    """Run vehicle detection on a video with LIVE preview window and save output."""
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / f"{video_path.stem}_detected.mp4"
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    # Resize for display if the video is very large
    display_width = min(width, 1280)
    display_height = int(height * (display_width / width))

    window_name = "YOLOv8 Vehicle Detection - LIVE"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, display_width, display_height)

    paused = False
    frame_num = 0
    prev_time = time.time()
    current_fps = 0.0

    print(f"\n{'='*60}")
    print(f"  LIVE Vehicle Detection")
    print(f"  Video : {video_path.name}")
    print(f"  Size  : {width}x{height} @ {fps:.1f} FPS")
    print(f"  Frames: {total_frames}")
    print(f"  Output: {output_path}")
    print(f"{'='*60}")
    print(f"  Controls: Q=Quit  P=Pause  S=Screenshot")
    print(f"{'='*60}\n")

    try:
        while True:
            if not paused:
                success, frame = capture.read()
                if not success:
                    print("\n[INFO] Video finished.")
                    break
                
                frame_num += 1

                # Run YOLO detection
                result = model.predict(
                    source=frame,
                    classes=VEHICLE_CLASS_IDS,
                    conf=0.25,
                    save=False,
                    verbose=False,
                )[0]

                # Count vehicles by type
                vehicle_counts = {}
                total_vehicles = 0

                if result.boxes is not None and len(result.boxes) > 0:
                    for box in result.boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        label = VEHICLE_NAMES.get(cls_id, f"Class {cls_id}")
                        color = VEHICLE_COLORS.get(cls_id, (200, 200, 200))
                        
                        draw_fancy_box(frame, x1, y1, x2, y2, color, label, conf)
                        
                        vehicle_counts[cls_id] = vehicle_counts.get(cls_id, 0) + 1
                        total_vehicles += 1

                # Calculate FPS
                curr_time = time.time()
                elapsed = curr_time - prev_time
                if elapsed > 0:
                    current_fps = 1.0 / elapsed
                prev_time = curr_time

                # Draw dashboard overlay
                draw_dashboard(frame, vehicle_counts, total_vehicles,
                               current_fps, frame_num, total_frames)

                # Write annotated frame to output
                writer.write(frame)

                # Display frame
                cv2.imshow(window_name, frame)

            # Key handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # q or ESC
                print("\n[INFO] Stopped by user.")
                break
            elif key == ord("p"):
                paused = not paused
                state = "PAUSED" if paused else "RESUMED"
                print(f"[INFO] {state}")
            elif key == ord("s"):
                screenshot_path = OUTPUT_DIR / f"screenshot_{frame_num}.png"
                cv2.imwrite(str(screenshot_path), frame)
                print(f"[INFO] Screenshot saved: {screenshot_path}")

    finally:
        capture.release()
        writer.release()
        cv2.destroyAllWindows()

    print(f"\n[DONE] Annotated video saved to: {output_path}")
    print(f"[DONE] Processed {frame_num}/{total_frames} frames.\n")


def detect_video(model: YOLO, video_path: Path) -> None:
    """Run vehicle detection on a video (no live preview, just saves output)."""
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / f"{video_path.stem}_detected.mp4"
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            result = model.predict(
                source=frame,
                classes=VEHICLE_CLASS_IDS,
                conf=0.25,
                save=False,
                verbose=False,
            )[0]
            writer.write(result.plot())
    finally:
        capture.release()
        writer.release()

    print(f"Annotated video saved to: {output_path}")


def detect_webcam(model: YOLO) -> None:
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        raise RuntimeError("Could not open the webcam.")

    print("Webcam detection started. Press q in the video window to stop.")
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            result = model.predict(
                source=frame,
                classes=VEHICLE_CLASS_IDS,
                conf=0.25,
                save=False,
                verbose=False,
            )[0]
            cv2.imshow("YOLOv8 Vehicle Detection", result.plot())
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect vehicles with YOLOv8 Nano.")
    parser.add_argument(
        "mode",
        nargs="?",
        default=DEFAULT_MODE,
        choices=["test", "image", "video", "live", "webcam"],
        help="test | image | video (save only) | live (video + live preview) | webcam",
    )
    parser.add_argument("source", nargs="?", help="path to an image or video file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "test":
        test_model()
        return

    if args.mode == "webcam":
        detect_webcam(load_model())
        return

    if args.mode in ("video", "live", "image") and not args.source:
        print(f"[INFO] No source given – using default: {DEFAULT_VIDEO}")
        source_path = DEFAULT_VIDEO
    else:
        source_path = Path(args.source).resolve()

    model = load_model()

    if args.mode == "image":
        detect_image(model, source_path)
    elif args.mode == "live":
        detect_video_live(model, source_path)
    else:
        detect_video(model, source_path)


if __name__ == "__main__":
    main()
