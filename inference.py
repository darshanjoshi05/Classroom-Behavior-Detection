from ultralytics import YOLO
import cv2
import numpy as np
from collections import Counter

# Per-class confidence thresholds
CLASS_CONF = {
    "hand_raising": 0.30,
    "using_phone":  0.15,   # lower to catch subtle phone use
    "sleeping":     0.25,
    "writing":      0.30,
    "reading":      0.30,
}

# Bounding box colors per class (BGR)
COLORS = {
    "hand_raising": (0, 255, 0),
    "using_phone":  (0, 0, 255),
    "sleeping":     (255, 0, 0),
    "writing":      (0, 255, 255),
    "reading":      (255, 165, 0),
}


def run_tiled_inference(frame, model, tile_size=640, overlap=150):
    """
    Run tiled inference on a single frame.
    Splits the frame into overlapping tiles, runs inference on each,
    maps results back to full-frame coordinates, and applies NMS.
    """
    h, w = frame.shape[:2]
    all_boxes = []

    for y in range(0, h, tile_size - overlap):
        for x in range(0, w, tile_size - overlap):
            y2 = min(y + tile_size, h)
            x2 = min(x + tile_size, w)
            tile = frame[y:y2, x:x2]

            result = model(tile, conf=0.10, verbose=False)[0]

            for box in result.boxes:
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                conf = float(box.conf[0])

                # Apply per-class threshold
                if conf < CLASS_CONF.get(cls_name, 0.25):
                    continue

                bx1, by1, bx2, by2 = box.xyxy[0].tolist()
                fx1 = int(bx1 + x)
                fy1 = int(by1 + y)
                fx2 = int(bx2 + x)
                fy2 = int(by2 + y)

                # Skip teacher area (bottom center of frame)
                box_cy = (fy1 + fy2) / 2
                box_cx = (fx1 + fx2) / 2
                if box_cy > h * 0.65 and 0.2 < box_cx / w < 0.8:
                    continue

                all_boxes.append([fx1, fy1, fx2, fy2, conf, cls_id, cls_name])

    if not all_boxes:
        return frame.copy(), []

    # Apply NMS across all tiles
    boxes_np = np.array([[b[0], b[1], b[2], b[3]] for b in all_boxes],
                        dtype=np.float32)
    scores_np = np.array([b[4] for b in all_boxes], dtype=np.float32)
    indices = cv2.dnn.NMSBoxes(
        boxes_np.tolist(), scores_np.tolist(), 0.10, 0.45)

    final_boxes = []
    if len(indices) > 0:
        for i in indices.flatten():
            final_boxes.append(all_boxes[i])

    # Draw boxes on frame
    annotated = frame.copy()
    for b in final_boxes:
        fx1, fy1, fx2, fy2, conf, cls_id, cls_name = b
        color = COLORS.get(cls_name, (128, 128, 128))
        cv2.rectangle(annotated, (fx1, fy1), (fx2, fy2), color, 2)
        cv2.putText(annotated, f"{cls_name} {conf:.2f}",
                    (fx1, fy1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 2)

    return annotated, final_boxes


def run_on_video(video_path, model_path, output_path, skip=5):
    """
    Run tiled inference on a full video.

    Args:
        video_path:  Path to input video
        model_path:  Path to best.pt weights
        output_path: Path to save annotated output video
        skip:        Process every Nth frame (5 = every 5th frame)
    """
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {total} frames | {fps:.0f} FPS | {w}x{h}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    frame_idx = 0
    processed = 0
    class_counts = Counter()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % skip == 0:
            annotated, boxes = run_tiled_inference(frame, model)
            for b in boxes:
                class_counts[b[6]] += 1
            processed += 1
            if processed % 50 == 0:
                print(f"  Processed {processed} frames "
                      f"({frame_idx}/{total})...")
        else:
            annotated = frame

        out.write(annotated)
        frame_idx += 1

    cap.release()
    out.release()

    print(f"\nDone! Processed {processed}/{total} frames")
    print(f"Output saved to: {output_path}")
    print("\nDetections per class:")
    total_det = sum(class_counts.values())
    for cls, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        pct = count / total_det * 100 if total_det > 0 else 0
        print(f"  {cls:<20} {count:>5}  ({pct:.1f}%)")

    return class_counts


def run_on_image(image_path, model_path, output_path=None, tiled=True):
    """
    Run inference on a single image.

    Args:
        image_path:  Path to input image
        model_path:  Path to best.pt weights
        output_path: Path to save annotated image (optional)
        tiled:       Use tiled inference (recommended for wide-angle)
    """
    model = YOLO(model_path)
    frame = cv2.imread(image_path)

    if tiled:
        annotated, boxes = run_tiled_inference(frame, model)
    else:
        result = model(frame, conf=0.35, verbose=False)[0]
        annotated = result.plot()
        boxes = []

    if output_path:
        cv2.imwrite(output_path, annotated)
        print(f"Saved to: {output_path}")

    return annotated, boxes


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  Video: python inference.py video.mp4 best.pt output.mp4")
        print("  Image: python inference.py image.jpg best.pt output.jpg")
        sys.exit(1)

    input_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else "best.pt"
    output_path = sys.argv[3] if len(sys.argv) > 3 else "output.mp4"

    if input_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        run_on_video(input_path, model_path, output_path)
    else:
        run_on_image(input_path, model_path, output_path)
