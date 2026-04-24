import os
import argparse
import time
import cv2

from perception.detector import Detector, draw_detections


def filter_by_labels(detections, labels_set):
    return [d for d in detections if d['label'] in labels_set]


def main():
    parser = argparse.ArgumentParser(description='Multi-model detection: separate outputs per model')
    parser.add_argument('--input', '-i', required=True, help='Input video file path')
    parser.add_argument('--output-dir', '-o', default='perception/output', help='Directory to save output clips')
    parser.add_argument('--conf', type=float, default=0.4, help='Confidence threshold')
    parser.add_argument('--device', default='cpu', help='Device for models: cpu or cuda')
    parser.add_argument('--person-backend', default='yolov5', choices=['yolov5', 'ultralytics'], help='Backend for person detector')
    parser.add_argument('--person-model', default=None, help='Model path/name for person detector (optional)')
    parser.add_argument('--obj-backend', default='ultralytics', choices=['yolov5', 'ultralytics'], help='Backend for ball/bat detector')
    parser.add_argument('--obj-model', default=None, help='Model path/name for ball/bat detector (optional)')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading person detector ({args.person_backend}) and object detector ({args.obj_backend})...")
    person_detector = Detector(backend=args.person_backend, model_path=args.person_model, device=args.device)
    obj_detector = Detector(backend=args.obj_backend, model_path=args.obj_model, device=args.device)

    # Labels to keep for each detector
    person_labels = set(['person'])
    # COCO labels for ball/bat candidates
    obj_labels = set(['sports ball', 'baseball bat', 'bat'])

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"Error: cannot open input {args.input}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    base_name = os.path.splitext(os.path.basename(args.input))[0]
    person_out = os.path.join(args.output_dir, f"{base_name}_person.mp4")
    obj_out = os.path.join(args.output_dir, f"{base_name}_objects.mp4")
    combined_out = os.path.join(args.output_dir, f"{base_name}_combined.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    person_writer = cv2.VideoWriter(person_out, fourcc, fps, (w, h))
    obj_writer = cv2.VideoWriter(obj_out, fourcc, fps, (w, h))
    combined_writer = cv2.VideoWriter(combined_out, fourcc, fps, (w, h))

    frame_idx = 0
    start = time.time()
    print(f"Processing frames and writing to {args.output_dir}...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # Run detectors (they are independent/backends may differ)
        try:
            person_dets = person_detector.detect(frame, conf=args.conf)
        except Exception as e:
            print(f"Person detector error on frame {frame_idx}: {e}")
            person_dets = []

        try:
            obj_dets = obj_detector.detect(frame, conf=args.conf)
        except Exception as e:
            print(f"Object detector error on frame {frame_idx}: {e}")
            obj_dets = []

        # Filter detections to only the classes we care about
        person_dets = filter_by_labels(person_dets, person_labels)
        obj_dets = filter_by_labels(obj_dets, obj_labels)

        # Draw and write
        person_frame = frame.copy()
        obj_frame = frame.copy()
        combined_frame = frame.copy()

        if person_dets:
            draw_detections(person_frame, person_dets)
        if obj_dets:
            draw_detections(obj_frame, obj_dets)

        # Combine both annotations on one frame
        combined_dets = person_dets + obj_dets
        if combined_dets:
            draw_detections(combined_frame, combined_dets)

        person_writer.write(person_frame)
        obj_writer.write(obj_frame)
        combined_writer.write(combined_frame)

        if frame_idx % 100 == 0:
            elapsed = time.time() - start
            print(f"Frame {frame_idx} processed ({elapsed:.1f}s)")

    cap.release()
    person_writer.release()
    obj_writer.release()
    combined_writer.release()

    total_time = time.time() - start
    print(f"Done. {frame_idx} frames processed in {total_time:.1f}s")
    print(f"Outputs:\n - {person_out}\n - {obj_out}\n - {combined_out}")


if __name__ == '__main__':
    main()
