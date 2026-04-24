# Perception utilities

multi_detect.py — Run two detectors in parallel and save separate output clips.

Usage example:

```
python -m perception.multi_detect --input sample_cricket.mp4 --output-dir perception/output --device cpu
```

This will produce three files in `perception/output`:
- `<basename>_person.mp4` — frames annotated with `person` detections (uses `--person-backend`)
- `<basename>_objects.mp4` — frames annotated with ball/bat detections (uses `--obj-backend`)
- `<basename>_combined.mp4` — both annotations together

You can change backends (choices: `yolov5`, `ultralytics`) and model paths with `--person-model` and `--obj-model`.
