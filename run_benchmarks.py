#!/usr/bin/env python

from ultralytics.utils.benchmarks import benchmark

models = [
    # "yolov5su.pt",

    "yolov8n.pt",
    # "yolov8s.pt",

    # "yolov8m.pt",
    # "yolov8l.pt",

    # "yolov9t.pt",
    # "yolov10n.pt",

    # "yolov10s.pt",

    # "yolov10m.pt",

    # "yolov10l.pt",

    # "yolo11n.pt",
    # "yolo11s.pt",
]

data = "coco128.yaml"
imgsz = 640
half = False
device = "musa"

for model in models:
    print(f"Benchmarking model: {model}")
    benchmark(model=model, data=data, imgsz=imgsz, half=half, device=device)