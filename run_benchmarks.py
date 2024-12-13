#!/usr/bin/env python

from ultralytics.utils.benchmarks import benchmark

models = [
    "yolov5n.pt",
    # "yolov5s.pt",
    # "yolov5m.pt",
    # "yolov5l.pt",

    # "yolov8n.pt",
    # "yolov8s.pt",
    # "yolov8m.pt",
    # "yolov8l.pt",

    # "yolov10n.pt",
    # "yolov10s.pt",
    # "yolov10m.pt",
    # "yolov10l.pt",

    # "yolo11n.pt",
    # "yolo11s.pt",
    # "yolo11m.pt",
    # "yolo11l.pt",
]

precision=[
    False,       # float32
    # True,       # float16
]

batchs = [
    # 1,
    # 8,
    16
]

dataset = "coco128.yaml"
imgsz = 640
device = "musa:0"

for model in models:
    for half in precision:
        for batch in batchs:
            print(f"Benchmarking model: {model} with half:{half} batch: {batch}")
            benchmark(model=model, data=dataset, imgsz=imgsz, batch=batch, half=half, device=device, pt_only=True)