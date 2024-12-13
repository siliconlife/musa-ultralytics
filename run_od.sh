#!/bin/bash

models=(
    "yolov5n.pt"
    "yolov5s.pt"
    "yolov5m.pt"
    "yolov5l.pt"

    "yolov8n.pt"
    "yolov8s.pt"
    "yolov8m.pt"
    "yolov8l.pt"

    "yolov10n.pt"
    "yolov10s.pt"
    "yolov10m.pt"
    "yolov10l.pt"

    "yolo11n.pt"
    "yolo11s.pt"
    "yolo11m.pt"
    "yolo11l.pt"
)

# False: fp32, True: fp16;
dtype=(
    "False"
    "True"
)

batchs=(
    "1"
    "8"
    "16"
    "32"
    "64"
)

dataset="coco128.yaml"
imgsz=640
device="musa:0"

tag=yolo_od
log_dir="logs"
log_file="bench_${tag}_$(date +%Y%m%d_%H%M%S).log"
if [ ! -d "$log_dir" ]; then
    mkdir "$log_dir"
fi
exec > >(tee -a "$log_dir/$log_file")

for half in "${dtype[@]}"; do
    for model in "${models[@]}"; do
        for batch in "${batchs[@]}"; do
            echo "Benchmarking model: $model with half:$half batch:$batch dataset:${dataset} imgsz:${imgsz}"
            python -c "from ultralytics.utils.benchmarks import benchmark; \
                       benchmark(model='${model}', data='${dataset}', imgsz=${imgsz}, batch=${batch}, half=${half}, device='${device}', pt_only=True, verbose=True)"
            sleep 5
        done
    done
done