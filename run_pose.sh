#!/bin/bash

# 定义模型列表
models=(
    "yolov8n-pose.pt"
    "yolov8s-pose.pt"
    "yolov8m-pose.pt"
    "yolov8l-pose.pt"
)

# 定义精度选项 (True 表示 float16, False 表示 float32)
precision=(
    "False"
    "True"
)

# 定义批量大小列表
batchs=(
    "1"
)

# 定义数据集路径和图像尺寸
dataset="coco8-pose.yaml"
imgsz=640
device="musa:0"

tag=yolo_pose
timestamp=$(date +"%Y%m%d_%H%M%S")
log_file=bench_${tag}_${timestamp}.log
exec > >(tee -a "$log_file")

# 循环遍历所有组合并调用 benchmark.py
for model in "${models[@]}"; do
    for half in "${precision[@]}"; do
        for batch in "${batchs[@]}"; do
            echo "Benchmarking model: $model with half:$half batch:$batch dataset:${dataset} imgsz:${imgsz}"
            python -c "from ultralytics.utils.benchmarks import benchmark; \
                       benchmark(model='${model}', data='${dataset}', imgsz=${imgsz}, batch=${batch}, half=${half}, device='${device}', pt_only=True, verbose=True)"
            sleep 5
        done
    done
done