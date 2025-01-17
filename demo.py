import threading
import argparse
import cv2
import os
from queue import Queue, Empty
from ultralytics import YOLO
from screeninfo import get_monitors
import time
import numpy as np

class FrameCaptureThread(threading.Thread):
    """
    A dedicated thread for capturing frames from the input source and distributing copies
    to different model queues. Starts only after all models are loaded and respects step mode.
    """
    def __init__(self, input, frame_queues, fps, step, loop, restart_event):
        threading.Thread.__init__(self)
        self.input = input
        self.frame_queues = frame_queues
        self.running = True
        self.cap = None
        self.fps = fps
        self.delay = 1.0 / fps if fps > 0 else 0
        self.step = step
        self.capture_step_event = threading.Event()  # Event for step-by-step control
        self.loop = loop
        self.restart_event = restart_event
        self._initialize_capture()

    def _initialize_capture(self):
        self.cap = None
        self.image_files = None
        self.image_iterator = None
        if self.input.isdigit():
            self.cap = cv2.VideoCapture(int(self.input))
        elif os.path.isfile(self.input):
            self.cap = cv2.VideoCapture(self.input)
        elif os.path.isdir(self.input):
            self.image_files = sorted([os.path.join(self.input, f) for f in os.listdir(self.input) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))])
            self.image_iterator = iter(self.image_files)
        elif self.input.startswith("rtsp://"):
            self.cap = cv2.VideoCapture(self.input)
        else:
            print(f"Error: Invalid input source: {self.input}")
            self.running = False

    def run(self):
        if not self.running:
            return
        print("FrameCaptureThread: All models loaded, starting capture.")

        while self.running:
            if self.restart_event.is_set():
                print("FrameCaptureThread: Restarting input source.")
                self._initialize_capture()
                self.restart_event.clear()
                if not self.running:
                    break # Exit if stop was called during restart

            frame = None
            start_time = time.time()
            if self.cap:
                ret, frame = self.cap.read()
                if not ret:
                    if self.loop:
                        self.restart_event.set() # Signal restart
                        continue
                    else:
                        break  # Handle end of video
            elif self.image_iterator:
                try:
                    image_path = next(self.image_iterator)
                    frame = cv2.imread(image_path)
                    if frame is None:
                        print(f"Error reading image: {image_path}")
                except StopIteration:
                    if self.loop:
                        self.restart_event.set() # Signal restart
                        continue
                    else:
                        break  # Handle end of image directory
            else:
                break # Should not happen if initialized correctly

            if frame is not None:
                for queue in self.frame_queues.values():
                    send_frame = frame.copy()
                    try:
                        queue.put(send_frame, block=False)  # Distribute copies
                    except Exception as e:
                        print(f"Error putting frame to queue: {e}")
                        send_frame = None

            elapsed_time = time.time() - start_time
            sleep_time = max(0, self.delay - elapsed_time)
            if not self.step:
                time.sleep(sleep_time)
            else:
                self.capture_step_event.wait()  # Wait for the signal to proceed
                self.capture_step_event.clear() # Reset the event

            if self.step and not self.running: # Check again after potential wait
                break

        if self.cap and self.cap.isOpened():
            self.cap.release()
        print("FrameCaptureThread finished.")

    def stop(self):
        self.running = False
        self.capture_step_event.set() # Ensure it's unblocked if waiting
        self.restart_event.set() # Ensure it's unblocked if waiting for restart

    def next_step(self):
        self.capture_step_event.set()

class ModelThread(threading.Thread):
    """
    A thread for running YOLO inference on a specific model, receiving frames from its dedicated queue.
    Signals when model is loaded.
    """
    def __init__(self, model_path, input_queue, output_queue, models_loaded_semaphore, display_region, model_index, loop, restart_event):
        threading.Thread.__init__(self)
        self.model_path = model_path
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.running = True
        self.model = None
        self.model_name = os.path.basename(model_path).split('.')[0]  # Extract model name
        self.display_region = display_region # (x, y, width, height) for this model
        self.models_loaded_semaphore = models_loaded_semaphore
        self.model_index = model_index
        self.loop = loop
        self.restart_event = restart_event

    def run(self):
        try:
            self.model = YOLO(self.model_path)
            print(f"Thread for {self.model_path} loaded model successfully.")
            self.models_loaded_semaphore.release()

            while self.running:
                try:
                    frame = self.input_queue.get(timeout=0.1)
                    if frame is None:  # Sentinel value for stopping
                        break

                    results = self.model.predict(frame)
                    annotated_frame = results[0].plot() if results else frame
                    self.output_queue.put((self.model_name, annotated_frame, self.display_region))  # Include region

                except Empty:
                    continue  # No frame available, try again or exit

        except Exception as e:
            print(f"Error in thread for {self.model_path}: {e}")
        finally:
            print(f"Thread for {self.model_path} finished.")

    def stop(self):
        self.running = False

def calculate_model_layouts(num_models, background_width, background_height):
    """Calculates display regions for models based on their count with smarter layout."""

    if num_models == 0:
        return []

    if num_models == 1:
        return [(0, 0, background_width, background_height)]

    elif 2 <= num_models <= 4:
        grid_size = 2
    elif 5 <= num_models <= 9:
        grid_size = 3
    elif 10 <= num_models <= 16:
        grid_size = 4
    else:
        # For more than 16, determine the smallest square grid that fits
        grid_size = int(np.ceil(np.sqrt(num_models)))

    cell_width = background_width // grid_size
    cell_height = background_height // grid_size
    layouts = []

    for i in range(num_models):
        row = i // grid_size
        col = i % grid_size
        x = col * cell_width
        y = row * cell_height
        layouts.append((x, y, cell_width, cell_height))

    return layouts

def main():
    parser = argparse.ArgumentParser(description="Multi-threaded YOLO inference with a static background and smart layout.")
    parser.add_argument("--input", type=str, default="./ultralytics/assets", help="Input source (camera index, video file, or directory of images).")
    parser.add_argument("--model", nargs='+', type=str, default=["yolov5n.pt", "yolov8m-pose.pt", "yolov8n-seg.pt"], help="Path(s) to the YOLO model(s).")
    parser.add_argument("--background", type=str, default=None, help="Path to the static background image. Defaults to black.")
    parser.add_argument("--loop", action="store_true", help="Enable loop mode for video or image directory.")
    parser.add_argument("--step", action="store_true", help="Enable single-step mode. Press 's' to advance.")
    parser.add_argument("--fps", type=int, default=1, help="Frames per second for input processing.")
    args = parser.parse_args()

    monitors = get_monitors()
    if monitors:
        primary_monitor = monitors[0]
        screen_width = primary_monitor.width // 2
        screen_height = primary_monitor.height // 2
    else:
        print("Warning: Could not detect any monitors. Using default resolution (1920x1080).")
        screen_width = 1920
        screen_height = 1080

    if args.background:
        background = cv2.imread(args.background)
        if background is None:
            print(f"Error: Could not load background image from {args.background}. Using a black background instead.")
            background = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
    else:
        background = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)

    bg_height, bg_width, _ = background.shape

    num_models = len(args.model)
    model_layouts = calculate_model_layouts(num_models, bg_width, bg_height)

    frame_queues = {f"model_{i}": Queue(maxsize=10) for i in range(num_models)}
    output_queue = Queue()
    threads = []
    models_loaded_semaphore = threading.Semaphore(0) # 初始化为 0，模型加载后释放
    restart_event = threading.Event() # Event to signal frame capture thread to restart

    # Create and start model threads FIRST
    for i, model_path in enumerate(args.model):
        display_region = model_layouts[i]
        model_thread = ModelThread(model_path, frame_queues[f"model_{i}"], output_queue, models_loaded_semaphore, display_region, i, args.loop, restart_event)
        threads.append(model_thread)
        model_thread.start()

    for _ in range(num_models):
        models_loaded_semaphore.acquire()

    # Start the frame capture thread, passing the queues and the semaphore
    capture_thread = FrameCaptureThread(args.input, frame_queues, fps=args.fps, step=args.step, loop=args.loop, restart_event=restart_event)
    threads.append(capture_thread)

    print("Waiting for all models to load...")
    capture_thread.start()

    window_name = "AIDemo"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, bg_width, bg_height)
    cv2.moveWindow(window_name, 0, 0) # Move to top-left

    while any(t.is_alive() for t in threads):
        updated_frames = {}
        try:
            while True:  # Process all available frames in the queue
                model_name, frame, region = output_queue.get_nowait()
                updated_frames[model_name] = (frame, region)
        except Empty:
            pass

        # Overlay the latest frames onto the background
        for model_name, (frame, (x, y, w, h)) in updated_frames.items():
            if frame is not None:
                try:
                    # 确保目标区域在背景图像的范围内
                    if y >= 0 and y + h <= background.shape[0] and x >= 0 and x + w <= background.shape[1]:
                        resized_frame = cv2.resize(frame, (w, h))

                        # 获取背景图像中要覆盖的区域
                        roi = background[y:y + h, x:x + w]

                        # 确保 resized_frame 和 roi 的形状相同
                        if roi.shape == resized_frame.shape:
                            # 直接将 resized_frame 复制到 roi
                            roi[:] = resized_frame
                        else:
                            print(f"Warning: Shape mismatch for {model_name}, skipping overlay.")
                    else:
                        print(f"Warning: Target region for {model_name} is out of background bounds, skipping overlay.")

                except Exception as e:
                    print(f"Error overlaying frame for {model_name}: {e}, Region: {(x,y,w,h)}, Frame Shape: {frame.shape if frame is not None else None}")

        cv2.imshow(window_name, background)

        delay = 1 if args.fps <= 0 else int(1000 / args.fps)
        key = cv2.waitKey(delay)

        if key & 0xFF == ord('q'):
            break
        elif args.step and key & 0xFF == ord('n'):
            capture_thread.next_step()

    # Stop all threads
    capture_thread.stop()
    for thread in threads:
        thread.stop()

    for thread in threads:
        thread.join()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()