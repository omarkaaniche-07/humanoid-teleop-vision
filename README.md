# humanoid-teleop-vision
## NAO Motion Capture Node — `motion_capture_nao_node.py`

### What it does
Runs YOLOv8 pose estimation on a video/webcam feed, extracts human joint
angles (shoulders, elbows, hips, knees, ankles, neck, spine + experimental
roll/yaw joints), filters them with a One Euro Filter, maps them to NAO's
joint ranges, and publishes each as a `std_msgs/Float64` on its own ROS2
topic (e.g. `/nao/LShoulderPitch/cmd_pos`).

### Before you run it
- **Weights**: needs `yolov8n-pose.pt`. `ultralytics` will auto-download it
  on first run if it's not present, so you don't need to fetch it manually
  — just make sure you have internet access the first time.
- **Dependencies**: `ultralytics`, `opencv-python`, `numpy`, `rclpy`,
  `std_msgs`. Install via `pip install ultralytics opencv-python numpy`
  (ROS2 deps come from your existing ROS2 install).
- **GPU**: the node hardcodes `.to("cuda")`. If you don't have an NVIDIA
  GPU / CUDA-enabled torch, this will crash — ping me and I'll add a CPU
  fallback, or edit that line locally to `.to("cpu")` for now.

### How to run it
```bash
# source your ROS2 workspace first as usual
python3 motion_capture_nao_node.py --source path/to/video.mp4 --conf 0.5
```
- `--source` — path to a video file, or a webcam index (e.g. `0`)
- `--conf` — detection confidence threshold (default 0.5)

There's no default video path baked in anymore — you must pass `--source`.

### What gets published
One topic per mapped joint under `/nao/<JointName>/cmd_pos`, in radians.
Unmapped joints (e.g. wrist yaw) are intentionally not published — see
comments in `JOINT_TO_NAO` for why.

### Known limitations
- 2D camera only — roll/yaw joints (marked "experimental" in
  `JOINT_MOTOR_MAP`) are rough proxies, not true 3D angles. Expect noise.
- `h` (human angle) ranges may need retuning against your own footage.
