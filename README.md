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
### Running with Gazebo simulation

1. **Install Gazebo + ROS2 integration** (skip if already set up):
```bash
   sudo apt install ros-<your-ros2-distro>-gazebo-ros-pkgs
```

2. **Get the NAO robot description/model package**:
   [PLACEHOLDER — repo/package name + install steps go here]

3. **Build your workspace**:
```bash
   cd ~/your_ros2_ws
   colcon build --symlink-install
   source install/setup.bash
```

4. **Launch the simulation** (spawns NAO in Gazebo):
```bash
   ros2 launch <nao_package_name> <launch_file_name>.launch.py
```

5. **In a separate terminal**, run the motion capture node so it starts
   publishing joint commands the simulated NAO will follow:
```bash
   python3 motion_capture_nao_node.py --source path/to/video.mp4 --conf 0.5
```

6. **Verify it's working**: the NAO model in Gazebo should mirror your
   movements from the video/webcam feed. If it doesn't move, check that:
   - The topic names in `JOINT_TO_NAO` match the joint controller topics
     Gazebo is actually subscribed to (`ros2 topic list` to confirm)
   - The node's terminal isn't printing `LOST` for the joints you're
     testing (means detection confidence is too low — try lowering
     `--conf` or improving lighting/camera framing)

### What gets published
One topic per mapped joint under `/nao/<JointName>/cmd_pos`, in radians.
Unmapped joints (e.g. wrist yaw) are intentionally not published — see
comments in `JOINT_TO_NAO` for why.

### Known limitations
- 2D camera only — roll/yaw joints (marked "experimental" in
  `JOINT_MOTOR_MAP`) are rough proxies, not true 3D angles. Expect noise.
- `h` (human angle) ranges may need retuning against your own footage.
