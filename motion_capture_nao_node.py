import argparse
import math
import time
from math import asin, atan2, sqrt, degrees

import cv2
import numpy as np
from ultralytics import YOLO

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64

SKELETON = [
    (0,1),(0,2),(1,3),(2,4),(5,6),
    (5,7),(7,9),(6,8),(8,10),(5,11),
    (6,12),(11,12),(11,13),(13,15),(12,14),(14,16)
]

BONE_COLOR  = (255, 165, 0)
BBOX_COLOR  = (0, 200, 255)
TEXT_COLOR  = (255, 255, 255)
ALERT_COLOR = (0, 0, 255)

JOINT_DEFS = {
    "left_shoulder":  dict(p1=11, vertex=5,  p3=7,  color=(255,100,100)),
    "right_shoulder": dict(p1=12, vertex=6,  p3=8,  color=(100,100,255)),
    "left_elbow":     dict(p1=5,  vertex=7,  p3=9,  color=(255,200, 50)),
    "right_elbow":    dict(p1=6,  vertex=8,  p3=10, color=( 50,200,255)),
    "left_hip":       dict(p1=5,  vertex=11, p3=13, color=(200, 50,255)),
    "right_hip":      dict(p1=6,  vertex=12, p3=14, color=( 50,255,200)),
    "left_knee":      dict(p1=11, vertex=13, p3=15, color=(255,255,  0)),
    "right_knee":     dict(p1=12, vertex=14, p3=16, color=(  0,255,255)),
    "left_ankle":     dict(p1=13, vertex=15, p3=16, color=(180,255,100)),
    "right_ankle":    dict(p1=14, vertex=16, p3=15, color=(100,180,255)),
    "torso_left":     dict(p1=5,  vertex=11, p3=12, color=(255,180,180)),
    "torso_right":    dict(p1=6,  vertex=12, p3=11, color=(180,180,255)),
}

PANEL_GROUPS = [
    ("UPPER BODY", ["left_shoulder","right_shoulder","left_elbow","right_elbow","neck"]),
    ("LOWER BODY", ["left_hip","right_hip","left_knee","right_knee",
                    "left_ankle","right_ankle"]),
    ("TORSO",      ["torso_left","torso_right","spine"]),
]

# "h" = expected human angle range (degrees) coming from angle_at_vertex()
# "m" = target NAO motor range (degrees) -- derived from NAO's real joint
#        limits (radians, from the Gazebo Joint Position Controller panel),
#        converted to degrees. Adjust "h" ranges to match your own
#        calibration/testing.
JOINT_MOTOR_MAP = {
    "left_shoulder":  dict(h=(30, 180), m=(119.8, -119.8)),   # LShoulderPitch (inverted)
    "right_shoulder": dict(h=(30, 180), m=(119.8, -119.8)),   # RShoulderPitch (inverted)
    "left_elbow":     dict(h=(0,  180), m=(-88.2,   -1.7)),   # LElbowRoll
    "right_elbow":    dict(h=(0,  180), m=(-88.2,   -1.7)),   # RElbowRoll
    "left_hip":       dict(h=(70, 180), m=(27.5,   -88.2)),   # LHipPitch (inverted)
    "right_hip":      dict(h=(70, 180), m=(27.5,   -88.2)),   # RHipPitch (inverted)
    "left_knee":      dict(h=(0,  180), m=( -5.2,  120.9)),   # LKneePitch
    "right_knee":     dict(h=(0,  180), m=( -5.2,  120.9)),   # RKneePitch
    "left_ankle":     dict(h=(60, 130), m=(-68.2,   52.7)),   # LAnklePitch
    "right_ankle":    dict(h=(60, 130), m=(-68.2,   52.7)),   # RAnklePitch
    "torso_left":     dict(h=(60, 120), m=(60, 120)),         # not mapped to NAO
    "torso_right":    dict(h=(60, 120), m=(60, 120)),         # not mapped to NAO
    "neck":           dict(h=(130, 180), m=(-38.4,  29.2)),   # HeadPitch
    "spine":          dict(h=(70,  110), m=(70,  110)),       # not mapped to NAO

    # --- Experimental lateral / rotation estimates below ---
    # These are NOT true 3-point joint angles. A single 2D camera can't
    # observe true rotation around a limb's own axis, so these are rough
    # proxies based on lateral pixel displacement. Expect more noise than
    # the Pitch joints above -- retune "h" ranges against your own footage.
    "head_yaw":         dict(h=(-45,  45), m=(-119.7, 119.7)),  # HeadYaw
    "left_sh_roll":     dict(h=(-10,  80), m=( -17.8,  76.2)),  # LShoulderRoll
    "right_sh_roll":    dict(h=(-80,  10), m=( -76.2,  17.8)),  # RShoulderRoll (mirrored)
    "left_hip_roll":    dict(h=(-30,  30), m=( -21.8,  45.3)),  # LHipRoll
    "right_hip_roll":   dict(h=(-30,  30), m=( -45.3,  21.8)),  # RHipRoll (mirrored)
    "left_ankle_roll":  dict(h=(-30,  30), m=( -22.9,  44.1)),  # LAnkleRoll
    "right_ankle_roll": dict(h=(-30,  30), m=( -44.1,  22.9)),  # RAnkleRoll (mirrored)
    "left_elbow_yaw":   dict(h=(-45,  45), m=(-119.7, 119.7)),  # LElbowYaw (rough proxy)
    "right_elbow_yaw":  dict(h=(-45,  45), m=(-119.7, 119.7)),  # RElbowYaw (rough proxy)
    "hip_yaw_pitch":    dict(h=(-30,  30), m=( -65.9,  42.4)),  # shared L/R HipYawPitch proxy
}

# Maps your detected joint name -> NAO joint name / ROS 2 topic suffix.
# Only joints listed here get published. Add more as you extend detection.
#
# NOTE: LWristYaw / RWristYaw are intentionally NOT included. A single 2D
# wrist keypoint carries no information about forearm rotation -- there is
# nothing to measure it from with the current keypoint set. They are left
# unpublished (joint stays at its default/last commanded position) rather
# than faking a signal that would just be noise.
JOINT_TO_NAO = {
    "left_shoulder":     "LShoulderPitch",
    "right_shoulder":    "RShoulderPitch",
    "left_elbow":        "LElbowRoll",
    "right_elbow":       "RElbowRoll",
    "left_hip":          "LHipPitch",
    "right_hip":         "RHipPitch",
    "left_knee":         "LKneePitch",
    "right_knee":        "RKneePitch",
    "left_ankle":        "LAnklePitch",
    "right_ankle":       "RAnklePitch",
    "neck":              "HeadPitch",

    # experimental additions
    "head_yaw":          "HeadYaw",
    "left_sh_roll":      "LShoulderRoll",
    "right_sh_roll":     "RShoulderRoll",
    "left_hip_roll":     "LHipRoll",
    "right_hip_roll":    "RHipRoll",
    "left_ankle_roll":   "LAnkleRoll",
    "right_ankle_roll":  "RAnkleRoll",
    "left_elbow_yaw":    "LElbowYaw",
    "right_elbow_yaw":   "RElbowYaw",
    # hip_yaw_pitch is handled specially -- published to BOTH L and R below
}


def map_angle(human_angle: float, joint_name: str) -> float:
    cfg = JOINT_MOTOR_MAP[joint_name]
    h_min, h_max = cfg["h"]
    m_min, m_max = cfg["m"]
    h_clamped = max(h_min, min(h_max, human_angle))
    if h_max == h_min:
        return float(m_min)
    ratio = (h_clamped - h_min) / (h_max - h_min)
    motor_angle = m_min + ratio * (m_max - m_min)
    return float(max(min(m_min, m_max), min(max(m_min, m_max), motor_angle)))


class OneEuroFilter:
    def __init__(self, te, mincutoff=1.0, beta=0.0, dcutoff=1.0):
        self.mincutoff = float(mincutoff)
        self.beta      = float(beta)
        self.dcutoff   = float(dcutoff)
        self.x_prev    = None
        self.dx_prev   = None
        self.t_prev    = float(te)

    def reset(self, t):
        self.x_prev  = None
        self.dx_prev = None
        self.t_prev  = float(t)

    def _alpha(self, cutoff, dt):
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, t, x):
        t = float(t)
        if self.x_prev is None:
            self.x_prev  = x
            self.dx_prev = 0.0
            self.t_prev  = t
            return x
        dt = t - self.t_prev
        if dt <= 0:
            return self.x_prev
        d_x     = (x - self.x_prev) / dt
        alpha_d = self._alpha(self.dcutoff, dt)
        dx_hat  = alpha_d * d_x + (1.0 - alpha_d) * self.dx_prev
        cutoff  = self.mincutoff + self.beta * abs(dx_hat)
        alpha   = self._alpha(cutoff, dt)
        x_hat   = alpha * x + (1.0 - alpha) * self.x_prev
        self.x_prev  = x_hat
        self.dx_prev = dx_hat
        self.t_prev  = t
        return x_hat


class MotionCapturePipeline(Node):
    def __init__(self):
        super().__init__('nao_joint_publisher')

        self.model = YOLO("yolov8n-pose.pt")
        self.model.to("cuda")
        self.joint_states = {
            name: {"angle": 0.0, "motor_angle": 0.0, "status": "LOST"}
            for name in JOINT_DEFS
        }

        t0 = time.time()
        self.filters = {
            name: OneEuroFilter(t0, mincutoff=0.8, beta=0.01)
            for name in JOINT_DEFS
        }

        self.joint_states["neck"]  = {"angle": 0.0, "motor_angle": 0.0, "status": "LOST"}
        self.joint_states["spine"] = {"angle": 0.0, "motor_angle": 0.0, "status": "LOST"}
        self.filters["neck"]  = OneEuroFilter(t0, mincutoff=0.8, beta=0.01)
        self.filters["spine"] = OneEuroFilter(t0, mincutoff=0.8, beta=0.01)

        # Experimental lateral/rotation joints -- see JOINT_MOTOR_MAP notes
        experimental_joints = [
            "head_yaw", "left_sh_roll", "right_sh_roll",
            "left_hip_roll", "right_hip_roll",
            "left_ankle_roll", "right_ankle_roll",
            "left_elbow_yaw", "right_elbow_yaw",
            "hip_yaw_pitch",
        ]
        for jname in experimental_joints:
            self.joint_states[jname] = {"angle": 0.0, "motor_angle": 0.0, "status": "LOST"}
            self.filters[jname] = OneEuroFilter(t0, mincutoff=0.8, beta=0.01)

        # ROS 2 publishers: one per NAO joint we're driving
        self.publishers_map = {}
        for human_joint, nao_joint in JOINT_TO_NAO.items():
            topic = f"/nao/{nao_joint}/cmd_pos"
            self.publishers_map[human_joint] = self.create_publisher(Float64, topic, 10)
            self.get_logger().info(f"Publishing {human_joint} -> {topic}")

        # hip_yaw_pitch is a special case: on real NAO hardware, L/RHipYawPitch
        # are mechanically coupled (one motor drives both). We approximate
        # that here by publishing the SAME value to both joints.
        self.hip_yaw_pitch_publishers = [
            self.create_publisher(Float64, "/nao/LHipYawPitch/cmd_pos", 10),
            self.create_publisher(Float64, "/nao/RHipYawPitch/cmd_pos", 10),
        ]
        self.get_logger().info("Publishing hip_yaw_pitch -> /nao/LHipYawPitch/cmd_pos + /nao/RHipYawPitch/cmd_pos")

    @staticmethod
    def angle_at_vertex(p1, p2, p3) -> float:
        a  = np.array(p1, dtype=float)
        b  = np.array(p2, dtype=float)
        c  = np.array(p3, dtype=float)
        ba = a - b
        bc = c - b
        denom = np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6
        cos_a = np.dot(ba, bc) / denom
        return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))

    @staticmethod
    def lateral_angle(a, b) -> float:
        """Approximate 'roll/abduction' angle (degrees, signed) of the
        vector a->b based on its horizontal (x) lean, using the standard
        sin-based abduction proxy: -90 (fully left) .. 0 (straight down/up)
        .. +90 (fully right). This is a 2D approximation, not a true
        3D joint angle -- expect noise, especially when the limb points
        toward/away from the camera."""
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        dist = sqrt(dx * dx + dy * dy) + 1e-6
        ratio = max(-1.0, min(1.0, dx / dist))
        return degrees(asin(ratio))

    def _publish_angle(self, human_joint_name):
        state = self.joint_states[human_joint_name]
        if state["status"] != "TRACKING":
            return  # don't publish stale/lost data
        rad = math.radians(state["motor_angle"])
        msg = Float64()
        msg.data = rad

        if human_joint_name == "hip_yaw_pitch":
            for pub in self.hip_yaw_pitch_publishers:
                pub.publish(msg)
            return

        if human_joint_name not in self.publishers_map:
            return  # not mapped to a NAO joint, skip
        self.publishers_map[human_joint_name].publish(msg)

    def _update_joint(self, name, kp_xy, kp_conf, conf_thr, t):
        d  = JOINT_DEFS[name]
        i1, iv, i3 = d["p1"], d["vertex"], d["p3"]
        prev_status = self.joint_states[name]["status"]

        if (kp_conf[i1] > conf_thr and
                kp_conf[iv] > conf_thr and
                kp_conf[i3] > conf_thr):
            raw      = self.angle_at_vertex(kp_xy[i1], kp_xy[iv], kp_xy[i3])
            if prev_status == "LOST":
                self.filters[name].reset(t)
            filtered = self.filters[name].filter(t, raw)
            motor    = map_angle(filtered, name)
            self.joint_states[name]["angle"]       = filtered
            self.joint_states[name]["motor_angle"] = motor
            self.joint_states[name]["status"]      = "TRACKING"
            self._publish_angle(name)
        else:
            self.joint_states[name]["status"] = "LOST"

    def _update_neck(self, kp_xy, kp_conf, conf_thr, t):
        if kp_conf[0] > conf_thr and kp_conf[5] > conf_thr and kp_conf[6] > conf_thr:
            raw = self.angle_at_vertex(kp_xy[5], kp_xy[0], kp_xy[6])
            if self.joint_states["neck"]["status"] == "LOST":
                self.filters["neck"].reset(t)
            filtered = self.filters["neck"].filter(t, raw)
            self.joint_states["neck"]["angle"]       = filtered
            self.joint_states["neck"]["motor_angle"] = map_angle(filtered, "neck")
            self.joint_states["neck"]["status"]      = "TRACKING"
            self._publish_angle("neck")
        else:
            self.joint_states["neck"]["status"] = "LOST"

    def _update_spine(self, kp_xy, kp_conf, conf_thr, t):
        if (kp_conf[5] > conf_thr and kp_conf[6] > conf_thr and
                kp_conf[11] > conf_thr and kp_conf[12] > conf_thr):
            mid_shoulder = (kp_xy[5]  + kp_xy[6])  / 2.0
            mid_hip      = (kp_xy[11] + kp_xy[12]) / 2.0
            vertical_ref = np.array([mid_hip[0], mid_hip[1] - 100.0])
            raw = self.angle_at_vertex(mid_shoulder, mid_hip, vertical_ref)
            if self.joint_states["spine"]["status"] == "LOST":
                self.filters["spine"].reset(t)
            filtered = self.filters["spine"].filter(t, raw)
            self.joint_states["spine"]["angle"]       = filtered
            self.joint_states["spine"]["motor_angle"] = map_angle(filtered, "spine")
            self.joint_states["spine"]["status"]      = "TRACKING"
            # spine has no NAO mapping -- _publish_angle would skip it anyway
        else:
            self.joint_states["spine"]["status"] = "LOST"

    def _update_lateral(self, name, a_idx, b_idx, kp_xy, kp_conf, conf_thr, t):
        """Generic updater for the experimental lateral/roll-type joints,
        using lateral_angle() between two keypoints a_idx -> b_idx."""
        if kp_conf[a_idx] > conf_thr and kp_conf[b_idx] > conf_thr:
            raw = self.lateral_angle(kp_xy[a_idx], kp_xy[b_idx])
            if self.joint_states[name]["status"] == "LOST":
                self.filters[name].reset(t)
            filtered = self.filters[name].filter(t, raw)
            self.joint_states[name]["angle"]       = filtered
            self.joint_states[name]["motor_angle"] = map_angle(filtered, name)
            self.joint_states[name]["status"]      = "TRACKING"
            self._publish_angle(name)
        else:
            self.joint_states[name]["status"] = "LOST"

    def _update_head_yaw(self, kp_xy, kp_conf, conf_thr, t):
        # nose=0, left_shoulder=5, right_shoulder=6
        if kp_conf[0] > conf_thr and kp_conf[5] > conf_thr and kp_conf[6] > conf_thr:
            mid_shoulder = (kp_xy[5] + kp_xy[6]) / 2.0
            raw = self.lateral_angle(mid_shoulder, kp_xy[0])
            if self.joint_states["head_yaw"]["status"] == "LOST":
                self.filters["head_yaw"].reset(t)
            filtered = self.filters["head_yaw"].filter(t, raw)
            self.joint_states["head_yaw"]["angle"]       = filtered
            self.joint_states["head_yaw"]["motor_angle"] = map_angle(filtered, "head_yaw")
            self.joint_states["head_yaw"]["status"]      = "TRACKING"
            self._publish_angle("head_yaw")
        else:
            self.joint_states["head_yaw"]["status"] = "LOST"

    def _update_hip_yaw_pitch(self, kp_xy, kp_conf, conf_thr, t):
        # Approximate torso twist using mid-hip -> mid-shoulder lateral lean
        if (kp_conf[5] > conf_thr and kp_conf[6] > conf_thr and
                kp_conf[11] > conf_thr and kp_conf[12] > conf_thr):
            mid_shoulder = (kp_xy[5] + kp_xy[6]) / 2.0
            mid_hip      = (kp_xy[11] + kp_xy[12]) / 2.0
            raw = self.lateral_angle(mid_hip, mid_shoulder)
            if self.joint_states["hip_yaw_pitch"]["status"] == "LOST":
                self.filters["hip_yaw_pitch"].reset(t)
            filtered = self.filters["hip_yaw_pitch"].filter(t, raw)
            self.joint_states["hip_yaw_pitch"]["angle"]       = filtered
            self.joint_states["hip_yaw_pitch"]["motor_angle"] = map_angle(filtered, "hip_yaw_pitch")
            self.joint_states["hip_yaw_pitch"]["status"]      = "TRACKING"
            self._publish_angle("hip_yaw_pitch")
        else:
            self.joint_states["hip_yaw_pitch"]["status"] = "LOST"

    def update_experimental_joints(self, kp_xy, kp_conf, conf_thr, t):
        # shoulder(5/6) -> elbow(7/8) lateral lean, as ShoulderRoll proxy
        self._update_lateral("left_sh_roll",  5, 7,  kp_xy, kp_conf, conf_thr, t)
        self._update_lateral("right_sh_roll", 6, 8,  kp_xy, kp_conf, conf_thr, t)
        # hip(11/12) -> knee(13/14) lateral lean, as HipRoll proxy
        self._update_lateral("left_hip_roll",  11, 13, kp_xy, kp_conf, conf_thr, t)
        self._update_lateral("right_hip_roll", 12, 14, kp_xy, kp_conf, conf_thr, t)
        # knee(13/14) -> ankle(15/16) lateral lean, as AnkleRoll proxy
        self._update_lateral("left_ankle_roll",  13, 15, kp_xy, kp_conf, conf_thr, t)
        self._update_lateral("right_ankle_roll", 14, 16, kp_xy, kp_conf, conf_thr, t)
        # elbow(7/8) -> wrist(9/10) lateral lean, as ElbowYaw rough proxy
        self._update_lateral("left_elbow_yaw",  7, 9,  kp_xy, kp_conf, conf_thr, t)
        self._update_lateral("right_elbow_yaw", 8, 10, kp_xy, kp_conf, conf_thr, t)
        # head yaw and shared hip yaw pitch
        self._update_head_yaw(kp_xy, kp_conf, conf_thr, t)
        self._update_hip_yaw_pitch(kp_xy, kp_conf, conf_thr, t)

    def print_joint_angles(self):
        print("\n--- Joint Angles ---")
        for name, state in self.joint_states.items():
            if state["status"] == "TRACKING":
                print(f"  {name:<18} human={state['angle']:6.1f}deg  "
                      f"motor={state['motor_angle']:6.1f}deg  "
                      f"({math.radians(state['motor_angle']):.3f} rad)")
            else:
                print(f"  {name:<18} LOST")

    def process_frame(self, frame, conf_thr=0.3):
        try:
            results = self.model(frame, verbose=False)
            result  = results[0]
        except Exception as e:
            print(f"[WARN] Inference error: {e}")
            for j in self.joint_states:
                self.joint_states[j]["status"] = "LOST"
            return frame

        t = time.time()

        if result.keypoints is None or result.boxes is None:
            for j in self.joint_states:
                self.joint_states[j]["status"] = "LOST"
            return frame

        kp_xy_all   = result.keypoints.xy.cpu().numpy()
        kp_conf_all = result.keypoints.conf.cpu().numpy()
        boxes_all   = result.boxes.xyxy.cpu().numpy()

        if len(boxes_all) > 0:
            kp_xy   = kp_xy_all[0]
            kp_conf = kp_conf_all[0]

            x1, y1, x2, y2 = map(int, boxes_all[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), BBOX_COLOR, 2)

            for (si, ei) in SKELETON:
                if kp_conf[si] > conf_thr and kp_conf[ei] > conf_thr:
                    pt1 = (int(kp_xy[si][0]), int(kp_xy[si][1]))
                    pt2 = (int(kp_xy[ei][0]), int(kp_xy[ei][1]))
                    cv2.line(frame, pt1, pt2, BONE_COLOR, 2)

            for name, d in JOINT_DEFS.items():
                iv = d["vertex"]
                if kp_conf[iv] > conf_thr:
                    cx = int(kp_xy[iv][0])
                    cy = int(kp_xy[iv][1])
                    cv2.circle(frame, (cx, cy), 6, d["color"], -1)

            for name in JOINT_DEFS:
                self._update_joint(name, kp_xy, kp_conf, conf_thr, t)
            self._update_neck(kp_xy, kp_conf, conf_thr, t)
            self._update_spine(kp_xy, kp_conf, conf_thr, t)
            self.update_experimental_joints(kp_xy, kp_conf, conf_thr, t)

            self._render_inline_labels(frame, kp_xy, kp_conf, conf_thr)
        else:
            for j in self.joint_states:
                self.joint_states[j]["status"] = "LOST"

        self._render_panel(frame)
        return frame

    def _render_inline_labels(self, frame, kp_xy, kp_conf, conf_thr):
        for name, d in JOINT_DEFS.items():
            iv    = d["vertex"]
            state = self.joint_states[name]
            if kp_conf[iv] <= conf_thr:
                continue
            x = int(kp_xy[iv][0]) + 10
            y = int(kp_xy[iv][1]) - 5
            if state["status"] == "TRACKING":
                label = f"{state['angle']:.0f}deg -> {state['motor_angle']:.0f}deg"
                color = d["color"]
            else:
                label = f"({state['angle']:.0f}deg)"
                color = ALERT_COLOR
            cv2.putText(frame, label, (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0,0,0), 3)
            cv2.putText(frame, label, (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1)

    def _render_panel(self, frame):
        h, w    = frame.shape[:2]
        panel_w = 330
        panel_x = w - panel_w - 10
        line_h  = 22
        pad_top = 10
        pad_l   = 8

        total_rows = sum(1 + len(j) for _, j in PANEL_GROUPS) + 1
        panel_h    = total_rows * line_h + pad_top * 2

        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x - 5, 10),
                      (panel_x + panel_w, 10 + panel_h), (20,20,20), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        y = 10 + pad_top
        for group_name, joint_names in PANEL_GROUPS:
            cv2.putText(frame, group_name,
                        (panel_x + pad_l, y + line_h - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1)
            y += line_h

            for jname in joint_names:
                state = self.joint_states[jname]
                d     = JOINT_DEFS.get(jname, {"color": (200, 200, 200)})
                cx = panel_x + pad_l + 6
                cy = y + line_h // 2 - 4
                cv2.circle(frame, (cx, cy), 5, d["color"], -1)
                short = jname.replace("_", " ")

                if state["status"] == "TRACKING":
                    val_str   = (f"{state['angle']:5.1f}"
                                 f" -> {state['motor_angle']:5.1f} deg")
                    val_color = TEXT_COLOR
                else:
                    val_str   = f"  LOST  ({state['angle']:.1f})"
                    val_color = ALERT_COLOR

                cv2.putText(frame, short,
                            (panel_x + pad_l + 16, y + line_h - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200,200,200), 1)
                cv2.putText(frame, val_str,
                            (panel_x + pad_l + 138, y + line_h - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, val_color, 1)
                y += line_h

            cv2.line(frame,
                     (panel_x, y + 2), (panel_x + panel_w, y + 2),
                     (60,60,60), 1)
            y += 6


def main():
    rclpy.init()

    parser = argparse.ArgumentParser(description="Motion Capture Viewer")
    parser.add_argument("--source", default=r"/home/omar-kaaniche/Downloads/jump.mp4")
    parser.add_argument("--conf",   type=float, default=0.5)
    args = parser.parse_args()

    source = args.source
    # If it's a digit, treat it as a webcam index
    if source.isdigit():
        source = int(source)
    cap      = cv2.VideoCapture(source)
    pipeline = MotionCapturePipeline()

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    prev_time = time.time()

    try:
        while rclpy.ok():
            ret, frame = cap.read()
            if not ret:
                print("[INFO] End of stream.")
                break

            frame = pipeline.process_frame(frame, conf_thr=args.conf)

            now       = time.time()
            fps       = 1.0 / (now - prev_time + 1e-9)
            prev_time = now
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

            cv2.imshow("Motion Capture", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            rclpy.spin_once(pipeline, timeout_sec=0)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        pipeline.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()


