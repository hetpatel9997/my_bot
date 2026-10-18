#!/usr/bin/env python3
"""
phone_bridge_node.py (v2)  -  an iPhone as the robot's sensors, for free.

Talks to BOTH phone clients:
  * the Safari web page (https://<robot>:8443/, needs the certificate from make_cert.sh)
  * the native WeedBotSensor app (ws://<robot>:8444/ws, no certificate): adds LiDAR depth,
    ARKit position tracking, battery/thermal state and tap-to-tag weeds.

Publishes
  /phone/image/compressed   sensor_msgs/CompressedImage  back camera, JPEG, frame phone_optical
  /phone/camera_info        sensor_msgs/CameraInfo       ARKit intrinsics scaled to the streamed size
                                                         (or camera_info_url if given)
  /phone/imu/data_raw       sensor_msgs/Imu              phone_link, rad/s and m/s^2 (+9.8 up at rest)
  /phone/fix                sensor_msgs/NavSatFix        GPS, accuracy -> covariance
  /phone/depth/image        sensor_msgs/Image 16UC1      LiDAR depth in mm, 0 = invalid, phone_optical
  /phone/depth/camera_info  sensor_msgs/CameraInfo       intrinsics for the depth image
  /phone/ar_pose            geometry_msgs/PoseStamped    phone_optical in ar_origin (ARKit, z up)
  /phone/ar_twist           geometry_msgs/TwistWithCovarianceStamped  base_link velocity from ARKit
                                                         (only while mode == mounted) -> EKF twist0
  /phone/odom               nav_msgs/Odometry            base_link in ar_origin + the twist above
  /weed_tags                geometry_msgs/PointStamped   tagged weed in the map frame (latched, keep 50)
  /weed_tags/json           std_msgs/String              full tag record incl. file names (latched)
  /phone/status             std_msgs/String              JSON: connection, rates, thermal, battery, track
  /phone/e_stop             std_msgs/Bool                optional twist_mux lock (thermal critical)
  TF: phone_link -> phone_optical (static, if publish_optical_tf), map -> ar_origin (after anchoring)

Services
  /phone/anchor_here   std_srvs/Trigger   ties the AR world to the map using the robot's pose now
                                          (phone mounted, robot localized). Also triggered by the
                                          app's Anchor button and automatically when the app
                                          switches mounted -> handheld.
  /phone/anchor_clear  std_srvs/Trigger

Frames: phone_link = phone body axes (x right edge, y top edge, z out of the screen).
phone_optical = back camera optical frame (x right, y down, z forward); in phone_link it is the
fixed rotation rpy = (pi, 0, -pi/2). ar_origin = ARKit world converted to ROS: (x, y, z)_ros =
(x, -z, y)_arkit. The robot's URDF gives base_link -> phone_link for the mount.

Needs: pip3 install aiohttp pyyaml ; python3-numpy
"""
import asyncio
import json
import math
import os
import ssl
import struct
import threading
import time
import zlib
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rcl_interfaces.msg import SetParametersResult
from sensor_msgs.msg import CompressedImage, CameraInfo, Imu, NavSatFix, NavSatStatus, Image
from std_msgs.msg import String, Bool
from geometry_msgs.msg import PoseStamped, TwistWithCovarianceStamped, PointStamped, TransformStamped
from nav_msgs.msg import Odometry
from std_srvs.srv import Trigger
import tf2_ros

try:
    from aiohttp import web
except ImportError:  # pragma: no cover
    web = None

DEG = math.pi / 180.0
# ROS <- ARKit world: x_ros = x_ar, y_ros = -z_ar, z_ros = y_ar
C_AR2ROS = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
# optical <- ARKit camera (x right, y up, z backward): flip y and z
D_CAM2OPT = np.diag([1.0, -1.0, -1.0])
# phone_optical expressed in phone_link (rpy = pi, 0, -pi/2)
R_LINK_OPT = np.array([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])


# ---------------------------------------------------------------------------- small linear algebra
def mat_from_cols16(T16):
    """ARKit simd_float4x4 flattened column-major -> 4x4 numpy."""
    return np.array(T16, dtype=float).reshape(4, 4, order='F')


def quat_from_mat(R):
    """(x, y, z, w) from a 3x3 rotation matrix."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    n = math.sqrt(x * x + y * y + z * z + w * w)
    return (x / n, y / n, z / n, w / n)


def mat_from_quat(x, y, z, w):
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def rotvec_from_mat(R):
    """axis * angle of a rotation matrix (small-angle safe)."""
    c = max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0))
    ang = math.acos(c)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if ang < 1e-6:
        return v / 2.0
    return v * (ang / (2.0 * math.sin(ang)))


def mat_from_transform(tr):
    """geometry_msgs/Transform -> 4x4."""
    T = np.eye(4)
    q = tr.rotation
    T[:3, :3] = mat_from_quat(q.x, q.y, q.z, q.w)
    T[:3, 3] = [tr.translation.x, tr.translation.y, tr.translation.z]
    return T


def inv(T):
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


def ar_pose_to_ros(T_ar):
    """ARKit camera pose in the ARKit world -> phone_optical pose in ar_origin (ROS axes)."""
    T = np.eye(4)
    T[:3, :3] = C_AR2ROS @ T_ar[:3, :3] @ D_CAM2OPT
    T[:3, 3] = C_AR2ROS @ T_ar[:3, 3]
    return T


def fill_pose(pose, T):
    pose.position.x, pose.position.y, pose.position.z = [float(v) for v in T[:3, 3]]
    x, y, z, w = quat_from_mat(T[:3, :3])
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = x, y, z, w


def fill_transform(tr, T):
    tr.translation.x, tr.translation.y, tr.translation.z = [float(v) for v in T[:3, 3]]
    x, y, z, w = quat_from_mat(T[:3, :3])
    tr.rotation.x, tr.rotation.y, tr.rotation.z, tr.rotation.w = x, y, z, w


class OffsetEstimator:
    """offset = min(arrival - sent) over a sliding window; stamp = sent + offset."""

    def __init__(self, window_s=20.0):
        self.window = window_s
        self.samples = deque()

    def update(self, arrival, sent):
        d = arrival - sent
        self.samples.append((arrival, d))
        while self.samples and arrival - self.samples[0][0] > self.window:
            self.samples.popleft()
        return min(s[1] for s in self.samples)

    def latency_ms(self):
        if len(self.samples) < 2:
            return None
        ds = [s[1] for s in self.samples]
        return (ds[-1] - min(ds)) * 1000.0

    def reset(self):
        self.samples.clear()


# ---------------------------------------------------------------------------- the node
class PhoneBridge(Node):
    def __init__(self):
        super().__init__('phone_bridge')
        dp = self.declare_parameter
        dp('https_port', 8443)                 # web page (0 = off)
        dp('ws_port', 8444)                    # native app, plain websocket (0 = off)
        dp('http_port', 8080)                  # certificate install page (0 = off)
        dp('cert_dir', os.path.expanduser('~/phone_bridge_certs'))
        dp('www_dir', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'www'))
        dp('frame_id', 'phone_link')
        dp('optical_frame', 'phone_optical')
        dp('gps_frame_id', 'phone_gps')
        dp('ar_frame', 'ar_origin')
        dp('base_frame', 'base_link')
        dp('map_frame', 'map')
        dp('publish_optical_tf', True)
        dp('gyro_sign', 1.0)
        dp('accel_sign', 1.0)
        dp('save_dir', os.path.expanduser('~/weedbot_data/phone_photos'))
        dp('tag_dir', os.path.expanduser('~/weedbot_data/weed_tags'))
        dp('camera_info_url', '')
        dp('twist_window_s', 0.15)             # pose differencing window for the velocity
        dp('twist_sigma_v', 0.03)              # m/s, tracking normal
        dp('twist_sigma_w', 0.02)              # rad/s
        dp('auto_anchor_on_handheld', True)
        dp('fallback_identity_anchor', False)  # sim/testing only: anchor without a robot pose
        dp('estop_on_thermal_critical', True)
        dp('estop_on_disconnect', False)
        # sent to the app as "set" (also when changed at runtime with ros2 param set)
        dp('image_hz', 10.0)
        dp('depth_hz', 5.0)
        dp('pose_hz', 30.0)
        dp('jpeg_quality', 0.6)
        dp('image_width', 640)

        p = lambda n: self.get_parameter(n).value
        self.frame_id = p('frame_id')
        self.optical_frame = p('optical_frame')
        self.gps_frame_id = p('gps_frame_id')
        self.ar_frame = p('ar_frame')
        self.base_frame = p('base_frame')
        self.map_frame = p('map_frame')
        self.gyro_sign = float(p('gyro_sign'))
        self.accel_sign = float(p('accel_sign'))
        self.save_dir = os.path.expanduser(p('save_dir'))
        self.tag_dir = os.path.expanduser(p('tag_dir'))
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.tag_dir, exist_ok=True)
        self.twist_window = float(p('twist_window_s'))
        self.sig_v = float(p('twist_sigma_v'))
        self.sig_w = float(p('twist_sigma_w'))
        self.auto_anchor = bool(p('auto_anchor_on_handheld'))
        self.fallback_identity_anchor = bool(p('fallback_identity_anchor'))
        self.estop_thermal = bool(p('estop_on_thermal_critical'))
        self.estop_disconnect = bool(p('estop_on_disconnect'))

        sensor_qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
        latched = QoSProfile(depth=50, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        latched1 = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST,
                              durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_img = self.create_publisher(CompressedImage, 'phone/image/compressed', sensor_qos)
        self.pub_info = self.create_publisher(CameraInfo, 'phone/camera_info', sensor_qos)
        self.pub_imu = self.create_publisher(Imu, 'phone/imu/data_raw', 50)
        self.pub_fix = self.create_publisher(NavSatFix, 'phone/fix', 10)
        self.pub_depth = self.create_publisher(Image, 'phone/depth/image', sensor_qos)
        self.pub_depth_info = self.create_publisher(CameraInfo, 'phone/depth/camera_info', sensor_qos)
        self.pub_pose = self.create_publisher(PoseStamped, 'phone/ar_pose', 10)
        self.pub_twist = self.create_publisher(TwistWithCovarianceStamped, 'phone/ar_twist', 10)
        self.pub_odom = self.create_publisher(Odometry, 'phone/odom', 10)
        self.pub_tag = self.create_publisher(PointStamped, 'weed_tags', latched)
        self.pub_tag_json = self.create_publisher(String, 'weed_tags/json', latched)
        self.pub_status = self.create_publisher(String, 'phone/status', 10)
        self.pub_estop = self.create_publisher(Bool, 'phone/e_stop', latched1)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.static_tf = tf2_ros.StaticTransformBroadcaster(self)
        if bool(p('publish_optical_tf')):
            self._publish_optical_tf()

        self.create_service(Trigger, 'phone/anchor_here', self._srv_anchor)
        self.create_service(Trigger, 'phone/anchor_clear', self._srv_anchor_clear)
        self.add_on_set_parameters_callback(self._on_params)

        self.lock = threading.Lock()          # around publishes from the web thread
        self.outbox = deque()                 # JSON strings to send to the phone
        self.offset = OffsetEstimator()
        self.calib_info = self._load_camera_info(p('camera_info_url'))
        self.connected = False
        self.app = None                       # 'web' or 'ios'
        self.sid = None                       # ARKit session id of the connected app
        self.mode = 'mounted'
        self.last_gps = None
        self.last_status = {}
        self.track = 'unknown'
        self.last_intrinsics = None           # (fx, fy, cx, cy, res_w, res_h, sent_w, sent_h)
        self.pose_hist = deque()              # (t_sec, T_ar_opt) for the twist
        self.last_pose = None                 # (t_sec, T_ar_opt, stamp)
        self.T_base_opt = None                # base_link -> phone_optical from the URDF (cached)
        self._tbo_next_try = 0.0
        self.anchor = None                    # T_map_ar
        self.anchor_sid = None
        self.counts = {'frames': 0, 'imu': 0, 'gps': 0, 'depth': 0, 'pose': 0, 'tags': 0}
        self._rate_ref = (time.time(), dict(self.counts))
        self._estop = None
        self.create_timer(1.0, self._publish_status)

    # ------------------------------------------------------------------ helpers
    def _now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _stamp(self, phone_ms):
        now = self._now_s()
        if phone_ms is None:
            return self.get_clock().now().to_msg()
        try:
            off = self.offset.update(now, float(phone_ms) / 1000.0)
            t = float(phone_ms) / 1000.0 + off
        except (TypeError, ValueError):
            t = now
        return rclpy.time.Time(seconds=t).to_msg()

    def _stamp_s(self, stamp):
        return stamp.sec + stamp.nanosec * 1e-9

    def _send(self, obj):
        self.outbox.append(json.dumps(obj))

    def _publish_optical_tf(self):
        tr = TransformStamped()
        tr.header.stamp = self.get_clock().now().to_msg()
        tr.header.frame_id = self.frame_id
        tr.child_frame_id = self.optical_frame
        T = np.eye(4)
        T[:3, :3] = R_LINK_OPT
        fill_transform(tr.transform, T)
        self.static_tf.sendTransform(tr)

    def _load_camera_info(self, url):
        if not url:
            return None
        path = url[7:] if url.startswith('file://') else url
        try:
            import yaml
            with open(os.path.expanduser(path)) as f:
                c = yaml.safe_load(f)
            info = CameraInfo()
            info.width = int(c['image_width'])
            info.height = int(c['image_height'])
            info.distortion_model = c.get('distortion_model', 'plumb_bob')
            info.d = [float(x) for x in c['distortion_coefficients']['data']]
            info.k = [float(x) for x in c['camera_matrix']['data']]
            info.r = [float(x) for x in c['rectification_matrix']['data']]
            info.p = [float(x) for x in c['projection_matrix']['data']]
            self.get_logger().info(f'Loaded camera calibration {path} ({info.width}x{info.height})')
            return info
        except Exception as e:  # noqa
            self.get_logger().warn(f'Could not load camera_info_url {url}: {e}')
            return None

    @staticmethod
    def _camera_info(w, h, fx, fy, cx, cy, header):
        info = CameraInfo()
        info.header = header
        info.width, info.height = int(w), int(h)
        info.distortion_model = 'plumb_bob'
        info.d = [0.0] * 5
        info.k = [float(fx), 0.0, float(cx), 0.0, float(fy), float(cy), 0.0, 0.0, 1.0]
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.p = [float(fx), 0.0, float(cx), 0.0, 0.0, float(fy), float(cy), 0.0, 0.0, 0.0, 1.0, 0.0]
        return info

    def _lookup(self, target, source, stamp=None, timeout_s=0.2):
        """4x4 of `source` in `target` at `stamp` (latest if None); waits up to timeout_s, then
        falls back to the latest transform. None if the TF does not exist."""
        t = rclpy.time.Time() if stamp is None else rclpy.time.Time.from_msg(stamp)
        deadline = time.time() + timeout_s
        while True:
            try:
                return mat_from_transform(self.tf_buffer.lookup_transform(target, source, t).transform)
            except Exception as e:  # LookupException / ExtrapolationException / ConnectivityException
                if time.time() >= deadline:
                    if stamp is not None:
                        try:
                            return mat_from_transform(self.tf_buffer.lookup_transform(target, source, rclpy.time.Time()).transform)
                        except Exception:
                            return None
                    self.get_logger().warn(f'TF {target} <- {source}: {e}', throttle_duration_sec=5.0)
                    return None
                time.sleep(0.02)

    def _base_opt(self):
        if self.T_base_opt is None and time.time() >= self._tbo_next_try:
            self.T_base_opt = self._lookup(self.base_frame, self.optical_frame, None, 0.05)
            self._tbo_next_try = time.time() + 3.0
            if self.T_base_opt is None:
                self.get_logger().warn(f'No TF {self.base_frame} -> {self.optical_frame}: add the phone mount to the URDF '
                                       '(see phone_mount.xacro). ARKit velocity is not published until then.',
                                       throttle_duration_sec=10.0)
        return self.T_base_opt

    def _on_params(self, params):
        if any(prm.name in ('image_hz', 'depth_hz', 'pose_hz', 'jpeg_quality', 'image_width') for prm in params):
            self._send({'t': 'set', **self._set_dict(params)})
        return SetParametersResult(successful=True)

    def _set_dict(self, params=None):
        vals = {n: self.get_parameter(n).value for n in ('image_hz', 'depth_hz', 'pose_hz', 'jpeg_quality', 'image_width')}
        if params:
            for prm in params:
                if prm.name in vals:
                    vals[prm.name] = prm.value
        return {'imageHz': float(vals['image_hz']), 'depthHz': float(vals['depth_hz']), 'poseHz': float(vals['pose_hz']),
                'quality': float(vals['jpeg_quality']), 'width': float(vals['image_width'])}

    # ------------------------------------------------------------------ status / e-stop (timer, executor thread)
    def _publish_status(self):
        t = time.time()
        t0, c0 = self._rate_ref
        dt = max(t - t0, 1e-3)
        rates = {k: round((self.counts[k] - c0[k]) / dt, 1) for k in self.counts}
        self._rate_ref = (t, dict(self.counts))
        lat = self.offset.latency_ms()
        st = self.last_status
        msg = {'connected': self.connected, 'app': self.app, 'mode': self.mode, 'sid': self.sid,
               'rates_hz': rates, 'latency_jitter_ms': None if lat is None else round(lat, 1),
               'track': self.track, 'thermal': st.get('thermal'), 'battery': st.get('battery'),
               'charging': st.get('charging'), 'lidar': st.get('lidar'),
               'anchored': self.anchor is not None and self.anchor_sid == self.sid,
               'last_gps_accuracy_m': None if self.last_gps is None else self.last_gps.get('acc')}
        estop = False
        reason = ''
        if self.estop_thermal and st.get('thermal') == 'critical':
            estop, reason = True, 'phone thermal critical'
        if self.estop_disconnect and not self.connected:
            estop, reason = True, 'phone disconnected'
        msg['e_stop'] = estop
        msg['e_stop_reason'] = reason
        with self.lock:
            self.pub_status.publish(String(data=json.dumps(msg)))
            if estop != self._estop:
                self._estop = estop
                self.pub_estop.publish(Bool(data=estop))
                if estop:
                    self.get_logger().warn(f'/phone/e_stop = true: {reason}')

    # ------------------------------------------------------------------ message handlers (web thread)
    def on_connect(self, remote):
        self.connected = True
        self.offset.reset()
        self.get_logger().info(f'WebSocket from {remote}')

    def on_disconnect(self):
        self.connected = False
        self.app = None
        self.pose_hist.clear()
        self.get_logger().info('WebSocket closed')

    def on_binary(self, data: bytes):
        """Returns a dict to send back to the phone, or None."""
        if len(data) < 10:
            return None
        kind = data[0]
        phone_ms = struct.unpack('>d', data[1:9])[0]
        body = data[9:]
        if kind == 1:
            self._on_image(body, phone_ms)
        elif kind == 2:
            self._save_photo(body, phone_ms)
        elif kind in (3, 4):
            if len(body) < 4:
                return None
            n = struct.unpack('>I', body[:4])[0]
            if len(body) < 4 + n:
                return None
            try:
                hdr = json.loads(body[4:4 + n].decode('utf-8'))
            except (ValueError, UnicodeDecodeError) as e:
                self.get_logger().warn(f'bad header in kind {kind}: {e}')
                return None
            payload = body[4 + n:]
            if kind == 3:
                self._on_depth(hdr, payload, phone_ms)
            else:
                return self._on_tag(hdr, payload, phone_ms)
        return None

    def on_json(self, m: dict):
        """Returns a dict to send back to the phone, or None."""
        t = m.get('t')
        if t == 'imu':
            self._on_imu(m)
        elif t == 'pose':
            self._on_pose(m)
        elif t == 'gps':
            self._on_gps(m)
        elif t == 'status':
            self._on_status(m)
        elif t == 'hello':
            self.app = m.get('app', 'web')
            self._new_session(m.get('sid'))
            self.mode = m.get('mode', self.mode)
            self.get_logger().info(f"Phone connected: app={self.app} {str(m.get('ua', ''))[:60]} lidar={m.get('lidar')}")
            return {'t': 'set', **self._set_dict()}
        elif t == 'anchor':
            ok, msg = self._anchor_here()
            return {'t': 'ack', 'kind': 'anchor', 'ref': m.get('ts'), 'ok': ok, 'msg': msg}
        elif t == 'stats':
            pass
        return None

    def _new_session(self, sid):
        if sid != self.sid:
            if self.anchor is not None:
                self.get_logger().warn('AR session changed: the map <-> AR anchor is no longer valid, anchor again')
            self.sid = sid
            self.pose_hist.clear()
            self.last_pose = None

    # ---- image / imu / gps (same as v1, plus ARKit intrinsics)
    def _on_image(self, jpeg, phone_ms):
        msg = CompressedImage()
        msg.header.stamp = self._stamp(phone_ms)
        msg.header.frame_id = self.optical_frame
        msg.format = 'jpeg'
        msg.data = jpeg
        info = None
        if self.calib_info is not None:
            info = self.calib_info
            info.header = msg.header
        elif self.last_intrinsics is not None:
            fx, fy, cx, cy, rw, rh, sw, sh = self.last_intrinsics
            sx, sy = sw / rw, sh / rh
            info = self._camera_info(sw, sh, fx * sx, fy * sy, cx * sx, cy * sy, msg.header)
        with self.lock:
            self.pub_img.publish(msg)
            if info is not None:
                self.pub_info.publish(info)
        self.counts['frames'] += 1

    def _on_imu(self, m):
        imu = Imu()
        imu.header.stamp = self._stamp(m.get('ts'))
        imu.header.frame_id = self.frame_id
        imu.orientation_covariance[0] = -1.0
        gs = self.gyro_sign * (1.0 if m.get('gu') == 'rad' else DEG)
        as_ = self.accel_sign
        imu.angular_velocity.x = gs * float(m.get('gx') or 0.0)
        imu.angular_velocity.y = gs * float(m.get('gy') or 0.0)
        imu.angular_velocity.z = gs * float(m.get('gz') or 0.0)
        imu.linear_acceleration.x = as_ * float(m.get('ax') or 0.0)
        imu.linear_acceleration.y = as_ * float(m.get('ay') or 0.0)
        imu.linear_acceleration.z = as_ * float(m.get('az') or 0.0)
        for i in (0, 4, 8):
            imu.angular_velocity_covariance[i] = 4e-5
            imu.linear_acceleration_covariance[i] = 1e-2
        with self.lock:
            self.pub_imu.publish(imu)
        self.counts['imu'] += 1

    def _on_gps(self, m):
        fix = NavSatFix()
        fix.header.stamp = self._stamp(m.get('ts'))
        fix.header.frame_id = self.gps_frame_id
        fix.status.status = NavSatStatus.STATUS_FIX
        fix.status.service = NavSatStatus.SERVICE_GPS
        fix.latitude = float(m.get('lat', 0.0))
        fix.longitude = float(m.get('lon', 0.0))
        fix.altitude = float(m['alt']) if m.get('alt') is not None else float('nan')
        acc = float(m.get('acc') or 10.0)
        alt_acc = float(m.get('altacc') or acc * 2)
        fix.position_covariance = [acc * acc, 0.0, 0.0, 0.0, acc * acc, 0.0, 0.0, 0.0, alt_acc * alt_acc]
        fix.position_covariance_type = NavSatFix.COVARIANCE_TYPE_APPROXIMATED
        with self.lock:
            self.pub_fix.publish(fix)
        self.last_gps = m
        self.counts['gps'] += 1

    def _on_status(self, m):
        self.last_status = m
        self.track = m.get('track', self.track)
        new_mode = m.get('mode', self.mode)
        if new_mode != self.mode:
            self.get_logger().info(f'phone mode: {self.mode} -> {new_mode}')
            if new_mode == 'handheld' and self.mode == 'mounted' and self.auto_anchor:
                ok, msg = self._anchor_here()
                self.get_logger().info(f'auto-anchor on handheld: {msg}')
                self._send({'t': 'ack', 'kind': 'anchor', 'ref': m.get('ts'), 'ok': ok, 'msg': 'auto-anchor: ' + msg})
            self.mode = new_mode
            self.pose_hist.clear()

    # ---- ARKit pose -> /phone/ar_pose, /phone/ar_twist, /phone/odom
    def _on_pose(self, m):
        try:
            T_ar = mat_from_cols16(m['T'])
        except (KeyError, ValueError, TypeError):
            return
        if m.get('sid') != self.sid:
            self._new_session(m.get('sid'))
        self.track = m.get('track', self.track)
        if 'mode' in m:
            self.mode = m['mode']
        try:
            res, sent = m.get('res'), m.get('sent') or m.get('res')
            self.last_intrinsics = (float(m['fx']), float(m['fy']), float(m['cx']), float(m['cy']),
                                    float(res[0]), float(res[1]), float(sent[0]), float(sent[1]))
        except (KeyError, TypeError, ValueError, IndexError):
            pass
        stamp = self._stamp(m.get('ts'))
        t = self._stamp_s(stamp)
        T = ar_pose_to_ros(T_ar)
        self.last_pose = (t, T, stamp)

        ps = PoseStamped()
        ps.header.stamp = stamp
        ps.header.frame_id = self.ar_frame
        fill_pose(ps.pose, T)
        with self.lock:
            self.pub_pose.publish(ps)
        self.counts['pose'] += 1

        if self.track == 'notAvailable':
            self.pose_hist.clear()
            return
        self.pose_hist.append((t, T))
        while len(self.pose_hist) > 2 and t - self.pose_hist[1][0] >= self.twist_window:
            self.pose_hist.popleft()
        if self.mode != 'mounted':
            return
        T_bo = self._base_opt()
        if T_bo is None or len(self.pose_hist) < 2:
            return
        t0, T0 = self.pose_hist[0]
        dt = t - t0
        if dt < 0.5 * self.twist_window or dt > 1.0:
            return
        R, R0 = T[:3, :3], T0[:3, :3]
        v_opt = R.T @ (T[:3, 3] - T0[:3, 3]) / dt
        w_opt = rotvec_from_mat(R0.T @ R) / dt
        R_bo, t_bo = T_bo[:3, :3], T_bo[:3, 3]
        w_base = R_bo @ w_opt
        v_base = R_bo @ v_opt - np.cross(w_base, t_bo)
        sv, sw = self.sig_v, self.sig_w
        if self.track != 'normal':
            sv, sw = sv * 10.0, sw * 10.0

        tw = TwistWithCovarianceStamped()
        tw.header.stamp = stamp
        tw.header.frame_id = self.base_frame
        tw.twist.twist.linear.x, tw.twist.twist.linear.y, tw.twist.twist.linear.z = [float(v) for v in v_base]
        tw.twist.twist.angular.x, tw.twist.twist.angular.y, tw.twist.twist.angular.z = [float(v) for v in w_base]
        cov = [0.0] * 36
        for i in range(3):
            cov[i * 7] = sv * sv
            cov[(i + 3) * 7] = sw * sw
        tw.twist.covariance = cov

        od = Odometry()
        od.header.stamp = stamp
        od.header.frame_id = self.ar_frame
        od.child_frame_id = self.base_frame
        fill_pose(od.pose.pose, T @ inv(T_bo))
        od.twist = tw.twist
        pcov = [0.0] * 36
        for i in range(6):
            pcov[i * 7] = 0.01 if self.track == 'normal' else 1.0
        od.pose.covariance = pcov
        with self.lock:
            self.pub_twist.publish(tw)
            self.pub_odom.publish(od)

    # ---- LiDAR depth
    def _on_depth(self, hdr, blob, phone_ms):
        try:
            w, h = int(hdr['w']), int(hdr['h'])
        except (KeyError, ValueError, TypeError):
            return
        try:
            raw = zlib.decompress(blob, -15)          # raw DEFLATE (Apple's COMPRESSION_ZLIB)
        except zlib.error:
            try:
                raw = zlib.decompress(blob)           # zlib-wrapped, just in case
            except zlib.error as e:
                self.get_logger().warn(f'depth inflate failed: {e}', throttle_duration_sec=5.0)
                return
        if len(raw) != w * h * 2:
            self.get_logger().warn(f'depth size mismatch: {len(raw)} bytes for {w}x{h}', throttle_duration_sec=5.0)
            return
        img = Image()
        img.header.stamp = self._stamp(phone_ms)
        img.header.frame_id = self.optical_frame
        img.height, img.width = h, w
        img.encoding = '16UC1'
        img.is_bigendian = 0
        img.step = w * 2
        img.data = raw
        info = self._camera_info(w, h, hdr.get('fx', 0.0), hdr.get('fy', 0.0), hdr.get('cx', 0.0), hdr.get('cy', 0.0), img.header)
        with self.lock:
            self.pub_depth.publish(img)
            self.pub_depth_info.publish(info)
        self.counts['depth'] += 1

    # ---- weed tags
    def _on_tag(self, hdr, jpeg, phone_ms):
        stamp = self._stamp(phone_ms)
        mode = hdr.get('mode', self.mode)
        sid = hdr.get('sid')
        p_cam = hdr.get('p_cam')
        p_world = hdr.get('p_world')
        p_map, method, reason = None, None, None

        if p_cam is not None and mode == 'mounted':
            p_opt = D_CAM2OPT @ np.array(p_cam, dtype=float)
            T_mo = self._lookup(self.map_frame, self.optical_frame, stamp, 0.3)
            if T_mo is not None:
                p_map = T_mo[:3, :3] @ p_opt + T_mo[:3, 3]
                method = 'tf'
            elif self.anchor is not None and self.anchor_sid == sid and p_world is not None:
                p_map = self.anchor[:3, :3] @ (C_AR2ROS @ np.array(p_world, dtype=float)) + self.anchor[:3, 3]
                method = 'anchor'
            else:
                reason = f'no TF {self.map_frame} <- {self.optical_frame} (robot not localized, or phone mount missing in URDF)'
        elif p_world is not None and mode == 'handheld':
            if self.anchor is not None and self.anchor_sid == sid:
                p_map = self.anchor[:3, :3] @ (C_AR2ROS @ np.array(p_world, dtype=float)) + self.anchor[:3, 3]
                method = 'anchor'
            else:
                reason = 'no anchor for this AR session: mount the phone on the robot and press Anchor, then walk'
        else:
            reason = 'no 3D point at the tap (no LiDAR depth and no ground plane): get closer or tap the ground next to it'

        name = self._unique_name(self.tag_dir, phone_ms)
        jpg_path = os.path.join(self.tag_dir, name + '.jpg')
        rec = {'name': name, 'stamp': self._stamp_s(stamp), 'phone_ms': phone_ms, 'mode': mode, 'sid': sid,
               'source': hdr.get('source'), 'track': hdr.get('track'), 'note': hdr.get('note', ''),
               'px': hdr.get('px'), 'py': hdr.get('py'), 'w': hdr.get('w'), 'h': hdr.get('h'),
               'fx': hdr.get('fx'), 'fy': hdr.get('fy'), 'cx': hdr.get('cx'), 'cy': hdr.get('cy'),
               'depth_m': hdr.get('depth_m'), 'p_cam': p_cam, 'p_world': p_world, 'T_ar': hdr.get('T'),
               'gps': {k: hdr.get(k) for k in ('lat', 'lon', 'acc')} if hdr.get('lat') is not None else None,
               'map_frame': self.map_frame,
               'map_xyz': None if p_map is None else [float(v) for v in p_map],
               'method': method, 'reason': reason, 'image': jpg_path, 'status': 'pending', 'kills': []}
        try:
            with open(jpg_path, 'wb') as f:
                f.write(jpeg)
            with open(os.path.join(self.tag_dir, name + '.json'), 'w') as f:
                json.dump(rec, f, indent=1)
            with open(os.path.join(self.tag_dir, 'index.jsonl'), 'a') as f:
                f.write(json.dumps(rec) + '\n')
        except OSError as e:
            self.get_logger().error(f'could not save tag: {e}')

        with self.lock:
            self.pub_tag_json.publish(String(data=json.dumps(rec)))
            if p_map is not None:
                pt = PointStamped()
                pt.header.stamp = stamp
                pt.header.frame_id = self.map_frame
                pt.point.x, pt.point.y, pt.point.z = [float(v) for v in p_map]
                self.pub_tag.publish(pt)
        self.counts['tags'] += 1
        if p_map is not None:
            msg = f'weed at map ({p_map[0]:.2f}, {p_map[1]:.2f}) via {method}'
            self.get_logger().info(f'tag {name}: {msg}')
            return {'t': 'ack', 'kind': 'tag', 'ref': phone_ms, 'ok': True, 'msg': msg}
        self.get_logger().warn(f'tag {name} saved without map position: {reason}')
        return {'t': 'ack', 'kind': 'tag', 'ref': phone_ms, 'ok': False, 'msg': reason}

    # ---- anchor
    def _anchor_here(self):
        if self.last_pose is None:
            return False, 'no AR pose from the phone yet'
        t, T_ar_opt, stamp = self.last_pose
        if self._now_s() - t > 1.0:
            return False, 'AR pose is stale (is the app streaming?)'
        if self.track != 'normal':
            return False, f'AR tracking is {self.track}: hold still until it is normal'
        T_mo = self._lookup(self.map_frame, self.optical_frame, stamp, 0.3)
        if T_mo is None:
            if self.fallback_identity_anchor:
                T_mo = T_ar_opt.copy()
                self.get_logger().warn('anchor: no robot pose, using identity map <- ar_origin (fallback_identity_anchor)')
            else:
                return False, f'no TF {self.map_frame} <- {self.optical_frame}: robot must be localized and the phone mounted'
        self.anchor = T_mo @ inv(T_ar_opt)
        self.anchor_sid = self.sid
        tr = TransformStamped()
        tr.header.stamp = self.get_clock().now().to_msg()
        tr.header.frame_id = self.map_frame
        tr.child_frame_id = self.ar_frame
        fill_transform(tr.transform, self.anchor)
        self.static_tf.sendTransform(tr)
        yaw = math.degrees(math.atan2(self.anchor[1, 0], self.anchor[0, 0]))
        msg = f'anchored: ar_origin at map ({self.anchor[0, 3]:.2f}, {self.anchor[1, 3]:.2f}), yaw {yaw:.0f} deg'
        self.get_logger().info(msg)
        return True, msg

    def _srv_anchor(self, req, res):
        res.success, res.message = self._anchor_here()
        return res

    def _srv_anchor_clear(self, req, res):
        self.anchor, self.anchor_sid = None, None
        res.success, res.message = True, 'anchor cleared'
        return res

    @staticmethod
    def _unique_name(folder, phone_ms):
        base = time.strftime('%Y%m%d_%H%M%S') + f'_{int(phone_ms) % 1000:03d}'
        name, n = base, 1
        while os.path.exists(os.path.join(folder, name + '.jpg')):
            name = f'{base}_{n}'
            n += 1
        return name

    def _save_photo(self, jpeg, phone_ms):
        name = self._unique_name(self.save_dir, phone_ms)
        path = os.path.join(self.save_dir, name + '.jpg')
        with open(path, 'wb') as f:
            f.write(jpeg)
        side = {'phone_ms': phone_ms, 'gps': self.last_gps, 'mode': self.mode, 'sid': self.sid,
                'ar_pose_T': None if self.last_pose is None else [float(v) for v in self.last_pose[1].flatten(order='F')]}
        with open(os.path.join(self.save_dir, name + '.json'), 'w') as f:
            json.dump(side, f)
        self.get_logger().info(f'Saved photo {path}')


# ---------------------------------------------------------------------------- web server
def make_app(node: PhoneBridge, www_dir: str, with_page: bool):
    index = os.path.join(www_dir, 'index.html')

    async def page(request):
        return web.FileResponse(index)

    async def ws_handler(request):
        ws = web.WebSocketResponse(max_msg_size=16 * 1024 * 1024, heartbeat=10)
        await ws.prepare(request)
        node.on_connect(request.remote)

        async def sender():
            try:
                while not ws.closed:
                    while node.outbox:
                        await ws.send_str(node.outbox.popleft())
                    await asyncio.sleep(0.05)
            except (ConnectionError, asyncio.CancelledError):
                pass

        task = asyncio.ensure_future(sender())
        try:
            async for msg in ws:
                reply = None
                if msg.type == web.WSMsgType.BINARY:
                    reply = node.on_binary(msg.data)
                elif msg.type == web.WSMsgType.TEXT:
                    try:
                        reply = node.on_json(json.loads(msg.data))
                    except (ValueError, TypeError) as e:
                        node.get_logger().warn(f'bad message: {e}')
                elif msg.type == web.WSMsgType.ERROR:
                    node.get_logger().warn(f'ws error: {ws.exception()}')
                if reply is not None:
                    await ws.send_str(json.dumps(reply))
        finally:
            task.cancel()
            node.on_disconnect()
        return ws

    app = web.Application()
    if with_page:
        app.router.add_get('/', page)
        app.router.add_get('/index.html', page)
    app.router.add_get('/ws', ws_handler)
    return app


def make_ca_app(cert_dir: str, https_port: int):
    ca = os.path.join(cert_dir, 'ca.crt')

    async def ca_file(request):
        return web.FileResponse(ca, headers={'Content-Type': 'application/x-x509-ca-cert',
                                             'Content-Disposition': 'attachment; filename="weedbot-ca.crt"'})

    async def help_page(request):
        host = request.host.split(':')[0]
        return web.Response(content_type='text/html', text=f"""<html><body style="font-family:system-ui;padding:20px">
<h2>WeedBot phone setup</h2>
<p>Using the WeedBotSensor app? Nothing to install: enter {host} in its settings.</p>
<p>Using Safari instead:</p>
<ol>
<li><a href="/ca.crt">Download the certificate</a> (tap Allow, then Close).</li>
<li>Settings &rarr; General &rarr; VPN &amp; Device Management &rarr; <b>WeedBot Robot CA</b> &rarr; Install.</li>
<li>Settings &rarr; General &rarr; About &rarr; Certificate Trust Settings &rarr; turn on <b>WeedBot Robot CA</b>.</li>
<li>Open <a href="https://{host}:{https_port}/">https://{host}:{https_port}/</a> and tap Start.</li>
</ol></body></html>""")

    app = web.Application()
    app.router.add_get('/', help_page)
    app.router.add_get('/ca.crt', ca_file)
    return app


async def serve(node: PhoneBridge):
    p = lambda n: node.get_parameter(n).value
    cert_dir = os.path.expanduser(p('cert_dir'))
    https_port, ws_port, http_port = int(p('https_port')), int(p('ws_port')), int(p('http_port'))
    runners = []

    if ws_port > 0:
        r = web.AppRunner(make_app(node, p('www_dir'), with_page=False))
        await r.setup()
        await web.TCPSite(r, '0.0.0.0', ws_port).start()
        runners.append(r)
        node.get_logger().info(f'App websocket: ws://<this-ip>:{ws_port}/ws')

    crt, key = os.path.join(cert_dir, 'server.crt'), os.path.join(cert_dir, 'server.key')
    if https_port > 0 and os.path.exists(crt) and os.path.exists(key):
        ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_ctx.load_cert_chain(crt, key)
        r = web.AppRunner(make_app(node, p('www_dir'), with_page=True))
        await r.setup()
        await web.TCPSite(r, '0.0.0.0', https_port, ssl_context=ssl_ctx).start()
        runners.append(r)
        node.get_logger().info(f'Phone web page: https://<this-ip>:{https_port}/')
        if http_port > 0:
            r = web.AppRunner(make_ca_app(cert_dir, https_port))
            await r.setup()
            await web.TCPSite(r, '0.0.0.0', http_port).start()
            runners.append(r)
            node.get_logger().info(f'Certificate install page: http://<this-ip>:{http_port}/')
    elif https_port > 0:
        node.get_logger().warn(f'No certificate in {cert_dir}: the Safari page is off (run make_cert.sh); the app still works.')

    if not runners:
        node.get_logger().error('nothing to serve: set ws_port and/or https_port')
        return
    while rclpy.ok():
        await asyncio.sleep(0.5)
    for r in runners:
        await r.cleanup()


def main():
    if web is None:
        raise SystemExit('aiohttp is not installed: pip3 install aiohttp')
    rclpy.init()
    node = PhoneBridge()
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()
    try:
        asyncio.run(serve(node))
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
