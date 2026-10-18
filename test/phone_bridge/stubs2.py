"""rclpy / message / tf2 stubs for testing phone_bridge_node.py v2 without ROS installed.
Run the tests with:  python3 tests/test_bridge_v2.py   (from the weedbot_ios folder)
"""
import sys, types, time as _time, math


class _Msg:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


class _V3:
    def __init__(self): self.x = self.y = self.z = 0.0


class _Q:
    def __init__(self): self.x = self.y = self.z = 0.0; self.w = 1.0


class _Hdr:
    def __init__(self): self.stamp = None; self.frame_id = ''


class Bool(_Msg): data = False
class Float32(_Msg): data = 0.0
class String(_Msg): data = ''


class Imu:
    def __init__(self):
        self.header = _Hdr(); self.orientation_covariance = [0.0] * 9; self.angular_velocity = _V3()
        self.linear_acceleration = _V3(); self.angular_velocity_covariance = [0.0] * 9; self.linear_acceleration_covariance = [0.0] * 9


class NavSatStatus: STATUS_FIX = 0; SERVICE_GPS = 1


class NavSatFix:
    COVARIANCE_TYPE_APPROXIMATED = 1
    def __init__(self):
        self.header = _Hdr(); self.status = _Msg(status=0, service=0); self.latitude = self.longitude = self.altitude = 0.0
        self.position_covariance = [0.0] * 9; self.position_covariance_type = 0


class CompressedImage:
    def __init__(self): self.header = _Hdr(); self.format = ''; self.data = b''


class Image:
    def __init__(self):
        self.header = _Hdr(); self.height = self.width = 0; self.encoding = ''; self.is_bigendian = 0; self.step = 0; self.data = b''


class CameraInfo:
    def __init__(self):
        self.header = _Hdr(); self.width = self.height = 0; self.distortion_model = ''; self.d = []; self.k = []; self.r = []; self.p = []


class Pose:
    def __init__(self): self.position = _V3(); self.orientation = _Q()


class PoseStamped:
    def __init__(self): self.header = _Hdr(); self.pose = Pose()


class Twist:
    def __init__(self): self.linear = _V3(); self.angular = _V3()


class TwistWithCovariance:
    def __init__(self): self.twist = Twist(); self.covariance = [0.0] * 36


class TwistWithCovarianceStamped:
    def __init__(self): self.header = _Hdr(); self.twist = TwistWithCovariance()


class PoseWithCovariance:
    def __init__(self): self.pose = Pose(); self.covariance = [0.0] * 36


class PoseWithCovarianceStamped:
    def __init__(self): self.header = _Hdr(); self.pose = PoseWithCovariance()


class PointStamped:
    def __init__(self): self.header = _Hdr(); self.point = _V3()


class Transform:
    def __init__(self): self.translation = _V3(); self.rotation = _Q()


class TransformStamped:
    def __init__(self): self.header = _Hdr(); self.child_frame_id = ''; self.transform = Transform()


class Odometry:
    def __init__(self):
        self.header = _Hdr(); self.child_frame_id = ''; self.pose = PoseWithCovariance(); self.twist = TwistWithCovariance()


class OccupancyGrid:
    def __init__(self):
        self.header = _Hdr(); self.info = _Msg(resolution=0.0, width=0, height=0, origin=Pose()); self.data = []


class Trigger:
    class Request: pass
    class Response:
        def __init__(self): self.success = False; self.message = ''


class SetParametersResult(_Msg): pass


class Pub:
    def __init__(self, topic): self.topic = topic; self.msgs = []
    def publish(self, m): self.msgs.append(m)


class _Logger:
    def info(self, m, **kw): print('[info]', m)
    def warn(self, m, **kw): print('[warn]', m)
    def error(self, m, **kw): print('[ERROR]', m)


class _Param:
    def __init__(self, v): self.value = v


class TimeStub:
    def __init__(self, seconds=0.0, nanoseconds=None):
        self.nanoseconds = int(seconds * 1e9) if nanoseconds is None else nanoseconds
    def to_msg(self): return _Msg(sec=int(self.nanoseconds // 1e9), nanosec=int(self.nanoseconds % 1e9))
    def __sub__(self, o): return TimeStub(nanoseconds=self.nanoseconds - o.nanoseconds)
    @staticmethod
    def from_msg(m): return TimeStub(seconds=m.sec + m.nanosec * 1e-9)


class DurationStub:
    def __init__(self, seconds=0.0): self.nanoseconds = int(seconds * 1e9)


class _Clock:
    def now(self): return TimeStub(seconds=_time.time())


class Node:
    PARAM_OVERRIDES = {}
    def __init__(self, name):
        self.name = name; self.params = {}; self.pubs = {}; self.timers = []; self.services = {}; self.param_cbs = []
    def declare_parameter(self, n, v): self.params[n] = Node.PARAM_OVERRIDES.get(n, v)
    def get_parameter(self, n): return _Param(self.params[n])
    def create_publisher(self, t, topic, qos): p = Pub(topic); self.pubs[topic] = p; return p
    def create_subscription(self, *a, **k): return None
    def create_service(self, t, name, cb): self.services[name] = cb; return None
    def create_timer(self, period, cb): self.timers.append(cb); return None
    def add_on_set_parameters_callback(self, cb): self.param_cbs.append(cb)
    def get_logger(self): return _Logger()
    def get_clock(self): return _Clock()
    def destroy_node(self): pass


class TfException(Exception): pass


class Buffer:
    """Test double: tests fill `frames[(target, source)] = TransformStamped`."""
    def __init__(self, *a, **k): self.frames = {}
    def lookup_transform(self, target, source, t, timeout=None):
        if (target, source) in self.frames:
            return self.frames[(target, source)]
        raise TfException(f'no {target} <- {source}')


class TransformListener:
    def __init__(self, b, n): pass


class StaticTransformBroadcaster:
    def __init__(self, node): self.sent = []
    def sendTransform(self, tr): self.sent.append(tr)


def install():
    rclpy = types.ModuleType('rclpy')
    rclpy.ok = lambda: True; rclpy.init = lambda *a, **k: None; rclpy.shutdown = lambda: None; rclpy.spin = lambda n: None
    node_mod = types.ModuleType('rclpy.node'); node_mod.Node = Node
    qos = types.ModuleType('rclpy.qos')
    for n in ['QoSProfile', 'ReliabilityPolicy', 'HistoryPolicy', 'DurabilityPolicy']:
        setattr(qos, n, type(n, (), {'__init__': lambda s, **k: None, 'BEST_EFFORT': 1, 'RELIABLE': 2, 'KEEP_LAST': 3, 'TRANSIENT_LOCAL': 4}))
    rtime = types.ModuleType('rclpy.time'); rtime.Time = TimeStub; rtime.Duration = DurationStub
    rdur = types.ModuleType('rclpy.duration'); rdur.Duration = DurationStub
    rclpy.time = rtime; rclpy.duration = rdur
    mods = {'rclpy': rclpy, 'rclpy.node': node_mod, 'rclpy.qos': qos, 'rclpy.time': rtime, 'rclpy.duration': rdur}
    sm = types.ModuleType('sensor_msgs.msg')
    for c in (Imu, NavSatFix, NavSatStatus, CompressedImage, CameraInfo, Image): setattr(sm, c.__name__, c)
    mods['sensor_msgs'] = types.ModuleType('sensor_msgs'); mods['sensor_msgs.msg'] = sm
    stdm = types.ModuleType('std_msgs.msg'); stdm.Bool = Bool; stdm.Float32 = Float32; stdm.String = String
    mods['std_msgs'] = types.ModuleType('std_msgs'); mods['std_msgs.msg'] = stdm
    gm = types.ModuleType('geometry_msgs.msg')
    for c in (PoseStamped, TwistWithCovarianceStamped, PointStamped, TransformStamped, PoseWithCovarianceStamped, Pose, Twist, Transform):
        setattr(gm, c.__name__, c)
    mods['geometry_msgs'] = types.ModuleType('geometry_msgs'); mods['geometry_msgs.msg'] = gm
    nm = types.ModuleType('nav_msgs.msg'); nm.OccupancyGrid = OccupancyGrid; nm.Odometry = Odometry
    mods['nav_msgs'] = types.ModuleType('nav_msgs'); mods['nav_msgs.msg'] = nm
    ss = types.ModuleType('std_srvs.srv'); ss.Trigger = Trigger
    mods['std_srvs'] = types.ModuleType('std_srvs'); mods['std_srvs.srv'] = ss
    ri = types.ModuleType('rcl_interfaces.msg'); ri.SetParametersResult = SetParametersResult
    mods['rcl_interfaces'] = types.ModuleType('rcl_interfaces'); mods['rcl_interfaces.msg'] = ri
    tf = types.ModuleType('tf2_ros'); tf.Buffer = Buffer; tf.TransformListener = TransformListener
    tf.StaticTransformBroadcaster = StaticTransformBroadcaster; mods['tf2_ros'] = tf
    aio = types.ModuleType('aiohttp'); aio.web = object(); mods['aiohttp'] = aio
    sys.modules.update(mods)


def make_tf(frame, child, x=0.0, y=0.0, z=0.0, qx=0.0, qy=0.0, qz=0.0, qw=1.0):
    tr = TransformStamped()
    tr.header.frame_id = frame; tr.child_frame_id = child
    tr.transform.translation.x, tr.transform.translation.y, tr.transform.translation.z = x, y, z
    tr.transform.rotation.x, tr.transform.rotation.y, tr.transform.rotation.z, tr.transform.rotation.w = qx, qy, qz, qw
    return tr


def quat_rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r / 2), math.sin(r / 2), math.cos(p / 2), math.sin(p / 2), math.cos(y / 2), math.sin(y / 2)
    return (sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy)
