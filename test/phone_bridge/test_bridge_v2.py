#!/usr/bin/env python3
"""Exercises phone_bridge_node.py v2 handlers without ROS (stubbed rclpy/tf2). No network, no aiohttp.
Run from the weedbot_ios folder:  python3 tests/test_bridge_v2.py
In my_bot:                        python3 test/phone_bridge/test_bridge_v2.py  (tests scripts/phone_bridge_node.py)
"""
import importlib.util, json, math, os, struct, sys, tempfile, time, zlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import stubs2
stubs2.install()

tmp = tempfile.mkdtemp(prefix='wb_')
stubs2.Node.PARAM_OVERRIDES = {'save_dir': tmp + '/photos', 'tag_dir': tmp + '/tags', 'twist_window_s': 0.1}
# my_bot layout (test/phone_bridge -> scripts/) first, else the weedbot_ios layout (tests -> bridge/)
NODE = os.path.join(HERE, '..', '..', 'scripts', 'phone_bridge_node.py')
if not os.path.exists(NODE):
    NODE = os.path.join(HERE, '..', 'bridge', 'phone_bridge_node.py')
spec = importlib.util.spec_from_file_location('pb', NODE)
pb = importlib.util.module_from_spec(spec); spec.loader.exec_module(pb)

node = pb.PhoneBridge()
now_ms = lambda: time.time() * 1000.0
# The test feeds messages faster than real time; the jitter-removing clock offset (tested in
# test_bridge2.py for v1) would collapse their stamps, so use the phone time as-is here.
node.offset.update = lambda arrival, sent: 0.0

# ---------------------------------------------------------------- static TF phone_link -> phone_optical
tr = node.static_tf.sent[0]
q = (tr.transform.rotation.x, tr.transform.rotation.y, tr.transform.rotation.z, tr.transform.rotation.w)
assert tr.header.frame_id == 'phone_link' and tr.child_frame_id == 'phone_optical'
assert np.allclose(np.abs(q), [0.7071, 0.7071, 0, 0], atol=1e-3) and q[0] * q[1] < 0, q
R = pb.mat_from_quat(*q)
assert np.allclose(R, pb.R_LINK_OPT, atol=1e-6)
# the same rotation as rpy (pi, 0, -pi/2), what phone_mount.xacro uses
assert np.allclose(pb.mat_from_quat(*stubs2.quat_rpy(math.pi, 0, -math.pi / 2)), pb.R_LINK_OPT, atol=1e-6)
print('static TF phone_link -> phone_optical: ok (rpy pi 0 -pi/2)')

# ---------------------------------------------------------------- hello -> set
node.on_connect('10.0.0.5')
reply = node.on_json({'t': 'hello', 'ts': now_ms(), 'app': 'ios', 'sid': 'S1', 'ua': 'iPhone iOS 18', 'lidar': True, 'mode': 'mounted'})
assert reply['t'] == 'set' and reply['imageHz'] == 10.0 and reply['width'] == 640.0, reply
assert node.sid == 'S1' and node.app == 'ios'
print('hello -> set:', reply)

# ---------------------------------------------------------------- imu in rad (app) and deg (web page)
node.on_json({'t': 'imu', 'ts': now_ms(), 'gu': 'rad', 'gx': 0, 'gy': 0, 'gz': 1.0, 'ax': 0, 'ay': 0, 'az': 9.81})
node.on_json({'t': 'imu', 'ts': now_ms(), 'gx': 0, 'gy': 0, 'gz': 57.2958, 'ax': 0, 'ay': 0, 'az': 9.81})
imu = node.pubs['phone/imu/data_raw'].msgs
assert abs(imu[0].angular_velocity.z - 1.0) < 1e-6 and abs(imu[1].angular_velocity.z - 1.0) < 1e-3
assert imu[0].header.frame_id == 'phone_link'
print('imu rad/deg handling: ok')
node.on_json({'t': 'gps', 'ts': now_ms(), 'lat': 32.28, 'lon': -86.10, 'alt': 70.0, 'acc': 4.5, 'altacc': None})
fix = node.pubs['phone/fix'].msgs[-1]
assert fix.position_covariance[0] == 4.5 ** 2 and fix.latitude == 32.28 and fix.header.frame_id == 'phone_gps'
print('gps -> NavSatFix: ok')

# ---------------------------------------------------------------- ARKit pose -> ar_pose / twist / odom
# mount: phone_optical in base_link = standard forward-looking optical frame, rpy (-pi/2, 0, -pi/2), 0.3 m up
node.tf_buffer.frames[('base_link', 'phone_optical')] = stubs2.make_tf('base_link', 'phone_optical', 0.2, 0.0, 0.3, *stubs2.quat_rpy(-math.pi / 2, 0, -math.pi / 2))

def T_ar(x, y, z, yaw_about_ar_y=0.0):
    c, s = math.cos(yaw_about_ar_y), math.sin(yaw_about_ar_y)
    T = np.eye(4); T[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]; T[:3, 3] = [x, y, z]
    return T

def pose_msg(T, t_ms, sid='S1', track='normal'):
    return {'t': 'pose', 'ts': t_ms, 'sid': sid, 'T': [float(v) for v in T.flatten(order='F')], 'track': track,
            'fx': 1400.0, 'fy': 1400.0, 'cx': 960.0, 'cy': 720.0, 'res': [1920, 1440], 'sent': [640, 480], 'mode': 'mounted'}

# camera drives forward along its viewing direction (ARKit -z) at 0.5 m/s: ROS vx must be +0.5
t0 = now_ms()
for i in range(12):
    node.on_json(pose_msg(T_ar(0, 0, -0.5 * i * 0.033), t0 + i * 33.0))
ps = node.pubs['phone/ar_pose'].msgs[-1]
assert ps.header.frame_id == 'ar_origin'
assert abs(ps.pose.position.y - 0.5 * 11 * 0.033) < 1e-6 and abs(ps.pose.position.z) < 1e-9, 'ARKit -z must become ROS +y'
tw = node.pubs['phone/ar_twist'].msgs[-1]
assert tw.header.frame_id == 'base_link'
assert abs(tw.twist.twist.linear.x - 0.5) < 0.02 and abs(tw.twist.twist.linear.y) < 0.02 and abs(tw.twist.twist.angular.z) < 0.02, \
    (tw.twist.twist.linear.x, tw.twist.twist.linear.y, tw.twist.twist.angular.z)
assert tw.twist.covariance[0] == 0.03 ** 2
print('forward drive: ar_pose y=%.3f, twist vx=%.3f vy=%.3f wz=%.3f: ok' % (ps.pose.position.y, tw.twist.twist.linear.x, tw.twist.twist.linear.y, tw.twist.twist.angular.z))

# turning left (counterclockwise seen from above = about ARKit +y) at 0.3 rad/s: ROS wz must be +0.3
node.pose_hist.clear()
t0 = now_ms() + 5000
for i in range(12):
    node.on_json(pose_msg(T_ar(0, 0, 0, 0.3 * i * 0.033), t0 + i * 33.0))
tw = node.pubs['phone/ar_twist'].msgs[-1]
assert abs(tw.twist.twist.angular.z - 0.3) < 0.02, tw.twist.twist.angular.z
# the base origin is 0.2 m behind the phone, so turning in place still gives a small vy, never a big vx
assert abs(tw.twist.twist.linear.x) < 0.02, tw.twist.twist.linear.x
od = node.pubs['phone/odom'].msgs[-1]
assert od.child_frame_id == 'base_link' and od.header.frame_id == 'ar_origin'
print('turn in place: twist wz=%.3f vx=%.3f: ok' % (tw.twist.twist.angular.z, tw.twist.twist.linear.x))

# camera_info from ARKit intrinsics, scaled to the streamed 640x480
jpeg = b'\xff\xd8' + b'\x00' * 300 + b'\xff\xd9'
node.on_binary(bytes([1]) + struct.pack('>d', now_ms()) + jpeg)
info = node.pubs['phone/camera_info'].msgs[-1]
assert info.width == 640 and abs(info.k[0] - 1400 / 3) < 1e-6 and abs(info.k[2] - 320) < 1e-6
assert node.pubs['phone/image/compressed'].msgs[-1].header.frame_id == 'phone_optical'
print('camera_info from ARKit intrinsics: fx=%.1f cx=%.1f for %dx%d: ok' % (info.k[0], info.k[2], info.width, info.height))

# no twist while handheld
node.on_json({'t': 'status', 'ts': now_ms(), 'thermal': 'nominal', 'battery': 0.8, 'charging': False, 'mode': 'handheld', 'track': 'normal'})
n_tw = len(node.pubs['phone/ar_twist'].msgs)
for i in range(6):
    m = pose_msg(T_ar(0.1 * i, 0, 0), now_ms() + 9000 + i * 33.0); m['mode'] = 'handheld'; node.on_json(m)
assert len(node.pubs['phone/ar_twist'].msgs) == n_tw, 'twist must not be published while handheld'
print('no twist while handheld: ok')
node.on_json({'t': 'status', 'ts': now_ms(), 'thermal': 'nominal', 'battery': 0.8, 'charging': False, 'mode': 'mounted', 'track': 'normal'})

# ---------------------------------------------------------------- LiDAR depth
w, h = 6, 4
depth = np.arange(w * h, dtype='<u2') * 100       # 0 .. 2300 mm
blob = zlib.compressobj(level=6, wbits=-15); comp = blob.compress(depth.tobytes()) + blob.flush()
hdr = json.dumps({'w': w, 'h': h, 'unit': 'mm', 'fx': 200.0, 'fy': 200.0, 'cx': 3.0, 'cy': 2.0, 'min_conf': 1}).encode()
node.on_binary(bytes([3]) + struct.pack('>d', now_ms()) + struct.pack('>I', len(hdr)) + hdr + comp)
di = node.pubs['phone/depth/image'].msgs[-1]
assert di.encoding == '16UC1' and di.width == w and di.height == h and di.step == w * 2 and di.is_bigendian == 0
back = np.frombuffer(bytes(di.data), dtype='<u2')
assert np.array_equal(back, depth) and di.header.frame_id == 'phone_optical'
dinfo = node.pubs['phone/depth/camera_info'].msgs[-1]
assert dinfo.k[0] == 200.0 and dinfo.width == w
print('depth: %dx%d 16UC1 round-trips through raw deflate: ok (max %d mm)' % (w, h, back.max()))

# ---------------------------------------------------------------- weed tags
def tag(hdr_dict):
    hb = json.dumps(hdr_dict).encode()
    return node.on_binary(bytes([4]) + struct.pack('>d', now_ms()) + struct.pack('>I', len(hb)) + hb + jpeg)

# robot localized: map <- phone_optical = phone looking along map +x from (2, 3, 0.3)
node.tf_buffer.frames[('map', 'phone_optical')] = stubs2.make_tf('map', 'phone_optical', 2.0, 3.0, 0.3, *stubs2.quat_rpy(-math.pi / 2, 0, -math.pi / 2))
base = {'sid': 'S1', 'px': 640.0, 'py': 480.0, 'w': 1280, 'h': 960, 'fx': 933.3, 'fy': 933.3, 'cx': 640.0, 'cy': 480.0,
        'T': [float(v) for v in np.eye(4).flatten()], 'lat': 32.28, 'lon': -86.10, 'acc': 4.0, 'note': 'dandelion', 'track': 'normal'}

# 1) mounted, LiDAR point 1 m straight ahead (ARKit camera z backward -> point at z = -1)
r = tag({**base, 'mode': 'mounted', 'source': 'lidar', 'depth_m': 1.0, 'p_cam': [0.0, 0.0, -1.0], 'p_world': [0.0, 0.0, -1.0]})
assert r['ok'] and 'map (3.00, 3.00)' in r['msg'], r
pt = node.pubs['weed_tags'].msgs[-1]
assert pt.header.frame_id == 'map' and abs(pt.point.x - 3.0) < 1e-6 and abs(pt.point.y - 3.0) < 1e-6 and abs(pt.point.z - 0.3) < 1e-6
rec = json.loads(node.pubs['weed_tags/json'].msgs[-1].data)
assert rec['method'] == 'tf' and rec['gps']['lat'] == 32.28 and rec['note'] == 'dandelion' and os.path.exists(rec['image'])
print('mounted tag -> map (%.2f, %.2f, %.2f) via TF: ok' % (pt.point.x, pt.point.y, pt.point.z))

# 2) handheld before any anchor: saved, but no map position
r = tag({**base, 'mode': 'handheld', 'source': 'lidar', 'depth_m': 1.0, 'p_cam': [0.0, 0.0, -1.0], 'p_world': [0.0, 0.0, -1.0]})
assert not r['ok'] and 'anchor' in r['msg'], r
assert json.loads(node.pubs['weed_tags/json'].msgs[-1].data)['map_xyz'] is None
print('handheld tag without anchor refused with reason: ok')

# 3) anchor while mounted (last AR pose = identity camera), then the same physical point handheld
node.on_json(pose_msg(T_ar(0, 0, 0), now_ms()))
ok, msg = node._anchor_here()
assert ok, msg
assert node.static_tf.sent[-1].header.frame_id == 'map' and node.static_tf.sent[-1].child_frame_id == 'ar_origin'
print('anchor:', msg)
del node.tf_buffer.frames[('map', 'phone_optical')]          # phone is off the robot now: no TF for it
r = tag({**base, 'mode': 'handheld', 'source': 'lidar', 'depth_m': 1.0, 'p_cam': [0.0, 0.0, -1.0], 'p_world': [0.0, 0.0, -1.0]})
assert r['ok'] and 'map (3.00, 3.00)' in r['msg'] and 'anchor' in r['msg'], r
print('handheld tag via anchor -> same map point: ok')

# 4) a point 2 m ahead and 1 m to the camera's right in the ARKit world: map (4, 2)
r = tag({**base, 'mode': 'handheld', 'source': 'plane', 'depth_m': 2.0, 'p_cam': [1.0, 0.0, -2.0], 'p_world': [1.0, 0.0, -2.0]})
assert 'map (4.00, 2.00)' in r['msg'], r
print('handheld tag right of the camera -> map (4.00, 2.00): ok')

# 5) no 3D point at all
r = tag({**base, 'mode': 'handheld', 'source': 'none', 'depth_m': None, 'p_cam': None, 'p_world': None})
assert not r['ok'] and 'no 3D point' in r['msg']
print('tag without depth refused with reason: ok')

# 6) AR session restarts -> anchor invalid
node.on_json(pose_msg(T_ar(0, 0, 0), now_ms(), sid='S2'))
r = tag({**base, 'sid': 'S2', 'mode': 'handheld', 'source': 'lidar', 'depth_m': 1.0, 'p_cam': [0.0, 0.0, -1.0], 'p_world': [0.0, 0.0, -1.0]})
assert not r['ok'] and 'anchor' in r['msg']
print('anchor invalidated by a new AR session: ok')

# 7) auto-anchor when the app switches mounted -> handheld
node.tf_buffer.frames[('map', 'phone_optical')] = stubs2.make_tf('map', 'phone_optical', 2.0, 3.0, 0.3, *stubs2.quat_rpy(-math.pi / 2, 0, -math.pi / 2))
node.on_json({'t': 'status', 'ts': now_ms(), 'thermal': 'nominal', 'battery': 0.8, 'charging': False, 'mode': 'mounted', 'track': 'normal'})
node.on_json(pose_msg(T_ar(0, 0, 0), now_ms(), sid='S2'))
node.on_json({'t': 'status', 'ts': now_ms(), 'thermal': 'nominal', 'battery': 0.8, 'charging': False, 'mode': 'handheld', 'track': 'normal'})
assert node.anchor is not None and node.anchor_sid == 'S2'
out = [json.loads(s) for s in node.outbox]
assert any(o['t'] == 'ack' and o['ok'] and 'auto-anchor' in o['msg'] for o in out), out
print('auto-anchor on mounted -> handheld, ack queued for the app: ok')

# anchor button from the app
r = node.on_json({'t': 'anchor', 'ts': 123.0})
assert r['t'] == 'ack' and r['ok'] and r['ref'] == 123.0

lines = open(tmp + '/tags/index.jsonl').read().strip().split('\n')
assert len(lines) == 6 and all(json.loads(l)['name'] for l in lines)
print('index.jsonl has %d tags: ok' % len(lines))

# ---------------------------------------------------------------- status + e-stop + runtime "set"
node.on_json({'t': 'status', 'ts': now_ms(), 'thermal': 'critical', 'battery': 0.2, 'charging': False, 'mode': 'handheld', 'track': 'normal'})
node.timers[0]()
st = json.loads(node.pubs['phone/status'].msgs[-1].data)
assert st['e_stop'] and st['thermal'] == 'critical' and st['anchored'] and st['app'] == 'ios'
assert node.pubs['phone/e_stop'].msgs[-1].data is True
node.on_json({'t': 'status', 'ts': now_ms(), 'thermal': 'fair', 'battery': 0.2, 'charging': False, 'mode': 'handheld', 'track': 'normal'})
node.timers[0]()
assert node.pubs['phone/e_stop'].msgs[-1].data is False
print('status / e_stop on thermal critical and release: ok')

node.outbox.clear()
res = node.param_cbs[0]([stubs2._Msg(name='image_hz', value=5.0)])
assert res.successful and json.loads(node.outbox[0]) == {'t': 'set', 'imageHz': 5.0, 'depthHz': 5.0, 'poseHz': 30.0, 'quality': 0.6, 'width': 640.0}
print('ros2 param set image_hz -> set message for the app: ok')

# photo with AR pose sidecar
node.on_binary(bytes([2]) + struct.pack('>d', now_ms()) + jpeg)
side = [f for f in os.listdir(tmp + '/photos') if f.endswith('.json')]
assert side and json.load(open(tmp + '/photos/' + side[0]))['ar_pose_T'] is not None
node.on_binary(b'\x03short'); node.on_json({'t': 'nonsense'}); node.on_binary(bytes([4]) + struct.pack('>d', 1.0) + b'\x00\x00\x00\x09' + b'{bad')
node.on_disconnect()
print('ALL BRIDGE V2 TESTS PASSED')
