# WeedBot — autonomous backyard weed-spraying robot

Read this first in every session. It is the project memory shared between the
VM, the Raspberry Pi, and the owner's chat sessions.

## Who you are working with
- The owner is a robotics student, not a programmer. Explain what you are doing
  in plain language, one step at a time, and say which machine a command runs on.
- You do the software engineering. Write complete files, not fragments. After
  any change, say exactly how to run it and what success looks like.
- Ask before: running `sudo`, changing Arduino firmware, deleting anything,
  or commanding the robot to move or spray. Never auto-start motion.

## Machines
| Name | What it is | How to reach it | Role |
|------|-----------|-----------------|------|
| vm   | `het@het-virtual-machine`: Ubuntu 22.04 in VMware on a Mac, x86_64, ROS 2 Humble, RViz, Gazebo, VS Code | you are here when the hostname is het-virtual-machine | main development, git, VS Code |
| pi   | `het@het-desktop`: Raspberry Pi 4, Ubuntu 22.04 arm64, ROS 2 Humble, IP 192.168.1.45 | `ssh pi` (passwordless from the VM) | the robot computer |
| arduino | Genuine Uno (USB 2341:0043) running ROSArduinoBridge (`~/ros_arduino_bridge` on the Pi), `/dev/ttyACM0` @ 57600 | USB serial on the Pi | wheels + encoders (L298N) |

The owner works only inside the VM (the Mac just hosts VMware). VS Code runs in the VM and
reaches the Pi through Remote-SSH using the same `Host pi` entry.

Workflow: edit and build in the VM, commit to git, then on the Pi
`cd ~/robot_ws/src/my_bot && git pull && cd ~/robot_ws && colcon build --symlink-install`.
From the VM you may run `ssh pi "<command>"` to build, launch, or read logs on the robot.
Keep ROS traffic on the LAN. Pi and VM both use `ROS_DOMAIN_ID=0`, FastDDS
(`rmw_fastrtps_cpp`) and the FastDDS Discovery Server on the Pi (`192.168.1.45:11811`),
all set in each machine's `~/.bashrc`. Both `.bashrc` files also set
`FASTRTPS_DEFAULT_PROFILES_FILE=$HOME/fastdds_super_client.xml` (SUPER_CLIENT profile, copy in
`config/fastdds_super_client.xml`), so every terminal sees all topics. After changing it run
`ros2 daemon stop`. VM alias `ros_local` unsets both and sets ROS_LOCALHOST_ONLY=1.
CycloneDDS is NOT in use (old XMLs exist, unused).

## Repository
- Package `my_bot` (ROS 2 Humble, Articulated Robotics layout): `description/` (xacro URDF),
  `launch/`, `config/` (`my_controllers.yaml`, `nav2_params.yaml`, `mapper_params_online_async.yaml`).
- Workspace: VM `~/dev_ws` (source in `~/dev_ws/src/my_bot`); Pi `~/robot_ws` (source in `~/robot_ws/src/my_bot`).
- Build: `colcon build --symlink-install && source install/setup.bash`.
- Prefer Python nodes (`rclpy`) for new perception/task nodes.
- The robot runs `my_bot` + `diffdrive_arduino` (ros2_control) from `~/robot_ws`:
  `ros2 launch my_bot launch_robot.launch.py`, talking to ROSArduinoBridge on `/dev/ttyACM0` @ 57600
  (commands `e` read encoders -> "L R", `m L R` motor speeds, `u` PID; verified 2026-10-02:
  `e` -> `0 0`). The Pi's `~/robot_ws/src/diffdrive_arduino` has uncommitted local edits (May 2026).
- Phase 2 EKF (`ekf.yaml`) reads `/diff_cont/odom` from this stack; `launch_robot.launch.py` includes it.
- LiDAR: `ros2 launch my_bot lidar.launch.py` (ldlidar_stl_ros2 from `~/robot_ws/src`, params in
  `config/ldlidar.yaml`: LD19 mode, 230400, `frame_id: laser_frame`, default port `/dev/ldlidar`
  = udev symlink to the CP2102 (10c4:ea60), currently ttyUSB0). No static TF: the URDF provides
  `laser_frame` (weedbot_body.xacro: base_link + x -0.2495, z 0.378 = scan plane, LD06 on the spray tank
  at the back, its 0 deg mark toward the arm: see "LD06 orientation" in "Robot model").
  Do not use the vendor `ld06.launch.py` (frame
  `base_laser` + its own static TF).
- Drive test (manual, nothing starts at boot). Pi terminal 1: `ros2 launch my_bot launch_robot.launch.py`
  (keep the robot still ~3 s for IMU calibration); Pi terminal 2: `ros2 launch my_bot lidar.launch.py`;
  Pi terminal 3 (owner only, wheels off the ground first): `ros2 run teleop_twist_keyboard
  teleop_twist_keyboard --ros-args -r cmd_vel:=/cmd_vel_keyboard -p speed:=0.1 -p turn:=0.5`;
  VM: `rviz2 -d ~/dev_ws/src/my_bot/config/drive_test.rviz` (fixed frame odom, robot model, /scan, TF).
- Drive direction check (CAD v21, NOT done yet; owner only, wheels OFF the ground): v21 made the
  drive-wheel end the front without rewiring, so `description/ros2_control.xacro` swaps
  left/right_wheel_name and sets `enc_counts_per_rev` -3436 (diffdrive_arduino uses
  2*pi/enc_counts_per_rev for the motor command AND the encoder angle, so the minus sign inverts both).
  Firmware unchanged. The Pi's diffdrive_arduino has uncommitted local edits that were not checked
  (Pi offline). Run the drive test above, then: hold `i` -> both wheels turn so the robot would drive
  toward the ARM/phone end, and `ros2 topic echo /diff_cont/odom --field twist.twist.linear` shows
  x > 0; hold `j` -> the left wheel (left when facing the arm end) turns backward and the right one
  forward (counterclockwise from above), angular z > 0. If only forward is wrong: undo both changes.
  If only the turn is wrong: undo the name swap. If both are wrong: change -3436 back to 3436 and
  test again; still both wrong = the driver ignores the sign: stop and tell Claude.
- Velocity path (sim AND robot): `launch/twist_mux.launch.py` (included by launch_robot and launch_sim)
  -> `/cmd_vel_mux` -> collision_monitor (`launch/safety.launch.py`) -> `/diff_cont/cmd_vel_unstamped`. Inputs (`config/twist_mux.yaml`): Nav2 `/cmd_vel` priority 70,
  keyboard `/cmd_vel_keyboard` 90, joystick `/cmd_vel_joy` 100, each with a 0.5 s timeout (teleop
  overrides Nav2 only while keys are held: hold `k` to stop). Lock `/e_stop` (std_msgs/Bool,
  priority 255, latched): `ros2 topic pub --once /e_stop std_msgs/msg/Bool "{data: true}"` blocks
  every input until `{data: false}`. Never publish straight to `/diff_cont/cmd_vel_unstamped`.
- `launch/safety.launch.py` (in launch_robot and launch_sim): collision_monitor (+ its lifecycle
  manager), `scan_self_filter.py` (`/scan` -> `/scan_filtered` without returns from the robot
  itself; ALL scan users read `/scan_filtered`: costmaps, AMCL, SLAM, collision monitor) and
  `safety_state_node.py` (`/safety_state`). If any of them is down, the robot cannot move.
- The Pi needs (apt) before the next launch_robot: `ros-humble-twist-mux`,
  `ros-humble-nav2-collision-monitor`, `python3-scipy`.
- Sim "person": `ros2 run my_bot sim_person.py spawn --ahead 0.5 [--left L] | cross --ahead 1.0 --speed 0.2 |
  spawn X Y | move X1 Y1 X2 Y2 [SPEED] | remove` (relative = from Gazebo's true robot pose)
  (0.4 m x 1.7 m cylinder; room.world has the gazebo_ros_state plugin). Sim LiDAR min range
  0.14 m since CAD v19 (`description/lidar_sim.xacro`): every ray must start outside the tank's outline
  seen from above, or Gazebo reports flickering false hits at the tank's edge (0.052 m) whenever
  the robot pitches a hair; 0.14 is inside scan_self_filter's box anyway (0.3 once hid the stop zone). Sim test 2026-10-03: Nav2 goal 1 m SUCCEEDED;
  keyboard override and /e_stop both stopped the robot, Nav2 resumed after release.
- Gazebo coverage test (VM only; run `ros_local` first in EVERY terminal so the sim stays off the
  Pi's discovery server): world `worlds/room.world` (5 x 4 m, table + couch boxes, robot spawns at
  the centre); `launch_sim.launch.py world:=... gui:=false`, `online_async_launch.py` (SLAM),
  `map_saver_cli`, `localization_launch.py` + `navigation_launch.py` with `map:=` and
  `use_sim_time:=true`, `coverage.launch.py use_sim_time:=true`, RViz `config/coverage_sim.rviz`.
  A cold first Gazebo start can exceed spawn_entity's 30 s; rerun. Maps (outside git): `~/dev_ws/room_map.yaml`
  (2026-10-03, used for AMCL) and `~/dev_ws/room_map_v21.yaml` (2026-10-19, N and E walls clipped).
- Sim and real share ONE odometry pipeline: `imu_ekf.launch.py` (robot_localization EKF, `ekf.yaml`)
  fuses `/diff_cont/odom` + `/imu/data_raw` and publishes `/odom` + odom->base_link; diff_cont's own
  odom TF is off in both. In Gazebo, `description/imu_sim.xacro` (included only when sim_mode) adds
  an IMU plugin on `imu_link` -> `/imu/data_raw` @ 50 Hz (gyro noise 0.002 rad/s + small random
  bias), and launch_sim passes `use_mpu6050:=false use_sim_time:=true`. The VM needs
  `ros-humble-robot-localization` (apt) or launch_sim will not start.
- `launch_sim.launch.py` args: `world`, `gui` (default true; false = no Gazebo window),
  `wheel_mu` (Gazebo wheel friction, default 1.0; 0.03 shows wheel slip, 0.01 heavy slip),
  `sim_camera` (default false; the Gazebo camera costs a lot of CPU, turn on for Phase 1).
- `ros2 run my_bot sim_odom_check.py` (sim only, refuses without /clock): drives 1 m and 360 deg,
  stopping on Gazebo's true pose (`/ground_truth/odom`, p3d plugin in sim-only
  `description/sim_ground_truth.xacro`), and compares `/odom` (EKF) and `/diff_cont/odom` with it.
  Results 2026-10-03 (error vs ground truth, after 1 m forward / after 360 deg turn):
  mu 1.0:  EKF -1.7 cm, -0.5 deg / -1.2 deg;  wheels -1.7 cm, 0.0 deg / -0.9 deg.
  mu 0.03: EKF -11.6 cm / -0.4 deg;  wheels -11.6 cm / +14.2 deg. The EKF fixes HEADING under slip
  (gyro yaw rate); it cannot fix DISTANCE (only wheel speed is fused), and neither sees sideways slide.
  It still publishes straight to `/diff_cont/cmd_vel_unstamped` (written before the safety chain):
  do not use it until it goes through `/cmd_vel_keyboard`.
- VM performance (2 cores, 3.8 GB, VMware SVGA3D GPU; driver crash traces in the desktop log):
  real-time factor 0.37 with the Gazebo window. Since 2026-10-03 (sim camera off by default,
  room.world physics 500 Hz): 1.00 headless, 0.99 headless + SLAM + Nav2 (load ~2).
  2026-10-03 a full stack with RViz + AMCL + Nav2 + coverage + teleops reached load 14 with
  229 MB free and gzserver FROZE (threads stuck in futex waits, /clock stopped, Nav2 lifecycle
  never finished -> all goals rejected). If goals do nothing: check `ros2 topic hz /clock` and
  `ros2 lifecycle get /bt_navigator` first. Use `gui:=false`, start RViz last, never set
  LIBGL_ALWAYS_SOFTWARE (forces llvmpipe). Best fix: give the VM 4 cores / 6-8 GB.

## Phone sensors (iPhone app) (added 2026-10-18)
- iPhone 16 Pro running the WeedBotSensor app streams camera, LiDAR depth, ARKit pose, IMU and GPS
  over ONE plain websocket `ws://<robot-ip>:8444/ws` to `scripts/phone_bridge_node.py` (v2), which
  publishes them in ROS (no ROS on the phone). Docs + Swift source on the VM in `~/weedbot_ios`
  (README.md, PROTOCOL.md = wire protocol, bridge/README.md, STEP_BY_STEP.md). Run on the Pi for the
  robot, on the VM for tests: `ros2 launch my_bot phone_bridge.launch.py` (args
  `fallback_identity_anchor` = test only, `camera_info_url`, `cert_dir`). Log line `App websocket:
  ws://<this-ip>:8444/ws`; enter that IP in the app (gear icon); green dot = connected; the app
  remembers it and reconnects by itself. Needs `pip3 install aiohttp pyyaml` (VM done 2026-10-18,
  Pi not yet). Port 8444 has no authentication: LAN only, never port-forward it. The Safari page
  (8443/8080) needs certificates (make_cert.sh) and a `www/` folder, neither exists; not needed.
- Topics: `/phone/image/compressed` (JPEG ~10 Hz) + `/phone/camera_info` (ARKit intrinsics scaled to
  the streamed size), `/phone/depth/image` (16UC1 mm, 0 = invalid, 256x192, ~5 Hz) +
  `/phone/depth/camera_info`, `/phone/ar_pose` (PoseStamped, phone_optical in ar_origin, ~30 Hz),
  `/phone/imu/data_raw` (phone_link, rad/s, m/s^2 +9.8 up, ~100 Hz), `/phone/fix` (NavSatFix,
  ~1 Hz), `/phone/ar_twist` + `/phone/odom` (base_link velocity from ARKit, Mounted mode only),
  `/weed_tags` (PointStamped in map, latched, last 50), `/weed_tags/json` (full record),
  `/phone/status` (JSON 1 Hz: connected, mode, track, anchored, rates, thermal, battery),
  `/phone/e_stop` (Bool, latched, true on thermal critical). Runtime: `ros2 param set /phone_bridge
  image_hz 5.0` (also depth_hz, pose_hz, jpeg_quality, image_width) is pushed to the app.
- Frames: `phone_link` (phone body: x right edge, y top edge, z out of the screen; IMU),
  `phone_optical` (back camera: x right, y down, z forward; images, depth, AR poses), `ar_origin`
  (ARKit world in ROS axes, z up: ROS (x, y, z) = ARKit (x, -z, y)); `map -> ar_origin` static TF
  once anchored. `description/weedbot_body.xacro` adds base_link -> phone_mount -> phone_optical ->
  phone_link (since 2026-10-18; `phone_mount.xacro` is kept but no longer included): phone in
  landscape, home-button end on the right seen from behind, camera forward. `phone_xyz` =
  (0.1273, 0.0546, 0.1941) from CAD v21 (camera 0.0546 m LEFT of the centre line; holder on a shelf on
  the front of the body, under the arm), in `weedbot_properties.xacro`, `phone_tilt` 35 deg; it sets the
  mounted tag positions and the ARKit twist lever arm. Mount check, robot still: `ros2 topic echo
  /phone/imu/data_raw --field linear_acceleration` ~ +8.0 x, +5.6 z (z wrong sign -> `accel_sign:
  -1.0`); turning CCW must give + on x (else `gyro_sign: -1.0`), both in phone_bridge.launch.py.
- EKF: `ekf.yaml` twist0 = `/phone/ar_twist` (vx, vy, vyaw; odom0/imu0 unchanged; reference copy
  `config/ekf_phone_twist.yaml`). Covariance x10 when track is limited, nothing when notAvailable.
  It is published whenever the app is in MOUNTED mode (base_link -> phone_optical always exists
  now), so: Mounted only when the phone is really on the robot, otherwise the phone's own motion
  goes into /odom. Never run the bridge in Mounted mode next to a Gazebo sim in the same domain
  (the sim EKF would fuse the real phone, with wall-clock stamps against sim time).
- Tagging a weed (tap it in the camera picture). MOUNTED: point -> TF map <- phone_optical at the
  photo time; needs the robot localized (AMCL/SLAM) and the xacro mount. HANDHELD: with the phone
  still on the localized robot and track normal, press Anchor (`/phone/anchor_here`; the bridge
  also auto-anchors when the app switches Mounted -> Handheld), then take the phone off, walk and
  tap; points go through the anchor. ANY new AR session kills the anchor (status `anchored:
  false`, tag ack "no anchor for this AR session"): Reset AR, an AR error, or any interruption
  (phone locked, Control Center/notifications, app switch, call: the app restarts AR with a new
  session id; happened 3 times in the 2026-10-18 test). Keep the app on screen from Anchor to the
  last tap. `/phone/anchor_clear` drops it. Every tap is saved, even without a position:
  `~/weedbot_data/weed_tags/<name>.jpg` + `.json` (map_xyz, method tf|anchor, reason if none,
  GPS, status pending, kills []) + a line in `index.jsonl`; Photo button -> `~/weedbot_data/phone_photos`.
  `ros2 topic echo /weed_tags --once` shows an OLD tag (50 are latched); newest = the bridge log
  line or `tail -1 ~/weedbot_data/weed_tags/index.jsonl`.
- Safety: the bridge publishes no velocity commands. `/phone/e_stop` is NOT wired into twist_mux
  (only add it, priority 253, if the phone becomes the camera for the spray decision); the LiDAR
  safety chain does not depend on the phone. Keep the phone on 5 V when mounted (test: battery
  45% -> 35% in ~25 min, thermal "fair").
- VM test (no robot; `ros_local` in EVERY terminal):
  `ros2 launch my_bot phone_bridge.launch.py fallback_identity_anchor:=true`, then
  `ros2 topic hz /phone/image/compressed` (~10), `/phone/ar_pose` (~30), `/phone/imu/data_raw`
  (~100), `/phone/depth/image` (~5), `ros2 topic echo /phone/status --once` (track normal, lidar
  true); `ros2 service call /phone/anchor_here std_srvs/srv/Trigger` -> success;
  `rviz2 -d ~/dev_ws/src/my_bot/config/phone_test.rviz` (fixed frame map; phone axes, blue = view
  direction; red balls = /weed_tags; depth panel dark = near, black = no reading). In this test
  mode map = ar_origin = where the app's AR session STARTED (app start or Reset AR), and the
  camera's forward direction at that moment is map +y, NOT +x (ARKit .gravity alignment); z is
  relative to the phone's height then (floor ~ -1.1 m). Handheld check: Reset AR at a start spot
  aimed at a floor marker, Anchor, walk, tap the marker -> (~0, marker distance, -phone height).
- Results 2026-10-18 (VM 192.168.1.82, phone 192.168.1.34, iOS 27.0.1): image 9.97, pose 30.1,
  imu 99.8, depth 5.00 Hz, track normal, lidar true; anchor_here success; handheld tag at map
  (-0.03, 0.99, -1.15) (sideways 3 cm; the phone then laid next to the marker read
  (0.02, 0.89, -1.11)); marker 1.00 m from the start (tape) -> tag 0.99 m, ~1 cm error (an
  earlier try read 0.95 m, ~5 cm); RViz pose and depth update live. `python3 test/phone_bridge/test_bridge_v2.py`
  (no ROS needed) -> ALL BRIDGE V2 TESTS PASSED. The 11 tags from that test in
  `~/weedbot_data/weed_tags` are test taps from several AR sessions, not weeds.
- The protected ROS communication setup is UNCHANGED by the phone work: FastDDS discovery server,
  fastdds.service, `~/fastdds_super_client.xml`, FASTRTPS_DEFAULT_PROFILES_FILE, both `.bashrc`
  files and ROS_DOMAIN_ID were not touched. The phone link is plain TCP 8444 outside DDS; VM ufw is
  not enabled (nothing to open); the Pi's firewall is not checked yet.
## How the robot actually runs (updated 2026-10-02)
- Boot: the ONLY robot-related systemd service is `fastdds.service` (FastDDS discovery server,
  `fastdds discovery --server-id 0`, listening on 0.0.0.0:11811). Nothing ROS starts at boot and
  nothing drives the motors; the robot stack is launched by hand.
- Retired (kept on disk, do not delete): `~/ros2_ws` (arduino_odom, robot_tf_odom, scripts) and its
  services `arduino_odom`, `arduino_tf_broadcaster`, `robot_autostart` (`~/start_robot_full.sh`),
  `ldlidar`, `slam_toolbox`, `robot-bringup`, all disabled 2026-10-02. Exact disable/re-enable
  commands: `~/boot_services_backup.txt` on the Pi. Its serial protocols match no existing sketch.
- `.bashrc` returns early for non-interactive shells, so `ssh pi "cmd"` does not get the DDS
  settings; use `ssh pi 'bash -ic "cmd"'` (or source the variables) for ROS commands over SSH.

## Protected setup — never modify without asking the owner first
Read freely; ask the owner before editing, disabling, restarting, or deleting any of these:
- Pi systemd units in `/etc/systemd/system/`: `fastdds.service` (must keep running), `arduino_odom.service`,
  `arduino_tf_broadcaster.service`, `robot_autostart.service`, `ldlidar.service`,
  `slam_toolbox.service`, `robot-bringup.service`.
- Pi `~/start_robot_full.sh` (and the other `~/start_*.sh` / `~/run_slam_pi.sh` scripts).
- Pi `~/ros2_ws` (retired, keep on disk; packages `arduino_odom`, `robot_tf_odom`, `mpu6050_driver`, `ldlidar_stl_ros2`).
- DDS / ROS environment: `~/.bashrc` on Pi and VM (ROS_DOMAIN_ID, RMW_IMPLEMENTATION,
  ROS_DISCOVERY_SERVER, ROS_IP, FASTRTPS_DEFAULT_PROFILES_FILE, aliases;
  backups `~/.bashrc.bak-2026-10-02`); `~/fastdds_super_client.xml` (Pi and VM); `~/cyclonedds.xml`, `~/cyclonedds_pi.xml` (Pi),
  `~/cyclonedds.xml`, `~/cyclonedds_vm.xml` (VM). Any new FastDDS profile XML counts too.
- Pi udev rules: `/etc/udev/rules.d/99-arduino-odom.rules`, `99-ldlidar.rules`,
  `99-usb-devices.rules`, `99-usb-serial.rules` (`/dev/arduino_odom`, `/dev/ldlidar`, `/dev/ld06`).
- Pi `~/ros_arduino_bridge` (the Uno firmware source) and Arduino firmware in general.
- Pi `~/boot_services_backup.txt`.

## Robot hardware (as built)
- Drive: differential drive. Two 12 V 100 RPM 1:45 encoder gearmotors (37 mm, 6 mm D-shaft)
  at the FRONT (CAD v21: front = drive-wheel end, arm and phone), L298N driver, printed spur gears to
  8 mm axles, 5-inch (127 mm) wheels. Back: single stem caster (after the drivetrain rebuild; before
  it, two fixed wheels). Until v19 the caster end was called the front.
- Gear ratio: 2:1 now (20T:40T). After the rebuild: 30T:40T = 1.33:1.
  (Only if the ros2_control stack is used: `enc_counts_per_rev` in diffdrive_arduino =
  motor counts per rev × gear ratio; its SIGN sets the motor/encoder direction, -3436 since v21.)
- `wheel_radius` 0.0635 m, `wheel_separation` 0.176 m (CAD v17: tyre centres y +-0.088). The URDF,
  gazebo_control.xacro and my_controllers.yaml all use these (see "Robot model").
- Sensors: LD06 2D LiDAR (on a puck on the spray tank at the back, scan plane 0.378 m: sees nothing
  lower; the arm hides ~+-10 deg straight AHEAD, see "Robot model"), MPU6050 IMU on Pi I2C bus 1 at 0x68
  (`scripts/mpu6050_node.py` -> `/imu/data_raw`; yaw rate fused with wheel odometry by the
  robot_localization EKF, `config/ekf.yaml`, which publishes `/odom` and odom->base_link;
  diff_cont has `enable_odom_tf: false` on the real robot, true in Gazebo), camera = the iPhone on
  the front phone mount (see "Phone sensors"; the old Pi cam / webcam plan and its `camera_link`
  frames are no longer in the URDF).
- Arm: 4 x MG996R (yaw, shoulder, elbow, wrist pitch) + a wrist roll servo, base on the top cover at
  the front, over the drive wheels (not wired yet; see "Robot model"). Spray: small 12 V pump switched
  by a relay; tank (3.6 kg full) on the top frame at the back.
- Power: 12 V Li-ion pack, 5 V for Pi. Battery not monitored yet.
- Frame: 2020/4020 aluminium extrusion, 360 × 240 × 160 mm, based on the "NXP robot platform" CAD.

## Robot model (CAD v21, 2026-10-18; FreeCAD "project 101 v21", final for now)
- FRONT = the DRIVE-WHEEL end (arm and phone); the caster is at the BACK. Up to CAD v19 the caster
  end was called the front: v21 turned the x axis round (same hardware; the phone holder moved to a
  shelf on the new front). The real drive direction is flipped in ros2_control.xacro, see "Drive
  direction check".
- `description/robot.urdf.xacro` includes `weedbot_properties.xacro` (ALL measurements in one file,
  from CAD v21; still MEASURE: IMU position (imu.xacro), masses other than the tank, caster trail,
  arm yaw range (MG996R 180 deg through a gear)), `weedbot_body.xacro` (bottom + top frame, body
  shell, drive wheels, caster, electronics plate, spray tank with the LD06 on a puck, phone mount),
  `weedbot_arm.xacro` (arm), plus `imu.xacro`, `ros2_control.xacro` / `gazebo_control.xacro` (wheel
  values from the properties), and sim-only `imu_sim.xacro`, `lidar_sim.xacro` (Gazebo LD06 on
  laser_frame -> /scan), `sim_ground_truth.xacro`. Kept on disk but NOT included: `robot_core.xacro`,
  `lidar.xacro`, `camera.xacro`, `phone_mount.xacro`.
- `base_link` is ON THE GROUND under the centre of the drive axle; `base_footprint` = the same point.
  Layout (x forward from the drive axle, toward the arm): bottom frame x -0.293..0.067, 0.14 wide;
  top frame same x, 0.24 wide; body shell x -0.299..0.073, 0.316 wide, z 0.04..0.237; wheels r 0.0635
  at y +-0.088; caster swivel axis x -0.419 on a bracket behind the frame = the TAIL (driving forward
  the wheel trails 0.037 behind the swivel, so the tail reaches x -0.52); spray tank x -0.294..-0.198,
  0.239 wide, on the top frame at the back, sump below, with the LD06 on a 50 x 50 x 8 mm puck on top
  (x -0.2495); electronics plate x -0.223..-0.003 on the deck; arm yaw axis at x 0.0066 (just in
  front of the axle), base plate set into the top cover, round base housing r 0.0715, 0.023 tall, on
  it (turns with the yaw); phone mount on a shelf on the front of the body: shelf x 0.073..0.145,
  z 0.086..0.142, wedge bracket x 0.085..0.158 up to 0.2095 (drawn as its bounding box), cradle slab
  tilted 35 deg x 0.074..0.158, top 0.225; phone mount and arm camera reach x 0.175.
- Heights above the ground: deck (top of the bottom rails) 0.0905, top frame 0.2105..0.2305, top cover
  0.2366 (= top of the body shell), arm plate 0.2305..0.2366 (yaw joint 0.2366), arm housing to
  0.2596, phone camera 0.194 (x 0.127, y +0.055), caster swivel 0.152, stem top 0.221, tank
  0.2305..0.3544 (filler cap to 0.3604, not modelled: position not given), puck 0.3474..0.3554 (7 mm
  into the tank top, as given), LD06 bottom 0.3554 -> scan plane (laser_frame) 0.378. Arm parked:
  shoulder 0.3073, elbow 0.4073, wrist axis x 0.107 at 0.4073, nozzle tip pointing down at x 0.107,
  0.276; arm top 0.448 in CAD v19 (model 0.429: smaller servo boxes). Straight up the spray_tip would
  be at 0.638.
- Names kept: base_link, base_footprint, laser_frame, left_wheel_joint, right_wheel_joint, imu_link,
  phone_optical, phone_link. `camera_link` / `camera_link_optical` are GONE (the phone is the camera):
  the uncommitted W1 `fake_weed_detector.py` still uses camera_link_optical + /camera/image_raw from
  the old Gazebo camera, so W1 needs phone_optical + a simulated phone camera before it runs again;
  the `sim_camera` launch arg is still accepted but does nothing.
- Arm joints `arm_base_yaw`, `shoulder_pitch`, `elbow_pitch`, `wrist_pitch`, `wrist_roll` (axis to axis
  0.0707 (from the arm base underside = yaw joint, on the top cover) / 0.100 / 0.100 / 0.131;
  wrist_roll sits `arm_roll_d` 0.0737 along the last link, axis along the link, child link `nozzle` ->
  `spray_tip`). -90..+90 deg each, 0 = straight up, + pitch tilts forward (toward the phone), + yaw =
  counterclockwise seen from above. The yaw servo hangs under the plate (inside the body) 0.0224 in
  front of the axis and drives a gear (yaw joint stays at arm_x). PARKED POSE (CAD v21, `arm_park_*`
  in weedbot_properties.xacro): elbow +90 deg (forearm forward, horizontal), wrist +90 deg (nozzle
  pointing down), others 0. In SIM the arm joints are FIXED in that pose (the joint origins carry the
  angles): with no simulated servos revolute joints let the arm collapse through the LiDAR plane. Add
  a Gazebo position controller before moving the arm in sim.
  `scripts/arm_joint_state_stub.py` (`config/arm_joint_state_stub.yaml`, started by launch_robot and
  launch_sim) publishes the five arm joints in the parked pose on /joint_states at 5 Hz (view_robot's
  sliders start there too), plus the real caster's
  `caster_swivel_joint` / `caster_wheel_joint` (TF only; ignored in sim, where the caster is a fixed
  frictionless ball). The servo driver replaces the arm part later (same joint names). Until then
  the real arm must sit in the parked pose, or TF, RViz and the scan filter show it in the wrong place.
  RViz + sliders: `ros2 launch my_bot view_robot.launch.py` (`config/view_robot.rviz`, fixed frame
  base_link; `sim_mode:=true` for the sim variant).
- RViz (Humble) draws ALL visuals of a link in the colour of the link's FIRST visual (inline colours
  do not help). A part with a different colour needs its own fixed link (top_cover, caster_socket,
  phone_shelf, phone_bracket, phone_cradle, phone_body, arm_yaw_servo, turret_servo, wrist_servos,
  shoulder_servo, elbow_servo).
- `rsp.launch.py` strips XML comments from robot_description, and every launch passes it as
  `ParameterValue(..., value_type=str)`: gazebo_ros2_control hands the URDF to its controller manager
  as a `--param robot_description:=...` argument that is parsed as YAML, so a "word: " in any comment
  stopped the wheel controllers (and launch_sim / launch_robot did not start at all from commit
  3b156e3, the phone_mount.xacro comment, until the URDF v2 commit).
- LD06 orientation: `lidar_yaw` 0 (weedbot_properties.xacro) = its 0 deg mark points forward (toward
  the arm); mounted the other way round -> `lidar_yaw` `${pi}`. Check on the robot (standing still,
  lidar.launch.py + RViz fixed frame base_link): a box ~0.5 m BEHIND the robot must show up behind it
  (if it shows up in front, set `${pi}`); a box on the robot's LEFT must show up on the left (else the
  scan direction is mirrored: `laser_scan_dir` in config/ldlidar.yaml).
- LiDAR self-hits (v21): only the parked arm is in the scan plane, IN FRONT of the LD06: the upper arm
  (0.246 m) and, behind it, the wrist (0.36 m): ~+-10 deg blind straight ahead (CAD). The upper arm's
  COLLISION box is widened to that envelope (`arm_scan_w` 0.087) so the sim shows the same 20 deg.
  The arm moved out of the parked pose (servo driver, later) can cross the scan plane elsewhere.
  `scan_self_filter.py` box (safety.yaml) x -0.53..0.18, y +-0.16 in base_link (the requested
  laser_filters LaserScanBoxFilter, z 0..0.6; laser_filters is not installed, so the existing node
  does the same in 2D). It sets self-hits to NaN, not +inf: the costmaps use inf_is_valid, so +inf
  rays clear the costmap out to 3 m beyond whatever robot part they hit.
- Nav2 footprint (both costmaps, robot_radius off, padding 0.01): [[0.175, 0.080], [0.073, 0.158],
  [-0.299, 0.158], [-0.520, 0.101], [-0.520, -0.101], [-0.299, -0.158], [0.073, -0.158],
  [0.175, -0.080]]. Farthest point 0.53 m (the caster tail) = the circle swept turning on the spot,
  so local inflation 0.55 and coverage clearances 0.288 m straight / 0.66 m in turns. Inscribed
  radius 0.168 (the sides + padding): SmacPlanner2D keeps the robot centre that far from obstacles,
  so it only plans through gaps the robot fits through. safety.yaml: StopBand x 0.18..0.275,
  SlowCorridor x 0.18..0.575 (both y +-0.16), stop_zone x -0.56..0.215, y +-0.198.
- SAFETY (v21): BLIND WEDGE STRAIGHT AHEAD. The LD06 is at the back and the parked arm hides +-10 deg
  around the centre line in front of it; the wedge widens with distance: 0.15-0.18 m wide at the stop
  band, as wide as the robot (0.32 m) at x 0.65 (0.47 m ahead of the front). Sim 2026-10-18, person
  (0.4 m cylinder) straight ahead, centre at x 0.40 -> 16 returns (only its edges), 12 in the StopBand,
  0 in stop_zone (so /safety_state does not say stop); x 0.60 -> 8 returns, 4 in the SlowCorridor;
  x 0.90 and 1.50 -> 0 returns (invisible). Driving straight at a standing person: stopped 0.09 m
  short (0.08 m/s cap on) / 0.08 m short (cap off, 0.15 m/s commanded). Driving at an 8 cm post on the
  centre line (0.08 m/s): NOT seen, the robot drove into it. So anything narrower than the wedge
  straight ahead (post, pole, picket, a leg) is invisible to the collision monitor, /safety_state and
  the costmaps, and Nav2/RPP cannot avoid it while heading at it. The phone camera looks there but is
  NOT part of the safety chain. Behind and beside the robot the view is clear (the rear is fully
  visible now: Nav2's BackUp recovery reverses into seen space). As before, nothing lower than the
  0.378 m scan plane is seen (pets, boxes, pots, edging, kerbs). Before the robot drives on its own
  (Nav2 / coverage) the wedge needs a fix, e.g. a parked pose that keeps the whole arm below the scan
  plane (check in CAD), the LD06 raised above the arm, or a phone-camera obstacle check in the safety
  chain. Person tests re-run in sim for v21 on 2026-10-19 (see "Sim regression v21"); still to do on
  the real robot.
- Sim test 2026-10-18 v21, arm parked (room.world, headless, domain 42): spawn and controllers OK,
  robot level (pitch 0.13 deg); raw /scan: 20 self-hit rays (-10..+10 deg, 0.246-0.250 m = the upper
  arm), no other gap; /scan_filtered: those 20 are NaN. Teleop input (/cmd_vel_keyboard) +0.10 m/s
  for 3 s -> +0.229 m along base_link +x = toward the arm (capped at 0.08 m/s: no map, so everything
  is "unknown"); +0.5 rad/s for 3 s -> +81 deg, counterclockwise. Person / post: see SAFETY above.
  (The v2 drive test, 0.240 m straight with EKF = wheels, used the old wheel values.)
- TAIL SWING (v21): the caster tail (x -0.52) sweeps a 0.54 m circle on every turn on the spot, and
  turning AWAY from a wall swings it INTO that wall. SmacPlanner2D keeps only the half width (0.168 m)
  and RPP's footprint check is the only guard against tail strikes, and it is marginal: it looks only
  ~0.1-0.15 m of tail travel ahead (eaten by command latency on the loaded VM), and once the tail is
  within ~1 cm of a wall the wall drops out of the local costmap (footprint clearing + the self-filter
  box), so a retry can turn into it. Give Nav2 goal poses >= 0.6 m of room around the robot centre.
- Sim regression v21, 2026-10-19 (domain 42, headless, committed room.world, positions and gaps from
  Gazebo truth; test scripts were throwaway):
  1 Teleop keys via /cmd_vel_keyboard: i (+0.10, capped 0.08 m/s without a map) 3 s -> +0.243 m toward
    the arm in /odom, /diff_cont/odom and truth, wheel joints +1.28 / +1.28 rad/s; j (+0.5 rad/s) 3 s
    -> +83 deg CCW, left wheel -0.66, right +0.63 rad/s. PASS.
  2 SLAM from scratch (Nav2 goals under SLAM) -> `~/dev_ws/room_map_v21.yaml/.pgm` (the saved
    `~/dev_ws/room_map.yaml` untouched): not mirrored (99% of its wall cells within 0.10 m of the saved
    map; mirrored / rotated copies 7-51%), every wall a single line (the saved map has doubled south
    and west walls), table within 0.015 m of the truth. But the N and E walls are MISSING: slam_toolbox
    sizes the map to the scan endpoints, the noiseless sim LiDAR puts them exactly on those walls and
    the last row / column falls off (the saved map is bigger only through stray points); the table's
    north face was never seen (the robot cannot turn in the 0.6 m gap north of it). AMCL therefore
    keeps using the saved room_map (its east wall is up to 0.10 m too close: AMCL vs truth 0.11-0.14 m).
  3 AMCL + Nav2, goals (1.6,-1.2), (-1.6,0.9), (1.9,-0.3): all SUCCEEDED, 0 recoveries, 0 controller
    aborts, min outline clearance 0.126 m (caster tail vs table, turning). PASS. (In step 2 a goal pose
    with the tail 8 cm from the table: the turn on the spot took it to 0.013 m before RPP stopped.)
  4 Coverage medium. Old planner: 66% / 52%, 3 / 6 controller aborts and 2 TAIL CONTACTS per run (east
    wall, table): lane ends where no U-turn fitted stopped lane_clear (0.29 m) from a wall and RPP
    turned on the spot there. Fixed in coverage_planner_node.py: both ends of every piece Nav2 joins
    live (and where a perimeter loop starts) are pulled back to turn_clear 0.66 m. Now 55% in 4.3 min,
    0 aborts, 0 contacts, min outline clearance 0.150 m, no RPP collision warnings. No Nav2 / safety /
    clearance parameter changed. The rest of the drop: the perimeter loop runs ~0.6 m from every wall
    (each bend is pushed to turn_clear and every pushed point leaves a kink the next pass pushes too),
    leaving a ~0.37 m band along all walls. OPEN: needs a loop that uses the clearance for its actual
    curve radius plus a safe way off a wall-hugging loop.
  5 Persons (AMCL + Nav2): standing --ahead 1.5, Nav2 goal through it: stopped 0.23 m short, then went
    around, min side gap 0.08 m. Exactly in front at 1.0 m, keyboard 0.15 m/s: stopped 0.09 m short. At
    1.0 m standing still the RAW /scan has 0 returns on the person (the arm hides it; the filter
    changed only the 20 arm rays, all <= 0.25 m); first returns at 0.96 m, 6 at 0.75, 11 at 0.55 (raw
    and filtered within 1 = one-scan timing). Crossing (cross --ahead 1.0 --speed 0.2) during a Nav2
    goal: min gap 0.19 m. 0 contacts.
  6 Pool guard: NOT TESTABLE, no pool guard, sim pool zone or keepout filter exists yet.
  7 Phone bridge (launch-file parameters, test port 18444, fallback_identity_anchor, fake app client):
    base_link -> phone_optical (0.127, 0.055, 0.194), rpy (-125, 0, -90) deg = camera 35 deg down.
    Handheld tag, no map: (0.00, 1.00, -1.10) as expected; map = base_link, anchored through the URDF
    mount: floor point 1 m ahead of the camera -> (1.1273, 0.0546, 0.0000), anchor level.

## Software stack
- SLAM Toolbox for mapping, AMCL for localization, Nav2 for navigation (all working indoors).
- Perception: YOLO instance segmentation (Ultralytics YOLO11n-seg / YOLOv8n-seg), classes
  `weed_broadleaf` and `keep_plant`; trained on Colab, exported to NCNN/ONNX for the Pi.
- Weed localization: calibrated camera + ground-plane ray cast (pixel -> base_footprint -> map).
- Simulation: Gazebo (free). Isaac Sim is optional, cloud only, later.
- Visualization from the Mac: Foxglove Studio or RViz in the VM.

## Planned ROS 2 nodes (build in this order)
mpu6050_node + ekf_node -> coverage_planner_node -> camera (v4l2_camera) -> weed_detector_node ->
weed_localization_node -> weed_manager_node -> mission_executor -> arm_planner_node -> precision_alignment_node ->
spray_controller_node. TF frames: map, odom, base_link, base_footprint, laser_frame, imu_link,
phone_optical, phone_link, arm_base, arm_turret, upper_arm, forearm, wrist, nozzle, spray_tip.

## Roadmap and current status
- Phase 1 (POSTPONED, resume after Phase 3): camera publishing + intrinsic calibration +
  camera in URDF/TF.
- Phase 2 (in progress): MPU6050 + robot_localization EKF (odom + IMU). Files: mpu6050_node.py,
  ekf.yaml, imu.xacro, imu_ekf.launch.py.
- Phase 3: coverage via coverage_planner_node (robot-vacuum style: perimeter laps, then lanes
  with smooth U-turns, all sent as ONE path to Nav2's FollowPath / Regulated Pure Pursuit; the
  planner only fills joins). Robot size comes from the local costmap footprint in nav2_params.
  INDOOR ONLY for now: it plans on /map + the global costmap and does not yet honour a keepout
  mask, so outdoor use needs the Nav2 keepout filter (pool + 1 m) wired in first.
  Options (per run; any change discards the old plan):
    ros2 launch my_bot coverage.launch.py use_sim_time:=true pattern:=dense lane_angle:=90 perimeter_laps:=2
    ros2 param set /coverage_planner pattern wide        # dense 0.25 | medium 0.35 (default) | wide 0.50 | custom
    ros2 param set /coverage_planner lane_spacing 0.30   # with pattern custom
    ros2 param set /coverage_planner lane_angle 45       # degrees, or auto (fewest lane pieces)
    ros2 param set /coverage_planner perimeter_laps 0    # 0, 1 or 2
    ros2 service call /coverage/plan std_srvs/srv/Trigger    # preview /coverage_path in RViz
    ros2 service call /coverage/start std_srvs/srv/Trigger   # /coverage/stop to cancel
  Room-world results 2026-10-03 (headless, perimeter_laps 1): dense 88% in 328 s, medium 89% in
  307 s, wide 80% in 226 s (wide lanes are 0.50 m apart, wider than the 0.45 m swath, so gaps are
  expected); 2-4 controller aborts per run, all recovered by retry/skip at the same two spots.
  v21 robot (2026-10-19): medium 55% in 4.3 min, 0 aborts, 0 contacts (lane ends trimmed for the
  caster tail; perimeter loop still ~0.6 m from the walls), see "Sim regression v21".
  Blocked path: waits blocked_wait (5 s), re-plans around if there is room, else skips 1 m into
  a revisit queue retried at the end; the final report (log + latched `/coverage/report`) lists
  anything still skipped with its map location. Needs python3-scipy (apt) on the Pi.
  Person tests 2026-10-04 (OLD robot geometry, re-run for the rebuilt robot - see "Robot model") (room world,
  medium, 0.15 m/s top speed, path 0.03 m further from walls):
  0 contacts. Person standing on the path: stopped 0.10 m short. Person stepping in 0.20 m in
  front of the bumper (verified) at 0.13 m/s: stopped 0.09 m short. 85% covered in 9.8 min;
  4 false stops (couch corner near the start, top wall, and two beside the table where the saved
  room_map lacks the table's front edge -> remap the room to remove those); all revisits succeeded.
  Do NOT disable RPP collision detection: tested, it is the only side/corner protection (a person
  crossing from the side hit the robot, gap -0.20 m; coverage fell to 38%).
- Phase 4: detector node on the Pi (low fps is fine). Needs Phase 1 camera.
- Phase 5: weed localization to map frame, verified within 10 cm at 1 m.
- Phase 6: weed_manager + NavigateToPose to a standoff pose.
- Phase 7–8: cloud sim / synthetic data (optional).
- Phase 9–10: arm, spray, precision alignment, TreatWeed action.
- Phase 11: mission executor, docking, battery, safety.
Each phase ends with a verification test; do not move on until it passes.

### Weed track W1-W5 (added 2026-10-04): simulation first, Pi hardware-in-the-loop check per milestone
Each milestone: works in Gazebo (headless test, own ROS domain, cleaned up), then a
hardware-in-the-loop check with the robot-side nodes on the Pi. Safety rules above still apply
(no spraying in sim or on the bench without the owner explicitly asking in that session).
- W1 weed memory foundation: sim weeds, fake detector (true positions + camera FOV, noise,
  misses, false alarms), SQLite weed database (positions, sightings, photo crops, status,
  treatment history), weed_manager merging repeat sightings, RViz markers.
- W2 web dashboard (weed map, photos, status, run history).
- W3 treat + verify loop in sim (drive to weed, treat, re-check later: verified dead / survived).
- W4 weather-aware spraying (no spraying in rain/wind; plan around the forecast).
- W5 self-improving detector pipeline (collect crops, label, retrain, evaluate, deploy).

### Drive calibration: PENDING until after the drivetrain rebuild (noted 2026-10-02)
Hardware is on the bench, not in the chassis; rebuild = caster in front, 30T:40T gears. Open items:
- (a) diff_cont odom showed 0.227 m/s for a 0.1 m/s teleop command on the bench. Check
  `loop_rate` in `ros2_control.xacro` (30) against `PID_RATE` in ROSArduinoBridge and the owner's
  uncommitted edits to `~/robot_ws/src/diffdrive_arduino/hardware/diffbot_system.cpp`.
- (b) Encoder counts per wheel turn (`enc_counts_per_rev`, now 3436) unverified.
- (c) IMU `axis_signs` (imu_ekf.launch.py) and imu.xacro pose to verify after mounting (imu_z is
  measured from the GROUND since base_link moved there on 2026-10-18; now 0.10). Forward flipped with
  CAD v21: an X arrow pointing at the caster = rpy 0 0 pi (the EKF uses only the z gyro: unaffected).
- (d) Occasional IMU I2C read errors when the motors start (check wiring/power/noise).
- DONE 2026-10-18: one set of wheel values for sim and real, from CAD v17: r 0.0635, sep 0.176 in
  `weedbot_properties.xacro` AND `my_controllers.yaml` (gazebo_control.xacro reads the properties;
  the sim-only override in `config/gaz_ros2_ctl_use_sim.yaml` is deleted). Check them on the real
  robot with a 1 m / 360 deg drive; change both files together.
- (e) Drive direction after CAD v21 (front = drive end): "Drive direction check" in "Repository".

## Rules for automated tests (Claude) on the VM
- Every headless test runs isolated from the owner's sessions: `ROS_DOMAIN_ID=42`,
  `ROS_LOCALHOST_ONLY=1`, `ROS_DISCOVERY_SERVER` and `FASTRTPS_DEFAULT_PROFILES_FILE` unset, and
  `GAZEBO_MASTER_URI=http://127.0.0.1:11346` (Gazebo's own transport ignores ROS domains; the
  owner's Gazebo uses the default port 11345).
- When a test ends (pass, fail or abort), stop everything it started:
  `~/dev_ws/src/my_bot/scripts/sim_cleanup.sh --domain 42`, and check it exits 0 ("Clean").
  Never stop processes in other domains; if the owner's sim is running and a test needs the CPU,
  ask the owner to stop it.
- `scripts/sim_cleanup.sh` (no arguments) is the owner's full reset: stops ALL sim/Nav2/RViz
  processes in every domain and clears `/dev/shm/fastrtps_*`. `--dry-run` lists only.

## Hard safety rules
- The backyard has a swimming pool. LiDAR cannot see water. Any outdoor navigation config
  MUST include a keepout zone around the pool with at least 1 m margin. Refuse to generate
  outdoor coverage paths without it.
- Never spray inside a keepout/flower-bed zone. Never enable the spray relay in simulation
  or on the bench without the owner explicitly asking in that session.
- Cap `max_vel_x` at 0.3 m/s and `max_vel_theta` at 1.0 rad/s in nav2_params. Current top speed
  is 0.15 m/s EVERYWHERE (diff_cont limits in my_controllers.yaml cap every source; RPP and the
  velocity smoother match), and 0.08 m/s whenever an unknown obstacle is within 1.5 m.
- SPRAY ONLY WHEN `/safety_state` IS `clear`. The spray controller must subscribe to
  `/safety_state` (std_msgs/String, latched, 10 Hz: clear | slow | stop) and close the valve
  immediately on anything else, on a missing/old message, or on `/e_stop` true.
- Every motion command goes through the collision monitor (sim and robot):
  twist_mux -> `/cmd_vel_mux` -> collision_monitor -> `/diff_cont/cmd_vel_unstamped`. Never
  publish straight to `/diff_cont/cmd_vel_unstamped`. Zones in `config/safety.yaml`: stop band
  from the front (phone mount, 0.175) to 0.10 m ahead (0.32 m wide); slow (40 %) corridor 0.40 m ahead
  of the front. The LiDAR (scan plane 0.378 m, at the back since v21) does not see anything lower
  than that, nor anything narrow straight ahead (the arm's blind wedge, see "Robot model"). Full outline+0.10 m boxes stopped the robot beside every wall; Nav2 "approach" mode
  never triggered in this Humble version. Side/rear contact while turning is left to RPP's
  footprint check. No LiDAR data for 1 s -> stop. Do not remove or bypass it.
  `/safety_state`: stop = anything within outline + 0.04 m, e-stop or no scan; slow = slow corridor
  OR an unknown obstacle (3+ scan points > 0.25 m from anything on /map) within 1.5 m.
  Command chain: twist_mux -> `/cmd_vel_mux` -> safety_state_node (speed governor: caps EVERY
  command at 0.08 m/s near unknown obstacles, also sends Nav2 `/speed_limit`; sends zeros at once
  on `/e_stop`) -> `/cmd_vel_capped` -> collision_monitor -> diff_cont. If it dies, nothing moves.
  Braking (sim, command -> standstill): ~2.5-3.5 cm at 0.12-0.14 m/s (~4 cm at 0.15), ~1.2-1.7 cm
  at 0.065 m/s (~2 cm at 0.08); e-stop the same. Plus up to ~0.2 s detection delay. diff_cont has
  no acceleration limit (fastest stop), cmd_vel_timeout 0.25 s. Re-measure on the real robot.
- Stop everything if a person or pet is within 2 m (to be implemented on top of /safety_state;
  the current zones only cover the robot's immediate surroundings).

## CAD rules (FreeCAD via the freecad MCP server, FreeCAD runs on the owner's Mac)
- Never overwrite the owner's file. Always save as a NEW version next to it, e.g.
  "project 101.FCStd" -> "project 101 v2.FCStd" (next free number); never save over an existing file.
- Add new parts as separate, clearly named bodies (e.g. `Body_CameraMount`); do not merge them
  into existing bodies or change the owner's existing features unless asked.
- Every printable part must fit the print bed 220 x 215 x 245 mm (X x Y x Z): check its bounding
  box in the orientation it will be printed and report the numbers.
- Assume PETG (wall thickness, clearances, overhangs and tolerances chosen for PETG).
- Report every change: which document and version file, which bodies/features were added or
  changed, key dimensions, and the bed-fit check.
- Connection: `claude mcp get freecad` (user scope, host 192.168.1.79, port 9875). The auth token
  lives only in the Claude Code MCP config (~/.claude.json) - never put it in git or this file.

## Conventions
- Explain code changes briefly; put long explanations in comments, not in chat walls.
- Launch files: one per subsystem (`camera.launch.py`, `imu.launch.py`, ...),
  all included by `launch_robot.launch.py`.
- Parameters live in `config/*.yaml`, never hard-coded.
- When a test is needed, give the exact command and the expected output.
