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
  `laser_frame` (chassis + x 0.175, z 0.175). Do not use the vendor `ld06.launch.py` (frame
  `base_laser` + its own static TF).
- Drive test (manual, nothing starts at boot). Pi terminal 1: `ros2 launch my_bot launch_robot.launch.py`
  (keep the robot still ~3 s for IMU calibration); Pi terminal 2: `ros2 launch my_bot lidar.launch.py`;
  Pi terminal 3 (owner only, wheels off the ground first): `ros2 run teleop_twist_keyboard
  teleop_twist_keyboard --ros-args -r cmd_vel:=/diff_cont/cmd_vel_unstamped -p speed:=0.1 -p turn:=0.5`;
  VM: `rviz2 -d ~/dev_ws/src/my_bot/config/drive_test.rviz` (fixed frame odom, robot model, /scan, TF).
  twist_mux is not installed on the Pi and its config (`use_stamped: true`) does not match diff_cont.

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
  on the rear, L298N driver, printed spur gears to 8 mm axles, 5-inch (127 mm) wheels.
  Front: single printed caster (after the drivetrain rebuild; before it, two fixed wheels).
- Gear ratio: 2:1 now (20T:40T). After the rebuild: 30T:40T = 1.33:1.
  (Only if the ros2_control stack is used: `enc_counts_per_rev` in diffdrive_arduino =
  motor counts per rev × gear ratio.)
- `wheel_radius` 0.0635 m, `wheel_separation` ≈ 0.206 m (measure after rebuild).
- Sensors: LD06 2D LiDAR (to move to rear-top post ~30 cm), MPU6050 IMU on Pi I2C bus 1 at 0x68
  (`scripts/mpu6050_node.py` -> `/imu/data_raw`; yaw rate fused with wheel odometry by the
  robot_localization EKF, `config/ekf.yaml`, which publishes `/odom` and odom->base_link;
  diff_cont has `enable_odom_tf: false` on the real robot, true in Gazebo), camera (Pi cam v2 or USB webcam,
  front, 25–40 cm high, tilted 30–45° down, frames `camera_link` -> `camera_link_optical`).
- Arm: 3-servo arm mounted over the front axle (not wired yet). A worm-gear motor is
  reserved for the arm lift joint later. Spray: small 12 V pump switched by a relay.
- Power: 12 V Li-ion pack, 5 V for Pi. Battery not monitored yet.
- Frame: 2020/4020 aluminium extrusion, 360 × 240 × 160 mm, based on the "NXP robot platform" CAD.

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
camera_link, camera_link_optical, arm_base_link ... spray_nozzle_link.

## Roadmap and current status
- Phase 1 (POSTPONED, resume after Phase 3): camera publishing + intrinsic calibration +
  camera in URDF/TF.
- Phase 2 (in progress): MPU6050 + robot_localization EKF (odom + IMU). Files: mpu6050_node.py,
  ekf.yaml, imu.xacro, imu_ekf.launch.py.
- Phase 3: coverage (boustrophedon lanes via coverage_planner_node, NavigateToPose per lane end,
  /coverage/plan|start|stop). INDOOR ONLY for now: the planner reads /map and does not yet honour
  a keepout mask, so outdoor use needs the Nav2 keepout filter (pool + 1 m) wired in first.
- Phase 4: detector node on the Pi (low fps is fine). Needs Phase 1 camera.
- Phase 5: weed localization to map frame, verified within 10 cm at 1 m.
- Phase 6: weed_manager + NavigateToPose to a standoff pose.
- Phase 7–8: cloud sim / synthetic data (optional).
- Phase 9–10: arm, spray, precision alignment, TreatWeed action.
- Phase 11: mission executor, docking, battery, safety.
Each phase ends with a verification test; do not move on until it passes.

## Hard safety rules
- The backyard has a swimming pool. LiDAR cannot see water. Any outdoor navigation config
  MUST include a keepout zone around the pool with at least 1 m margin. Refuse to generate
  outdoor coverage paths without it.
- Never spray inside a keepout/flower-bed zone. Never enable the spray relay in simulation
  or on the bench without the owner explicitly asking in that session.
- Cap `max_vel_x` at 0.3 m/s and `max_vel_theta` at 1.0 rad/s in nav2_params.
- Stop everything if a person or pet is within 2 m (to be implemented; keep the hook).

## Conventions
- Explain code changes briefly; put long explanations in comments, not in chat walls.
- Launch files: one per subsystem (`camera.launch.py`, `imu.launch.py`, ...),
  all included by `launch_robot.launch.py`.
- Parameters live in `config/*.yaml`, never hard-coded.
- When a test is needed, give the exact command and the expected output.
