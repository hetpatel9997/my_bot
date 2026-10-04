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
  teleop_twist_keyboard --ros-args -r cmd_vel:=/cmd_vel_keyboard -p speed:=0.1 -p turn:=0.5`;
  VM: `rviz2 -d ~/dev_ws/src/my_bot/config/drive_test.rviz` (fixed frame odom, robot model, /scan, TF).
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
  0.12 m (0.3 hid the stop zone; <= 0.05 starts rays on the laser housing and returns nothing). Sim test 2026-10-03: Nav2 goal 1 m SUCCEEDED;
  keyboard override and /e_stop both stopped the robot, Nav2 resumed after release.
- Gazebo coverage test (VM only; run `ros_local` first in EVERY terminal so the sim stays off the
  Pi's discovery server): world `worlds/room.world` (5 x 4 m, table + couch boxes, robot spawns at
  the centre); `launch_sim.launch.py world:=... gui:=false`, `online_async_launch.py` (SLAM),
  `map_saver_cli`, `localization_launch.py` + `navigation_launch.py` with `map:=` and
  `use_sim_time:=true`, `coverage.launch.py use_sim_time:=true`, RViz `config/coverage_sim.rviz`.
  A cold first Gazebo start can exceed spawn_entity's 30 s; rerun.
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
- VM performance (2 cores, 3.8 GB, VMware SVGA3D GPU; driver crash traces in the desktop log):
  real-time factor 0.37 with the Gazebo window. Since 2026-10-03 (sim camera off by default,
  room.world physics 500 Hz): 1.00 headless, 0.99 headless + SLAM + Nav2 (load ~2).
  2026-10-03 a full stack with RViz + AMCL + Nav2 + coverage + teleops reached load 14 with
  229 MB free and gzserver FROZE (threads stuck in futex waits, /clock stopped, Nav2 lifecycle
  never finished -> all goals rejected). If goals do nothing: check `ros2 topic hz /clock` and
  `ros2 lifecycle get /bt_navigator` first. Use `gui:=false`, start RViz last, never set
  LIBGL_ALWAYS_SOFTWARE (forces llvmpipe). Best fix: give the VM 4 cores / 6-8 GB.
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
  Blocked path: waits blocked_wait (5 s), re-plans around if there is room, else skips 1 m into
  a revisit queue retried at the end; the final report (log + latched `/coverage/report`) lists
  anything still skipped with its map location. Needs python3-scipy (apt) on the Pi.
  Person tests 2026-10-04 (room world, medium, 0.15 m/s top speed, path 0.03 m further from walls):
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

### Drive calibration: PENDING until after the drivetrain rebuild (noted 2026-10-02)
Hardware is on the bench, not in the chassis; rebuild = caster in front, 30T:40T gears. Open items:
- (a) diff_cont odom showed 0.227 m/s for a 0.1 m/s teleop command on the bench. Check
  `loop_rate` in `ros2_control.xacro` (30) against `PID_RATE` in ROSArduinoBridge and the owner's
  uncommitted edits to `~/robot_ws/src/diffdrive_arduino/hardware/diffbot_system.cpp`.
- (b) Encoder counts per wheel turn (`enc_counts_per_rev`, now 3436) unverified.
- (c) IMU `axis_signs` (imu_ekf.launch.py) and imu.xacro pose to verify after mounting.
- (d) Occasional IMU I2C read errors when the motors start (check wiring/power/noise).
- After the rebuild, update the URDF (`robot_core.xacro`: wheel radius, wheel positions/separation,
  caster) to the real robot's measured dimensions, matching `my_controllers.yaml`. Then delete the
  sim-only `wheel_separation`/`wheel_radius` override in `config/gaz_ros2_ctl_use_sim.yaml`, so sim
  and real share one set of dimensions. (Today URDF wheels: r 0.05, sep 0.35; controller: r 0.06985,
  sep 0.1895.)

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
  from the front bumper to 0.10 m ahead (0.40 m wide); slow (40 %) corridor 0.40 m ahead of the
  bumper. Full outline+0.10 m boxes stopped the robot beside every wall; Nav2 "approach" mode
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

## Conventions
- Explain code changes briefly; put long explanations in comments, not in chat walls.
- Launch files: one per subsystem (`camera.launch.py`, `imu.launch.py`, ...),
  all included by `launch_robot.launch.py`.
- Parameters live in `config/*.yaml`, never hard-coded.
- When a test is needed, give the exact command and the expected output.
