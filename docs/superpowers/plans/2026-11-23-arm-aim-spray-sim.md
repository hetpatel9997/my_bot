# Arm Aim + Spray in Simulation - Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** In Gazebo, a target point on the ground makes the WeedBot arm aim its nozzle at it (MoveIt 2), a simulated spray relay switches on for 1 s under six safety checks, and the arm returns to park.

**Architecture:** A pure-Python closed-form aim solver turns a target into ranked joint solutions; the aim node checks them with MoveIt, plans and executes through `arm_controller` (joint_trajectory_controller in gazebo_ros2_control), then asks the spray controller (the only relay owner) to spray, then parks. Same command interface (`/arm_controller/follow_joint_trajectory`) for the real servo driver later.

**Tech Stack:** ROS 2 Humble, rclpy, MoveIt 2 (move_group, OMPL, KDL, moveit_configs_utils), ros2_control + gazebo_ros2_control (Gazebo Classic 11), numpy, pytest 6.

**Spec:** `docs/superpowers/specs/2026-11-23-arm-aim-spray-sim-design.md` (read it first; this plan argues from it).

## Global Constraints

- Package `my_bot` (ament_cmake). New Python executables go into `install(PROGRAMS ...)` in `CMakeLists.txt`; helper modules are installed the same way and imported by sibling scripts (`from aim_solver import ...`, as `weed_manager.py` does with `weed_db`). `config/` and `launch/` are installed recursively already.
- Build: `cd ~/dev_ws && colcon build --symlink-install --packages-select my_bot && source install/setup.bash`.
- Arm joint order everywhere: `('arm_base_yaw', 'shoulder_pitch', 'elbow_pitch', 'wrist_pitch', 'wrist_roll')`. Radians internally, degrees only in reports/logs.
- Park = `[0, 0, pi/2, pi/2, 0]` (`arm_park_*` in `weedbot_properties.xacro`).
- Frames: `base_link` (on the ground under the drive axle), `arm_base` (child of `base_link`), `spray_tip` (its +z points out of the nozzle), `laser_frame`, `map`, `odom`.
- Numbers from the spec (use exactly): aim error <= 0.01 m; arrived = every joint within 1 deg for 0.3 s, timeout 15 s; `spray_seconds` 1.0 (tests accept +-0.15 s); hard cap `MAX_SPRAY_SECONDS = 3.0`; relay off <= 0.1 s after e-stop / safety_state change; `/safety_state` must be `clear` and <= 0.5 s old; robot still = |v| < 0.01 m/s and |w| < 0.02 rad/s for 0.5 s, no `/odom` for 1.0 s = refuse; arm still = < 0.02 rad/s for 0.3 s; no `/joint_states` for 0.5 s = refuse; nozzle line must hit the ground within 0.8 m of `arm_base`; tilt 0-50 deg in 1 deg steps; standoff 0.08-0.25 m in 0.01 m steps; tip >= 0.03 m above the target; <= 20 candidates; LiDAR keep-clear band +-0.03 m around the `laser_frame` height, box = `scan_self_filter.box` from `config/safety.yaml` shrunk by 0.02 m; MoveIt limits 1.0 rad/s and 2.0 rad/s^2, scaling 0.5; ground box top at z 0.02 in `base_link`.
- Exact reason strings (tests assert them): solver `invalid target`, `behind the arm`, `too far`, `too close / under the arm`, `would block the LiDAR`; aim node `no transform <frame> -> base_link`, `MoveIt not ready`, `blocked by the robot body`, `planning failed`, `did not reach the aim pose`, `e-stop`, `PARK FAILED`; spray controller `not armed`, `busy`, `no joint states`, `arm moving`, `nozzle not pointing at the ground`, `no odom`, `robot moving`, `inside keepout zone <name>`, `keepout check impossible: no transform map <- base_link`, `e-stop`, `safety_state <state>`, `safety_state missing or old`; abort = `aborted: <reason>`; success = `sprayed <s:.2f> s`.
- Nothing touches the real robot: no `ssh pi`, no change to the `RealRobot` ros2_control block, firmware or protected setup. The `gpio` relay backend is written but never run in this plan.
- Every ROS test runs isolated: `source test/arm_aim/sim_env.sh` (Task 2) first, and ends with `~/dev_ws/src/my_bot/scripts/sim_cleanup.sh --domain 42` printing `Clean`, pass or fail. Gazebo only with `gui:=false` in tests; never run Nav2 at the same time (VM: 4 cores, 4.8 GB).
- `CMakeLists.txt`, `package.xml`, `description/ros2_control.xacro` and `CLAUDE.md` already hold uncommitted changes: add only the needed lines, never stage them in this plan's commits.
- Commits only if the owner OKs them: on branch `arm-aim-sim` (created from `main` with the working tree as is), staging only the paths listed in the task.

## Review Focus

1. A target in a frame TF cannot resolve (e.g. `map` before localisation): refused with `no transform <frame> -> base_link`, arm does not move. Test: Task 6, case 7.
2. A new target while a cycle runs (a detector will publish many): ignored and logged, the running cycle is unaffected, nothing is queued. Test: Task 6, case 8.
3. The spray controller is shut down mid-spray (Ctrl-C, launch stop): relay set off and the last `/spray/relay` is `false`. Test: Task 4, `test_shutdown_during_spray`.
4. `/joint_states` stops (broadcaster or driver dead): spray refused `no joint states`, never treated as "still". Test: Task 4, `test_no_joint_states`.
5. Non-finite target coordinates (bad depth pixel): `invalid target`, no exception. Test: Task 1, `test_invalid_target`.

---

### Task 1: Aim solver

**Files:**
- Create: `scripts/aim_solver.py`
- Create: `test/arm_aim/conftest.py`, `test/arm_aim/test_aim_solver.py`
- Modify: `CMakeLists.txt` (add `scripts/aim_solver.py` to `install(PROGRAMS`)

**Interfaces:**
- Consumes: the URDF from `description/robot.urdf.xacro`.
- Produces (used by Tasks 3, 5, 6):

```python
JOINT_NAMES = ('arm_base_yaw', 'shoulder_pitch', 'elbow_pitch', 'wrist_pitch', 'wrist_roll')

@dataclass(frozen=True)
class ArmGeometry:
    base_in_base_link: tuple   # (x, y, z) of arm_base in base_link (arm_base_joint origin)
    yaw_z: float               # arm_base -> yaw joint (arm_base_yaw origin z)
    l_shoulder: float          # yaw joint -> shoulder axis   (0.0707)
    l_upper: float             # shoulder -> elbow            (0.100)
    l_fore: float              # elbow -> wrist pitch         (0.100)
    l_tip: float               # wrist pitch -> spray tip     (0.131 = wrist_roll z + spray_tip_joint z)
    lower: tuple               # 5 joint limits, rad, JOINT_NAMES order
    upper: tuple
    @staticmethod
    def from_urdf(urdf_xml: str) -> 'ArmGeometry'   # ValueError if a needed joint is missing or an origin has non-zero rpy

@dataclass(frozen=True)
class AimLimits:
    tilt_max_deg: float = 50.0
    tilt_step_deg: float = 1.0
    standoff_min: float = 0.08
    standoff_max: float = 0.25
    standoff_step: float = 0.01
    tip_clearance: float = 0.03
    max_candidates: int = 20
    scan_z: Optional[float] = None       # laser_frame height in base_link; None = no keep-clear check
    keep_box: Optional[tuple] = None     # (xmin, xmax, ymin, ymax) in base_link, ALREADY shrunk
    scan_band: float = 0.03

@dataclass(frozen=True)
class AimSolution:
    joints: tuple        # 5 rad, JOINT_NAMES order, roll = 0.0
    tilt: float          # rad from straight down
    standoff: float      # m, tip -> target
    tip: tuple           # (x, y, z) in arm_base

def solve(target, geom: ArmGeometry, lim: AimLimits) -> tuple[list[AimSolution], str]
    # target (x, y, z) in arm_base; returns (solutions ranked best first, '') or ([], reason)
def nozzle_ray(joints, geom: ArmGeometry) -> tuple[np.ndarray, np.ndarray]   # tip, unit direction, arm_base
def arm_points(joints, geom: ArmGeometry, step: float = 0.01) -> np.ndarray   # (N, 3) samples along yaw joint -> shoulder -> elbow -> wrist -> tip, arm_base
def ray_point_distance(point, origin, direction) -> float                     # perpendicular distance; inf if the point is behind origin
```

- [ ] **Step 1: Write the failing tests**

`conftest.py`: put `scripts/` on `sys.path`; fixture `urdf_xml` = `xacro.process_file('description/robot.urdf.xacro', mappings={'use_ros2_control': 'true', 'sim_mode': 'false', 'wheel_mu': '1.0', 'sim_camera': 'false'}).toxml()`; fixture `geom` = `ArmGeometry.from_urdf(urdf_xml)`; helper `bl_to_arm(p, geom)` subtracts `base_in_base_link`; fixture `keep` = `AimLimits(scan_z=0.378, keep_box=(-0.51, 0.16, -0.14, 0.14))`.

`test_aim_solver.py`:
- `test_from_urdf_matches_cad`: `l_shoulder` 0.0707, `l_upper` 0.100, `l_fore` 0.100, `l_tip` 0.131, every limit +-pi/2 (all `abs=1e-6`).
- `test_random_points_exact`: 50 points, `numpy.random.default_rng(0)`, x in [0.27, 0.42], y in [-0.12, 0.12], z 0 (base_link). For every point at least one solution; for every solution `ray_point_distance(target, *nozzle_ray(s.joints, geom)) < 1e-6`, all joints within limits, `s.tip[2] >= target[2] + 0.03 - 1e-9`, `s.joints[4] == 0.0`.
- `test_ahead_point`: (0.30, 0, 0) base_link -> solutions, `sols[0].tilt < radians(30)`.
- `test_behind`: (-0.30, 0, 0) -> `([], 'behind the arm')`.
- `test_too_far`: (1.0, 0, 0) -> `([], 'too far')`.
- `test_under_arm`: (0.05, 0, 0) -> `([], 'too close / under the arm')`.
- `test_invalid_target`: (nan, 0, 0) and (0.3, inf, 0) -> `([], 'invalid target')`, no exception.
- `test_ranking`: for (0.34, 0.05, 0) the `(tilt, standoff)` pairs are non-decreasing in list order and `len(sols) <= 20`.
- `test_keep_clear_grid`: with `keep`, each of the 9 points x in (0.28, 0.34, 0.40) x y in (-0.10, 0, 0.10) has solutions, and for every solution no `arm_points` sample (shifted to base_link) has `|z - 0.378| < 0.03` with (x, y) outside the box.

- [ ] **Step 2: Run, expect failure**

Run: `source ~/dev_ws/install/setup.bash && cd ~/dev_ws/src/my_bot && python3 -m pytest test/arm_aim/test_aim_solver.py -v`
Expected: FAIL / errors with `ModuleNotFoundError: No module named 'aim_solver'`.

- [ ] **Step 3: Implement `scripts/aim_solver.py`**

`from_urdf`: parse with `xml.etree.ElementTree`; `l_tip` = `wrist_roll` origin z + `spray_tip_joint` origin z. The solve maths (the core, spelled out because the signature does not fix it):

```
yaw = atan2(y, x); r = hypot(x, y); h = z - (yaw_z + l_shoulder)       # target relative to the shoulder
for tilt in 0 .. tilt_max (step):         th = pi - tilt               # nozzle angle from straight up, leaning outward
  d = (sin th, cos th)                                                  # (radial, up)
  for s in standoff_min .. standoff_max:  tip = (r, h) - s*d ; wr = tip - l_tip*d
    c = (|wr|^2 - l_upper^2 - l_fore^2) / (2 l_upper l_fore); skip if |c| > 1
    for qe in (+acos c, -acos c):
      qs = atan2(wr.r, wr.up) - atan2(l_fore sin qe, l_upper + l_fore cos qe); qw = th - qs - qe
      keep if all within limits, tip height >= z + tip_clearance, and keep-clear passes
rank by (tilt, standoff, elbow height); cut to max_candidates
```

Reasons in this order: non-finite -> `invalid target`; yaw outside the yaw limits -> `behind the arm`; no candidate but some failed only on keep-clear -> `would block the LiDAR`; else `r >= l_upper + l_fore` -> `too far`, otherwise `too close / under the arm`.

- [ ] **Step 4: Run, expect pass**

Same command. Expected: 9 passed.

- [ ] **Step 5: Commit** (if approved; create the branch first: `git switch -c arm-aim-sim`)

```bash
git add scripts/aim_solver.py test/arm_aim/conftest.py test/arm_aim/test_aim_solver.py
git commit -m "arm: closed-form aim solver with LiDAR keep-clear rule"
```

### Task 2: Arm moving in Gazebo

**Files:**
- Modify: `description/weedbot_arm.xacro:20-33` (sim branch: `revolute`, all `q_*` 0, update the comment; add sim-only `<gazebo reference="...">` with `<gravity>false</gravity>` for `arm_turret`, `upper_arm`, `forearm`, `wrist`, `nozzle`)
- Modify: `description/ros2_control.xacro:66-94` (sim `GazeboSystem` block only: 5 arm joints with `command_interface position`, `state_interface position` with `<param name="initial_value">${arm_park_*}</param>`, `state_interface velocity`; second `<parameters>$(find my_bot)/config/arm_controllers_sim.yaml</parameters>` in the plugin)
- Modify: `launch/launch_sim.launch.py` (spawner `arm_controller` after `joint_broad`; stub parameter override `{'joints': ['caster_swivel_joint', 'caster_wheel_joint'], 'positions': [0.0, 0.0]}`)
- Create: `config/arm_controllers_sim.yaml`, `test/arm_aim/arm_test.world`, `test/arm_aim/sim_env.sh`, `test/arm_aim/start_sim.sh`, `test/arm_aim/check_arm_hold.py`, `test/arm_aim/check_drive_regression.py`

**Interfaces:**
- Produces: in sim, `/joint_states` carries the 5 arm joints; action `/arm_controller/follow_joint_trajectory` (`control_msgs/action/FollowJointTrajectory`); `test/arm_aim/sim_env.sh` and `start_sim.sh` used by Tasks 3-6.

`sim_env.sh` (sourced; exact values from CLAUDE.md "Rules for automated tests"):

```bash
source /opt/ros/humble/setup.bash && source ~/dev_ws/install/setup.bash
export ROS_DOMAIN_ID=42 ROS_LOCALHOST_ONLY=1 GAZEBO_MASTER_URI=http://127.0.0.1:11346
unset ROS_DISCOVERY_SERVER FASTRTPS_DEFAULT_PROFILES_FILE
export ARM_TEST_LOG=${ARM_TEST_LOG:-$HOME/.ros/arm_aim_test}; mkdir -p "$ARM_TEST_LOG"
```

`start_sim.sh`: sources `sim_env.sh`, starts `ros2 launch my_bot launch_sim.launch.py gui:=false world:=<abs path>/test/arm_aim/arm_test.world` in the background (log `$ARM_TEST_LOG/sim.log`), waits up to 120 s until `ros2 control list_controllers` shows `arm_controller`, `diff_cont`, `joint_broad` all `active`; on timeout (cold Gazebo start) runs `sim_cleanup.sh --domain 42` and retries once; exits non-zero if still not up.

`arm_controllers_sim.yaml`: `controller_manager.ros__parameters.arm_controller.type: joint_trajectory_controller/JointTrajectoryController`; `arm_controller.ros__parameters`: `joints` (5, order above), `command_interfaces: [position]`, `state_interfaces: [position, velocity]`, `allow_partial_joints_goal: false`, `constraints: {stopped_velocity_tolerance: 0.05, goal_time: 2.0, <each joint>: {goal: 0.02}}`. Controller manager update rate stays 10 Hz (from `my_controllers.yaml`).

`arm_test.world`: `sun`, `ground_plane`, and the `gazebo_ros_state` plugin block copied from `worlds/room.world:10-13`.

- [ ] **Step 1: Write `check_arm_hold.py`** (rclpy, `use_sim_time`): waits for TF `base_link <- spray_tip`, samples it for 30 s: every sample within 0.005 m of (0.107, z 0.276) in x and z; then sends a FollowJointTrajectory goal (yaw 0.5 rad, others park, 3 s) and asserts `/joint_states` yaw within 0.02 rad of 0.5 within 6 s; sends park back, same check at 0. Prints `ARM HOLD + MOVE PASS`, exit 0; else prints the failing check, exit 1.

- [ ] **Step 2: Run it against the current sim, expect failure**

Run: `bash test/arm_aim/start_sim.sh` -> fails waiting for `arm_controller` (the controller does not exist yet). Then `scripts/sim_cleanup.sh --domain 42` -> `Clean`.

- [ ] **Step 3: Make the xacro, yaml and launch changes listed under Files**

- [ ] **Step 4: Build and run**

```bash
cd ~/dev_ws && colcon build --symlink-install --packages-select my_bot
cd ~/dev_ws/src/my_bot && bash test/arm_aim/start_sim.sh && source test/arm_aim/sim_env.sh && python3 test/arm_aim/check_arm_hold.py
```
Expected: `ARM HOLD + MOVE PASS`.

- [ ] **Step 5: Driving regression**

`check_drive_regression.py`: publishes `/cmd_vel_keyboard` linear x 0.10 at 10 Hz for 3 s (through the safety chain; no map, so capped at 0.08 m/s), waits 2 s, asserts the `/ground_truth/odom` displacement is 0.243 +- 0.02 m (2026-10-19 baseline) and `/odom` agrees with it within 0.02 m. Prints `DRIVE REGRESSION PASS`.
Run: `python3 test/arm_aim/check_drive_regression.py` (same sim). Expected: `DRIVE REGRESSION PASS`.

- [ ] **Step 6: Clean up**

Run: `scripts/sim_cleanup.sh --domain 42`. Expected: `Clean`, exit 0.

- [ ] **Step 7: Commit** (if approved)

```bash
git add description/weedbot_arm.xacro launch/launch_sim.launch.py config/arm_controllers_sim.yaml test/arm_aim/arm_test.world test/arm_aim/sim_env.sh test/arm_aim/start_sim.sh test/arm_aim/check_arm_hold.py test/arm_aim/check_drive_regression.py
git commit -m "sim: arm joints position-controlled in Gazebo (arm_controller)"
```

### Task 3: MoveIt config

**Files:**
- Create: `config/moveit/weedbot.srdf`, `config/moveit/kinematics.yaml`, `config/moveit/joint_limits.yaml`, `config/moveit/moveit_controllers.yaml`
- Create: `launch/arm_aim.launch.py`, `config/arm_aim.rviz`
- Create: `scripts/arm_config.py`, `test/arm_aim/gen_disable_collisions.py`, `test/arm_aim/test_moveit_config.py`, `test/arm_aim/check_moveit.py`
- Modify: `package.xml` (exec_depend `moveit_ros_move_group`, `moveit_configs_utils`, `moveit_planners_ompl`, `moveit_simple_controller_manager`, `moveit_kinematics`, `moveit_ros_visualization`, `moveit_msgs`, `joint_trajectory_controller`, `control_msgs`, `trajectory_msgs`), `CMakeLists.txt` (install `scripts/arm_config.py`)

**Interfaces:**
- Consumes: Task 2's `arm_controller` and sim helpers.
- Produces: `arm_aim.launch.py` (args `use_sim_time` default `true`, `rviz` default `false`) starting `move_group`: services `/check_state_validity` (`moveit_msgs/srv/GetStateValidity`), `/apply_planning_scene`, action `/move_action` (`moveit_msgs/action/MoveGroup`); SRDF group `arm` (chain `arm_base` -> `spray_tip`), group_state `park`.
- `scripts/arm_config.py`: `load_park(srdf_path: str) -> tuple` (5 rad, JOINT_NAMES order, from group_state `park`); `load_keep_box(safety_yaml_path: str, shrink: float) -> tuple` (`scan_self_filter.ros__parameters.box` shrunk on every side).

Settings: `kinematics.yaml` arm: `kdl_kinematics_plugin/KDLKinematicsPlugin`, `kinematics_solver_search_resolution: 0.005`, `kinematics_solver_timeout: 0.05`, `position_only_ik: true`. `joint_limits.yaml`: `default_velocity_scaling_factor: 0.5`, `default_acceleration_scaling_factor: 0.5`, each joint `has_velocity_limits: true, max_velocity: 1.0, has_acceleration_limits: true, max_acceleration: 2.0`. `moveit_controllers.yaml`: `moveit_controller_manager: moveit_simple_controller_manager/MoveItSimpleControllerManager`, controller `arm_controller`, `type: FollowJointTrajectory`, `action_ns: follow_joint_trajectory`, `default: true`, the 5 joints.

Launch: `OpaqueFunction` resolving `use_sim_time` to a string, then `MoveItConfigsBuilder('weedbot', package_name='my_bot').robot_description(file_path='description/robot.urdf.xacro', mappings={'use_ros2_control': 'true', 'sim_mode': <use_sim_time>, 'wheel_mu': '1.0', 'sim_camera': 'false'}).robot_description_semantic(file_path='config/moveit/weedbot.srdf').robot_description_kinematics(file_path='config/moveit/kinematics.yaml').joint_limits(file_path='config/moveit/joint_limits.yaml').trajectory_execution(file_path='config/moveit/moveit_controllers.yaml').planning_pipelines(pipelines=['ompl']).to_moveit_configs()` (OMPL comes from moveit_configs_utils' defaults: no `config/ompl_planning.yaml` exists); `move_group` node with `moveit_config.to_dict()` + `use_sim_time`; RViz (`config/arm_aim.rviz`: fixed frame `base_link`, RobotModel, MotionPlanning display for group `arm`) only if `rviz` is true, with the same MoveIt parameters.

`gen_disable_collisions.py`: from the sim URDF, groups links into rigid groups (connected by fixed joints); prints a `<disable_collisions link1 link2 reason="Adjacent"/>` for every pair inside one group or in two groups one movable joint apart, sorted; `--write` replaces the block between `<!-- BEGIN generated -->` and `<!-- END generated -->` in `weedbot.srdf`.

- [ ] **Step 1: Install the launch helper** (owner approved 2026-11-23)

Run: `sudo apt install -y ros-humble-moveit-configs-utils`, then `python3 -c "import moveit_configs_utils; print('ok')"`. Expected: `ok`.

- [ ] **Step 2: Write the failing tests** (`test_moveit_config.py`, pytest, no ROS)

- `test_srdf_group`: group `arm` is a chain `base_link="arm_base"` `tip_link="spray_tip"`.
- `test_park_matches_xacro_and_stub`: `load_park('config/moveit/weedbot.srdf')` equals the `arm_park_*` values in `weedbot_properties.xacro` (evaluate `${pi/2}` with `pi = math.pi`) and the first 5 `positions` in `config/arm_joint_state_stub.yaml`, `abs=1e-6`.
- `test_joint_limits_all_joints`: all 5 joints, `max_velocity == 1.0`, `max_acceleration == 2.0`.
- `test_disable_list_up_to_date`: `gen_disable_collisions.py` output equals the SRDF's generated block.
- `test_load_keep_box`: `load_keep_box('config/safety.yaml', 0.02) == pytest.approx((-0.51, 0.16, -0.14, 0.14))`.

Run: `python3 -m pytest test/arm_aim/test_moveit_config.py -v`. Expected: FAIL (files missing).

- [ ] **Step 3: Write the MoveIt files, `arm_config.py`, the generator; run `python3 test/arm_aim/gen_disable_collisions.py --write`**

- [ ] **Step 4: Run the tests, expect pass** - same command, 5 passed.

- [ ] **Step 5: Write `check_moveit.py`** (rclpy, sim + `arm_aim.launch.py` running): park -> `/check_state_validity` `valid` true and no contacts; goal G = (0.3, 0.3, 1.2, 1.2, 0.0) rad valid; MoveGroup plan+execute to G (joint constraints, tolerance 0.001) -> `error_code.val == 1` and `/joint_states` within 1 deg of G within 15 s; back to park, same checks. Prints `MOVEIT PASS`. If park is invalid because of a contact between links that are NOT in neighbouring groups, stop and report it (model problem, do not hide it in the SRDF).

- [ ] **Step 6: Run it**

```bash
cd ~/dev_ws && colcon build --symlink-install --packages-select my_bot
cd ~/dev_ws/src/my_bot && bash test/arm_aim/start_sim.sh && source test/arm_aim/sim_env.sh
ros2 launch my_bot arm_aim.launch.py > $ARM_TEST_LOG/arm_aim.log 2>&1 &
python3 test/arm_aim/check_moveit.py
scripts/sim_cleanup.sh --domain 42
```
Expected: `MOVEIT PASS`, then `Clean`.

- [ ] **Step 7: Commit** (if approved)

```bash
git add config/moveit launch/arm_aim.launch.py config/arm_aim.rviz scripts/arm_config.py test/arm_aim/gen_disable_collisions.py test/arm_aim/test_moveit_config.py test/arm_aim/check_moveit.py
git commit -m "arm: MoveIt 2 config (group arm, park) and arm_aim launch"
```

### Task 4: Spray controller

**Files:**
- Create: `scripts/spray_controller_node.py`, `config/arm_aim.yaml` (section `spray_controller`)
- Create: `test/arm_aim/test_spray_controller.py`, `test/arm_aim/run_unit_tests.sh` (sources `sim_env.sh`, runs `python3 -m pytest "$@" -v`, then `sim_cleanup.sh --domain 42`)
- Modify: `CMakeLists.txt` (install `scripts/spray_controller_node.py`)

**Interfaces:**
- Consumes: `polygon_distance(poly, x, y)` from `scripts/keepout_zones_node.py` (import the module; it is installed next to this one). Zones file format = `config/backyard_zones.yaml` (`frame_id`, `zones: [{name, kind, polygon, margin}]`); inside = `polygon_distance(...) <= margin`.
- Produces (used by Tasks 5, 6): node `spray_controller`; services `/spray/arm` (`std_srvs/SetBool`), `/spray/trigger` (`std_srvs/Trigger`, answers after the spray ends or is refused; `success` = sprayed, `message` = reason string from Global Constraints); topics `/spray/relay` (`std_msgs/Bool`, transient local, depth 1), `/spray/status` (`std_msgs/String` JSON `{armed, relay, last}`, 1 Hz).

```python
MAX_SPRAY_SECONDS = 3.0
def initial_armed(backend: str, start_armed: bool) -> bool      # True only for backend 'sim' and start_armed
class SimRelay:  def set(self, on: bool) -> None; def close(self) -> None
class GpioRelay: def __init__(self, chip: int, pin: int, active_low: bool)  # RuntimeError if pin < 0 or lgpio missing
class SprayController(Node): def __init__(self, relay=None, **node_kwargs)  # relay injected by tests; else from `backend`
def main()   # MultiThreadedExecutor
```

Parameters (`config/arm_aim.yaml`): `backend: sim`, `start_armed: false`, `spray_seconds: 1.0`, `zones_file: ''` (re-read when the parameter changes), `odom_topic: /odom`, `joint_states_topic: /joint_states`, `e_stop_topic: /e_stop`, `safety_state_topic: /safety_state`, `safety_max_age: 0.5`, `still_linear: 0.01`, `still_angular: 0.02`, `still_time: 0.5`, `odom_timeout: 1.0`, `arm_still_speed: 0.02`, `arm_still_time: 0.3`, `joint_states_timeout: 0.5`, `ground_reach: 0.8`, `base_frame: base_link`, `map_frame: map`, `arm_base_frame: arm_base`, `nozzle_frame: spray_tip`, `gpio_chip: 0`, `gpio_pin: -1`, `active_low: false`.

Behaviour the tests pin: relay set off first thing in `__init__`, in `destroy_node()` and in a `finally` around every spray; the e-stop, `/safety_state` and `/spray/arm` false callbacks switch the relay off directly (not only the 20 Hz monitor); trigger service in its own callback group, subscriptions in a reentrant group; arm stillness from position differences over `arm_still_time` (not the velocity field); nozzle check from TF `base_link <- spray_tip`: +z must have z component < -0.1, ground hit point within `ground_reach` of `arm_base` in x-y; the keepout point is that ground hit converted to `map`. Checks run in the order of the spec (armed, arm, robot, keepout, e-stop, safety_state); a trigger while spraying -> `busy`.

- [ ] **Step 1: Write the failing tests** (pytest + rclpy in-process; fixture `world` = a helper node publishing fake `/odom` (zero twist, 20 Hz), `/joint_states` (park, 20 Hz), `/safety_state` `clear` (10 Hz), `/e_stop` false, static TF `base_link -> arm_base` (0.0066, 0, 0.2336), `base_link -> spray_tip` at (0.30, 0, 0.20) with +z pointing down, and optionally `map -> base_link` identity; fixture `ctl` = `SprayController(relay=FakeRelay())` with `start_armed: true` (`FakeRelay` records `(time, on)` calls); both spun in one MultiThreadedExecutor thread; timing measured on `FakeRelay` calls)

- `test_relay_off_at_start`: first FakeRelay call is `False`; first `/spray/relay` message is `false`.
- `test_initial_armed`: `initial_armed('sim', True)` True; `('sim', False)`, `('gpio', True)`, `('gpio', False)` False.
- `test_refuses_when_disarmed`: after `/spray/arm` false -> `(False, 'not armed')`, relay never on.
- `test_sprays_for_spray_seconds`: `(True, 'sprayed 1.00 s')`, on-time 1.0 +- 0.15 s.
- `test_robot_moving`: odom vx 0.05 -> `'robot moving'`.
- `test_no_odom`: odom stopped 1.5 s -> `'no odom'`.
- `test_arm_moving`: yaw position ramps 0.1 rad/s -> `'arm moving'`.
- `test_no_joint_states`: joint states stopped 1.0 s -> `'no joint states'`.
- `test_nozzle_up`: spray_tip +z up -> `'nozzle not pointing at the ground'`.
- `test_keepout`: zones file with zone `test_bed` = square of side 0.2 around (0.30, 0) in `map`, `map -> base_link` identity -> `'inside keepout zone test_bed'`.
- `test_keepout_no_map`: same zones file, no `map` transform -> `'keepout check impossible: no transform map <- base_link'`.
- `test_safety_not_clear` (parametrize): `slow` -> `'safety_state slow'`; `stop` -> `'safety_state stop'`; publisher stopped 1 s -> `'safety_state missing or old'`.
- `test_estop_during_spray`: `spray_seconds` 2.0, `/e_stop` true 0.5 s after the relay goes on -> relay off within 0.1 s of the publish, response `'aborted: e-stop'`.
- `test_safety_stop_during_spray`: same with `/safety_state` `stop` -> off within 0.1 s, `'aborted: safety_state stop'`.
- `test_abort_during_spray` (parametrize): 0.5 s into a 2 s spray, `/spray/arm` false -> `'aborted: not armed'`; odom vx 0.05 -> `'aborted: robot moving'`; yaw ramping 0.1 rad/s -> `'aborted: arm moving'`; each relay off by 0.5 s after the change.
- `test_cap_3s`: `spray_seconds` 10.0 -> on-time <= 3.1 s.
- `test_busy`: a second trigger during a spray -> `(False, 'busy')`, first spray unaffected.
- `test_shutdown_during_spray`: `ctl.destroy_node()` 0.3 s into a spray -> last FakeRelay call `False` and last `/spray/relay` message `false`.

Run: `bash test/arm_aim/run_unit_tests.sh test/arm_aim/test_spray_controller.py`. Expected: FAIL (`No module named 'spray_controller_node'`), then `Clean`.

- [ ] **Step 2: Implement `scripts/spray_controller_node.py` and the `spray_controller` section of `config/arm_aim.yaml`**

- [ ] **Step 3: Run, expect pass** - same command. Expected: 22 passed (the two parametrized tests count 3 each), then `Clean`.

- [ ] **Step 4: Commit** (if approved)

```bash
git add scripts/spray_controller_node.py config/arm_aim.yaml test/arm_aim/test_spray_controller.py test/arm_aim/run_unit_tests.sh
git commit -m "spray: relay gatekeeper with the six safety checks (sim backend)"
```

### Task 5: Aim node

**Files:**
- Create: `scripts/weed_aim_node.py`, `test/arm_aim/test_weed_aim_helpers.py`, `test/arm_aim/fixtures.py`, `test/arm_aim/check_aim_once.py`
- Modify: `config/arm_aim.yaml` (section `weed_aim`), `launch/arm_aim.launch.py` (add `weed_aim` and `spray_controller`; `spray_controller` gets `start_armed: true` only when `use_sim_time` is true), `config/arm_aim.rviz` (MarkerArray `/aim/markers`), `CMakeLists.txt` (install `scripts/weed_aim_node.py`)

**Interfaces:**
- Consumes: Task 1 `ArmGeometry.from_urdf`, `AimLimits`, `solve`, `ray_point_distance`, `JOINT_NAMES`; Task 3 `load_park`, `load_keep_box`, `/check_state_validity`, `/apply_planning_scene`, `/move_action`; Task 4 `/spray/trigger`, `/spray/relay`.
- Produces (used by Task 6 and sub-project 2): node `weed_aim`; inputs `/aim_target` and `/clicked_point` (`geometry_msgs/PointStamped`, any frame); `/aim/result` (`std_msgs/String` JSON with keys `target` {`frame`, `base_link_xyz`}, `reachable`, `reason`, `joints_deg`, `tilt_deg`, `standoff_m`, `aim_error_m`, `sprayed`, `spray_reason`, `durations` {`solve`, `move`, `spray`, `park`}, `state` (`ok` | `refused` | `error`)); `/aim/markers` (`visualization_msgs/MarkerArray`: target sphere, nozzle line, spray cone only while `/spray/relay` is true); service `/aim/park` (`std_srvs/Trigger`).
- Helpers in `weed_aim_node.py` (tested without ROS): `reached(current: dict, goal: Sequence[float], tol_rad: float) -> bool` (False if any arm joint is missing); `joint_goal_constraints(goal: Sequence[float], tol: float = 0.001) -> moveit_msgs.msg.Constraints`.
- `test/arm_aim/fixtures.py`: `start_map_fixture(node: Node) -> None` publishes a latched empty `/map` (`nav_msgs/OccupancyGrid`, 10 x 10 m, 0.05 m, all 0, origin (-5, -5), frame `map`) and a static TF `map -> odom` identity.

Parameters (`weed_aim` in `config/arm_aim.yaml`): `base_frame: base_link`, `arm_base_frame: arm_base`, `laser_frame: laser_frame`, `safety_yaml: ''` and `srdf_path: ''` (the launch file passes the installed `config/safety.yaml` and `config/moveit/weedbot.srdf`), `keep_clear_shrink: 0.02`, `scan_band: 0.03`, `tilt_max_deg: 50.0`, `standoff_min: 0.08`, `standoff_max: 0.25`, `tip_clearance: 0.03`, `spray: true`, `velocity_scaling: 0.5`, `acceleration_scaling: 0.5`, `arrive_tolerance_deg: 1.0`, `arrive_hold: 0.3`, `arrive_timeout: 15.0`, `planning_time: 5.0`, `ground_top: 0.02`.

Cycle (one at a time; a target while busy or in error is logged and dropped, no result): target -> TF to `base_link` (else `no transform <frame> -> base_link`) -> `arm_base` -> `solve` -> candidates through `/check_state_validity` in order (none valid -> `blocked by the robot body`) -> MoveGroup goal (`group_name 'arm'`, joint constraints, `num_planning_attempts 3`, `allowed_planning_time`, the scalings, `plan_only false`; not available -> `MoveIt not ready`, failed -> `planning failed`) -> `reached` held `arrive_hold` within `arrive_timeout` (else `did not reach the aim pose`, park) -> aim error = `ray_point_distance(target_bl, tip, +z of spray_tip)` from TF -> `/spray/trigger` if `spray` -> park (fail -> `PARK FAILED`, state `error` until `/aim/park` succeeds) -> publish result. At start: robot description from `/robot_description` (transient local), ground box (`id 'ground'`, 4 x 4 x 0.1 m, top at `ground_top`, frame `base_link`) via `/apply_planning_scene`. `/e_stop` true during a cycle: cancel the MoveGroup goal, no spray, result `e-stop`, state `error`; `/aim/park` refuses with `e-stop` while it is still true.

- [ ] **Step 1: Write the failing helper tests** (`test_weed_aim_helpers.py`)

- `test_reached_within_tolerance`: goal park, current park + 0.9 deg on each joint -> True; + 1.1 deg on one -> False.
- `test_reached_missing_joint`: current without `wrist_roll` -> False.
- `test_joint_goal_constraints`: 5 `JointConstraint`s in JOINT_NAMES order, positions equal the goal, tolerances 0.001, weight 1.0.

Run: `bash test/arm_aim/run_unit_tests.sh test/arm_aim/test_weed_aim_helpers.py`. Expected: FAIL (module missing).

- [ ] **Step 2: Implement `weed_aim_node.py`, the `weed_aim` parameters, the launch and RViz additions, `fixtures.py`**

- [ ] **Step 3: Run the helper tests, expect pass** - same command, 3 passed.

- [ ] **Step 4: Write `check_aim_once.py`**: `start_map_fixture`, wait for `/safety_state` `clear` (30 s), publish `/aim_target` (0.34, 0, 0) in `base_link`; within 40 s one `/aim/result` with `reachable` true, `aim_error_m <= 0.01`, `sprayed` true, `spray_reason == 'sprayed 1.00 s'`, `state == 'ok'`; afterwards `/joint_states` within 1 deg of park. Prints `AIM ONCE PASS`.

- [ ] **Step 5: Run it**

```bash
cd ~/dev_ws && colcon build --symlink-install --packages-select my_bot
cd ~/dev_ws/src/my_bot && bash test/arm_aim/start_sim.sh && source test/arm_aim/sim_env.sh
ros2 launch my_bot arm_aim.launch.py > $ARM_TEST_LOG/arm_aim.log 2>&1 &
python3 test/arm_aim/check_aim_once.py
scripts/sim_cleanup.sh --domain 42
```
Expected: `AIM ONCE PASS`, then `Clean`.

- [ ] **Step 6: Commit** (if approved)

```bash
git add scripts/weed_aim_node.py config/arm_aim.yaml launch/arm_aim.launch.py config/arm_aim.rviz test/arm_aim/test_weed_aim_helpers.py test/arm_aim/fixtures.py test/arm_aim/check_aim_once.py
git commit -m "arm: weed_aim node (aim -> spray -> park cycle with report)"
```

### Task 6: Full headless test, docs, owner demo

**Files:**
- Create: `test/arm_aim/run_sim_aim_test.sh`, `test/arm_aim/sim_aim_check.py`
- Modify: `CLAUDE.md` (new section "Arm aim + spray (sim)"; in "Robot model" replace "In SIM the arm joints are FIXED in that pose ... Add a Gazebo position controller before moving the arm in sim." with the new state; mention `arm_joint_state_stub` publishes only the caster joints in sim)

**Interfaces:**
- Consumes: everything above (`start_sim.sh`, `arm_aim.launch.py`, `fixtures.start_map_fixture`, `/aim/result`, `/spray/relay`, `/aim/park`, `sim_person.py`, `check_drive_regression.py`).

`run_sim_aim_test.sh`: `trap` that always runs `sim_cleanup.sh --domain 42` and fails the run if it does not print `Clean`; `start_sim.sh`; `arm_aim.launch.py` in the background; `python3 test/arm_aim/check_arm_hold.py`; `python3 test/arm_aim/sim_aim_check.py`; `python3 test/arm_aim/check_drive_regression.py`; exit non-zero if any failed.

- [ ] **Step 1: Write `sim_aim_check.py`** (`start_map_fixture`; wait for `/safety_state` `clear`; records `/spray/relay` edges, `/joint_states`, raw `/scan` in `base_link` and `/safety_state` throughout). Cases, each printed `PASS`/`FAIL <why>`, final line `ARM AIM SIM TEST: PASS (n/n)` or `FAIL (k/n)`:
  1. 9 targets x in (0.28, 0.34, 0.40) x y in (-0.10, 0, 0.10), z 0, `base_link`: `reachable`, `aim_error_m <= 0.01`, `sprayed`, relay rising edge after the joints were within 1 deg of `joints_deg`, on-time 1.0 +- 0.15 s, joints back within 1 deg of park; during each aim no raw `/scan` return outside (-0.53, 0.18, -0.16, 0.16) and `/safety_state` stays `clear`.
  2. (0.80, 0, 0) -> `too far`; (-0.30, 0, 0) -> `behind the arm`; (0.05, 0, 0) -> `too close / under the arm` or `blocked by the robot body`; for all three every joint stays within 0.5 deg of park and the relay never goes on.
  3. Keepout: write a temp zones file (zone `test_bed`, square of side 0.1 around map (0.34, 0.10), margin 0), set `/spray_controller zones_file` to it, target (0.34, 0.10, 0) -> `sprayed` false, `spray_reason` starts with `inside keepout zone`, arm parks; set `zones_file` back to `''`.
  4. Robot moving: publish `/cmd_vel_keyboard` x 0.05 at 10 Hz from the target until the result -> `spray_reason` is `robot moving` or `aborted: robot moving`; stop, wait 2 s.
  5. E-stop: `spray_seconds` 2.0; target (0.34, 0, 0); on the relay rising edge publish `/e_stop` true -> relay off within 0.1 s (by message receipt), `spray_reason` `aborted: e-stop`, `state` `error`, `/aim/park` refused while e-stop is on; then `/e_stop` false, `/aim/park` succeeds, joints within 1 deg of park; `spray_seconds` back to 1.0.
  6. Person: `ros2 run my_bot sim_person.py spawn --ahead 1.0`, wait until `/safety_state` is not `clear`, target (0.34, 0, 0) -> `spray_reason` starts with `safety_state`, arm parks; `sim_person.py remove`, wait for `clear`.
  7. Target (0.34, 0, 0) in frame `nowhere` -> `reason` `no transform nowhere -> base_link`, joints never leave park.
  8. Target A (0.34, 0, 0), 0.5 s later target B (0.28, 0.10, 0): exactly one result (A's, `ok`) and no second result in the 5 s after it; then target B again -> `ok`.

- [ ] **Step 2: Run the whole suite**

```bash
cd ~/dev_ws/src/my_bot && bash test/arm_aim/run_unit_tests.sh test/arm_aim
bash test/arm_aim/run_sim_aim_test.sh
```
Expected: all unit tests pass; `ARM HOLD + MOVE PASS`, `ARM AIM SIM TEST: PASS (8/8)`, `DRIVE REGRESSION PASS`, `Clean`. Any failure: fix in the owning task's code, re-run.

- [ ] **Step 3: Update `CLAUDE.md`** with the measured numbers from Step 2 (aim errors, on-times), the demo commands below, the test commands, the topics/services, and the safety notes (the real robot's spray controller starts disarmed; the `gpio` backend is unused until sub-project 4).

Owner demo (VM, `ros_local` in each terminal): terminal 1 `ros2 launch my_bot launch_sim.launch.py world:=$HOME/dev_ws/src/my_bot/test/arm_aim/arm_test.world`; terminal 2 `ros2 launch my_bot arm_aim.launch.py rviz:=true`; terminal 3 `python3 ~/dev_ws/src/my_bot/test/arm_aim/fixtures.py` (runs `start_map_fixture` standalone so `/safety_state` turns `clear`); in RViz use "Publish Point" on the ground 0.25-0.45 m in front of the robot. Success: the arm aims, an orange spray cone shows for 1 s, the arm parks, `ros2 topic echo /aim/result` shows `aim_error_m` under 0.01. (`fixtures.py` therefore needs a `__main__` that starts the fixture and spins.)

- [ ] **Step 4: Commit** (if approved; `CLAUDE.md` is not staged, see Global Constraints)

```bash
git add test/arm_aim/run_sim_aim_test.sh test/arm_aim/sim_aim_check.py test/arm_aim/fixtures.py
git commit -m "test: headless arm aim + spray suite"
```
