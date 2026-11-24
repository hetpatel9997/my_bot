# Arm aims and sprays a weed, in simulation (sub-project 1) - design

Date: 2026-11-23. Status: Approved 2026-11-23; implemented on branch arm-aim-sim (2026-11-24).

**Amendment 2026-11-24 (owner, branch arm-4-servos): the arm has FOUR servos, no wrist roll.**
Wherever this document says 5 joints / `wrist_roll`, read 4 joints (arm_base_yaw, shoulder_pitch,
elbow_pitch, wrist_pitch). The nozzle is fixed straight on the wrist-pitch servo with its tip about
0.060 m from that axis (was 0.131 m through a roll servo), so the arm reaches about 10 cm less: the
9-target test grid moved from x 0.28 / 0.34 / 0.40 to 0.24 / 0.29 / 0.34 m. The aiming rules (tilt
0-50 deg, standoff 0.08-0.25 m) and everything else are unchanged.

## 1. What this is

The WeedBot arm work is split into four sub-projects, built in this order:

1. **Arm aims and sprays in simulation** (THIS document): give the arm a target point on the
   ground, it aims the nozzle at it, the spray relay switches on, then off, then the arm parks.
2. Camera to weed position: simulated phone camera + fake weeds in Gazebo, a detection in the
   picture turned into a ground point next to the robot, robot drives up so the weed is in reach.
3. Weed detector training: YOLO (any kind of weed, one general `weed` class), public dataset +
   the owner's own lawn photos, trained on Colab, exported for the Pi.
4. The real arm: servo power and driver, relay wiring, servo calibration, arm-aware LiDAR filter.

Sub-project 2 later replaces the "clicked point" input of this one with the detector's output;
nothing else in this design changes for that.

### What the owner said (2026-11-23)
- Use the iPhone BACK camera (it faces the robot's front and has the LiDAR depth sensor).
- Detect any kind of weed in general. Training data: a public dataset plus own photos.
- The arm is built; there is no servo board yet (owner planned to wire servos to the Pi directly;
  decided in sub-project 4, see section 9).
- Spray at an angle while the weed is still in the camera's view (not drive-over-and-spray-down).
- Use MoveIt 2 for the arm.
- When the arm reaches its aim pose it switches on the spray relay, the weed is sprayed, and when
  done the arm goes back to its normal (parked) position.
- OK to `sudo apt install ros-humble-moveit-configs-utils`.

### Assumptions (owner can correct)
- The robot stands still while the arm aims and sprays (stop, aim, spray, park, then drive on).
- The weed is on flat ground (z = 0 in `base_link`). Sub-project 2 supplies real heights.
- In simulation there is no pump: the "relay" is a ROS signal and a spray cone drawn in RViz.

### Success (all must hold, checked by the automatic tests in section 7)
- 9 reachable targets: the nozzle's centre line passes within **1 cm** of each target, the relay
  turns on only after the arm has arrived, stays on 1.0 s (+-0.15 s), and the arm returns to park
  (every joint within 1 deg).
- 3 impossible targets: refused with a reason, the arm never moves, the relay never turns on.
- Every refusal case in section 5 leaves the relay off. An e-stop during a spray, or
  `/safety_state` leaving `clear`, turns the relay off within 0.1 s.
- The arm holds the parked pose in Gazebo, and normal driving in the sim is unchanged.

## 2. Key facts behind the design

- Arm (from `weedbot_properties.xacro`, CAD v21): yaw axis at x 0.0066 on the top cover (z 0.2366);
  shoulder 0.0707 above it (z 0.3073); upper arm 0.100, forearm 0.100, wrist pitch axis to spray
  tip 0.131; wrist roll turns the nozzle about its own axis (irrelevant for aiming, kept at 0).
  All joints -90..+90 deg, 0 = straight up, + pitch tilts forward, + yaw = counterclockwise.
  The three pitch axes are parallel and the shoulder sits on the yaw axis, so aiming is exact
  closed-form maths: yaw toward the point, then a flat 3-link problem.
- Camera (back camera, 1280 x 960, fx 891 -> 71 x 57 deg field of view), at (0.127, 0.055, 0.194),
  35 deg down: it sees the ground from about **0.23 m** ahead of `base_link` outwards.
- The nozzle can point straight down only over ground about **0.15-0.20 m** ahead: the camera
  cannot see there. Pointing 10-45 deg from vertical, from 16-24 cm away, it reaches ground about
  **0.25-0.45 m ahead and +-0.15 m sideways**, which the camera does see. (Throwaway estimate with
  rough body boxes, `reach.py` in the 2026-11-23 session scratchpad; MoveIt refines it with the
  real collision shapes.) So: angled spraying, weed in view.

## 3. Pieces and how they connect

| Piece | File(s) | One job |
|---|---|---|
| Aim solver | `scripts/aim_solver.py` (plain Python + numpy, no ROS) | target point -> ranked list of joint-angle solutions, or a reason why not |
| Sim arm | `description/weedbot_arm.xacro`, `description/ros2_control.xacro`, `config/arm_controllers_sim.yaml`, `launch/launch_sim.launch.py` | arm joints move in Gazebo, driven by `arm_controller` |
| MoveIt config | `config/moveit/` (SRDF + yaml), `launch/arm_aim.launch.py` | collision-free planning and execution for the arm |
| Aim node | `scripts/weed_aim_node.py`, `config/arm_aim.yaml` | runs one aim -> spray -> park cycle per target and reports it |
| Spray controller | `scripts/spray_controller_node.py`, `config/arm_aim.yaml` | the ONLY thing that switches the relay; enforces the safety checks |
| Tests | `test/arm_aim/` | unit tests + one headless Gazebo test |

```
RViz "Publish Point" (/clicked_point) or /aim_target
        |
        v
  weed_aim_node --(target in arm_base frame)--> aim_solver --> ranked joint solutions
        |  first solution MoveIt says is collision-free
        v
  move_group (MoveIt 2) --plan + execute--> /arm_controller/follow_joint_trajectory
        |                                         (Gazebo now; real servo driver in sub-project 4)
        v
  /joint_states --> arm arrived within 1 deg? --> /spray/trigger --> spray_controller
                                                                      | checks 1-6 (section 5)
                                                                      v
                                                            relay ON 1.0 s -> relay OFF
        <------------------------------------------------------------ result
  move_group: back to "park"  -->  /aim/result (JSON report)
```

The MoveIt config lives in a folder of the `my_bot` package (`config/moveit/`), not in a separate
package as first said in chat: colcon does not find packages nested inside `my_bot`, and a sibling
package would not reach the Pi with the usual `git pull`. MoveIt's launch helper
(`moveit_configs_utils`) accepts files from any package, so nothing is lost.

## 4. Each piece in detail

### 4.1 Aim solver (`scripts/aim_solver.py`)
- Built from the arm's numbers, which the aim node reads from `robot_description` at start
  (joint origins and limits of `shoulder_pitch`, `elbow_pitch`, `wrist_pitch`, `wrist_roll`,
  `spray_tip_joint`), so no measurement is copied into a second file.
- Input: target point in the `arm_base` frame. Output: up to 20 candidates, each with the 5 joint
  angles (roll = 0), the tilt from straight down, the standoff (tip to target), and the tip height;
  or no candidates plus a reason: `behind the arm` (yaw beyond the limit), `too far`,
  `too close / under the arm`.
- Method: yaw = atan2(y, x). In the arm's vertical plane, for each tilt from 0 to 50 deg (1 deg
  steps, nozzle leaning away from the yaw axis) and each standoff from 0.08 to 0.25 m (1 cm steps):
  tip = target - standoff * direction; wrist axis = tip - (wrist-to-tip length) * direction;
  two-link solution for shoulder and elbow (both elbow-up and elbow-down); wrist = tilt angle
  minus shoulder minus elbow. Keep it if every joint is inside its limits, the tip is at least
  3 cm above the target, and the arm stays out of the LiDAR's view (next point).
- LiDAR keep-clear: wherever an arm link comes within 3 cm of the scan plane (height of
  `laser_frame` from TF, 0.378 m), it must be inside the `scan_self_filter` box shrunk by 2 cm
  (box read from `config/safety.yaml`, today x -0.53..0.18, y -0.16..0.16 in `base_link`).
  Otherwise the LiDAR would see the arm, `/safety_state` would go to `stop`, and the spray would
  be refused anyway (section 4.5, check 6). The 2026-11-23 estimate shows this rule does not
  shrink the reachable patch.
- Ranking: smallest tilt first (closest to spraying straight down), then smallest standoff, then
  the elbow solution with the lower elbow (keeps the arm out of the LiDAR plane where possible).
- Exactness: the nozzle line of every returned solution, recomputed forwards, passes within
  1e-6 m of the target (unit test).
- Limits (tilt range, standoff range, min tip clearance) are parameters in `config/arm_aim.yaml`;
  the real spray jet's useful reach is measured in sub-project 4 and may change the standoff range.

### 4.2 Arm moving in Gazebo
- `weedbot_arm.xacro`: in `sim_mode` the five arm joints become `revolute` with 0 in their
  origins, exactly like the real robot (today they are `fixed` in the parked pose in sim).
- `ros2_control.xacro`, sim block only (`GazeboSystem`): add the five arm joints with a `position`
  command interface and `position` + `velocity` state interfaces, `initial_value` = the parked
  angles (`arm_park_*`), so the arm spawns parked and holds there before any controller starts.
  The real-robot block (`RealRobot`, diffdrive_arduino) is NOT touched.
- `config/arm_controllers_sim.yaml` (new, sim only, passed as an extra `<parameters>` file to the
  gazebo_ros2_control plugin): `arm_controller` = `joint_trajectory_controller/JointTrajectoryController`
  for the five joints. `my_controllers.yaml` (shared with the real robot) is not changed; the
  controller manager keeps its 10 Hz update rate (if arm motion turns out too jerky in the test,
  the sim-only file can raise it; noted as a risk, not planned).
- Sim only: the moving arm links (`arm_turret`, `upper_arm`, `forearm`, `wrist`, `nozzle`) get
  `<gravity>false</gravity>` in Gazebo. gazebo_ros2_control sets joint positions only at the
  controller rate (10 Hz); in between, gravity would make the arm sag. Real servos hold stiffly,
  so a gravity-free, position-held arm behaves like the real one.
- `launch_sim.launch.py`: spawns `arm_controller` after `joint_broad`; in sim
  `arm_joint_state_stub` is no longer started (the joint state broadcaster now publishes the arm
  joints, and the caster joints exist only in the real-robot URDF: `weedbot_body.xacro`'s
  `xacro:unless sim_mode` block, so publishing them in sim only produces "joint not found"
  errors). The real robot keeps the full stub until the servo driver exists (sub-project 4).

### 4.3 MoveIt config (`config/moveit/`)
- `weedbot.srdf`: planning group `arm` = chain `arm_base` -> `spray_tip` (5 joints); named state
  `park` (yaw 0, shoulder 0, elbow 90 deg, wrist 90 deg, roll 0); disabled collision pairs for
  links that are fixed to each other (body, frame, tank, phone mount, wheels...) and adjacent arm
  links, generated once by a small script and checked in.
- `kinematics.yaml`: KDL with position-only IK (only used by RViz's drag handle; the aim node uses
  the aim solver and plans to joint goals).
- `joint_limits.yaml`: max 1.0 rad/s and 2.0 rad/s^2 per joint (slow and gentle; MG996R no-load
  speed is ~5 rad/s). Velocity scaling 0.5 in the aim node.
- `moveit_controllers.yaml`: `moveit_simple_controller_manager`, controller `arm_controller`,
  `FollowJointTrajectory`, action namespace `follow_joint_trajectory`.
- Planning: OMPL (RRTConnect) from moveit_configs_utils' defaults.
- Ground: the aim node adds a ground box to the planning scene (top at z 0.02 in `base_link`:
  grass margin), so no plan ever puts the arm into the grass.
- `launch/arm_aim.launch.py` (args `use_sim_time`, `rviz`): `move_group` (robot description built
  from the same xacro with `sim_mode` set to match), `weed_aim_node`, `spray_controller_node`,
  and optionally RViz with the MotionPlanning panel (`config/arm_aim.rviz`).

### 4.4 Aim node (`scripts/weed_aim_node.py`)
- Inputs: `/aim_target` and `/clicked_point` (both `geometry_msgs/PointStamped`, any frame TF can
  convert). One cycle at a time: a target that arrives while busy is ignored and logged.
- Cycle: SOLVE (aim solver) -> VALIDATE (ask MoveIt's `/check_state_validity` for each candidate,
  in rank order, until one is collision-free) -> MOVE (MoveGroup action, joint goal, plan and
  execute) -> ARRIVED? (all joints within 1 deg for 0.3 s AND settled: each joint moved no more
  than 0.01 rad/s over those 0.3 s — half the spray controller's stillness limit, added
  2026-11-24 after the end-to-end test; timeout 15 s) -> SPRAY (call
  `/spray/trigger`, only if parameter `spray` is true) -> PARK (MoveGroup to `park`) -> REPORT.
- Measures the aim error after arriving: distance from the target to the nozzle's centre line,
  taken from the `spray_tip` TF (not from the solver's own numbers).
- Outputs: `/aim/result` (`std_msgs/String`, JSON: target, reachable, reason, joints, tilt,
  standoff, aim error, sprayed, spray reason, durations); `/aim/markers` (target sphere, nozzle
  line, spray cone shown only while `/spray/relay` is true). Service `/aim/park`
  (`std_srvs/Trigger`) parks the arm on demand.
- E-stop: if `/e_stop` turns true during a cycle, the node cancels the MoveIt goal (the arm stops
  where it is), does not spray, and stays in ERROR until the e-stop is released and `/aim/park`
  is called. This holds at ANY point of a cycle, even before the arm moved, and a short
  true -> false pulse is remembered (added 2026-11-24).
- Start gate (added 2026-11-24, final review): a cycle only STARTS when the latest
  `/safety_state` is fresh (<= 0.5 s) and not `stop`; otherwise the target is refused
  (`safety_state stop` / `safety_state missing or old`) and nothing moves. safety_state_node
  reports an e-stop as `stop`, so an e-stop pressed before weed_aim started still blocks it.
  `/e_stop` is read both volatile and transient-local. `/aim/park` is not gated on
  `/safety_state` (an arm stopped mid-move can be seen by the LiDAR itself).

### 4.5 Spray controller (`scripts/spray_controller_node.py`)
- The only code that switches the relay. Only standard ROS message/service types (no new
  interface package).
- Services: `/spray/arm` (`std_srvs/SetBool`: true = armed, false = disarmed); `/spray/trigger`
  (`std_srvs/Trigger`: answers after the spray finished or was refused; `success` = sprayed,
  `message` = reason). Topics: `/spray/relay` (`std_msgs/Bool`, latched, true only while on);
  `/spray/status` (JSON, 1 Hz: armed, relay, last result).
- Backends (parameter `backend`): `sim` (publishes the relay state only) and `gpio` (a Pi GPIO pin
  through `lgpio`, pin and active level as parameters; written now, first run on hardware in
  sub-project 4). `start_armed` is honoured ONLY with the `sim` backend; with `gpio` the node
  always starts disarmed.
- Spray time: parameter `spray_seconds` (default 1.0). A hard cap of 3.0 s is a constant in the
  code; the parameter can lower it but never raise it.
- Checks before switching on (any failure -> relay stays off, `message` says which):
  1. Armed.
  2. Arm in place: all arm joints still (speed below 0.02 rad/s for 0.3 s), and the nozzle line
     (from TF) hits the ground within 0.8 m of the arm base, i.e. it points down at the ground,
     not up or out. (The aim node has already checked the joints are within 1 deg of the goal.)
     Added 2026-11-24 (final review): that ground-hit point must lie OUTSIDE `robot_box`
     (base_link, x -0.56..0.21, y -0.19..0.19 = the LiDAR self-filter box + 0.03 m), else
     `nozzle aimed at the robot` — e.g. the parked nozzle points straight down onto the phone
     holder. Real aims hit 0.25 m or more ahead, so they are unaffected.
  3. Robot still: `/odom` speed below 0.01 m/s and turn rate below 0.02 rad/s for 0.5 s; no
     `/odom` in the last 1 s also counts as a failure.
  4. Not in a keepout zone: if `zones_file` is set (same format as `config/backyard_zones.yaml`,
     the pool guard's file), the point where the nozzle line hits the ground is converted to
     `map` and must lie outside every zone and its margin (reusing the pool guard's
     `polygon_distance` code). Zones set but no `map` transform -> refuse. No zones file -> this
     check is skipped and the log says so.
  5. No e-stop: `/e_stop` is not true.
  6. Safety state clear (CLAUDE.md hard rule): the latest `/safety_state` (`std_msgs/String`,
     latched, 10 Hz: clear | slow | stop) is `clear` and newer than 0.5 s. `slow`, `stop`, a
     missing message or an old one -> refuse.
- While on (checked at 20 Hz): e-stop, disarm, `/safety_state` not `clear` or older than 0.5 s,
  robot moving, or arm moving -> relay off at once, result `aborted: <reason>`. The relay also goes off at node start, shutdown, and on any
  exception.

## 5. What happens when things go wrong

| Situation | Arm | Relay | Report says |
|---|---|---|---|
| Target out of reach / behind | does not move | off | the solver's reason |
| Every candidate collides with the robot | does not move | off | `blocked by the robot body` |
| MoveIt cannot plan or is not running | does not move | off | `planning failed` / `MoveIt not ready` |
| Arm does not arrive within 1 deg in 15 s | parks | off | `did not reach the aim pose` |
| Spray refused (disarmed, keepout, robot moving, arm moving, no odom, safety state not clear) | parks | off | the spray controller's reason |
| Safety state leaves `clear` during the spray | parks | off within 0.1 s | `aborted: safety_state <state>` |
| E-stop during motion or spray | stops where it is | off within 0.1 s | `e-stop`; ERROR until released + `/aim/park` |
| Park fails | wherever it is | off | `PARK FAILED` (stays ERROR until `/aim/park` succeeds) |

## 6. Safety rules this design keeps (from CLAUDE.md)
- Never spray in simulation or on the bench without the owner asking in that session: the sim
  has no pump; on the real robot the controller starts disarmed and the owner arms it.
- Never spray inside a keepout / flower-bed zone: check 4, and refuse when it cannot be checked.
- Spray only when `/safety_state` is `clear`, relay off on anything else, a missing or old
  message, or `/e_stop`: checks 5 and 6, monitored during the spray too.
- No motion starts by itself: the arm moves only when a target is given or `/aim/park` is called.
- Nothing in this sub-project runs on the Pi or touches the real robot's launch, firmware or
  protected setup.

## 7. Testing

All automatic tests follow "Rules for automated tests" in CLAUDE.md (ROS_DOMAIN_ID 42, localhost
only, discovery server unset, Gazebo on port 11346, `sim_cleanup.sh --domain 42` must print
"Clean" at the end, pass or fail).

A. Aim solver unit tests (`test/arm_aim/test_aim_solver.py`, pytest, no ROS):
   - 50 random reachable points: every returned solution's nozzle line passes within 1e-6 m of
     the target and every joint is inside its limits.
   - Fixed cases: (0.30, 0, 0) ahead -> reachable with tilt under 30 deg; a point behind the robot
     -> `behind the arm`; 1 m ahead -> `too far`; under the arm base -> `too close / under the arm`.
B. Spray controller tests (`test/arm_aim/test_spray_controller.py`, ROS, no Gazebo; fake
   `/odom`, `/joint_states`, TF, `/e_stop` and `/safety_state`): refuses when disarmed, robot
   moving, arm moving, nozzle pointing up, inside a test keepout zone, zones without `map`,
   safety state `slow` / `stop` / missing / older than 0.5 s; e-stop or safety state `stop` during
   a spray -> relay off within 0.1 s; a 10 s request is cut at 3 s; `gpio` backend refuses
   `start_armed`.
C. Headless Gazebo test (`test/arm_aim/run_sim_aim_test.sh`), in an empty test world
   (`test/arm_aim/arm_test.world`: ground, sun, and the gazebo_ros_state plugin that
   `sim_person.py` needs). `safety_state_node` reports `slow` while there is no map (by design),
   so the test publishes an empty `/map` and a `map -> odom` transform; with nothing near the
   robot `/safety_state` is then `clear`:
   - in every aim pose the raw `/scan` has no returns outside the self-filter box (the LiDAR
     keep-clear rule works) and `/safety_state` stays `clear`;
   - after spawning, the arm stays parked for 30 s (spray tip within 5 mm of its parked height);
   - 9 targets at x 0.28 / 0.34 / 0.40 and y -0.10 / 0 / +0.10 (base_link, ground): each meets
     the success criteria in section 1;
   - 3 impossible targets (0.80 m ahead, 0.30 m behind, 0.05 m ahead under the arm): refused,
     arm never moved;
   - keepout: a test zones file covering one target (`map` = `odom` by a static transform in the
     test) -> aims, spray refused `keepout`, parks;
   - robot moving at 0.05 m/s (`/cmd_vel_keyboard`) when the spray is due -> refused;
   - e-stop during a spray -> relay off within 0.1 s, arm stops;
   - a sim person (`sim_person.py spawn --ahead 1.0 --left 0.3`) -> `/safety_state` not `clear`
     -> spray refused, arm parks. The person stands 0.3 m to the side (ruling R19): straight ahead
     at 1.0 m the parked arm hides it completely from the LD06 (the blind wedge, CLAUDE.md "Robot
     model"), so `/safety_state` would stay `clear`;
   - driving regression: 3 s at 0.10 m/s through `/cmd_vel_keyboard` (the safety chain; capped
     at 0.08 m/s) still moves 0.243 m +- 0.02 (2026-10-19 baseline) with `/odom` within 0.02 m of
     Gazebo's truth (`sim_odom_check.py` is not used: CLAUDE.md says it bypasses the safety chain).
D. Owner demo (not automatic): `launch_sim` + `arm_aim.launch.py rviz:=true`, click points on the
   ground in RViz, watch the arm aim, the spray cone, and the arm park.

## 8. Files

New: `scripts/aim_solver.py`, `scripts/arm_config.py` (reads park from the SRDF and the
scan-filter box from safety.yaml), `scripts/weed_aim_node.py`, `scripts/spray_controller_node.py`,
`config/arm_controllers_sim.yaml`, `config/arm_aim.yaml`, `config/arm_aim.rviz`,
`config/moveit/{weedbot.srdf,kinematics.yaml,joint_limits.yaml,moveit_controllers.yaml}`,
`launch/arm_aim.launch.py`, `test/arm_aim/*`.

Changed: `description/weedbot_arm.xacro` (sim branch only), `description/ros2_control.xacro`
(sim block only), `launch/launch_sim.launch.py`, `CMakeLists.txt` (install the new scripts),
`package.xml` (MoveIt and controller exec_depends), `CLAUDE.md` (new "Arm aim + spray (sim)"
section).

`CMakeLists.txt`, `package.xml` and `ros2_control.xacro` already hold uncommitted changes (the
VM's working copy has many uncommitted files): only the needed lines are added, and nothing is
committed without the owner's OK (then only Claude's lines).

Install on the VM (owner approved): `sudo apt install ros-humble-moveit-configs-utils`.

## 9. Not in this sub-project (for later)
- Sub-project 2: simulated phone camera (1280 x 960, fx 891, at `phone_optical`) and fake weeds in
  Gazebo; detection -> ground point (depth or ground-plane ray) -> `/aim_target`; drive up so the
  weed lies in the reachable patch.
- Sub-project 3: detector training (general `weed` class; public dataset + own photos; Colab;
  NCNN/ONNX for the Pi).
- Sub-project 4, real arm, including two hardware points already raised with the owner:
  - Servo POWER: never from the Pi's 5 V pin (five MG996R can draw several amps); a separate
    5-6 V supply of 5 A or more (e.g. a buck converter from the 12 V pack), grounds joined.
  - Servo SIGNAL: the Pi has only two hardware PWM channels for five servos (software timing
    makes servos jitter); a PCA9685 board on I2C bus 1 (0x40, no clash with the MPU6050 at 0x68)
    is the recommended alternative. Owner decides then.
  - A servo driver node serving `/arm_controller/follow_joint_trajectory` (so the MoveIt config
    stays the same), servo zero/direction calibration, measured yaw range, relay on a GPIO pin
    with the `gpio` backend, a relay that fails OFF if the spray controller process is killed
    (a GPIO pin keeps its level when a process dies: pull-down / normally-open wiring or a
    hardware watchdog), the spray jet's real reach (standoff range), and checking on the real
    LiDAR that the keep-clear rule (section 4.1) really keeps the arm out of the scan.
