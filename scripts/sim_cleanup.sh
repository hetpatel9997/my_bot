#!/usr/bin/env bash
# sim_cleanup.sh - stop simulation / Nav2 / RViz processes on this machine (VM) and clear
# stale FastDDS shared-memory files.
#
#   sim_cleanup.sh                 stop ALL matching processes (any ROS domain), then remove
#                                  /dev/shm/fastrtps_* and /dev/shm/sem.fastrtps_*
#   sim_cleanup.sh --domain 42     stop only processes whose ROS_DOMAIN_ID is 42 (unset = 0);
#                                  shared memory is left alone (other sessions may use it).
#                                  This is what automated tests use (they run in domain 42).
#   sim_cleanup.sh --dry-run ...   only list what would be stopped
#
# Each process gets Ctrl+C (SIGINT), then SIGTERM after 5 s, then SIGKILL after 3 more s.
# Exit code 0 = nothing left, 1 = something survived (listed), 3 = could not search processes.
# Keyboard teleops and processes started from the Pi are NOT touched.
set -u

DOMAIN=""
DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --domain) DOMAIN="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

# Everything a sim / Nav2 / RViz session starts (launch files, Gazebo, Nav2 servers, SLAM,
# EKF, twist_mux, ros2_control spawners, my_bot nodes, the ros2 CLI daemon).
PATTERN='gzserver|gzclient|rviz2|ros2 launch|spawn_entity\.py|_ros2_daemon|/opt/ros/humble/lib/(nav2_[a-z_]+|slam_toolbox|robot_localization|twist_mux|controller_manager|robot_state_publisher|joint_state_publisher|tf2_ros)/|/install/my_bot/lib/my_bot/'

# PIDs to never touch: this script and its ancestors
SELF=" $$ "
p=$$
while [ "$p" -gt 1 ]; do
  p=$(awk '{print $4}' "/proc/$p/stat" 2>/dev/null || echo 1)
  SELF="$SELF$p "
done

domain_of() {   # ROS_DOMAIN_ID of a PID (unset = 0)
  local d
  d=$(tr '\0' '\n' < "/proc/$1/environ" 2>/dev/null | sed -n 's/^ROS_DOMAIN_ID=//p')
  echo "${d:-0}"
}

find_targets() {
  local pid pids rc
  pids=$(pgrep -u "$(id -u)" -f "$PATTERN"); rc=$?      # pgrep patterns are extended regex
  if [ "$rc" -gt 1 ]; then                              # 1 = no match; 2/3 = pgrep error
    echo "ERROR: pgrep failed (exit $rc); not reporting a false 'clean'" >&2
    exit 3
  fi
  for pid in $pids; do
    case "$SELF" in *" $pid "*) continue ;; esac
    if [ -n "$DOMAIN" ] && [ "$(domain_of "$pid")" != "$DOMAIN" ]; then continue; fi
    echo "$pid"
  done
}

describe() {
  local pid
  for pid in "$@"; do
    printf '  %-7s domain %-3s %s\n' "$pid" "$(domain_of "$pid")" \
      "$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | cut -c1-100)"
  done
}

TARGETS=$(find_targets) || exit 3
if [ -z "$TARGETS" ]; then
  echo "No sim/Nav2/RViz processes${DOMAIN:+ in domain $DOMAIN}."
else
  echo "Stopping${DOMAIN:+ (domain $DOMAIN only)}:"
  describe $TARGETS
  if [ "$DRY" = 1 ]; then exit 0; fi
  kill -INT $TARGETS 2>/dev/null
  for sig in TERM KILL; do
    for _ in $(seq 1 10); do
      LEFT=$(for p in $TARGETS; do [ -d "/proc/$p" ] && echo "$p"; done)
      [ -z "$LEFT" ] && break
      sleep 0.5
    done
    [ -z "$LEFT" ] && break
    [ "$sig" = KILL ] && sleep 1
    kill -"$sig" $LEFT 2>/dev/null
  done
fi
[ "$DRY" = 1 ] && exit 0

# Anything (re)spawned meanwhile or still alive?
sleep 1
LEFT=$(find_targets) || exit 3
if [ -n "$LEFT" ]; then
  echo "STILL RUNNING:"
  describe $LEFT
  exit 1
fi

if [ -z "$DOMAIN" ]; then
  n=$(ls /dev/shm 2>/dev/null | grep -cE '^(sem\.)?fastrtps_')
  rm -f /dev/shm/fastrtps_* /dev/shm/sem.fastrtps_* 2>/dev/null
  echo "Removed $n FastDDS shared-memory file(s) from /dev/shm."
fi
echo "Clean: no sim/Nav2/RViz processes left${DOMAIN:+ in domain $DOMAIN}."
exit 0
