#!/usr/bin/env bash
# run_sim_aim_test.sh - the whole headless Gazebo test of the arm aim + spray work (SIMULATION ONLY).
#
#   bash ~/dev_ws/src/my_bot/test/arm_aim/run_sim_aim_test.sh
#
# Takes about 10 minutes. Everything runs isolated from the owner's sessions (sim_env.sh: ROS
# domain 42, this machine only, Gazebo port 11346). Logs in $ARM_TEST_LOG (~/.ros/arm_aim_test):
# sim.log (Gazebo + robot), arm_aim.log (move_group, weed_aim, spray_controller).
#
# Steps, in this order:
#   1. refuse to start if domain 42 already runs a sim, weed_aim, spray_controller or move_group
#      (they would answer this test's messages too; e.g. left over from an aborted run: then run
#      ~/dev_ws/src/my_bot/scripts/sim_cleanup.sh --domain 42 and stop the move_group, see below);
#   2. start_sim.sh: headless Gazebo in test/arm_aim/arm_test.world, waits for the controllers;
#   3. arm_aim.launch.py in the background (move_group + weed_aim + spray_controller; the spray
#      relay is only a ROS message in the sim, there is no pump);
#   4. check_arm_hold.py        -> "ARM HOLD + MOVE PASS"   (arm holds park 30 s, moves, comes back)
#   5. check_drive_regression.py -> "DRIVE REGRESSION PASS" (drives 0.24 m; BEFORE step 6 on
#      purpose: its 0.243 m baseline was measured without a map, and step 6 publishes one that
#      stays for the rest of the session)
#   6. sim_aim_check.py         -> "ARM AIM SIM TEST: PASS (8/8)" (aim, spray, park, refusals,
#      keepout, robot moving, e-stop, person, unknown frame, busy)
# Cleanup ALWAYS runs at the end (pass, fail, Ctrl-C): stop the arm_aim launch (its whole process
# group: SIGINT, then SIGTERM, then SIGKILL), sim_cleanup.sh --domain 42 (must print "Clean"),
# stop any move_group left in domain 42 (sim_cleanup.sh does not know move_group), and
# `ros2 daemon stop` (the ros2 CLI daemon of domain 42).
# Last line: "RUN SIM AIM TEST: PASS" (exit 0) or "RUN SIM AIM TEST: FAIL (...)" (exit 1);
# exit 2 = refused to start (nothing was started, nothing stopped).

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG="$(cd "$HERE/../.." && pwd)"
CLEANUP="$PKG/scripts/sim_cleanup.sh"

source "$HERE/sim_env.sh"

domain42_pids() {   # PIDs (this user) whose command line matches $1 and that run in domain 42
  local p
  for p in $(pgrep -u "$(id -u)" -f "$1"); do
    grep -qzx ROS_DOMAIN_ID=42 "/proc/$p/environ" 2>/dev/null && echo "$p"
  done
}

# ---- 1. refuse to start on top of something already running in domain 42
busy=$("$CLEANUP" --domain 42 --dry-run)
if ! echo "$busy" | grep -q '^No sim'; then
  echo "$busy"
  echo "run_sim_aim_test: domain 42 already has sim processes (above); not starting." \
       "If they are left over from your own test: $CLEANUP --domain 42"
  exit 2
fi
for name in weed_aim_node spray_controller_node move_group; do
  pids=$(domain42_pids "$name")
  if [ -n "$pids" ]; then
    echo "run_sim_aim_test: $name already runs in domain 42 (PID $pids); not starting."
    exit 2
  fi
done
others=$(pgrep -u "$(id -u)" -x gzserver)
if [ -n "$others" ]; then
  echo "run_sim_aim_test: NOTE another Gazebo (gzserver PID $others, not domain 42) is running;" \
       "it does not interfere (port 11345 vs 11346) but shares the CPU, which can slow this test."
fi

# ---- cleanup, always
LAUNCH_PGID=""
RESULTS=()
FAILED=0

stop_group() {   # stop a whole process group: SIGINT, then SIGTERM after 15 s, SIGKILL after 5 s
  local pgid=$1 sig waited
  for sig in INT TERM KILL; do
    pgrep -g "$pgid" >/dev/null || return 0
    kill -s "$sig" -- "-$pgid" 2>/dev/null
    [ "$sig" = INT ] && waited=30 || waited=10      # half-seconds
    while (( waited-- > 0 )) && pgrep -g "$pgid" >/dev/null; do sleep 0.5; done
  done
  ! pgrep -g "$pgid" >/dev/null
}

cleanup() {
  local rc=$? out p left rc_clean
  trap - EXIT
  trap '' INT TERM      # a second Ctrl-C must not cut the cleanup short
  echo "---- cleanup"
  if [ -n "$LAUNCH_PGID" ]; then
    if stop_group "$LAUNCH_PGID"; then
      echo "arm_aim.launch.py stopped (process group $LAUNCH_PGID)"
    else
      echo "WARNING: arm_aim.launch.py process group $LAUNCH_PGID survived SIGKILL"
      FAILED=1
    fi
  fi
  out=$("$CLEANUP" --domain 42 2>&1); rc_clean=$?
  echo "$out"
  if [ "$rc_clean" -ne 0 ] || ! echo "$out" | grep -q '^Clean'; then
    RESULTS+=("sim_cleanup.sh --domain 42 not Clean (exit $rc_clean)")
    FAILED=1
  fi
  for p in $(domain42_pids move_group); do
    echo "stopping leftover move_group (PID $p, domain 42)"
    kill "$p" 2>/dev/null
  done
  sleep 2
  left=$(domain42_pids move_group)
  if [ -n "$left" ]; then
    kill -9 $left 2>/dev/null
    RESULTS+=("move_group $left needed SIGKILL")
    FAILED=1
  fi
  ros2 daemon stop >/dev/null 2>&1 || true
  echo "ros2 daemon (domain 42) stopped"
  echo "---- summary"
  printf '  %s\n' "${RESULTS[@]}"
  if [ "$FAILED" -ne 0 ] || [ "$rc" -ne 0 ]; then
    echo "RUN SIM AIM TEST: FAIL (see above; logs in $ARM_TEST_LOG)"
    exit 1
  fi
  echo "RUN SIM AIM TEST: PASS"
  exit 0
}
trap cleanup EXIT
trap 'echo "run_sim_aim_test: interrupted"; FAILED=1; exit 130' INT TERM

# ---- 2. Gazebo
if ! bash "$HERE/start_sim.sh"; then
  RESULTS+=("start_sim.sh: sim not ready (see $ARM_TEST_LOG/sim.log)")
  FAILED=1
  exit 1
fi

# ---- 3. arm_aim.launch.py in its own process group (setsid), with SIGINT at its default:
# a background job of a non-interactive shell starts with SIGINT ignored, and then ros2 launch
# never hears Ctrl-C / kill -INT (seen once: two weed_aim nodes in one run). env resets it.
echo "starting arm_aim.launch.py, log $ARM_TEST_LOG/arm_aim.log"
setsid env --default-signal=INT ros2 launch my_bot arm_aim.launch.py \
  > "$ARM_TEST_LOG/arm_aim.log" 2>&1 < /dev/null &
launch_pid=$!
sleep 1
pgid=$(ps -o pgid= -p "$launch_pid" 2>/dev/null | tr -d ' ')
if [ "$pgid" != "$launch_pid" ]; then
  RESULTS+=("arm_aim.launch.py did not start in its own process group (pid $launch_pid, pgid '$pgid')")
  FAILED=1
  exit 1
fi
LAUNCH_PGID=$pgid
# SIGINT (signal 2 = bit 1 of SigIgn) must not be ignored, or SIGINT cannot stop it
sigign=$(awk '/^SigIgn:/ {print $2}' "/proc/$launch_pid/status")
if (( (16#$sigign >> 1) & 1 )); then
  RESULTS+=("arm_aim.launch.py ignores SIGINT (SigIgn $sigign)")
  FAILED=1
  exit 1
fi

run_check() {   # run_check <script> <pass line>; output also in $ARM_TEST_LOG/<script>.log
  local script=$1 want=$2 log="$ARM_TEST_LOG/${1%.py}.log" status
  echo "==== $script"
  python3 "$HERE/$script" 2>&1 | tee "$log"
  status=${PIPESTATUS[0]}
  if [ "$status" -eq 0 ] && grep -qxF "$want" "$log"; then
    RESULTS+=("$want")
  else
    RESULTS+=("$script FAILED (exit $status): $(tail -1 "$log")")
    FAILED=1
  fi
}

# ---- 4.-6. the checks (order: see the header)
run_check check_arm_hold.py 'ARM HOLD + MOVE PASS'
run_check check_drive_regression.py 'DRIVE REGRESSION PASS'
run_check sim_aim_check.py 'ARM AIM SIM TEST: PASS (8/8)'

exit "$FAILED"
