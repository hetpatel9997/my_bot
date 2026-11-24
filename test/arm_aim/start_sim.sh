#!/usr/bin/env bash
# start_sim.sh - start the headless WeedBot simulation for the arm tests and wait until it is ready.
#
#   bash test/arm_aim/start_sim.sh
#
# What it does, and why:
#   - uses the test settings from sim_env.sh (ROS domain 42, Gazebo port 11346), so the owner's
#     own sim or robot is never touched;
#   - starts launch_sim.launch.py with gui:=false (no Gazebo window: far lighter on the VM) in the
#     empty test world test/arm_aim/arm_test.world, in the background, output in $ARM_TEST_LOG/sim.log;
#   - waits up to 120 s until the three ros2_control controllers (arm_controller, diff_cont,
#     joint_broad) are all "active": only then can the arm and the wheels move;
#   - a cold first Gazebo start can take longer than spawn_entity's 30 s and the robot never
#     appears, so on a timeout it stops everything in domain 42 and tries ONE more time
#     (the first try's log is kept as sim_try1.log).
# Exit 0 = sim ready and left running. Non-zero = not ready; the sim may still be running so its
# log can be read. In every case, when the test is over:
#   ~/dev_ws/src/my_bot/scripts/sim_cleanup.sh --domain 42      (must print "Clean")

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLEANUP="$(cd "$HERE/../.." && pwd)/scripts/sim_cleanup.sh"
WORLD="$HERE/arm_test.world"
WAIT_S=120
CONTROLLERS="arm_controller diff_cont joint_broad"

source "$HERE/sim_env.sh"

# Refuse to start a second sim on top of one that is already running in the test domain
# (two Gazebos on port 11346 do not work, and those processes might belong to another test).
running=$("$CLEANUP" --domain 42 --dry-run)
if ! echo "$running" | grep -q '^No sim'; then
  echo "$running"
  echo "start_sim: domain 42 already has sim processes (above). If they are left over from" \
       "your own test, run $CLEANUP --domain 42 first."
  exit 2
fi

controllers_active() {
  # ros2 control prints colour codes; strip them, then every controller needs a line
  # "<name>  <type>  active". timeout: the CLI waits forever while there is no controller manager.
  local out c
  out=$(timeout 20 ros2 control list_controllers 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g')
  for c in $CONTROLLERS; do
    echo "$out" | grep -Eq "^$c[[:space:]]+[^[:space:]]+[[:space:]]+active[[:space:]]*$" || return 1
  done
  return 0
}

try_start() {
  echo "start_sim: starting Gazebo (headless, world $WORLD), log $ARM_TEST_LOG/sim.log"
  # setsid + nohup: the sim keeps running after this script exits and is not hit by the
  # caller's Ctrl+C; sim_cleanup.sh --domain 42 finds and stops it.
  setsid nohup ros2 launch my_bot launch_sim.launch.py gui:=false world:="$WORLD" \
    > "$ARM_TEST_LOG/sim.log" 2>&1 < /dev/null &
  local launch_pid=$! t0=$SECONDS
  while (( SECONDS - t0 < WAIT_S )); do
    if ! kill -0 "$launch_pid" 2>/dev/null; then
      echo "start_sim: ros2 launch exited after $((SECONDS - t0)) s (see $ARM_TEST_LOG/sim.log)"
      return 1
    fi
    if controllers_active; then
      echo "start_sim: controllers active ($CONTROLLERS) after $((SECONDS - t0)) s"
      return 0
    fi
    sleep 5
  done
  echo "start_sim: controllers not all active after $WAIT_S s:"
  timeout 20 ros2 control list_controllers 2>&1 | sed 's/\x1b\[[0-9;]*m//g; s/^/    /'
  return 1
}

if try_start; then
  echo "start_sim: READY"
  exit 0
fi
echo "start_sim: first try failed (a cold Gazebo start can do that); cleaning up and retrying once"
"$CLEANUP" --domain 42
mv -f "$ARM_TEST_LOG/sim.log" "$ARM_TEST_LOG/sim_try1.log"
if try_start; then
  echo "start_sim: READY (second try)"
  exit 0
fi
echo "start_sim: FAILED - sim not ready after two tries. Logs: $ARM_TEST_LOG/sim.log," \
     "$ARM_TEST_LOG/sim_try1.log. Run $CLEANUP --domain 42 when done."
exit 1
