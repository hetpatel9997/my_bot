#!/usr/bin/env bash
# run_unit_tests.sh - run arm-aim ROS unit tests (pytest + rclpy, no Gazebo) isolated, then clean up.
#
#   bash test/arm_aim/run_unit_tests.sh test/arm_aim/test_spray_controller.py [more pytest args]
#
# Why a wrapper: automated tests must never talk to the owner's own sim or robot (CLAUDE.md
# "Rules for automated tests (Claude) on the VM"). sim_env.sh puts every node of the test in ROS
# domain 42 on this machine only. When pytest ends (pass, fail or Ctrl-C) everything the test may
# have left in domain 42 is stopped with sim_cleanup.sh --domain 42 (must print "Clean"), and the
# ros2 CLI daemon of domain 42 is stopped too (sim_cleanup.sh does not catch it).
# Exit code = pytest's exit code.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLEANUP="$(cd "$HERE/../.." && pwd)/scripts/sim_cleanup.sh"

source "$HERE/sim_env.sh"

cleanup() {
  "$CLEANUP" --domain 42 || echo "WARNING: sim_cleanup.sh --domain 42 did not finish clean" >&2
  ros2 daemon stop >/dev/null 2>&1 || true
}
trap cleanup EXIT   # runs on normal exit and on Ctrl-C; keeps pytest's exit code

python3 -m pytest "$@" -v
exit $?
