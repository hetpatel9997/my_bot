# sim_env.sh - SOURCE this (do not run it) before any arm test: `source test/arm_aim/sim_env.sh`
#
# Why: automated tests must never talk to the owner's own sim or robot (CLAUDE.md "Rules for
# automated tests (Claude) on the VM"). This puts every ROS node of the test in its own ROS
# domain (42), keeps DDS traffic on this machine, drops the discovery-server settings, and moves
# Gazebo to its own port (11346; Gazebo ignores ROS domains and the owner's Gazebo uses 11345).
# ARM_TEST_LOG = folder for the test logs (sim.log etc.), default ~/.ros/arm_aim_test.
# When a test ends (pass, fail or abort): ~/dev_ws/src/my_bot/scripts/sim_cleanup.sh --domain 42
source /opt/ros/humble/setup.bash && source ~/dev_ws/install/setup.bash
export ROS_DOMAIN_ID=42 ROS_LOCALHOST_ONLY=1 GAZEBO_MASTER_URI=http://127.0.0.1:11346
unset ROS_DISCOVERY_SERVER FASTRTPS_DEFAULT_PROFILES_FILE
export ARM_TEST_LOG=${ARM_TEST_LOG:-$HOME/.ros/arm_aim_test}; mkdir -p "$ARM_TEST_LOG"
