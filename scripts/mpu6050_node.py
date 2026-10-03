#!/usr/bin/env python3
"""
mpu6050_node.py  -  reads a GY-521 / MPU6050 over I2C and publishes sensor_msgs/Imu.

Publishes:  imu/data_raw  (sensor_msgs/Imu)  angular velocity (rad/s) and linear
            acceleration (m/s^2) in the imu_link frame. No orientation (covariance[0] = -1),
            robot_localization fuses the yaw rate only.

Parameters:
  i2c_bus (1)             Pi 4 uses bus 1
  i2c_address (0x68)      0x69 if AD0 is tied to 3.3 V
  frame_id (imu_link)
  rate_hz (50.0)
  calibration_samples (300)   robot must be still for ~3 s at startup
  axis_signs ([1,1,1])    flip an axis if the board is mounted rotated (x fwd, y left, z up)

Wiring (GY-521 -> Pi 4 header): VCC->pin1 (3.3V), GND->pin6, SCL->pin5, SDA->pin3.
"""
import math
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu

try:
    from smbus2 import SMBus
except ImportError:  # pragma: no cover
    SMBus = None

REG_PWR_MGMT_1 = 0x6B
REG_SMPLRT_DIV = 0x19
REG_CONFIG = 0x1A
REG_GYRO_CONFIG = 0x1B
REG_ACCEL_CONFIG = 0x1C
REG_DATA = 0x3B          # ACCEL_XOUT_H, 14 bytes through GYRO_ZOUT_L

ACCEL_LSB_PER_G = 16384.0   # +/- 2 g range
GYRO_LSB_PER_DPS = 131.0    # +/- 250 deg/s range
G = 9.80665


def _s16(hi, lo):
    v = (hi << 8) | lo
    return v - 65536 if v > 32767 else v


class Mpu6050Node(Node):
    def __init__(self):
        super().__init__('mpu6050_node')
        self.declare_parameter('i2c_bus', 1)
        self.declare_parameter('i2c_address', 0x68)
        self.declare_parameter('frame_id', 'imu_link')
        self.declare_parameter('rate_hz', 50.0)
        self.declare_parameter('calibration_samples', 300)
        self.declare_parameter('axis_signs', [1.0, 1.0, 1.0])

        self.addr = int(self.get_parameter('i2c_address').value)
        self.frame_id = self.get_parameter('frame_id').value
        self.signs = [float(s) for s in self.get_parameter('axis_signs').value]
        rate = float(self.get_parameter('rate_hz').value)

        if SMBus is None:
            raise RuntimeError('smbus2 is not installed: pip3 install smbus2')
        self.bus = SMBus(int(self.get_parameter('i2c_bus').value))

        # Wake the chip and configure ranges / filtering.
        self.bus.write_byte_data(self.addr, REG_PWR_MGMT_1, 0x00)
        time.sleep(0.1)
        self.bus.write_byte_data(self.addr, REG_CONFIG, 0x03)        # DLPF ~44 Hz
        self.bus.write_byte_data(self.addr, REG_GYRO_CONFIG, 0x00)   # +/-250 dps
        self.bus.write_byte_data(self.addr, REG_ACCEL_CONFIG, 0x00)  # +/-2 g
        self.bus.write_byte_data(self.addr, REG_SMPLRT_DIV, 0x09)    # 1 kHz / 10 = 100 Hz

        self.gyro_bias = [0.0, 0.0, 0.0]
        self.accel_bias = [0.0, 0.0, 0.0]
        self._calibrate(int(self.get_parameter('calibration_samples').value))

        self.pub = self.create_publisher(Imu, 'imu/data_raw', 10)
        self.timer = self.create_timer(1.0 / rate, self._tick)
        self.get_logger().info(f'MPU6050 at 0x{self.addr:02x} publishing imu/data_raw at {rate:.0f} Hz')

    def _read_raw(self):
        d = self.bus.read_i2c_block_data(self.addr, REG_DATA, 14)
        ax, ay, az = _s16(d[0], d[1]), _s16(d[2], d[3]), _s16(d[4], d[5])
        gx, gy, gz = _s16(d[8], d[9]), _s16(d[10], d[11]), _s16(d[12], d[13])
        return ax, ay, az, gx, gy, gz

    def _calibrate(self, n):
        """Average n samples while the robot is still. Removes gyro bias and the
        small accelerometer offsets on x/y (assumes the board is mounted flat)."""
        self.get_logger().info(f'Calibrating IMU, keep the robot still ({n} samples)...')
        sums = [0.0] * 6
        for _ in range(n):
            raw = self._read_raw()
            for i in range(6):
                sums[i] += raw[i]
            time.sleep(0.01)
        means = [s / n for s in sums]
        self.accel_bias = [means[0], means[1], 0.0]
        self.gyro_bias = [means[3], means[4], means[5]]
        az_g = means[2] / ACCEL_LSB_PER_G
        if abs(az_g - 1.0) > 0.15:
            self.get_logger().warn(
                f'Z acceleration at rest is {az_g:.2f} g, expected ~1.0. '
                'Is the board mounted flat with Z pointing up?')
        self.get_logger().info(
            'Gyro bias (raw LSB): x=%.1f y=%.1f z=%.1f' % tuple(self.gyro_bias))

    def _tick(self):
        try:
            ax, ay, az, gx, gy, gz = self._read_raw()
        except OSError as e:
            self.get_logger().warn(f'I2C read failed: {e}', throttle_duration_sec=2.0)
            return
        msg = Imu()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        # No orientation estimate from this node.
        msg.orientation_covariance[0] = -1.0

        sx, sy, sz = self.signs
        msg.angular_velocity.x = sx * math.radians((gx - self.gyro_bias[0]) / GYRO_LSB_PER_DPS)
        msg.angular_velocity.y = sy * math.radians((gy - self.gyro_bias[1]) / GYRO_LSB_PER_DPS)
        msg.angular_velocity.z = sz * math.radians((gz - self.gyro_bias[2]) / GYRO_LSB_PER_DPS)
        msg.linear_acceleration.x = sx * (ax - self.accel_bias[0]) / ACCEL_LSB_PER_G * G
        msg.linear_acceleration.y = sy * (ay - self.accel_bias[1]) / ACCEL_LSB_PER_G * G
        msg.linear_acceleration.z = sz * az / ACCEL_LSB_PER_G * G

        gyro_var = 1e-4      # (0.01 rad/s)^2, conservative for this chip
        accel_var = 1e-2     # the accelerometer is noisy; the EKF does not fuse it anyway
        for i in (0, 4, 8):
            msg.angular_velocity_covariance[i] = gyro_var
            msg.linear_acceleration_covariance[i] = accel_var
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = Mpu6050Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
