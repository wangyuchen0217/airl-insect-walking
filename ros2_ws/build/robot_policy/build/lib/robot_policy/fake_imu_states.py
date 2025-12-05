# fake_imu_node.py

import time
import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu


def euler_to_quaternion(roll, pitch, yaw):
    """
    将欧拉角 (roll, pitch, yaw) 转成四元数 (x, y, z, w)
    旋转顺序假设为 XYZ (roll → pitch → yaw)
    """
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return x, y, z, w


class FakeImuNode(Node):
    def __init__(self):
        super().__init__('fake_imu')

        self.pub = self.create_publisher(Imu, '/imu/data', 10)

        self.rate_hz = 100.0
        self.timer = self.create_timer(1.0 / self.rate_hz, self.on_timer)

        self.t0 = time.time()

        self.get_logger().info("FakeImuNode started, publishing /imu/data")

    def on_timer(self):
        t = time.time() - self.t0

        # 小幅滚转/俯仰/偏航（弧度）
        roll = 0.05 * math.sin(0.3 * t)          # ±0.05 rad ≈ ±3°
        pitch = 0.03 * math.sin(0.4 * t + 0.5)   # ±0.03 rad ≈ ±1.7°
        yaw = 0.20 * math.sin(0.1 * t)           # ±0.2 rad ≈ ±11°

        qx, qy, qz, qw = euler_to_quaternion(roll, pitch, yaw)

        msg = Imu()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "imu_link"

        msg.orientation.x = qx
        msg.orientation.y = qy
        msg.orientation.z = qz
        msg.orientation.w = qw

        # 线加速度、角速度可以先设 0
        msg.angular_velocity.x = 0.0
        msg.angular_velocity.y = 0.0
        msg.angular_velocity.z = 0.0

        msg.linear_acceleration.x = 0.0
        msg.linear_acceleration.y = 0.0
        msg.linear_acceleration.z = 0.0

        self.pub.publish(msg)


def main():
    rclpy.init()
    node = FakeImuNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
