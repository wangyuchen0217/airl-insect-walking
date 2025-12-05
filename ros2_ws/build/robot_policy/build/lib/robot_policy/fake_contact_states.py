# fake_contact_node.py

import time
import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


class FakeContactNode(Node):
    def __init__(self):
        super().__init__('fake_contact')

        # 发布 6 维 contact: [FL, ML, HL, FR, MR, HR]
        self.pub = self.create_publisher(Float64MultiArray, '/foot_contacts', 10)

        self.rate_hz = 100.0
        self.timer = self.create_timer(1.0 / self.rate_hz, self.on_timer)

        self.t0 = time.time()

        # 每条腿一个相位，模拟简单的步态
        self.phases = [
            0.0,         # FL
            0.8 * math.pi,  # ML
            1.6 * math.pi,  # HL
            math.pi,        # FR
            1.8 * math.pi,  # MR
            2.6 * math.pi,  # HR
        ]

        # 一个步态周期的角速度
        self.omega = 2.0  # rad/s

        self.get_logger().info("FakeContactNode started, publishing /foot_contacts")

    def on_timer(self):
        t = time.time() - self.t0

        contacts = []
        for phase in self.phases:
            x = math.sin(self.omega * t + phase)
            # x > 0 表示 stance，相当于接触=1；x <= 0 表示 swing=0
            c = 1.0 if x > 0.0 else 0.0
            contacts.append(c)

        msg = Float64MultiArray()
        msg.data = contacts
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = FakeContactNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
