#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import serial

PORT = '/dev/ttyESP32'
BAUD = 115200

class SerialReader(Node):
    def __init__(self):
        super().__init__('red_mirror_serial_reader')

        self.ser = serial.Serial(PORT, BAUD, timeout=0.2)

        self.imu_pub = self.create_publisher(
            Float64MultiArray,
            'red_mirror/imu',
            10
        )

        self.fsr_pub = self.create_publisher(
            Float64MultiArray,
            'red_mirror/foot_contact',
            10
        )

        self.timer = self.create_timer(0.02, self.read_serial)  # 50Hz update

        self.get_logger().info("Serial reader started (Euler only).")

    def read_serial(self):
        if not self.ser.in_waiting:
            return

        line = self.ser.readline().decode('utf-8').strip()
        if not line:
            return

        parts = line.split(',')
        if len(parts) != 9:  # 9 IMU + 6 analog
            self.get_logger().warn("Malformed data size= " + str(len(parts)))
            return

        try:
            roll, pitch, yaw = map(float, parts[:3])
            fsr = list(map(float, parts[3:]))

            # Publish IMU as Euler + raw
            imu_msg = Float64MultiArray()
            imu_msg.data = [roll, pitch, yaw]
            self.imu_pub.publish(imu_msg)

            # Publish analogs
            fsr_msg = Float64MultiArray()
            fsr_msg.data = fsr
            self.fsr_pub.publish(fsr_msg)

        except ValueError:
            self.get_logger().warn("Failed to parse values")


def main(args=None):
    rclpy.init(args=args)
    node = SerialReader()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
