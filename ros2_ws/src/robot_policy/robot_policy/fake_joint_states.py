import math
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

class FakeJointStates(Node):
    def __init__(self):
        super().__init__('fake_joint_states')
        self.pub = self.create_publisher(JointState, '/joint_states', 10)
        self.joint_names = self.declare_parameter(
            'joint_names',
            ['LF_ThC','LF_CTr','LF_FTi','LM_ThC','LM_CTr','LM_FTi',
             'LH_ThC','LH_CTr','LH_FTi','RF_ThC','RF_CTr','RF_FTi',
             'RM_ThC','RM_CTr','RM_FTi','RH_ThC','RH_CTr','RH_FTi']
        ).get_parameter_value().string_array_value
        self.t0 = time.time()
        self.timer = self.create_timer(0.01, self.on_timer)  # 100 Hz

    def on_timer(self):
        t = time.time() - self.t0
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = self.joint_names
        # Generate fake joint positions and velocities (e.g., sine wave motion)
        pos = [0.2*math.sin(0.5*t + i*0.1) for i in range(len(self.joint_names))]
        vel = [0.2*0.5*math.cos(0.5*t + i*0.1) for i in range(len(self.joint_names))]
        js.position = pos
        js.velocity = vel
        self.pub.publish(js)

def main():
    rclpy.init()
    node = FakeJointStates()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
