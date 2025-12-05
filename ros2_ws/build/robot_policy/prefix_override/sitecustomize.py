import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/yuchen/airl-insect-walking/ros2_ws/install/robot_policy'
