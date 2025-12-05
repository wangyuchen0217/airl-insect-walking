#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

from dynamixel_sdk import *

# ---------------------------------------------------------
# USER CONFIGURATION
# ---------------------------------------------------------

DXL_ID_LIST =[11,12,13, # LF
              31,32,33, # LM
              51,52,53, # LH
              21,22,23, # RF
              41,42,43, # RM
              61,62,63] # RH

DXL_ID_ThC_LIST = [11 ,31, 51, 21, 41, 61]

# Initial pose (rad) mapped 1-to-1 with DXL_ID_LIST

# DXL_init_positions = [0.8,   0.25 -0.35, -1.3-0.4,  # Front left
#                       -0.8,  -0.25 +0.35,  1.3+0.4,  # Front right 
#                       0.0,    0.0  -0.5, -1.0-0.2,    # Middle left
#                       0.0,   -0.0  +0.5,  1.0+0.2,   # Middle right
#                       -0.8,   0.15 -0.1, -1.2-0.4,  # Hind left
#                       0.8,   -0.15 +0.1,  1.2+0.4]    # Hind right

# DXL_init_positions = [0.8,   -0.1, -1.7,  # LF
#                       0.0,    -0.5, -1.2,    # LM
#                       -0.8,   0.05, -1.6,  # LH
#                       -0.8,   0.1, 1.7,   # RF
#                       0.0,    0.5, 1.2,    # RM
#                       0.8,   -0.05, 1.6]    # RH

# DXL_init_direction = [   1,  1,  1,
#                         1, 1, 1,   
#                         1,  1,  1,
#                         1, 1, 1,   
#                         1,  1,  1,
#                         1, 1, 1]

DXL_init_positions = [
                        0.5235988,   0.1658063,  -1.0471976,  # 30,   9.5, -60
                        0.0000000,  -0.0436332,  -1.0471976,  #  0,  -2.5, -60
                        -0.6981317,   0.1658063,  -1.0471976,  # -40,  9.5, -60
                        0.5235988,   0.1658063,  -1.0471976,  # 30,   9.5, -60
                        0.0000000,  -0.0436332,  -1.0471976,  #  0,  -2.5, -60
                        -0.6981317,   0.1658063,  -1.0471976,  # -40,  9.5, -60
]

DXL_init_direction = [
                       1,  1,  1,
                        1, 1, 1,   
                        1,  1,  1,
                        -1, -1, -1,   
                        -1,  -1,  -1,
                        -1, -1, -1]

ADDR_TORQUE_ENABLE      = 64
ADDR_GOAL_VELOCITY      = 104
ADDR_GOAL_POSITION      = 116
ADDR_PRESENT_POSITION   = 132
ADDR_PROFILE_VELOCITY   = 112

LEN_PRESENT_POSITION    = 4

TORQUE_ENABLE           = 1
TORQUE_DISABLE          = 0

PROTOCOL_VERSION        = 2.0
BAUDRATE                = 2000000
DEVICENAME              = '/dev/ttyU2D2'

# Velocity profile written once on connect
DEFAULT_VELOCITY_TICKS  = 20 # 50

RAD_TO_TICK = 4096 / (2 * 3.14159)

# ---------------------------------------------------------

class RedMirrorDXLController(Node):
    def __init__(self):
        super().__init__('red_mirror_dynamixel_node')

        # ROS interfaces
        self.cmd_sub = self.create_subscription(
            Float64MultiArray,
            'red_mirror/DXL_cmd_ID_positions',
            self.cmd_callback,
            10
        )

        self.pos_pub = self.create_publisher(
            Float64MultiArray,
            'red_mirror/DXL_cur_positions',
            10
        )

        # Dynamixel setup
        self.portHandler = PortHandler(DEVICENAME)
        self.packetHandler = PacketHandler(PROTOCOL_VERSION)

        self._open_port()
        self._scan_motors()
        self._enable_torque()   ################################################
        self._set_default_velocity()
        self._move_to_init_position()

        self.groupSyncWritePosition = GroupSyncWrite(
            self.portHandler, self.packetHandler, ADDR_GOAL_POSITION, 4
        )

        self.timer = self.create_timer(0.01, self.publish_positions)

        self.get_logger().info("✅ Red Mirror Controller Ready.")
        
        
        # Create Group Sync Read
        self.groupSyncRead = GroupSyncRead(
            self.portHandler,
            self.packetHandler,
            ADDR_PRESENT_POSITION,
            LEN_PRESENT_POSITION)

        # Add all motor IDs only once
        for dxl_id in DXL_ID_ThC_LIST:
            dxl_add_result = self.groupSyncRead.addParam(dxl_id)
            if not dxl_add_result:
                self.get_logger().warn(f"Failed to add motor {dxl_id} to sync read.")


    # ---------------------------------------------------------
    def _open_port(self):
        if not self.portHandler.openPort():
            self.get_logger().error("❌ Failed to open port!")
            raise RuntimeError()
        if not self.portHandler.setBaudRate(BAUDRATE):
            self.get_logger().error("❌ Failed to set baudrate!")
            raise RuntimeError()
        self.get_logger().info("Port opened OK.")

    # ---------------------------------------------------------
    def _scan_motors(self):
        missing = []
        for dxl_id in list(DXL_ID_LIST):
            _, result, _ = self.packetHandler.ping(self.portHandler, dxl_id)
            if result != COMM_SUCCESS:
                missing.append(dxl_id)

        if missing:
            print(f"\nMissing motors: {missing}")
            choice = input("Ignore missing and continue? [y/N]: ")
            if choice.lower() != 'y':
                print("Aborting.")
                raise SystemExit()
            for m in missing:
                DXL_ID_LIST.remove(m)

        self.get_logger().info(f"Active IDs: {DXL_ID_LIST}")

    # ---------------------------------------------------------
    def _enable_torque(self):
        for dxl_id in DXL_ID_LIST:
            self.packetHandler.write1ByteTxRx(
                self.portHandler, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
            
        # ---------------------------------------------------------
    def _disable_torque(self):
        for dxl_id in DXL_ID_LIST:
            self.packetHandler.write1ByteTxRx(
                self.portHandler, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)

    # ---------------------------------------------------------
    def _set_default_velocity(self):
        for dxl_id in DXL_ID_LIST:
            self.packetHandler.write4ByteTxRx(
                self.portHandler, dxl_id, ADDR_PROFILE_VELOCITY,
                DEFAULT_VELOCITY_TICKS
            )
        self.get_logger().info(f"Velocity set = {DEFAULT_VELOCITY_TICKS}")

    # ---------------------------------------------------------
    def _move_to_init_position(self):
        self.get_logger().info("Moving to initial pose...")

        sync_init = GroupSyncWrite(
            self.portHandler, self.packetHandler, ADDR_GOAL_POSITION, 4
        )

        for idx, dxl_id in enumerate(DXL_ID_LIST):
            goal_rad = DXL_init_positions[idx] * DXL_init_direction[idx]
            goal_tick = int(goal_rad * RAD_TO_TICK + 2048)

            pbytes = [
                DXL_LOBYTE(DXL_LOWORD(goal_tick)),
                DXL_HIBYTE(DXL_LOWORD(goal_tick)),
                DXL_LOBYTE(DXL_HIWORD(goal_tick)),
                DXL_HIBYTE(DXL_HIWORD(goal_tick)),
            ]

            sync_init.addParam(dxl_id, pbytes)

        sync_init.txPacket()

    # ---------------------------------------------------------
    def cmd_callback(self, msg):
        data = msg.data
        n = len(data)

        if n % 2 != 0:
            self.get_logger().warn("Expected [ID,pos,ID,pos,...]")
            return

        self.groupSyncWritePosition.clearParam()
        i = 0

        while i < n:
            dxl_id = int(data[i])
            goal_rad = data[i+1]
            goal_tick = int(goal_rad * RAD_TO_TICK + 2048)

            pbytes = [
                DXL_LOBYTE(DXL_LOWORD(goal_tick)),
                DXL_HIBYTE(DXL_LOWORD(goal_tick)),
                DXL_LOBYTE(DXL_HIWORD(goal_tick)),
                DXL_HIBYTE(DXL_HIWORD(goal_tick)),
            ]

            self.groupSyncWritePosition.addParam(dxl_id, pbytes)
            i += 2

        self.groupSyncWritePosition.txPacket()

    # ---------------------------------------------------------
    def publish_positions(self):
        # msg = Float64MultiArray()
        # arr = []

        # for dxl_id in DXL_ID_ThC_LIST: # DXL_ID_LIST
        #     pos_tick, _, _ = self.packetHandler.read4ByteTxRx(self.portHandler, dxl_id, ADDR_PRESENT_POSITION)
        #     rad = (pos_tick - 2048) / RAD_TO_TICK
        #     arr.append(rad)

        # msg.data = arr   
        # self.pos_pub.publish(msg)
        
         # Efficiently read and publish all motor positions using GroupSyncRead.
         # Reads all motors (same address) in one broadcast and publishes once.
        
        msg = Float64MultiArray()
        arr = []

        # One broadcast read for all motors
        dxl_comm_result = self.groupSyncRead.txRxPacket()
        if dxl_comm_result != COMM_SUCCESS:
            self.get_logger().warn(f"SyncRead failed: {dxl_comm_result}")
            return

        # for dxl_id in DXL_ID_ThC_LIST:
        for dxl_id in DXL_ID_LIST:
            if self.groupSyncRead.isAvailable(dxl_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION):
                pos_tick = self.groupSyncRead.getData(dxl_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
                rad = (pos_tick - 2048) / RAD_TO_TICK
                arr.append(rad)
            else:
                arr.append(float('nan'))

        msg.data = arr
        self.pos_pub.publish(msg)
        

    # ---------------------------------------------------------
    def destroy_node(self):
        for dxl_id in DXL_ID_LIST:
            self.packetHandler.write1ByteTxRx(self.portHandler, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
        
        self._disable_torque(self)
        self.portHandler.closePort()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RedMirrorDXLController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

# ======================================= #

# Test this node
# ros2 topic pub --once red_mirror/DXL_cmd_ID_positions std_msgs/Float64MultiArray "data: [1, 3]"