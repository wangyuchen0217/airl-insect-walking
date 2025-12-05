import rosbag2_py
import rclpy
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message
import numpy as np

# 当前目录就是 rosbag2_xxx_... 的目录
bag_path = "."

# 初始化 rclpy（为了使用反序列化）
rclpy.init(args=None)

# 打开 bag
reader = rosbag2_py.SequentialReader()
storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3")
converter_options = rosbag2_py.ConverterOptions("", "")
reader.open(storage_options, converter_options)

# 获取所有 topic 和类型
topic_types_info = reader.get_all_topics_and_types()
topic_type_map = {t.name: t.type for t in topic_types_info}
print("Topics:", topic_type_map)

# 为每个 topic 准备对应的消息类型
msg_type_class = {}
for topic, type_str in topic_type_map.items():
    try:
        msg_type_class[topic] = get_message(type_str)
    except Exception as e:
        print(f"Failed to get message type for {topic} ({type_str}): {e}")

DXL_cmd = []
DXL_cur = []
foot = []
imu = []

while reader.has_next():
    topic, data, stamp = reader.read_next()

    # 只关心这几个 Float64MultiArray 话题
    if topic not in (
        "/red_mirror/DXL_cmd_ID_positions",
        "/red_mirror/DXL_cur_positions",
        "/red_mirror/foot_contact",
        "/red_mirror/imu",
    ):
        continue

    msg_cls = msg_type_class[topic]
    msg = deserialize_message(data, msg_cls)  # 反序列化为真正的 ROS 消息对象

    # 这些都是 std_msgs/msg/Float64MultiArray，所以直接用 msg.data
    arr = np.array(msg.data, dtype=np.float64)

    if topic == "/red_mirror/DXL_cmd_ID_positions":
        DXL_cmd.append(arr)
    elif topic == "/red_mirror/DXL_cur_positions":
        DXL_cur.append(arr)
    elif topic == "/red_mirror/foot_contact":
        foot.append(arr)
    elif topic == "/red_mirror/imu":
        imu.append(arr)

# 保存为 CSV（注意要检查是否有数据）
if DXL_cmd:
    np.savetxt("DXL_cmd.csv", np.vstack(DXL_cmd), delimiter=",")
    print("Saved DXL_cmd.csv")

if DXL_cur:
    np.savetxt("DXL_cur.csv", np.vstack(DXL_cur), delimiter=",")
    print("Saved DXL_cur.csv")

if foot:
    np.savetxt("foot_contact.csv", np.vstack(foot), delimiter=",")
    print("Saved foot_contact.csv")

if imu:
    np.savetxt("imu.csv", np.vstack(imu), delimiter=",")
    print("Saved imu.csv")

rclpy.shutdown()
