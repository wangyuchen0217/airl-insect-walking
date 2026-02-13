import rosbag2_py
import rclpy
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message
import numpy as np

# Current directory is the rosbag2_xxx_... directory
bag_path = "."

# Initialize rclpy (for deserialization)
rclpy.init(args=None)

# open bag
reader = rosbag2_py.SequentialReader()
storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3")
converter_options = rosbag2_py.ConverterOptions("", "")
reader.open(storage_options, converter_options)

# Acquire all topic and type information
topic_types_info = reader.get_all_topics_and_types()
topic_type_map = {t.name: t.type for t in topic_types_info}
print("Topics:", topic_type_map)

# Prepare message type classes for each topic
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

    # Focus on following Float64MultiArray topics
    if topic not in (
        "/red_mirror/DXL_cmd_ID_positions",
        "/red_mirror/DXL_cur_positions",
        "/red_mirror/foot_contact",
        "/red_mirror/imu",
    ):
        continue

    msg_cls = msg_type_class[topic]
    msg = deserialize_message(data, msg_cls)  # Deserialize to actual ROS message object

    # These are all std_msgs/msg/Float64MultiArray, so we can directly use msg.data
    arr = np.array(msg.data, dtype=np.float64)

    if topic == "/red_mirror/DXL_cmd_ID_positions":
        DXL_cmd.append(arr)
    elif topic == "/red_mirror/DXL_cur_positions":
        DXL_cur.append(arr)
    elif topic == "/red_mirror/foot_contact":
        foot.append(arr)
    elif topic == "/red_mirror/imu":
        imu.append(arr)

# Save to CSV (make sure to check if there is data)
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
