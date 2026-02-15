import argparse
import csv
import os
from rclpy.serialization import deserialize_message
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, Image
from beamng_msgs.msg import StateSensor, ElectricsSensor
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions


def save_bag_topics(reader, save_dir):
    """
    Iterates through all topics in a rosbag reader and saves selected ones to CSV or image files.

    Supports:
      - /sensor_publisher/sensors/state        → StateSensor (CSV)
      - /sensor_publisher/sensors/imu          → Imu (CSV)
      - /odom                                  → Odometry (CSV)
      - /sensor_publisher/sensors/electrics    → ElectricsSensor (CSV)
      - /sensor_publisher/sensors/camera/colour_throttle → Image (PNG)
    """

    # Ensure output directory exists
    os.makedirs(save_dir, exist_ok=True)
    img_dir = os.path.join(save_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    writers = {}
    bridge = None
    cv2 = None
    img_warning_shown = False
    saved_topics = set()

    def get_writer(filename, header):
        """Create a CSV writer lazily."""
        path = os.path.join(save_dir, filename)
        if filename not in writers:
            f = open(path, "w", newline="")
            writer = csv.writer(f)
            writer.writerow(header)
            writers[filename] = (f, writer)
        return writers[filename][1]

    # Loop through topics in the bag
    while reader.has_next():
        topic, data, _ = reader.read_next()

        # --- State Sensor ---
        if topic == "/sensor_publisher/sensors/state":
            msg = deserialize_message(data, StateSensor)
            ros_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

            writer = get_writer(
                "state_sensor.csv",
                [
                    "time",
                    "pos_x", "pos_y", "pos_z",
                    "dir_x", "dir_y", "dir_z",
                    "up_x", "up_y", "up_z",
                    "vel_x", "vel_y", "vel_z"
                ],
            )
            writer.writerow([
                ros_time,
                msg.position.x, msg.position.y, msg.position.z,
                msg.dir.x, msg.dir.y, msg.dir.z,
                msg.up.x, msg.up.y, msg.up.z,
                msg.velocity.x, msg.velocity.y, msg.velocity.z,
            ])
            saved_topics.add(topic)

        # --- IMU ---
        elif topic == "/sensor_publisher/sensors/imu":
            msg = deserialize_message(data, Imu)
            ros_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

            writer = get_writer(
                "imu_angular_velocity.csv",
                ["time", "ang_vel_x", "ang_vel_y", "ang_vel_z"],
            )
            writer.writerow([
                ros_time,
                msg.angular_velocity.x,
                msg.angular_velocity.y,
                msg.angular_velocity.z,
            ])
            saved_topics.add(topic)

        # --- Odometry ---
        elif topic == "/odom":
            msg = deserialize_message(data, Odometry)
            ros_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

            writer = get_writer(
                "odom.csv",
                [
                    "time",
                    "pos_x", "pos_y", "pos_z",
                    "quat_x", "quat_y", "quat_z", "quat_w",
                    "vel_x", "vel_y", "vel_z",
                    "ang_vel_x", "ang_vel_y", "ang_vel_z"
                ],
            )
            writer.writerow([
                ros_time,
                msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z,
                msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w,
                msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z,
                msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z,
            ])
            saved_topics.add(topic)

        # --- Electrics ---
        elif topic == "/sensor_publisher/sensors/electrics":
            msg = deserialize_message(data, ElectricsSensor)
            ros_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

            writer = get_writer(
                "controls.csv",
                [
                    "time",
                    "accxsmooth", "accysmooth", "acczsmooth",
                    "brake_input", "parkingbrake_input", "steering_input" ,"throttle_input",
                    "engine_rpm",
                    "wheelspeed"
                ],
            )
            writer.writerow([
                ros_time,
                msg.accxsmooth, msg.accysmooth, msg.acczsmooth,
                msg.brake_input, msg.parkingbrake_input, msg.steering_input, msg.throttle_input,
                msg.rpm,
                msg.wheelspeed
            ])
            saved_topics.add(topic)

        # --- Camera Images ---
        elif topic == "/sensor_publisher/sensors/camera/colour_throttle":
            msg = deserialize_message(data, Image)
            ros_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

            if bridge is None:
                try:
                    import cv2 as _cv2
                    from cv_bridge import CvBridge
                except ImportError:
                    if not img_warning_shown:
                        print("\n⚠️ cv_bridge/cv2 not available; skipping camera images.")
                        img_warning_shown = True
                    continue
                cv2 = _cv2
                bridge = CvBridge()

            try:
                cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                img_filename = os.path.join(img_dir, f"{ros_time}.png")
                cv2.imwrite(img_filename, cv_img)
                saved_topics.add(topic)
            except Exception as e:
                print(f"\n⚠️ Error converting image: {e}")


    # --- Close all CSV files ---
    for f, _ in writers.values():
        f.close()

    expected_topics = [
        '/sensor_publisher/sensors/state',
        '/sensor_publisher/sensors/imu',
        '/odom',
        '/sensor_publisher/sensors/electrics',
        '/sensor_publisher/sensors/camera/colour_throttle',
    ]
    missing_topics = [t for t in expected_topics if t not in saved_topics]

    if missing_topics:
        print(f"\n❌ Missing {len(missing_topics)} topic(s):")
        for t in sorted(missing_topics):
            print(f"   • {t}")
    else:
        print("\n✅ All topics extracted.")
    print(f"\n✅ Saved to {save_dir}")




def main():

    print("\n=================== EXTRACTING ROS DATA ===================\n")

    parser = argparse.ArgumentParser()
    parser.add_argument("--terrain", type=str, default='')
    parser.add_argument("--bag_name", type=str, default='')
    parser.add_argument("--output_name", type=str, default='')
    parser.add_argument("--transmission", type=str, default='')
    parser.add_argument("--type", type=str, default='')
    parser.add_argument("--bag_path", type=str, default='', help='Custom bag path (overrides default path construction)')
    args = parser.parse_args()

    # Path to your bag
    if args.bag_path:
        bag_path = args.bag_path
    else:
        home = os.path.expanduser("~")
        bag_path = f"{home}/ros2_ws/src/beamng_autonomy/bags/{args.terrain}/{args.transmission}/{args.type}/{args.bag_name}"

    # Set up bag reader
    storage_options = StorageOptions(uri=bag_path, storage_id="sqlite3")
    converter_options = ConverterOptions("", "")
    reader = SequentialReader()
    reader.open(storage_options, converter_options)

    # Prepare output directory
    save_dir = f"function_encoder_beamng/data/{args.transmission}/{args.type}/{args.output_name}"

    # Save the state sensor data.
    save_bag_topics(reader, save_dir)

    # Save bag name in text file
    bag_name_file = os.path.join(save_dir, "bag_name.txt")
    with open(bag_name_file, "w") as f:
        f.write(args.bag_name + "\n")


if __name__ == '__main__':
    main()
