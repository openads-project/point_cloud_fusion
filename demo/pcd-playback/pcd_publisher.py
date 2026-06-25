#!/usr/bin/env python3

# Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
# SPDX-License-Identifier: Apache-2.0

import os
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, List, Tuple

import rclpy
from builtin_interfaces.msg import Time
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from tf2_msgs.msg import TFMessage

_PCD_TO_POINTFIELD_DATATYPES = {
    ("F", 4): PointField.FLOAT32,
    ("F", 8): PointField.FLOAT64,
    ("I", 1): PointField.INT8,
    ("I", 2): PointField.INT16,
    ("I", 4): PointField.INT32,
    ("U", 1): PointField.UINT8,
    ("U", 2): PointField.UINT16,
    ("U", 4): PointField.UINT32,
}


@dataclass(frozen=True)
class Sensor:
    """Describe one LiDAR playback stream."""

    name: str
    file_suffix: str
    topic: str
    frame_id: str


@dataclass(frozen=True)
class PcdHeader:
    """Store the PCD schema required to construct a PointCloud2 message."""

    fields: List[str]
    sizes: List[int]
    types: List[str]
    counts: List[int]
    points: int
    data_mode: str
    source_stamp_ns: int

    @property
    def point_stride(self) -> int:
        """Return the number of bytes occupied by one point."""
        return sum(size * count for size, count in zip(self.sizes, self.counts))


SENSORS = (
    Sensor("front-left", "fl", "/demo/points/fl", "drivers/ouster_lidar_fl/lidar"),
    Sensor("front-right", "fr", "/demo/points/fr", "drivers/ouster_lidar_fr/lidar"),
    Sensor("rear-left", "rl", "/demo/points/rl", "drivers/ouster_lidar_rl/lidar"),
    Sensor("rear-right", "rr", "/demo/points/rr", "drivers/ouster_lidar_rr/lidar"),
)

# parent, child, translation xyz, rotation xyzw
STATIC_TRANSFORMS = (
    ("base_link", "rack", (-0.636, -0.695, 1.91), (0.0, 0.0, 0.0, 1.0)),
    (
        "rack",
        "drivers/ouster_lidar_fl/sensor",
        (2.868657363720639, 1.4092181596270053, 0.09646722214988167),
        (-0.06537965928597113, 0.1561986797443087, 0.3813849862049891, 0.9087755305328185),
    ),
    (
        "rack",
        "drivers/ouster_lidar_fr/sensor",
        (2.89, 0.04, 0.1),
        (0.0664522957366524, 0.16043007069025503, -0.3768695318387221, 0.9098437452559152),
    ),
    (
        "rack",
        "drivers/ouster_lidar_rl/sensor",
        (0.04, 1.444, 0.1),
        (-0.16042991606895812, 0.06645266902434228, 0.9098428683553489, 0.3768716488574853),
    ),
    (
        "rack",
        "drivers/ouster_lidar_rr/sensor",
        (0.04, 0.04, 0.1),
        (-0.16042991606895812, -0.06645266902434228, 0.9098428683553489, -0.3768716488574853),
    ),
    (
        "drivers/ouster_lidar_fl/sensor",
        "drivers/ouster_lidar_fl/lidar",
        (0.0, 0.0, 0.038),
        (0.0, 0.0, 0.999999999999985, -1.7320510330969933e-07),
    ),
    (
        "drivers/ouster_lidar_fr/sensor",
        "drivers/ouster_lidar_fr/lidar",
        (0.0, 0.0, 0.038),
        (0.0, 0.0, 0.999999999999985, -1.7320510330969933e-07),
    ),
    (
        "drivers/ouster_lidar_rl/sensor",
        "drivers/ouster_lidar_rl/lidar",
        (0.0, 0.0, 0.038),
        (0.0, 0.0, 0.999999999999985, -1.7320510330969933e-07),
    ),
    (
        "drivers/ouster_lidar_rr/sensor",
        "drivers/ouster_lidar_rr/lidar",
        (0.0, 0.0, 0.038),
        (0.0, 0.0, 0.999999999999985, -1.7320510330969933e-07),
    ),
)


def _parse_pcd_header(handle: BinaryIO) -> PcdHeader:
    metadata = {}
    source_stamp_ns = None
    while True:
        raw_line = handle.readline()
        if not raw_line:
            raise ValueError("PCD header ended before DATA declaration")
        line = raw_line.decode("ascii").strip()
        if line.startswith("# source_stamp_ns "):
            source_stamp_ns = int(line.removeprefix("# source_stamp_ns "))
            continue
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        key = parts[0].upper()
        values = parts[1:]
        if key == "DATA":
            metadata["DATA"] = values[0].lower()
            break
        metadata[key] = values

    fields = [field.lower() for field in metadata["FIELDS"]]
    sizes = [int(value) for value in metadata["SIZE"]]
    types = [value.upper() for value in metadata["TYPE"]]
    counts = [int(value) for value in metadata.get("COUNT", ["1"] * len(fields))]
    points = int(metadata["POINTS"][0])
    if not (len(fields) == len(sizes) == len(types) == len(counts)):
        raise ValueError("PCD header field metadata lengths do not match")
    if source_stamp_ns is None:
        raise ValueError("PCD header is missing '# source_stamp_ns'")
    return PcdHeader(fields, sizes, types, counts, points, metadata["DATA"], source_stamp_ns)


def _point_fields(header: PcdHeader) -> List[PointField]:
    fields = []
    offset = 0
    for name, size, field_type, count in zip(header.fields, header.sizes, header.types, header.counts):
        datatype = _PCD_TO_POINTFIELD_DATATYPES.get((field_type, size))
        if datatype is None:
            raise ValueError(f"Unsupported PCD field {name}: TYPE={field_type} SIZE={size}")
        fields.append(PointField(name=name, offset=offset, datatype=datatype, count=count))
        offset += size * count
    return fields


def _load_pcd(path: Path, frame_id: str) -> Tuple[PointCloud2, int]:
    with path.open("rb") as handle:
        header = _parse_pcd_header(handle)
        fields = _point_fields(header)
        ros_header = Header(frame_id=frame_id, stamp=Time())

        if header.data_mode == "binary":
            expected_size = header.points * header.point_stride
            data = handle.read(expected_size)
            if len(data) != expected_size:
                raise ValueError(f"Binary payload mismatch in {path}: expected {expected_size}, got {len(data)} bytes")
            return (
                PointCloud2(
                    header=ros_header,
                    height=1,
                    width=header.points,
                    fields=fields,
                    is_bigendian=False,
                    point_step=header.point_stride,
                    row_step=expected_size,
                    data=data,
                    is_dense=False,
                ),
                header.source_stamp_ns,
            )

        if header.data_mode == "ascii":
            points: List[Tuple[object, ...]] = []
            for raw_line in handle:
                values = raw_line.decode("ascii").strip().split()
                if not values:
                    continue
                cursor = 0
                point: List[object] = []
                for field_type, count in zip(header.types, header.counts):
                    parser = float if field_type == "F" else int
                    for _ in range(count):
                        point.append(parser(values[cursor]))
                        cursor += 1
                points.append(tuple(point))
            return point_cloud2.create_cloud(ros_header, fields, points), header.source_stamp_ns

        raise ValueError(f"Unsupported PCD DATA mode '{header.data_mode}' in {path}")


def _make_static_tf(stamp: Time) -> TFMessage:
    transforms = []
    for parent, child, xyz, xyzw in STATIC_TRANSFORMS:
        transform = TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = parent
        transform.child_frame_id = child
        transform.transform.translation.x = xyz[0]
        transform.transform.translation.y = xyz[1]
        transform.transform.translation.z = xyz[2]
        transform.transform.rotation.x = xyzw[0]
        transform.transform.rotation.y = xyzw[1]
        transform.transform.rotation.z = xyzw[2]
        transform.transform.rotation.w = xyzw[3]
        transforms.append(transform)
    return TFMessage(transforms=transforms)


def _discover_frame_paths(pcd_dir: Path) -> List[Tuple[Path, ...]]:
    paths_by_sensor = {sensor.file_suffix: sorted(pcd_dir.glob(f"*_{sensor.file_suffix}.pcd")) for sensor in SENSORS}
    frame_indices = {suffix: {path.name.rsplit("_", 1)[0]: path for path in paths} for suffix, paths in paths_by_sensor.items()}
    all_indices = set.union(*(set(indices) for indices in frame_indices.values()))
    if not all_indices:
        raise ValueError(f"No indexed PCD files found in {pcd_dir}")

    incomplete = {
        index: [sensor.file_suffix for sensor in SENSORS if index not in frame_indices[sensor.file_suffix]]
        for index in all_indices
    }
    incomplete = {index: missing for index, missing in incomplete.items() if missing}
    if incomplete:
        raise ValueError(f"Incomplete synchronized PCD sets: {incomplete}")

    ordered_indices = sorted(all_indices, key=int)
    return [tuple(frame_indices[sensor.file_suffix][index] for sensor in SENSORS) for index in ordered_indices]


class PcdPublisher(Node):
    """Publish synchronized LiDAR PCD sets in alternating directions."""

    def __init__(self) -> None:
        """Load all complete PCD sets and create their ROS publishers."""
        super().__init__("pcd_publisher")
        pcd_dir = Path(os.environ.get("PCD_DIR", "/data"))
        publish_rate_hz = float(os.environ.get("PCD_PUBLISH_RATE_HZ", "10.0"))
        if publish_rate_hz <= 0.0:
            raise ValueError("PCD_PUBLISH_RATE_HZ must be greater than zero")

        sensor_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        static_tf_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        frame_paths = _discover_frame_paths(pcd_dir)
        self._frames = [tuple(_load_pcd(path, sensor.frame_id) for path, sensor in zip(paths, SENSORS)) for paths in frame_paths]
        self._frame_index = 0
        self._playback_direction = 1
        self._publishers = [self.create_publisher(PointCloud2, sensor.topic, sensor_qos) for sensor in SENSORS]
        self._tf_publisher = self.create_publisher(TFMessage, "/tf_static", static_tf_qos)
        self._tf_publisher.publish(_make_static_tf(self.get_clock().now().to_msg()))
        self._timer = self.create_timer(1.0 / publish_rate_hz, self._publish)

        total_points = sum(cloud.width * cloud.height for frame in self._frames for cloud, _ in frame)
        self.get_logger().info(
            f"Loaded {len(self._frames)} synchronized PCD sets "
            f"({len(self._frames) * len(SENSORS)} files, {total_points} points total); "
            f"publishing forward and backward at {publish_rate_hz:.2f} Hz"
        )

    def _publish(self) -> None:
        frame = self._frames[self._frame_index]
        playback_start_ns = self.get_clock().now().nanoseconds
        source_start_ns = min(source_stamp_ns for _, source_stamp_ns in frame)
        for publisher, (cloud, source_stamp_ns) in zip(self._publishers, frame):
            stamp_ns = playback_start_ns + source_stamp_ns - source_start_ns
            cloud.header.stamp = Time(sec=stamp_ns // 1_000_000_000, nanosec=stamp_ns % 1_000_000_000)
            publisher.publish(cloud)

        if len(self._frames) > 1:
            next_index = self._frame_index + self._playback_direction
            if next_index < 0 or next_index >= len(self._frames):
                self._playback_direction *= -1
                next_index = self._frame_index + self._playback_direction
            self._frame_index = next_index


def main() -> None:
    """Run the PCD playback node."""
    rclpy.init()
    node = PcdPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception:
        if rclpy.ok():
            raise
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
