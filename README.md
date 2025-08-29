# point_cloud_fusion

This package provides a node that fuses two input point clouds and publishes the fused point cloud. It uses a synchronization policy to ensure that the point clouds are processed in a timely manner. Also, the input point clouds are transformed to a target frame before fusion.

- [Container Images](#container-images)
- [point_cloud_fusion](#point_cloud_fusion)


### Container Images

| Description | Image:Tag | Default Command |
| --- | --- | -- |
| Used for deployment | `gitlab.ika.rwth-aachen.de:5050/fb-fi/its-modules/perception/point_cloud_fusion:latest` | `ros2 launch point_cloud_fusion point_cloud_fusion.launch.py` |
| Used for development | `gitlab.ika.rwth-aachen.de:5050/fb-fi/its-modules/perception/point_cloud_fusion:latest-dev` | `/bin/bash` |


## `point_cloud_fusion`

### Subscribed Topics

| Topic | Type | Description |
| --- | --- | --- |
| `~/input1` | `sensor_msgs::msg::PointCloud2` | Point cloud (1/2) to be fused |
| `~/input2` | `sensor_msgs::msg::PointCloud2` | Point cloud (2/2) to be fused |

### Published Topics

| Topic | Type | Description |
| --- | --- | --- |
| `~/output` | `sensor_msgs::msg::PointCloud2` | Managed by `point_cloud_transport`, so additional transport-specific output topics are automatically created (e.g., `~/output/compressed`, `~/output/draco`). |

### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `target_frame` | `string` | Target frame to which the input point clouds are transformed before fusion. |
| `point_cloud_transport1` | `string` | Transport method for the first point cloud subscriber (`raw\|draco\|zlib\|zstd\|cloudini`). |
| `point_cloud_transport2` | `string` | Transport method for the second point cloud subscriber (`raw\|draco\|zlib\|zstd\|cloudini`). |
