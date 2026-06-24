# point_cloud_fusion

The package provides a C++ ROS 2 component node that subscribes to multiple point-cloud topics, transforms all point clouds into a common target frame, and publishes a single fused point cloud. It supports CPU fusion and optional CUDA acceleration.

## Node

| Package | Node | Description |
| --- | --- | --- |
| `point_cloud_fusion` | `point_cloud_fusion` | ROS 2 node for point-cloud fusion |

### Subscribed Topics

| Topic | Type | Description |
| --- | --- | --- |
| Configured in `input_topics` | `sensor_msgs/msg/PointCloud2` | Point clouds to transform and fuse. |

### Published Topics

| Topic | Type | Description |
| --- | --- | --- |
| `~/output` | `sensor_msgs/msg/PointCloud2` | Fused point cloud. Managed by `point_cloud_transport`, so transport-specific output topics can also be created. |

## Launch File Usage

The provided launch file supports launch arguments for node naming, parameters, logging, simulation time, tracing, and topic remapping.

```bash
ros2 launch point_cloud_fusion point_cloud_fusion.launch.py \
    namespace:=/perception \
    params:=/docker-ros/ws/src/target/point_cloud_fusion/config/params.yml
```

### Launch Arguments

| Argument | Type | Description |
| --- | --- | --- |
| `name` | `string` | Node name. Default: `point_cloud_fusion`. |
| `namespace` | `string` | Node namespace. Default: empty. |
| `params` | `string` | Path to parameter file. Default: package `config/params.yml`. |
| `log_level` | `string` | ROS logging level (`debug`, `info`, `warn`, `error`, `fatal`). Default: `info`. |
| `use_sim_time` | `bool` | Use simulation clock. Default: `false`. |
| `trace` | `bool` | Enable ROS tracing. Default: `false`. |
| `point_cloud_topic` | `string` | Remap for `~/point_cloud`. Default: `~/point_cloud`. |

## Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `target_frame` | `string` | Target frame to which input point clouds are transformed before fusion. Required. |
| `input_topics` | `string array` | Input point-cloud topics. Configure between 1 and 9 entries. Required. |
| `input_transport_hints` | `string array` | Transport hints corresponding to each input topic. May be omitted; unspecified entries default to `raw`. |
| `sync_queue_size` | `integer` | Queue depth for the approximate-time synchronizer. Default: `3`. |
| `output_queue_size` | `integer` | Queue depth for the fused output publisher. Default: `10`. |
| `max_time_diff_sec` | `double` | Maximum allowed timestamp difference across synchronized input clouds. Default: `0.05`. |
| `age_penalty` | `double` | ApproximateTime age penalty. Default: `0.1`. |
| `output_fields` | `string array` | Optional subset of point fields to keep in the fused cloud. Empty publishes all incoming fields. |
| `output_stamp_mode` | `string` | Fused header timestamp mode: `earliest`, `latest`, `mean`, or `input0`. Default: `earliest`. |
| `use_cuda` | `bool` | Enable CUDA acceleration when compiled with CUDA support. Default: `true`. |
| `fixed_points_per_input_cloud` | `integer` | Runtime-reconfigurable limit for valid points processed per input cloud. `0` disables the limit. Updates apply atomically between fusion batches. Valid range: `[0, 10000000]`. Default: `0`. |

## Point-Cloud Transport

The fused output publisher is managed by `point_cloud_transport`. Configure enabled output transport plugins and their plugin-specific parameters in the node parameter file.
