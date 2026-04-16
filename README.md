# point_cloud_fusion

This package provides the `point_cloud_fusion` node, which subscribes to multiple point-cloud topics, transforms all point clouds into a common target frame, and publishes a single fused point cloud. The node supports both CPU and GPU (CUDA) acceleration for high-performance fusion of multi-LiDAR sensor streams.

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
| List of topics defined in params. | `sensor_msgs::msg::PointCloud2` | N point clouds to be fused |

### Published Topics

| Topic | Type | Description |
| --- | --- | --- |
| `~/output` | `sensor_msgs::msg::PointCloud2` | Managed by `point_cloud_transport`, so additional transport-specific output topics are automatically created (e.g., `~/output/cloudini`, `~/output/draco`). |

### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `target_frame` | `string` | Target frame to which the input point clouds are transformed before fusion (no default; must be provided). |
| `input_topics` | `list of strings` | List of input topics to subscribe to. Configure between 1 and 9 entries (inclusive); the length determines how many point clouds are fused (no default; must be provided). |
| `input_transport_hints` | `list of strings` | Transport hints corresponding to each input topic. May be omitted; unspecified entries fall back to the default transport hint (`raw`). |
| `sync_queue_size` | `integer` | Queue depth provided to the ApproximateTime synchronizer. This is the maximum number of pending samples kept **per input topic** while searching for matches (default `3`). |
| `output_queue_size` | `integer` | Queue depth for the fused output publisher managed by `point_cloud_transport` (default `10`). Larger values tolerate short subscriber slowdowns at the cost of latency and memory. |
| `max_time_diff_sec` | `double` | Maximum allowed timestamp difference (in seconds) across input point clouds for fusion (default `0.05`). |
| `age_penalty` | `double` | Optional ApproximateTime age penalty (`>= 0`). Higher values bias the matcher toward newer messages over waiting for tighter timestamp alignment. If omitted, the internal ApproximateTime default is used. |
| `output_fields` | `list of strings` | Optional subset of point fields to keep in the fused cloud. When unset (default), all incoming fields are preserved. Typical entries include `x`, `y`, `z`, `intensity`, `t`, `reflectivity`, `ring`, `ambient`, `range`. |
| `output_stamp_mode` | `string` | Controls the timestamp written to the fused cloud header. Supported values: `earliest` (default; stamp of the oldest cloud in the batch), `latest` (stamp of the newest cloud in the batch), `mean` (midpoint between earliest and latest), `input0` (timestamp from `input_topics[0]` in the synchronized batch). This only affects the published header stamp, not the transformation of each input cloud. |
| `use_cuda` | `boolean` | Enable GPU acceleration via CUDA (default `true` when compiled with CUDA support). Falls back to CPU implementation if CUDA is unavailable or disabled. |
| `fixed_points_per_input_cloud` | `integer` | If greater than 0, limit each input cloud to at most this many **valid** points (NaN/Inf filtered out) before fusing. Processing stops once the limit is reached for each cloud. Default `0` (disabled; process all points from each cloud). |
| `range_limits.enable` | `boolean` | Enable XYZ range filtering in `target_frame`. When `false` (default), no range filtering is applied and all finite points pass through. |
| `range_limits.x_min` | `double` | Minimum x coordinate to keep in `target_frame`, in meters (default `-1000.0`). Must be less than `range_limits.x_max`. Valid range: `[-1000, 1000]` m. |
| `range_limits.x_max` | `double` | Maximum x coordinate to keep in `target_frame`, in meters (default `1000.0`). Must be greater than `range_limits.x_min`. Valid range: `[-1000, 1000]` m. |
| `range_limits.y_min` | `double` | Minimum y coordinate to keep in `target_frame`, in meters (default `-1000.0`). Must be less than `range_limits.y_max`. Valid range: `[-1000, 1000]` m. |
| `range_limits.y_max` | `double` | Maximum y coordinate to keep in `target_frame`, in meters (default `1000.0`). Must be greater than `range_limits.y_min`. Valid range: `[-1000, 1000]` m. |
| `range_limits.z_min` | `double` | Minimum z coordinate to keep in `target_frame`, in meters (default `-20.0`). Must be less than `range_limits.z_max`. Valid range: `[-20, 20]` m. |
| `range_limits.z_max` | `double` | Maximum z coordinate to keep in `target_frame`, in meters (default `20.0`). Must be greater than `range_limits.z_min`. Valid range: `[-20, 20]` m. |
