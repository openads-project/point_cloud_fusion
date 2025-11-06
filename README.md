# point_cloud_fusion

This package provides a node that fuses N input point clouds and publishes the fused point cloud. It uses a synchronization policy to ensure that the point clouds are processed in a timely manner. Also, the input point clouds are transformed to a target frame before fusion.

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
| `target_frame` | `string` | Target frame to which the input point clouds are transformed before fusion (default `base_link`). |
| `input_topics` | `list of strings` | List of input topics to subscribe to. The length of this list determines the number of input point clouds to be fused (no default; must be provided). |
| `input_transport_hints` | `list of strings` | Transport hints corresponding to each input topic. Defaults to an empty list, which makes every subscription use `raw`. |
| `sync_queue_size` | `integer` | Queue depth provided to the ApproximateTime synchronizer. This is the maximum number of pending samples kept **per input topic** while searching for matches (default `3`). |
| `max_time_diff_sec` | `double` | Maximum allowed timestamp difference (in seconds) across input point clouds for fusion (default `0.05`). |
| `output_fields` | `list of strings` | Optional subset of point fields to keep in the fused cloud. When unset (default), all incoming fields are preserved. |
| `output_stamp_mode` | `string` | Controls the timestamp written to the fused cloud header. Supported values: `earliest` (default; stamp of the oldest cloud in the batch), `latest` (stamp of the newest cloud in the batch), `mean` (midpoint between earliest and latest), `reference` (timestamp of the first cloud received by the synchronizer). This only affects the published header stamp—each input cloud is still transformed to the target frame using its own timestamp. |
