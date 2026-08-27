# Point cloud image

Build and optionally publish the point cloud data image mounted by the demo's
PCD playback service at `/data`.

Place the `.pcd` files in this directory, then run from the repository root:

```bash
PCD_VERSION=v1.0.0
PCD_IMAGE="ghcr.io/openads-project/point_cloud_fusion/pcds:${PCD_VERSION}"

docker build \
  --build-arg IMAGE_VERSION="${PCD_VERSION}" \
  --tag "${PCD_IMAGE}" \
  demo/pcds
```

The default image name is already configured in
[`demo/docker-compose.yml`](../docker-compose.yml). To use another local image,
set `PCD_IMAGE` when running Docker Compose.

To publish the image, log in with your GitHub username and a `write:packages`
token as the password, then push:

```bash
docker login ghcr.io
docker push "${PCD_IMAGE}"
```
