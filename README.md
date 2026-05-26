# Namazu

A simple blob server for use with [tanuki](https://github.com/nlfiedler/tanuki) to store and retrieve assets.

## Features

- Supports `GET`, `PUT`, and `DELETE` on assets identified by a base64url-encoded path.
- Stores files in the directory structure defined by the provided identifiers.
- Provides a `/thumbnail` endpoint for producing JPEG-formatted thumbnails of images.
- Provides a `/preview` endpoint for producing JPEG previews constrained to a target displayed `width` or `height` (one or the other) in pixels, preserving aspect ratio.
- Provides a `/metadata` endpoint for retrieving image and video metadata using an EXIF reader and `ffprobe` as appropriate.
- Provides a `/synthetic` endpoint that runs ML inference (image classification + face detection + embeddings) on stored image blobs. See [doc/specs/0001-synthetic-data.md](doc/specs/0001-synthetic-data.md) for the contract.
- Generates a unique `ETag` value and responds to `If-None-Match` with a 304 to support browser caching.
- Supports `Range` request header and responds with a 206 which benefits browser requests for video files.

## Requirements

To produce thumbnails and previews for video assets, `ffmpeg` will be invoked via a command shell.

The synthetic-data endpoint requires four model artifacts (MobileNetV2, SCRFD-2.5g, MobileFaceNet, and a labels-map JSON) listed in `model-manifest.json`. `build.rs` downloads them from the Tanuki project's GitHub Releases at build time, into the `models/` directory at the repo root. The Docker build does this inside the builder stage; the resulting `models/` is copied into the final image.

The synthetic endpoint adds about **~60 MB** of resident memory at steady state (model weights + the ORT runtime + per-request working buffers). On Linux the runtime needs `libgomp1` (already installed by the project Dockerfile).

## Configuration

- **ASSETS_PATH**
  - Path to the location where assets will be stored.
- **HOST**
  - Bind address for the HTTP listener, defaults to `127.0.0.1`
- **PORT**
  - Port number on which to listen for connections, defaults to `3000`
- **RUST_LOG**
  - Value interpreted by `env_logger` to set logging levels. The basic levels are `error`, `warn`, `info`, `debug`, `trace`, and `off`.
- **NAMAZU_MODELS_PATH**
  - Override the directory that the synthetic-data endpoint reads ML model files from. Resolution order at startup is: this env var, then `<binary-dir>/models`, then `<repo>/models` (for `cargo run` / `cargo test`).
- **NAMAZU_SKIP_MODEL_FETCH**
  - Set to `1` at build time to make `build.rs` skip all network access. Files already present in `models/` are still SHA256-verified; missing files emit warnings instead of failing the build. Intended for offline development, sandboxed CI, and Docker builds that pre-stage `models/` by other means.

## Uploading files

Files are uploaded via a `PUT` request to the `/assets/{id}` endpoint -- the body of the request is the file content. The `{id}` is substituted with the base64url-encoded path of the destination for the asset, including its destined filename. This operation can be performed with the `curl` command, as shown below.

The paths and names shown below are examples.

```shell
$ echo -n '2003-08-30/01kd0r0qa6p0s8g5nms0bb8m5p.jpg' | base64 | tr '+/' '-_' | tr -d '='
MjAwMy0wOC0zMC8wMWtkMHIwcWE2cDBzOGc1bm1zMGJiOG01cC5qcGc

$ curl -T tests/fixtures/f2t.jpg http://localhost:3000/assets/MjAwMy0wOC0zMC8wMWtkMHIwcWE2cDBzOGc1bm1zMGJiOG01cC5qcGc
HTTP/1.1 201 Created
content-length: 0
date: Mon, 29 Dec 2025 05:04:26 GMT

$ curl -I http://localhost:3000/assets/MjAwMy0wOC0zMC8wMWtkMHIwcWE2cDBzOGc1bm1zMGJiOG01cC5qcGc
HTTP/1.1 200 OK
content-length: 441
content-disposition: inline; filename="01kd0r0qa6p0s8g5nms0bb8m5p.jpg"
etag: "b934c81:1b9:69520bda:1421a568"
last-modified: Mon, 29 Dec 2025 05:04:26 GMT
accept-ranges: bytes
content-type: image/jpeg
cache-control: public, max-age=31536000
date: Mon, 29 Dec 2025 05:05:19 GMT
```

## Downloading as "attachment"

To retrieve an asset in a manner that will encourage the web browser to save the file to disk, pass the `?attachment=yes` query parameter to the `GET /assets/{id}` route. The value for `attachment` can be anything, as long as it is present the `Content-Disposition: attachment` header will be added to the response.

```shell
$ curl -I 'http://localhost:3000/assets/MjAwMy0wOC0zMC8wMWtkMHIwcWE2cDBzOGc1bm1zMGJiOG01cC5qcGc?attachment=1'
HTTP/1.1 200 OK
content-length: 441
etag: "b93f69e:1b9:6952e6d9:4d7e08b"
x-download-options: noopen
accept-ranges: bytes
content-disposition: attachment; filename="01kd0r0qa6p0s8g5nms0bb8m5p.jpg"
content-type: image/jpeg
last-modified: Mon, 29 Dec 2025 20:38:49 GMT
cache-control: public, max-age=31536000
date: Thu, 01 Jan 2026 04:10:07 GMT
```

## Deploying with Docker

```shell
docker build -t namazu-app .
docker image rm 192.168.1.4:5000/namazu
docker image tag namazu-app 192.168.1.4:5000/namazu
docker push 192.168.1.4:5000/namazu
```

## Origin of the name

The [namazu](https://en.wikipedia.org/wiki/Namazu) is a mythical catfish of Japan that purportedly causes earthquakes. That has nothing to do with this project, but the name is short and easy to type.
