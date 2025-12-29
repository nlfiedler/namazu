# Namazu

A simple blob server for use with [tanuki](https://github.com/nlfiedler/tanuki) to store and retrieve assets.

## Features

* Supports `GET`, `PUT`, and `DELETE` on assets identified by a base64-encoded path.
* Provides a `/thumbnail` endpoint for producing JPEG-formatted thumbnails of images.
* Generates a unique `ETag` value and responds to `If-None-Match` with a 304 to support browser caching.
* Supports `Range` header and responds with a 206 which serves browser requests for video files.

## Configuration

- **ASSETS_PATH**
  - Path to the location where assets will be stored.
- **HOST**
  - Bind address for the HTTP listener, defaults to `127.0.0.1`
- **PORT**
  - Port number on which to listen for connections, defaults to `3000`
- **RUST_LOG**
  - Value interpreted by `env_logger` to set logging levels. The basic levels are `error`, `warn`, `info`, `debug`, `trace`, and `off`.

## Uploading files

Files are uploaded using `multipart/form-data` via a `PUT` request to the `/assets/{id}` endpoint. The `{id}` is substituted with the base64-encoded path of the destination for the asset, including its destined filename. This can be done with `curl` as shown below.

The paths and names shown below are examples.

```shell
$ echo -n '2003-08-30/01kd0r0qa6p0s8g5nms0bb8m5p.jpg' | base64
MjAwMy0wOC0zMC8wMWtkMHIwcWE2cDBzOGc1bm1zMGJiOG01cC5qcGc=

$ curl -X PUT -F content=@tests/fixtures/f2t.jpg http://localhost:3000/assets/MjAwMy0wOC0zMC8wMWtkMHIwcWE2cDBzOGc1bm1zMGJiOG01cC5qcGc=
HTTP/1.1 201 Created
content-length: 0
date: Mon, 29 Dec 2025 05:04:26 GMT

$ curl -I http://localhost:3000/assets/MjAwMy0wOC0zMC8wMWtkMHIwcWE2cDBzOGc1bm1zMGJiOG01cC5qcGc=
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

## Deploying with Docker

```shell
docker build -t namazu-app .
docker image rm 192.168.1.4:5000/namazu
docker image tag namazu-app 192.168.1.4:5000/namazu
docker push 192.168.1.4:5000/namazu
```
