#
# build the application binary
#
FROM rust:latest AS builder
ENV DEBIAN_FRONTEND="noninteractive"
RUN apt-get -q update && \
    apt-get -q -y install clang
WORKDIR /build
COPY Cargo.toml .
COPY build.rs .
COPY model-manifest.json .
COPY src src/
# `cargo build` triggers build.rs, which fetches the model artifacts
# listed in model-manifest.json into /build/models. The build host must
# have outbound network access to GitHub Releases at this step, unless
# you stage models/ by other means and set NAMAZU_SKIP_MODEL_FETCH=1.
RUN cargo build --release

#
# build the healthcheck binary
#
FROM rust:latest AS healthy
WORKDIR /build
COPY healthcheck/Cargo.toml .
COPY healthcheck/src src/
RUN cargo build --release

#
# build the final image
#
FROM debian:latest
ENV DEBIAN_FRONTEND="noninteractive"
# libgomp1 is required by the prebuilt ONNX Runtime that `ort` links
# against at runtime.
RUN apt-get -q update && \
    apt-get -q -y install --no-install-recommends ffmpeg libgomp1 && \
    rm -rf /var/lib/apt/lists/*
ARG HOST="0.0.0.0"
ARG PORT="3000"
WORKDIR /app
COPY --from=builder /build/target/release/namazu .
COPY --from=healthy /build/target/release/healthcheck .
COPY --from=builder /build/models /app/models
COPY public public/
VOLUME /assets
ENV ASSETS_PATH="/assets/blobstore"
ENV HEALTHCHECK_PATH="/liveness"
ENV RUST_LOG="info"
EXPOSE ${PORT}
HEALTHCHECK CMD ./healthcheck
ENTRYPOINT ["./namazu"]
