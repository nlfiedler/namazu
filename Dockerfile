#
# build the application binary
#
FROM rust:latest AS builder
ENV DEBIAN_FRONTEND="noninteractive"
RUN apt-get -q update && \
    apt-get -q -y install clang
WORKDIR /build
COPY Cargo.toml .
COPY src src/
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
ARG HOST="0.0.0.0"
ARG PORT="3000"
WORKDIR /app
COPY --from=builder /build/target/release/namazu .
COPY --from=healthy /build/target/release/healthcheck .
COPY public public/
VOLUME /assets
ENV ASSETS_PATH="/assets/blobstore"
ENV HEALTHCHECK_PATH="/liveness"
ENV RUST_LOG="info"
EXPOSE ${PORT}
HEALTHCHECK CMD ./healthcheck
ENTRYPOINT ["./namazu"]
