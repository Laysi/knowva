# The production image with a knowva-server binary from this branch, built on
# the dev machine by the root Dockerfile's backend-builder stage:
#   podman build --platform linux/arm64 --target backend-builder -t knowva-builder .
#   podman cp "$(podman create knowva-builder)":/tmp/knowva-server .
# The binary must be executable before it is copied in.
FROM ghcr.io/laysi/knowva:dev-page-vision-20261003-2
COPY --chown=knowva:knowva knowva-server /app/backend/knowva-server
