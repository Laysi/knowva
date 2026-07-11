# syntax=docker/dockerfile:1.7

ARG NODE_VERSION=24.12.0
ARG RUST_VERSION=1.96.1
ARG PDFIUM_VERSION=7763

FROM debian:bookworm-slim AS pdfium
ARG TARGETARCH
ARG PDFIUM_VERSION
ARG PDFIUM_SHA256_AMD64=e3f0c66b2daad710cb6c8edd4a8c45c8902995e359dc0775917fc16e2e56349d
ARG PDFIUM_SHA256_ARM64=1518d09bbf09a1bef27a6f93656d95b55bd22bcaecda597679e05f4e82c5fbfa
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && case "${TARGETARCH}" in \
         amd64) pdfium_arch=x64; pdfium_sha256="${PDFIUM_SHA256_AMD64}" ;; \
         arm64) pdfium_arch=arm64; pdfium_sha256="${PDFIUM_SHA256_ARM64}" ;; \
         *) echo "unsupported TARGETARCH: ${TARGETARCH}" >&2; exit 1 ;; \
       esac \
    && curl -fsSL \
         "https://github.com/bblanchon/pdfium-binaries/releases/download/chromium/${PDFIUM_VERSION}/pdfium-linux-${pdfium_arch}.tgz" \
         -o /tmp/pdfium.tgz \
    && echo "${pdfium_sha256}  /tmp/pdfium.tgz" | sha256sum -c - \
    && mkdir -p /opt/pdfium \
    && tar -xzf /tmp/pdfium.tgz -C /opt/pdfium \
    && rm /tmp/pdfium.tgz

FROM node:${NODE_VERSION}-bookworm-slim AS frontend-builder
WORKDIR /src/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN --mount=type=cache,target=/root/.npm npm ci
COPY frontend/ ./
RUN npm run build

FROM rust:${RUST_VERSION}-bookworm AS backend-builder
WORKDIR /src/backend
COPY backend/Cargo.toml backend/Cargo.lock ./
COPY backend/src ./src
COPY backend/migrations ./migrations
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/backend/target \
    cargo build --locked --release \
    && cp target/release/km-server /tmp/km-server

# Explicit verification target. Build with:
# docker build --target native-test -t ai-km-demo:native-test .
FROM backend-builder AS native-test
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ca-certificates \
       fontconfig \
       fonts-noto-cjk \
       libreoffice-calc \
       libreoffice-core \
       libreoffice-impress \
       libreoffice-writer \
       libstdc++6 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=pdfium /opt/pdfium/lib/libpdfium.so /usr/local/lib/libpdfium.so
RUN ldconfig
ENV PDFIUM_LIBRARY_PATH=/usr/local/lib/libpdfium.so \
    LIBREOFFICE_BIN=/usr/bin/libreoffice \
    DOCUMENT_CONVERSION_TIMEOUT_SECS=120
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/backend/target \
    cargo test --locked \
      page_artifacts::tests::generated_pdf_and_office_fixtures_preserve_page_order \
      -- --ignored --exact --nocapture

FROM debian:bookworm-slim AS runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ca-certificates \
       fontconfig \
       fonts-noto-cjk \
       libreoffice-calc \
       libreoffice-core \
       libreoffice-impress \
       libreoffice-writer \
       libstdc++6 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 km \
    && useradd --uid 10001 --gid km --create-home --home-dir /app km
COPY --from=pdfium /opt/pdfium/lib/libpdfium.so /usr/local/lib/libpdfium.so
COPY --from=pdfium /opt/pdfium/LICENSE /usr/share/doc/pdfium/LICENSE
RUN ldconfig

WORKDIR /app/backend
COPY --from=backend-builder /tmp/km-server ./km-server
COPY --from=frontend-builder /src/frontend/dist /app/frontend/dist
RUN mkdir -p /app/backend/data/files \
    && chown -R km:km /app

USER km
ENV PORT=8080 \
    FILES_DIR=/app/backend/data/files \
    PDFIUM_LIBRARY_PATH=/usr/local/lib/libpdfium.so \
    LIBREOFFICE_BIN=/usr/bin/libreoffice
EXPOSE 8080
STOPSIGNAL SIGTERM
CMD ["./km-server"]
