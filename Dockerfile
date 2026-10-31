FROM ubuntu:24.04 AS builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        cmake \
        g++ \
        git \
        make \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . .

RUN cmake -B build -S . \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=OFF \
        -DRADIXFORGE_METAL=OFF \
        -DGGML_METAL=OFF \
        -DGGML_NATIVE=OFF \
        -DRADIXFORGE_BUILD_TESTS=OFF \
        -DRADIXFORGE_BUILD_TOOLS=OFF \
    && cmake --build build --parallel 2

FROM ubuntu:24.04 AS runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system radixforge \
    && useradd --system --gid radixforge --create-home radixforge

COPY --from=builder /src/build/radixforge /usr/local/bin/radixforge

USER radixforge
WORKDIR /models
EXPOSE 8400

ENTRYPOINT ["radixforge", "--host", "0.0.0.0"]
