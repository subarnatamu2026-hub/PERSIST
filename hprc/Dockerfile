# Charliecloud/Docker build of the Craftium (Minetest) headless system image.
# Used on FASTER via `ch-image build`. Same system deps as hprc/craftium.def.
# The Python venv (torch + craftium) is built separately into $SCRATCH/PERSIST/.venv.
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
# apt in an unprivileged (rootless) container can't drop to the _apt user, which
# breaks its download method. Disable the apt sandbox so it runs as-is.
RUN echo 'APT::Sandbox::User "root";' > /etc/apt/apt.conf.d/00no-sandbox \
 && apt-get update && apt-get install -y \
        g++ make libc6-dev cmake fontconfig libpng-dev libjpeg-dev \
        libgl1-mesa-dev libgles2-mesa-dev libsqlite3-dev libglfw3-dev \
        libogg-dev libvorbis-dev libopenal-dev libcurl4-gnutls-dev \
        libfreetype6-dev zlib1g-dev libgmp-dev libjsoncpp-dev libzstd-dev \
        libluajit-5.1-dev gettext libsdl2-dev libpython3-dev python3-dev \
        xvfb git curl ca-certificates build-essential pkg-config ffmpeg \
 && rm -rf /var/lib/apt/lists/*

# Craftium expects a recent SDL2; build 2.28.5 from source.
RUN cd /tmp && git clone https://github.com/libsdl-org/SDL.git SDL2 && cd SDL2 \
 && git checkout release-2.28.5 && mkdir -p build && cd build \
 && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local \
 && make -j"$(nproc)" && make install && ldconfig && rm -rf /tmp/SDL2

# uv (Python/deps manager).
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh

# Bind mount points must exist in the (read-only) image so ch-run doesn't try to
# mkdir them at runtime (fails on FASTER). /tmp already exists in the base image.
RUN mkdir -p /scratch

ENV PATH=/usr/local/bin:$PATH LC_ALL=C
