# Repackage the FightingICE game into a runnable image.
#
# Why: the upstream `ghcr.io/teamfightingice/fightingice:latest` is distroless
# (no shell, no package manager) and is missing `libharfbuzz` so the JVM's
# font subsystem crashes during boot, even with `--lightweight-mode`.
#
# Fix: stage 1 carries the upstream /app payload; stage 2 is a real
# Ubuntu/Temurin JRE image with the missing native libs and Xvfb installed.
# Xvfb gives the JVM a virtual display so font init has somewhere to write.

# Stage 1 — pull the FightingICE payload (jars + assets).
FROM ghcr.io/teamfightingice/fightingice:latest AS upstream

# Stage 2 — runtime with libs + Xvfb.
FROM eclipse-temurin:17-jre-jammy

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libharfbuzz0b \
        libfreetype6 \
        fontconfig \
        xvfb \
        libxrender1 \
        libxi6 \
        libxtst6 \
        libxext6 \
        libxrandr2 \
        libasound2 \
        libgl1 \
        libglu1-mesa \
        libopenal1 \
        x11vnc \
        ca-certificates \
        curl \
        unzip \
    && rm -rf /var/lib/apt/lists/*

# LWJGL native libraries. Upstream FightingICE image ships only the Java-side
# JARs, so OpenGL rendering (LIGHTWEIGHT_MODE=0) crashes at boot trying to
# load liblwjgl.so. Pull the matching natives from Maven Central. Both
# x86_64 (Linux/RTX-3080 box) and arm64 (Apple Silicon) — LWJGL's
# SharedLibraryLoader picks the right one for the JVM's platform.
# Version must match the Java JARs already in /app/lib (3.3.3).
ARG LWJGL_VERSION=3.3.3
RUN mkdir -p /opt/lwjgl-natives && cd /opt/lwjgl-natives \
    && for arch in linux linux-arm64; do \
         for mod in lwjgl lwjgl-glfw lwjgl-opengl lwjgl-openal; do \
           curl -fsSL -o "${mod}-${LWJGL_VERSION}-natives-${arch}.jar" \
             "https://repo.maven.apache.org/maven2/org/lwjgl/${mod}/${LWJGL_VERSION}/${mod}-${LWJGL_VERSION}-natives-${arch}.jar"; \
         done; \
       done

WORKDIR /app
COPY --from=upstream /app /app

# Sprite/character assets needed for full rendering (LIGHTWEIGHT_MODE=0).
# The upstream container ships only gSetting.txt + Motion.csv per character;
# the official release `resource-X.Y.zip` contains the rest. Download once
# at build time and merge over /app/data.
ARG FIGHTINGICE_VERSION=7.1
RUN curl -fsSL -o /tmp/resource.zip \
        "https://github.com/TeamFightingICE/FightingICE/releases/download/v${FIGHTINGICE_VERSION}/resource-${FIGHTINGICE_VERSION}.zip" \
    && unzip -q -o /tmp/resource.zip -d /app \
    && rm /tmp/resource.zip

ENV DISPLAY=:99
EXPOSE 31415

# Start Xvfb in the background, wait briefly for it, then exec java.
# Using `exec` keeps java as PID 1 so docker stop sends SIGTERM correctly,
# and stdout/stderr from java land in `docker logs` (xvfb-run wraps them
# so they don't show up).
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
