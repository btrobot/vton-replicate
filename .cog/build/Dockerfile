#syntax=docker/dockerfile:1.4
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib64:/usr/local/nvidia/bin
ENV NVIDIA_DRIVER_CAPABILITIES=all
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked set -eux; \
apt-get update -qq && \
apt-get install -qqy --no-install-recommends curl; \
rm -rf /var/lib/apt/lists/*; \
TINI_VERSION=v0.19.0; \
TINI_ARCH="$(dpkg --print-architecture)"; \
curl -sSL -o /sbin/tini "https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini-${TINI_ARCH}"; \
chmod +x /sbin/tini
ENTRYPOINT ["/sbin/tini", "--"]
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked apt-get update -qq && apt-get install -qqy libgl1-mesa-glx libglib2.0-0 ffmpeg git wget && rm -rf /var/lib/apt/lists/*
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked apt-get update -qq && apt-get install -qqy --no-install-recommends \
	wget \
	curl \
	xz-utils \
	git \
	ca-certificates \
	&& rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.9.26 /uv /uvx /usr/local/bin/
ENV UV_SYSTEM_PYTHON=true
RUN uv python install 3.11 && \
	ln -sf $(uv python find 3.11) /usr/bin/python3
ENV UV_PYTHON=3.11
ENV PATH="/usr/local/bin:$PATH"
COPY --from=cog_build requirements.txt /tmp/requirements.txt
ENV CFLAGS="-O3 -funroll-loops -fno-strict-aliasing -flto -S"
RUN --mount=type=cache,target=/root/.cache/pip uv run pip install --cache-dir /root/.cache/pip -r /tmp/requirements.txt
ENV CFLAGS=
ENV CFLAGS="-O3 -funroll-loops -fno-strict-aliasing -flto -S"
RUN --mount=type=cache,target=/root/.cache/uv uv pip install --no-cache cog
ENV CFLAGS=
RUN find / -type f -name "*python*.so" -printf "%h\n" | sort -u > /etc/ld.so.conf.d/cog.conf && ldconfig
RUN pip install git+https://github.com/facebookresearch/segment-anything.git
RUN pip install groundingdino-py==0.4.0
RUN git clone --depth 1 https://github.com/TemryL/ComfyUI-IDM-VTON.git /tmp/ComfyUI-IDM-VTON
RUN cp -r /tmp/ComfyUI-IDM-VTON/src/idm_vton /idm_vton_code
RUN rm -rf /tmp/ComfyUI-IDM-VTON
RUN mkdir -p /models
RUN cp setup.sh /setup.sh && chmod +x /setup.sh
WORKDIR /src
EXPOSE 5000
ENV COG_PREDICT_TYPE_STUB="predict.py:Predictor"
CMD ["python", "-m", "cog.server.http"]
COPY . /src