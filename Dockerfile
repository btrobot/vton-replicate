FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="/idm_vton_code"

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3-pip \
    libgl1-mesa-glx libglib2.0-0 ffmpeg git wget ca-certificates \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
# Use Aliyun PyPI mirror + retry
RUN pip install --no-cache-dir --retries 5 --timeout 120 \
    -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com \
    -r /tmp/requirements.txt \
    && pip install --no-cache-dir --retries 5 --timeout 120 \
    -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com \
    git+https://github.com/facebookresearch/segment-anything.git \
    groundingdino-py==0.4.0 \
    && rm -rf /root/.cache

RUN git clone --depth 1 https://github.com/TemryL/ComfyUI-IDM-VTON.git /tmp/idm-vton \
    && cp -r /tmp/idm-vton/src/idm_vton /idm_vton_code \
    && rm -rf /tmp/idm-vton

RUN mkdir -p /models/grounding-dino /models/sam /models/densepose /models/idm-vton

COPY download_models.py /project/download_models.py
COPY setup.sh /project/setup.sh
COPY predict.py /project/predict.py
RUN chmod +x /project/setup.sh

WORKDIR /project
CMD ["/bin/bash", "-c", "/project/setup.sh && python3 /project/predict.py"]
