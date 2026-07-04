FROM mirror.ccs.tencentyun.com/nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip \
    libgl1-mesa-glx libglib2.0-0 ffmpeg git wget \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

# Python packages (single layer, clean cache)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && pip install --no-cache-dir \
       git+https://github.com/facebookresearch/segment-anything.git \
       groundingdino-py==0.4.0 \
    && rm -rf /tmp/requirements.txt /root/.cache

# IDM-VTON custom model code
RUN git clone --depth 1 https://github.com/TemryL/ComfyUI-IDM-VTON.git /tmp/idm-vton \
    && cp -r /tmp/idm-vton/src/idm_vton /idm_vton_code \
    && rm -rf /tmp/idm-vton

# Model weights - download in one layer then clean caches
RUN python3 -c "
from huggingface_hub import snapshot_download, hf_hub_download
import subprocess, os

# IDM-VTON (~8GB)
snapshot_download(repo_id='yisol/IDM-VTON', local_dir='/models/idm-vton',
                  local_dir_use_symlinks=False, ignore_patterns=['*.md','*.txt','.git*'])

# GroundingDINO (~700MB)
hf_hub_download(repo_id='ShilongLiu/GroundingDINO', filename='groundingdino_swint_ogc.pth',
                 local_dir='/models/grounding-dino')
hf_hub_download(repo_id='ShilongLiu/GroundingDINO', filename='GroundingDINO_SwinT_OGC.cfg.py',
                 local_dir='/models/grounding-dino')

# DensePose TorchScript (~300MB)
hf_hub_download(repo_id='LayerNorm/DensePose-TorchScript-with-hint-image',
                 filename='densepose_r101_fpn_dl.torchscript', local_dir='/models/densepose')

# SAM ViT-H (~2.5GB)
os.makedirs('/models/sam', exist_ok=True)
subprocess.run(['wget', '-q', '-O', '/models/sam/sam_vit_h_4b8939.pth',
                 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth'], check=True)

# Cleanup HF cache
import shutil
for d in ['/root/.cache/huggingface', '/tmp/pip-*']:
    try: shutil.rmtree(d)
    except: pass
print('All models downloaded')
"

COPY predict.py /project/predict.py
WORKDIR /project

ENV PYTHONPATH="/idm_vton_code:$PYTHONPATH"
CMD ["python3", "predict.py"]
