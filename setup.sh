#!/bin/bash
set -e
MODELS_DIR="/models"
DONE_FLAG="$MODELS_DIR/.download_complete"

if [ -f "$DONE_FLAG" ]; then
    echo "✅ Models already downloaded, skipping"
    exit 0
fi

echo "⏳ Downloading models (first run only, ~12GB)..."

python3 -c "
from huggingface_hub import snapshot_download, hf_hub_download
import subprocess, os

print('  [1/4] IDM-VTON (~8GB)...')
snapshot_download(repo_id='yisol/IDM-VTON', local_dir='/models/idm-vton',
                  local_dir_use_symlinks=False, ignore_patterns=['*.md','*.txt','.git*'])

print('  [2/4] GroundingDINO (~700MB)...')
hf_hub_download(repo_id='ShilongLiu/GroundingDINO', filename='groundingdino_swint_ogc.pth',
                 local_dir='/models/grounding-dino')
hf_hub_download(repo_id='ShilongLiu/GroundingDINO', filename='GroundingDINO_SwinT_OGC.cfg.py',
                 local_dir='/models/grounding-dino')

print('  [3/4] DensePose (~300MB)...')
hf_hub_download(repo_id='LayerNorm/DensePose-TorchScript-with-hint-image',
                 filename='densepose_r101_fpn_dl.torchscript', local_dir='/models/densepose')

print('  [4/4] SAM ViT-H (~2.5GB)...')
os.makedirs('/models/sam', exist_ok=True)
subprocess.run(['wget','-q','-O','/models/sam/sam_vit_h_4b8939.pth',
                 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth'], check=True)
"

touch "$DONE_FLAG"
echo "✅ All models downloaded"
