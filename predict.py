"""
IDM-VTON Virtual Try-On Pipeline for Replicate
Chains: GroundingDINO → SAM → DensePose → IDM-VTON
"""
import os
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from cog import BasePredictor, Input, Path
from PIL import Image
from torchvision import transforms

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IDM_VTON_WEIGHTS = "/models/idm-vton"
GROUNDING_DINO_DIR = "/models/grounding-dino"
SAM_CHECKPOINT = "/models/sam/sam_vit_h_4b8939.pth"
DENSEPOSE_MODEL = "/models/densepose/densepose_r101_fpn_dl.torchscript"
IDM_VTON_CODE = "/idm_vton_code"

# Add IDM-VTON custom model code to path
sys.path.insert(0, IDM_VTON_CODE)


class Predictor(BasePredictor):

    def _ensure_models(self):
        """Download models on first run."""
        import subprocess
        done_flag = "/models/.download_complete"
        if os.path.exists(done_flag):
            print("✅ Models already downloaded")
            return
        
        print("⏳ Downloading models (~12GB, first run only)...")
        
        from huggingface_hub import snapshot_download, hf_hub_download
        
        print("  [1/4] IDM-VTON (~8GB)...")
        snapshot_download(repo_id="yisol/IDM-VTON", local_dir="/models/idm-vton",
                          local_dir_use_symlinks=False, 
                          ignore_patterns=["*.md", "*.txt", ".git*"])
        
        print("  [2/4] GroundingDINO (~700MB)...")
        hf_hub_download(repo_id="ShilongLiu/GroundingDINO", 
                        filename="groundingdino_swint_ogc.pth",
                        local_dir="/models/grounding-dino")
        hf_hub_download(repo_id="ShilongLiu/GroundingDINO",
                        filename="GroundingDINO_SwinT_OGC.cfg.py",
                        local_dir="/models/grounding-dino")
        
        print("  [3/4] DensePose (~300MB)...")
        hf_hub_download(
            repo_id="LayerNorm/DensePose-TorchScript-with-hint-image",
            filename="densepose_r101_fpn_dl.torchscript",
            local_dir="/models/densepose")
        
        print("  [4/4] SAM ViT-H (~2.5GB)...")
        os.makedirs("/models/sam", exist_ok=True)
        subprocess.run([
            "wget", "-q", "-O", "/models/sam/sam_vit_h_4b8939.pth",
            "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
        ], check=True)
        
        Path(done_flag).touch()
        print("✅ All models downloaded")

    def setup(self):
        """Load all 4 models. Runs once at container startup."""
        # Download models if not already present
        self._ensure_models()
        
        print("⏳ Loading models...")
        t0 = time.time()

        # ── 1. GroundingDINO ──
        print("  [1/4] GroundingDINO...")
        from groundingdino.util.inference import load_model as load_dino_model
        from groundingdino.util.inference import predict as dino_predict
        self.dino_model = load_dino_model(
            os.path.join(GROUNDING_DINO_DIR, "GroundingDINO_SwinT_OGC.cfg.py"),
            os.path.join(GROUNDING_DINO_DIR, "groundingdino_swint_ogc.pth"),
        )
        self.dino_predict = dino_predict

        # ── 2. SAM ──
        print("  [2/4] SAM...")
        from segment_anything import sam_model_registry, SamPredictor
        sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT)
        sam.to(DEVICE)
        self.sam_predictor = SamPredictor(sam)

        # ── 3. DensePose (TorchScript) ──
        print("  [3/4] DensePose...")
        self.densepose_model = torch.jit.load(DENSEPOSE_MODEL, map_location="cpu")
        self.densepose_model.to(DEVICE)
        self.densepose_model.eval()

        # ── 4. IDM-VTON Pipeline ──
        print("  [4/4] IDM-VTON pipeline...")
        from diffusers import AutoencoderKL, DDPMScheduler
        from transformers import (
            AutoTokenizer, CLIPImageProcessor,
            CLIPVisionModelWithProjection,
            CLIPTextModelWithProjection, CLIPTextModel,
        )
        from unet_hacked_tryon import UNet2DConditionModel
        from unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
        from tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline

        dtype = torch.float16

        noise_scheduler = DDPMScheduler.from_pretrained(
            IDM_VTON_WEIGHTS, subfolder="scheduler"
        )
        vae = AutoencoderKL.from_pretrained(
            IDM_VTON_WEIGHTS, subfolder="vae", torch_dtype=dtype
        ).requires_grad_(False).eval().to(DEVICE)

        unet = UNet2DConditionModel.from_pretrained(
            IDM_VTON_WEIGHTS, subfolder="unet", torch_dtype=dtype
        ).requires_grad_(False).eval().to(DEVICE)

        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            IDM_VTON_WEIGHTS, subfolder="image_encoder", torch_dtype=dtype
        ).requires_grad_(False).eval().to(DEVICE)

        unet_encoder = UNet2DConditionModel_ref.from_pretrained(
            IDM_VTON_WEIGHTS, subfolder="unet_encoder", torch_dtype=dtype
        ).requires_grad_(False).eval().to(DEVICE)

        text_encoder_one = CLIPTextModel.from_pretrained(
            IDM_VTON_WEIGHTS, subfolder="text_encoder", torch_dtype=dtype
        ).requires_grad_(False).eval().to(DEVICE)

        text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
            IDM_VTON_WEIGHTS, subfolder="text_encoder_2", torch_dtype=dtype
        ).requires_grad_(False).eval().to(DEVICE)

        tokenizer_one = AutoTokenizer.from_pretrained(
            IDM_VTON_WEIGHTS, subfolder="tokenizer", use_fast=False
        )
        tokenizer_two = AutoTokenizer.from_pretrained(
            IDM_VTON_WEIGHTS, subfolder="tokenizer_2", use_fast=False
        )

        self.pipe = TryonPipeline.from_pretrained(
            IDM_VTON_WEIGHTS,
            unet=unet, vae=vae,
            feature_extractor=CLIPImageProcessor(),
            text_encoder=text_encoder_one,
            text_encoder_2=text_encoder_two,
            tokenizer=tokenizer_one,
            tokenizer_2=tokenizer_two,
            scheduler=noise_scheduler,
            image_encoder=image_encoder,
            torch_dtype=dtype,
        )
        self.pipe.unet_encoder = unet_encoder
        self.pipe = self.pipe.to(DEVICE)
        self.pipe.weight_dtype = dtype

        print(f"✅ All models loaded in {time.time()-t0:.1f}s")

    def run(
        self,
        person_image: Path = Input(description="人物照片"),
        garment_image: Path = Input(description="衣物图片"),
        clothing_type: str = Input(
            description="衣物类型提示（用于自动分割定位衣物区域）",
            default="tshirt",
            choices=["tshirt", "dress", "jacket", "pants", "shirt", "clothing", "skirt", "coat"],
        ),
        garment_description: str = Input(
            description="衣物描述（英文，影响生成质量）",
            default="a t-shirt",
        ),
        negative_prompt: str = Input(
            description="负面提示词",
            default="monochrome, lowres, bad anatomy, worst quality, low quality",
        ),
        width: int = Input(description="输出宽度", default=768, ge=384, le=1536),
        height: int = Input(description="输出高度", default=1024, ge=512, le=2048),
        num_inference_steps: int = Input(description="推理步数", default=30, ge=10, le=100),
        guidance_scale: float = Input(description="引导强度", default=2.0, ge=1.0, le=10.0),
        segmentation_threshold: float = Input(
            description="分割置信度阈值", default=0.3, ge=0.1, le=0.9
        ),
        mask_expand_px: int = Input(
            description="分割 mask 膨胀像素数", default=60, ge=0, le=200
        ),
        seed: int = Input(description="随机种子（-1 为随机）", default=-1),
    ) -> Path:
        """Run the full VTON pipeline."""

        if seed == -1:
            seed = torch.randint(0, 2**32, (1,)).item()

        # Load images
        person_img = Image.open(person_image).convert("RGB")
        garment_img = Image.open(garment_image).convert("RGB")
        orig_w, orig_h = person_img.size

        # ── Step 1: GroundingDINO 检测衣物区域 ──
        print(f"🔍 Step 1: Detecting '{clothing_type}' with GroundingDINO...")
        import groundingdino.datasets.transforms as T
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        img_tensor, _ = transform(person_img, None)
        boxes, logits, phrases = self.dino_predict(
            self.dino_model,
            img_tensor,
            clothing_type,
            box_threshold=segmentation_threshold,
            text_threshold=segmentation_threshold,
        )
        if boxes.shape[0] == 0:
            raise ValueError(
                f"未检测到 '{clothing_type}'，请降低 segmentation_threshold 或更换 clothing_type"
            )
        # Use the highest-confidence box
        best_idx = logits.argmax()
        box = boxes[best_idx].numpy()
        # Convert from center format to corner format, scale to original image size
        h, w = img_tensor.shape[1], img_tensor.shape[2]
        cx, cy, bw, bh = box
        x1 = int((cx - bw / 2) * orig_w)
        y1 = int((cy - bh / 2) * orig_h)
        x2 = int((cx + bw / 2) * orig_w)
        y2 = int((cy + bh / 2) * orig_h)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(orig_w, x2), min(orig_h, y2)
        print(f"  Box: ({x1},{y1})-({x2},{y2})")

        # ── Step 2: SAM 精确分割 ──
        print("✂️  Step 2: SAM segmentation...")
        person_np = np.array(person_img)
        self.sam_predictor.set_image(person_np)
        box_np = np.array([[x1, y1, x2, y2]])
        masks, scores, _ = self.sam_predictor.predict(
            box=box_np,
            multimask_output=True,
        )
        # Take the mask with highest score
        best_mask = masks[scores.argmax()].astype(np.uint8)  # (H, W) 0/1

        # Expand mask
        if mask_expand_px > 0:
            kernel = np.ones((mask_expand_px, mask_expand_px), np.uint8)
            best_mask = cv2.dilate(best_mask, kernel, iterations=1)

        # Convert mask to PIL (white = garment area to keep, black = area to replace)
        mask_pil = Image.fromarray(best_mask * 255).convert("RGB")

        # ── Step 3: DensePose ──
        print("🧍 Step 3: DensePose estimation...")
        dp_input = torch.from_numpy(person_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        dp_input = dp_input.to(DEVICE)
        with torch.no_grad():
            dp_output = self.densepose_model(dp_input)
        # Convert DensePose output to image
        dp_result = dp_output.squeeze().cpu()
        if dp_result.dim() == 3:
            dp_result = dp_result[0]  # Take first channel
        dp_result = (dp_result - dp_result.min()) / (dp_result.max() - dp_result.min() + 1e-8)
        dp_result = (dp_result * 255).byte().numpy()
        dp_colored = cv2.applyColorMap(dp_result, cv2.COLORMAP_VIRIDIS)
        dp_colored = cv2.cvtColor(dp_colored, cv2.COLOR_BGR2RGB)
        pose_img = Image.fromarray(dp_colored).resize((width, height))

        # ── Step 4: IDM-VTON 推理 ──
        print("👕 Step 4: IDM-VTON inference...")
        transform_norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        person_resized = person_img.resize((width, height))
        garment_resized = garment_img.resize((width, height))
        mask_resized = mask_pil.resize((width, height))

        with torch.no_grad(), torch.cuda.amp.autocast(), torch.inference_mode():
            prompt = "model is wearing " + garment_description
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.pipe.encode_prompt(
                prompt,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )

            prompt_c = ["a photo of " + garment_description]
            neg_c = [negative_prompt]
            (prompt_embeds_c, _, _, _) = self.pipe.encode_prompt(
                prompt_c,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
                negative_prompt=neg_c,
            )

            pose_tensor = transform_norm(pose_img).unsqueeze(0).to(DEVICE, self.pipe.dtype)
            garment_tensor = transform_norm(garment_resized).unsqueeze(0).to(DEVICE, self.pipe.dtype)

            generator = torch.Generator(DEVICE).manual_seed(seed)

            images = self.pipe(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                num_inference_steps=num_inference_steps,
                generator=generator,
                strength=1.0,
                pose_img=pose_tensor,
                text_embeds_cloth=prompt_embeds_c,
                cloth=garment_tensor,
                mask_image=mask_resized,
                image=person_resized,
                height=height,
                width=width,
                ip_adapter_image=garment_resized,
                guidance_scale=guidance_scale,
            )[0]

        result = images[0]
        output_path = Path("/tmp/output.png")
        result.save(str(output_path))
        print("✅ Done!")
        return output_path
