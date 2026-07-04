---
title: VTON - Virtual Try-On
emoji: 👕
colorFrom: purple
colorTo: pink
sdk: cog
pinned: false
license: apache-2.0
---

# IDM-VTON Virtual Try-One

一键换衣：上传一张人物照片 + 一张衣物照片，自动输出换衣后的效果图。

基于 [IDM-VTON](https://github.com/BoyuanJiang/IDM-VTON) (ECCV 2024)，集成全自动分割和姿态提取。

## Pipeline

```
人物照片 + 衣物照片
    │
    ├─ GroundingDINO → 自动定位衣物区域
    ├─ SAM → 精确分割 mask
    ├─ DensePose → 人体姿态 UV 图
    └─ IDM-VTON → 生成换衣结果
```

## Usage

```python
import replicate

output = replicate.run(
    "your-username/vton-tryon",
    input={
        "person_image": open("person.jpg", "rb"),
        "garment_image": open("garment.jpg", "rb"),
        "clothing_type": "tshirt",
        "garment_description": "a white t-shirt",
    }
)
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| person_image | image | required | 人物照片 |
| garment_image | image | required | 衣物照片 |
| clothing_type | enum | tshirt | 衣物类型（用于自动分割） |
| garment_description | string | a t-shirt | 衣物描述（英文） |
| width | int | 768 | 输出宽度 |
| height | int | 1024 | 输出高度 |
| num_inference_steps | int | 30 | 推理步数 |
| guidance_scale | float | 2.0 | 引导强度 |
| seed | int | -1 | 随机种子 |

## Hardware

Requires A40 (48GB) GPU. ~12GB VRAM usage.

## License

Apache 2.0
