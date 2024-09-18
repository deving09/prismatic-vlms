"""
open_clip_vit.py
"""

from prismatic.models.backbones.vision.base_vision import  OpenCLIPVisionViTBackbone

# Registry =>> Supported CLIP Vision Backbones (from TIMM)
OPEN_CLIP_VISION_BACKBONES = {
    "recap-clip-vit-l": "hf-hub:UCSC-VLAA/ViT-L-16-HTxt-Recap-CLIP"
}



class OpenCLIPViTBackbone(OpenCLIPVisionViTBackbone):
    def __init__(self, vision_backbone_id: str, image_resize_strategy: str, default_image_size: int = 224) -> None:
        super().__init__(
            vision_backbone_id,
            OPEN_CLIP_VISION_BACKBONES[vision_backbone_id],
            image_resize_strategy,
            default_image_size=default_image_size,
            override_act_layer="quick_gelu" if OPEN_CLIP_VISION_BACKBONES[vision_backbone_id].endswith(".openai") else None,
        )