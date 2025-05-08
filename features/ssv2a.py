from features.base import BaseFeatureNetwork
import torch.nn as nn
import torch
import json
import clip

from audio_model.ssv2a.model.pipeline import Pipeline
from dataclasses import dataclass

@dataclass
class SSV2ANetworkConfig:
    config_path: str = "/work/chu980802/ssv2a/ssv2a.json"
    ckpt_path: str = "/work/chu980802/ssv2a/ssv2a.pth"
    dalle2_cfg: str = "/work/chu980802/ssv2a/dalle2_prior_config.json"
    dalle2_ckpt: str = "/work/chu980802/ssv2a/dalle2_prior.pth"
    clip_version: str = "ViT-L/14"

class SSV2ANetwork(BaseFeatureNetwork):
    def __init__(self, config: SSV2ANetworkConfig):
        super().__init__()
        self.config = config
        self.ssv2a_pipeline = Pipeline(config.config_path, config.ckpt_path, device="cuda").eval()

        self.clip_model, self.preprocess = clip.load(config.clip_version, device="cuda")
        self.clip_model.eval()
    
    @torch.no_grad()
    def encode_image(self, imgs):
        embs = self.clip_model.encode_image(imgs)
        embs /= embs.norm(dim=-1, keepdim=True)
        embs = embs.float()
        
        embs = self.ssv2a_pipeline.clips2foldclaps(embs, 64)
        
        return embs

    @property
    def embedding_dim(self):
        return 512
