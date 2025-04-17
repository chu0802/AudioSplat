from dataclasses import dataclass
from typing import Tuple

from PIL import Image

import torch
import torch.nn as nn
import torchvision as tv



from audio_clip.utils.transforms import ToTensor1D
from audio_clip.model import AudioCLIP
from features.base import BaseFeatureNetwork


@dataclass
class AudioCLIPNetworkConfig:
    model_filename: str = 'AudioCLIP-Full-Training.pt'
    sample_rate: int = 44100
    image_size: int = 224
    image_mean: Tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073)
    image_std: Tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711)

    pretrained_model_path = f"./ckpts/audio_clip/{model_filename}"
    
    
class AudioCLIPNetwork(BaseFeatureNetwork):
    def __init__(self, config: AudioCLIPNetworkConfig):
        super().__init__()
        self.config = config
        
        self.audio_transforms = ToTensor1D()
        self.image_transforms = tv.transforms.Compose([
            tv.transforms.ToTensor(),
            tv.transforms.Resize(config.image_size, interpolation=Image.BICUBIC),
            tv.transforms.CenterCrop(config.image_size),
            tv.transforms.Normalize(config.image_mean, config.image_std)
        ])
        
        self.model = AudioCLIP(pretrained=config.pretrained_model_path)
        self.model.to("cuda")

    @torch.no_grad()
    def encode_image(self, input):
        ((_, image_features, _), _), _ = self.model(image=input)
        return image_features.half()

    @property
    def embedding_dim(self) -> int:
        return 1024