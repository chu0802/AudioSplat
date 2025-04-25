
import torchvision
from features.base import BaseFeatureNetwork
from dataclasses import dataclass
import torch
import clip
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode


@dataclass
class OpenAICLIPNetworkConfig:
    version: str = "ViT-L/14"
    device = "cuda"

class OpenAICLIPNetwork(BaseFeatureNetwork):
    def __init__(self, config: OpenAICLIPNetworkConfig):
        super().__init__()
        self.config = config
        self.model, _ = clip.load(config.version, device=config.device)
        self.image_preprocess = Compose([
            Resize(224, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(224),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
    
    @torch.no_grad()
    def encode_text(self, text):
        input = self.tokenizer(text, return_tensors="pt", padding=True).to("cuda")
        text_features = self.model.get_text_features(**input)

        return text_features

    @torch.no_grad()
    def encode_image(self, image):
        inputs = self.image_preprocess(image)
        image_features = self.model.encode_image(inputs)
        
        return image_features

    @property
    def embedding_dim(self) -> int:
        return 768