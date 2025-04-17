from transformers import CLIPModel, AutoTokenizer
import torchvision
from features.base import BaseFeatureNetwork
from dataclasses import dataclass
import torch


@dataclass
class OpenAICLIPNetworkConfig:
    repo_id: str = "openai/clip-vit-large-patch14"

class OpenAICLIPNetwork(BaseFeatureNetwork):
    def __init__(self, config: OpenAICLIPNetworkConfig):
        super().__init__()
        self.config = config
        self.model = CLIPModel.from_pretrained(config.repo_id)
        self.tokenizer = AutoTokenizer.from_pretrained(config.repo_id)
        self.image_processor = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )
    
    @torch.no_grad()
    def encode_text(self, text):
        input = self.tokenizer(text, return_tensors="pt", padding=True).to("cuda")
        text_features = self.model.get_text_features(**input)

        return text_features

    @torch.no_grad()
    def encode_image(self, image):
        inputs = self.image_processor(image)
        image_features = self.model.get_image_features(pixel_values=inputs)
        
        return image_features

    @property
    def embedding_dim(self) -> int:
        return 768