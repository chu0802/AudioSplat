import argparse
from pathlib import Path
from audio_model.ssv2a.model.pipeline import Pipeline
from audio_model.ssv2a.model.aldm import build_audioldm, emb_to_audio
from audio_model.ssv2a.data.utils import save_wave
from audio_model.ssv2a.model.dalle2_prior import Dalle2Prior
from features.ssv2a import SSV2ANetworkConfig
import clip
from PIL import Image
from torchvision.transforms import Resize, ToTensor
import numpy as np
import torch

def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--total_seconds", type=int, default=10)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--num_steps", type=int, default=100)
    parser.add_argument("--output_dir", type=Path, default=Path("output/audiox"))
    parser.add_argument("--sample_path", type=Path, default=Path("audio_model/example/frames/frame_001.png"))
    return parser.parse_args()

SAMPLE_PATHS = [
    Path("customized_dataset/data/living_room/language_features_train/intermediate_results/DSC_1621/seg_img_004.png"),
    # Path("customized_dataset/data/living_room/language_features_train/intermediate_results/DSC_1536/seg_img_005.png")
    # Path("customized_dataset/data/living_room/language_features_train/intermediate_results/DSC_1621/seg_img_012.png"),
    # Path("customized_dataset/data/living_room/language_features_train/intermediate_results/DSC_1621/seg_img_022.png"),
    # Path("customized_dataset/data/living_room/images/DSC_1536.JPG")
]

BASE_OUTPUT_DIR = Path("output/ssv2a/living_room/DSC_1621")
OBJECT_NAME = "kettle"

TEXT_PROMPTS = [
    "boiling",
    "whistling",
    "pouring",
    "placing",
    "lifting",
    
    # "opening",
    # "closing",
    # "knocking",
    # "rattling",
    # "cleaning",
    # "breaking",
    # "shattering",
    # "opening a window",
    # "closing a window",
    # "knocking a window",
    # "rattling a window",
    # "cleaning a window",
    # "breaking a window",
    # "shattering a window",
]

# TEXT_PROMPTS = [
#     "slamming",
#     "creaking",
#     "opening",
#     "knocking",
#     "closing",
#     "hitting",
#     "pushing",
#     "pulling",
#     "shutting",
#     "sliding",
# ]

@torch.no_grad()
def get_image_feature(clip_model, preprocess, img):
    img = preprocess(img).unsqueeze(0).to("cuda")
    embs = clip_model.encode_image(img)
    embs /= embs.norm(dim=-1, keepdim=True)
    embs = embs.float()

    return embs

@torch.no_grad()
def get_text_feature(dalle2_model, texts, bs=64, n_samples_per_batch=2, cond_scale=1):
    text_features = dalle2_model.sample(texts, n_samples_per_batch=n_samples_per_batch, cond_scale=cond_scale)
    return text_features

if __name__ == "__main__":
    args = argument_parser()
    meta_config = SSV2ANetworkConfig()
    var_samples = 64
    
    clip_model, preprocess = clip.load(meta_config.clip_version, device="cuda")
    clip_model.eval()
    
    dalle2_model = Dalle2Prior(meta_config.dalle2_cfg, meta_config.dalle2_ckpt, device="cuda")
    
    model = Pipeline(meta_config.config_path, meta_config.ckpt_path, device="cuda").eval()

    rng = np.random.default_rng(args.seed)
    
    audioldm_v = model.config['audioldm_version']
    audio_ldm = build_audioldm(model_name=audioldm_v, device="cuda")
    
    text_features = get_text_feature(dalle2_model, TEXT_PROMPTS)
    text_bembs = model.manifold.fold_clips(text_features, var_samples=var_samples, normalize=False)

    with torch.no_grad():
        for sample_path in SAMPLE_PATHS:
            img = Image.open(sample_path)
            image_feature = get_image_feature(clip_model, preprocess, img)
            
            image_bembs = model.manifold.fold_clips(image_feature, var_samples=var_samples, normalize=False)
            
            for text_prompt, text_feature, text_bemb in zip(TEXT_PROMPTS, text_features, text_bembs):
                remixer_src = torch.zeros((1, model.remixer.slot, model.fold_dim), device="cuda", dtype=image_feature.dtype)
                remixer_clip = torch.zeros((1, model.remixer.slot, model.clip_dim), device="cuda", dtype=image_feature.dtype)
                
                remixer_src[:, 1, ...] = image_bembs
                remixer_src[:, 2, ...] = text_bemb
                
                remixer_clip[:, 1, ...] = image_feature
                remixer_clip[:, 2, ...] = text_feature
                
                remix_clap = model.cycle_mix(
                    remixer_clip,
                    fixed_src=remixer_src,
                    its=64,
                    var_samples=var_samples,
                    samples=64,
                    shuffle=True,
                )
                
                seed = rng.integers(100000)

                local_waves = emb_to_audio(audio_ldm, remix_clap, batchsize=64, duration=10, n_candidate_gen_per_text=5, return_multiple_samples=True, seed=seed)
                
                del remixer_src, remixer_clip, remix_clap
                
                image_id = int(sample_path.stem.split("_")[-1])
                output_dir = BASE_OUTPUT_DIR / f"{image_id:03d}_{OBJECT_NAME}" / text_prompt.replace(" ", "_")
                output_dir.mkdir(parents=True, exist_ok=True)

                save_wave(local_waves, output_dir, f"output_{seed}")
