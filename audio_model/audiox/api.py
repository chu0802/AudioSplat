import torch
import torchaudio
from einops import rearrange
from .stable_audio_tools.inference.generation import generate_diffusion_cond
from .stable_audio_tools.data.utils import read_video, merge_video_audio
from .stable_audio_tools.data.utils import load_and_process_audio

from .stable_audio_tools.models.factory import create_model_from_config
from .stable_audio_tools.models.utils import load_ckpt_state_dict
import json
from pathlib import Path
import argparse
import numpy as np


MODEL_CONFIG_PATH = "audio_model/audiox/config.json"
MODEL_CKPT_PATH = "/work/chu980802/audiox/model/model.ckpt"



def load_model_from_config(config_path=MODEL_CONFIG_PATH, ckpt_path=MODEL_CKPT_PATH, device="cuda"):
    with open(config_path) as f:
        model_config = json.load(f)
    model = create_model_from_config(model_config)
    model.load_state_dict(load_ckpt_state_dict(ckpt_path))
    model = model.to(device)
    return model
    
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
    # Path("customized_dataset/data/living_room/images/DSC_1536.JPG")
    # Path("audio_model/example/frames/frame_009.png")
    # Path("customized_dataset/data/cars/language_features_train/intermediate_results/Camera_000/seg_img_002.png"),
    # Path("customized_dataset/data/living_room/language_features_train/intermediate_results/DSC_1621/seg_img_022.png"),
    # Path("customized_dataset/data/living_room/language_features_train/intermediate_results/DSC_1536/seg_img_005.png")
    Path("customized_dataset/data/living_room/language_features_train/intermediate_results/DSC_1621/seg_img_004.png")
]

BASE_OUTPUT_DIR = Path("output/audiox/living_room/DSC_1621")
# BASE_OUTPUT_DIR = Path("output/audiox/test_examples")
OBJECT_NAME = [
    "kettle",
]

TEXT_PROMPTS = [
    "generate general audio of this image",
    "boiling the kettle",
    "whistling the kettle",
    "pouring from the kettle",
    "placing the kettle",
    "lifting the kettle",
]
    # "tapping the table",
    # "tapping the object",
    # "tapping the object in this image",
    # "dragging the table",
    # "dragging the object",
    # "dragging the object in this image",
    # "scratching the table",
    # "scratching the object",
    # "scratching the object in this image",
    # "wiping the table",
    # "wiping the object",
    # "wiping the object in this image",
    # "hitting the table",
    # "hitting the object",
    # "hitting the object in this image",
    # "pushing the table",
    # "pushing the object",
    # "pushing the object in this image",

if __name__ == "__main__":
    args = argument_parser()
    device = "cuda"
    
    with open(MODEL_CONFIG_PATH) as f:
        model_config = json.load(f)
    
    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]
    target_fps = model_config["video_fps"]
    seconds_start = 0
    # text_prompt = args.output_dir.stem.replace("_", " ")
    audio_path = None
    
    model = create_model_from_config(model_config)
    model.load_state_dict(load_ckpt_state_dict(MODEL_CKPT_PATH))

    model = model.to(device)
    
    rng = np.random.default_rng(args.seed)

    for sample_path, obj_name in zip(SAMPLE_PATHS, OBJECT_NAME):
        video_tensor = read_video(sample_path, seek_time=0, duration=args.total_seconds, target_fps=target_fps)
        audio_tensor = load_and_process_audio(audio_path, sample_rate, seconds_start, args.total_seconds)
        
        for text_prompt in TEXT_PROMPTS:
            

            conditioning = [{
                "video_prompt": [video_tensor.unsqueeze(0)],        
                "text_prompt": text_prompt,
                "audio_prompt": audio_tensor.unsqueeze(0),
                "seconds_start": seconds_start,
                "seconds_total": args.total_seconds
            }]
            
            seed_list = rng.choice(np.arange(1000000), size=args.num_samples, replace=False)
                
            # Generate stereo audio
            for seed in seed_list:
                output = generate_diffusion_cond(
                    model,
                    steps=args.num_steps,
                    cfg_scale=7,
                    conditioning=conditioning,
                    sample_size=sample_size,
                    sigma_min=0.3,
                    sigma_max=500,
                    seed=seed,
                    sampler_type="dpmpp-3m-sde",
                    device=device
                )

                # Rearrange audio batch to a single sequence
                output = rearrange(output, "b d n -> d (b n)")

                # Peak normalize, clip, convert to int16, and save to file
                output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
                
                image_id = int(sample_path.stem.split("_")[-1])
                output_dir = BASE_OUTPUT_DIR / f"{image_id:03d}_{obj_name}" / text_prompt.replace(" ", "_")
                output_dir.mkdir(parents=True, exist_ok=True)
                
                torchaudio.save(output_dir / f"output_{seed}.wav", output, sample_rate)
