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
    parser.add_argument("--seed", type=int, default=1102)
    parser.add_argument("--total_seconds", type=int, default=10)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--num_steps", type=int, default=100)
    parser.add_argument("--output_dir", type=Path, default=Path("output/audiox"))
    parser.add_argument("--sample_path", type=Path, default=Path("audio_model/example/frames/frame_001.png"))
    return parser.parse_args()

if __name__ == "__main__":
    args = argument_parser()
    device = "cuda"
    
    with open(MODEL_CONFIG_PATH) as f:
        model_config = json.load(f)
    
    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]
    target_fps = model_config["video_fps"]
    seconds_start = 0
    text_prompt = "closing this object gently" 
    audio_path = None
    
    model = create_model_from_config(model_config)
    model.load_state_dict(load_ckpt_state_dict(MODEL_CKPT_PATH))

    model = model.to(device)

    video_tensor = read_video(args.sample_path, seek_time=0, duration=args.total_seconds, target_fps=target_fps)
    audio_tensor = load_and_process_audio(audio_path, sample_rate, seconds_start, args.total_seconds)

    conditioning = [{
        "video_prompt": [video_tensor.unsqueeze(0)],        
        "text_prompt": text_prompt,
        "audio_prompt": audio_tensor.unsqueeze(0),
        "seconds_start": seconds_start,
        "seconds_total": args.total_seconds
    }]
        
    # Generate stereo audio
    for i in range(args.num_samples):
        current_seed = args.seed + i
        output = generate_diffusion_cond(
            model,
            steps=args.num_steps,
            cfg_scale=7,
            conditioning=conditioning,
            sample_size=sample_size,
            sigma_min=0.3,
            sigma_max=500,
            seed=current_seed,
            sampler_type="dpmpp-3m-sde",
            device=device
        )

        # Rearrange audio batch to a single sequence
        output = rearrange(output, "b d n -> d (b n)")

        # Peak normalize, clip, convert to int16, and save to file
        output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
        
        args.output_dir.mkdir(parents=True, exist_ok=True)
        
        torchaudio.save(args.output_dir / f"output_{current_seed}.wav", output, sample_rate)
