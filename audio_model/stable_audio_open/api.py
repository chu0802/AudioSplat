import torch
import soundfile as sf
from diffusers import StableAudioPipeline
from pathlib import Path
import json
import argparse
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--prompt_path", type=str, default="/home/chu980802/langsplat/audio_model/stable_audio_open/prompt.json")
    parser.add_argument("-o", "--output_dir", type=str, default="output_test/stable_audio_open/")
    parser.add_argument("-d", "--duration", type=float, default=10.0)
    args  = parser.parse_args()
    return args

def main():
    args = parse_args()

    pipe = StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0", torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    negative_prompt = "Low quality."
    seed = 1102

    with open(args.prompt_path, "r") as f:
        prompts = json.load(f)

    for object_name, prompt_list in prompts.items():
        for prompt in tqdm(prompt_list):
            generator = torch.Generator("cuda").manual_seed(seed)

            # run the generation
            audio = pipe(
                prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=100,
                audio_end_in_s=args.duration,
                num_waveforms_per_prompt=1,
                generator=generator,
            ).audios

            output = audio[0].T.float().cpu().numpy()
            output_dir = Path(args.output_dir) / object_name.replace(" ", "_")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_name = prompt.replace(" ", "_") + f"_{seed}.wav"
            sf.write(output_dir / output_name, output, pipe.vae.sampling_rate)

if __name__ == "__main__":
    main()
