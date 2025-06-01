import torch
import soundfile as sf
from diffusers import StableAudioPipeline
from pathlib import Path
import json
import argparse
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--prompt_path", type=Path, default=Path("/home/chu980802/langsplat/audio_model/stable_audio_open/prompt.json"))
    parser.add_argument("-o", "--output_dir", type=Path, default=Path("output_test/stable_audio_open/"))
    parser.add_argument("-d", "--duration", type=float, default=10.0)
    args  = parser.parse_args()
    return args


def main():
    args = parse_args()

    pipe = StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0", torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    negative_prompt = "Low quality."
    seed = 1102

    with args.prompt_path.open("r") as f:
        prompts = json.load(f)

    for object_name, prompt_list in prompts.items():
        prompt_list.insert(0, object_name.replace("_", " "))
        for i, prompt in tqdm(enumerate(prompt_list)):
            generator = torch.Generator("cuda").manual_seed(seed)

            # run the generation
            latent = pipe(
                prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=100,
                audio_end_in_s=args.duration,
                num_waveforms_per_prompt=1,
                generator=generator,
                output_type="latent",
            ).audios

            output_path = args.output_dir / object_name /f"{i}_{prompt.replace(' ', '_')}.pt"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(latent.detach().cpu(), output_path)

if __name__ == "__main__":
    main()
