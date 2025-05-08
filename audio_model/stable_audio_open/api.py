import torch
import soundfile as sf
from diffusers import StableAudioPipeline
from pathlib import Path

pipe = StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0", torch_dtype=torch.float16)
pipe = pipe.to("cuda")


negative_prompt = "Low quality."
seed = 1102

# define the prompts
# prompts = [
#     "The sound of a wooden door opening",
#     "The sound of a wooden door closing",
#     "The sound of a wooden door creaking",
#     "The sound of a wooden door slamming",
#     "The sound of a wooden door knocking",
#     "The sound of a wooden door hitting",
#     "The sound of a wooden door pushing",
# ]

# prompts = [
#     "The sound of opening a window",
#     "The sound of closing a window",
#     "The sound of knocking a window",
#     "The sound of rattling a window",
#     "The sound of cleaning a window",
#     "The sound of breaking a window",
#     "The sound of shattering a window",
#     "The sound of smashing a window",
# ]

OBJECT_NAME = "kettle"

prompts = [
    "The sound of boiling the kettle",
    "The sound of whistling the kettle",
    "The sound of pouring from the kettle",
    "The sound of placing the kettle",
    "The sound of lifting the kettle",
]

for prompt in prompts:
    # set the seed for generator
    generator = torch.Generator("cuda").manual_seed(seed)

    # run the generation
    audio = pipe(
        prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=100,
        audio_end_in_s=10.0,
        num_waveforms_per_prompt=3,
        generator=generator,
    ).audios

    output = audio[0].T.float().cpu().numpy()
    output_dir = Path("output") / "stable_audio_open" / OBJECT_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_name = prompt.replace(" ", "_") + f"_{seed}.wav"
    sf.write(output_dir / output_name, output, pipe.vae.sampling_rate)
