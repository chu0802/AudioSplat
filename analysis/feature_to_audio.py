from pathlib import Path
import torch
from diffusers import StableAudioPipeline
import soundfile as sf
from tqdm import tqdm

@torch.no_grad()
def main():
    pipe = StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0", torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    feature_dir = Path("output/")
    duration = 10.0

    audio_dir = Path("audio/")

    for method_path in tqdm(sorted(feature_dir.iterdir())):
        for object_path in sorted(method_path.iterdir()):
            for prompt_path in sorted(object_path.iterdir()):
                feature = torch.load(prompt_path, weights_only=True).to("cuda")
                audio = pipe.vae.decode(feature).sample

                waveform_start = int(0 * pipe.vae.config.sampling_rate)
                waveform_end = int(duration * pipe.vae.config.sampling_rate)
                audio = audio[:, :, waveform_start:waveform_end]

                output_dir = audio_dir / method_path.stem / object_path.stem
                output_dir.mkdir(parents=True, exist_ok=True)

                output = audio[0].T.float().cpu().numpy()

                breakpoint()

                sf.write(output_dir / f"{prompt_path.stem}.wav", output, pipe.vae.sampling_rate)

if __name__ == "__main__":
    main()
