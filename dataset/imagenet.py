from gptwrapper import GPTWrapper
from gptwrapper.response import ObjectRecognitionResponse, InteractionResponse
from gptwrapper.config.system_prompt import RECOGNITION_SYSTEM_PROMPT, INTERACTION_SYSTEM_PROMPT

from pathlib import Path
import json
from tqdm import tqdm
# cd langsplat && ca langsplat
# python -m audio_model.stable_audio_open.api --prompt_path split/split_part_10.json
RECOG_PROMPT = "What is this?"
INTERACTION_PROMPT = "show me 5 different interactions to interact with {object_name} that can produce unique and interesting sounds. If the object is a static object, the format should be action + object_name. e.g., opening the door. If the object can produce sound by itself, the format should be object_name + action, e.g., dog barking"

imagenet_path = Path("imagenet_1000.json")
image_root = Path("/work/chu980802/data/classification/imagenet/images")
image_annotation_path = Path("/work/chu980802/data/classification/imagenet/imagenet_annotations.json")
def main():
    gpt = GPTWrapper(model_name="gemini-2.5-flash-preview-04-17")
    
    with imagenet_path.open("r") as f:
        imagenet_data = json.load(f)
    with image_annotation_path.open("r") as f:
        image_annotation_data = json.load(f)["class_names"]
    results = {}
    
    # add the cost as suffix to the results
    for label, path in tqdm(imagenet_data.items(), postfix=f"cost: {gpt.show_cost().cost:.5f}"):
        image_path = image_root / path
        
        recog_res = gpt.ask(
            image=image_path,
            text=RECOG_PROMPT,
            system_message=RECOGNITION_SYSTEM_PROMPT,
            response_format=ObjectRecognitionResponse,
        )
        
        object_name = recog_res.name
        
        interaction_res = gpt.ask(
            text=INTERACTION_PROMPT.format(object_name=object_name),
            system_message=INTERACTION_SYSTEM_PROMPT,
            response_format=InteractionResponse,
        )
        
        results[label] = {
            "image_path": image_path.as_posix(),
            "class_name": image_annotation_data[int(label)],
            "object_name": object_name,
            "interaction": interaction_res.to_list(),
        }
    
        with open("imagenet_1000_results.json", "w") as f:
            json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
