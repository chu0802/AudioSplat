from pathlib import Path
import torch

def main():

    dir = Path("output/")

    feature_dict = {}

    for method_path in sorted(dir.iterdir()):
        for object_path in sorted(method_path.iterdir()):
            if object_path.stem not in feature_dict:
                feature_dict[object_path.stem] = {}

            for prompt_path in sorted(object_path.iterdir()):
                feature_dict[object_path.stem][prompt_path.stem] = torch.load(prompt_path, weights_only=True).squeeze()

    features = torch.stack([
        torch.stack([
            prompt_feature for prompt_feature in object_feature.values()
        ])
        for object_feature in feature_dict.values()
    ])

    base_features = features[:, 0, ...].reshape(-1, 64*1024)
    prompt_features = features[:, 1:, ...].reshape(-1, *features.shape[2:]).reshape(-1, 64*1024)

    # Calculate distance between base features and prompt features

    distances = torch.stack([
        torch.norm(prompt_features - base_feature[None], dim=1)
        for base_feature in base_features
    ])

    obj_distances = distances.reshape(7, 7, 20).mean(dim=2)

    breakpoint()

if __name__ == "__main__":
    main()