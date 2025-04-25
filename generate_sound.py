import numpy as np
from audio_model.ssv2a.model.pipeline import Pipeline
from audio_model.ssv2a.model.aldm import build_audioldm, emb_to_audio
from audio_model.ssv2a.data.utils import save_wave
from features.ssv2a import SSV2ANetworkConfig
from autoencoder.model import get_model
import torch
from pathlib import Path
import argparse


def decompress_feature(args, feature):
    ae = get_model(args.feature_size)
    ckpt_path = f"autoencoder/ckpt/{args.dataset}/best_ckpt.pth"
    ae.load_state_dict(torch.load(ckpt_path))
    ae = ae.eval().cuda()
    
    with torch.no_grad():
        feature = ae.decode(feature)
    
    return feature
    
def aggregate_compressed_align_feature(args):
    selected_feature_path = f"output/{args.dataset}_3/{args.mode}/ours_None/renders_npy/{args.name}.npy"
    corresponding_seg_map_path = f"customized_dataset/data/{args.dataset}/language_features_{args.mode}/{args.name}_s.npy"
    
    # this should only happen in the colmap-style dataset and the selected view is a test view.
    if not Path(corresponding_seg_map_path).exists():
        corresponding_seg_map_path = corresponding_seg_map_path.replace("test", "train")
    
    seg_map = np.load(corresponding_seg_map_path)[-1, :, :]
    selected_feature = np.load(selected_feature_path)
    features = []
    for i in range(seg_map.max()):
        mask = seg_map == i
        selected_features = selected_feature[mask]
        
        features.append(selected_features.mean(axis=0))
        
        print(selected_features.shape)

    feature = torch.from_numpy(np.stack(features, axis=0)).float().cuda()
    return feature

def get_feature(args):
    if args.type == "uncompressed" or args.type == "compressed":
        type = "language_features" if args.type == "uncompressed" else "language_features_dim3"
        name = f"{args.name}_f"
        selected_feature_path = Path(f"customized_dataset/data/{args.dataset}/{type}_{args.mode}/{name}.npy")
        feature = torch.from_numpy(np.load(selected_feature_path)).float().cuda()
    else:
        feature = aggregate_compressed_align_feature(args)
    
    if args.type == "compressed" or args.type == "compressed_aligned":
        feature = decompress_feature(args, feature)
    
    return feature

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="living_room")
    parser.add_argument("--type", type=str, choices=["uncompressed", "compressed", "compressed_aligned"], default="uncompressed")
    parser.add_argument("--mode", type=str, choices=["train", "test"], default="train")
    parser.add_argument("--name", type=str, default="DSC_1536")
    parser.add_argument("--feature_size", type=int, default=512)
    
    return parser.parse_args()

def main():
    args = parse_args()
    meta_config = SSV2ANetworkConfig()
    model = Pipeline(meta_config.config_path, meta_config.ckpt_path, device="cuda").eval()

    feature = get_feature(args)
    
    audioldm_v = model.config['audioldm_version']
    audio_ldm = build_audioldm(model_name=audioldm_v, device="cuda")

    local_waves = emb_to_audio(audio_ldm, feature, batchsize=64, duration=10)

    save_dir = Path("output_test_audio") / args.dataset / args.type / args.mode / args.name
    save_dir.mkdir(parents=True, exist_ok=True)

    save_wave(local_waves, save_dir, [f"{i:03d}" for i in range(feature.shape[0])])


if __name__ == "__main__":
    main()