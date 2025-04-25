import os

import random
from argparse import ArgumentParser
import numpy as np
import torch

from autoencoder.model import get_model as get_autoencoder
from preprocess import get_model as get_lang_model
from features.api import AudioCLIPNetwork, AudioCLIPNetworkConfig
import librosa
import glob
import cv2
from pathlib import Path
from tqdm import tqdm

from audio_model.ssv2a.api import DEFAULT_CONFIG_PATH, DEFAULT_PRETRAINED_PATH


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

@torch.no_grad()
def evaluate(feat_dir, image_dir, output_path, audio_path, autoencoder, mask_thresh):
    device = torch.device("cuda")
    
    feat_paths = sorted(glob.glob(os.path.join(feat_dir, '*.npy')))
    image_paths = sorted(glob.glob(os.path.join(image_dir, '*.png')))
    
    compressed_sem_feats = [
        np.load(feat_path)
        for feat_path in feat_paths
    ]
    
    aclip_model = AudioCLIPNetwork(AudioCLIPNetworkConfig).eval()
    
    for feat, image_path in tqdm(zip(compressed_sem_feats, image_paths), total=len(compressed_sem_feats)):
        sem_feat = torch.from_numpy(feat).float().to(device)
        
        h, w, _ = sem_feat.shape
        restored_feat = autoencoder.decode(sem_feat).reshape(h, w, -1)
        
        # query image by text
        
        input_text = [["car"]]
        ((_, _, text_features), _), _ = aclip_model.model(text=input_text)
        
        text_features /= text_features.norm(dim=-1, keepdim=True)        
        
        scale_image_text = torch.clamp(aclip_model.model.logit_scale.exp(), min=1.0, max=100.0)
        
        text_image_values = scale_image_text * torch.einsum("bk,ijk->bij", text_features, restored_feat)
        
        ti_min_value, ti_max_value = text_image_values.min(), text_image_values.max()
        text_image_values = (text_image_values - ti_min_value) / (ti_max_value - ti_min_value)
        
        # calculate the 90th percentile of the text_image_values
        text_image_values = text_image_values.squeeze().cpu().numpy()
        threshold = np.percentile(text_image_values, 80)
        
        text_image_values[text_image_values < threshold] = 0
        text_image_values[text_image_values >= threshold] = 1
        
        # use heat map to visualize the values
        text_image_values = (text_image_values * 255).astype(np.uint8)
        text_image_values = cv2.applyColorMap(text_image_values, cv2.COLORMAP_JET)
        
        ti_output_dir = Path(output_path) / "text_image"
        ti_output_dir.mkdir(parents=True, exist_ok=True)
        
        cv2.imwrite((ti_output_dir / os.path.basename(image_path)).as_posix(), text_image_values)
        
        # query image by audio
        
        track, _ = librosa.load(audio_path, sr=44100, dtype=np.float32)
        audio_feat = aclip_model.audio_transforms(track.reshape(1, -1))[None]

        ((audio_features, _, _), _), _ = aclip_model.model(audio=audio_feat)
        audio_features /= audio_features.norm(dim=-1, keepdim=True)
        
        scale_audio_image = torch.clamp(aclip_model.model.logit_scale_ai.exp(), min=1.0, max=100.0)
        
        audio_image_values = scale_audio_image * torch.einsum("bk,ijk->bij", audio_features, restored_feat)

        ai_min_value, ai_max_value = audio_image_values.min(), audio_image_values.max()
        audio_image_values = (audio_image_values - ai_min_value) / (ai_max_value - ai_min_value)
        
        
        # calculate the 90th percentile of the audio_image_values
        audio_image_values = audio_image_values.squeeze().cpu().numpy()
        threshold = np.percentile(audio_image_values, 80)
        
        audio_image_values[audio_image_values < threshold] = 0
        audio_image_values[audio_image_values >= threshold] = 1
        
        audio_image_values = (audio_image_values * 255).astype(np.uint8)
        audio_image_values = cv2.applyColorMap(audio_image_values, cv2.COLORMAP_JET)
        
        ai_output_dir = Path(output_path) / "audio_image"
        ai_output_dir.mkdir(parents=True, exist_ok=True)
        
        cv2.imwrite((ai_output_dir / os.path.basename(image_path)).as_posix(), audio_image_values)

@torch.no_grad()
def get_feature_by_point(compressed_features, ae, points):
    h, w, _ = compressed_features[0].shape
    
    res = []

    for feat in tqdm(compressed_features):
        selected_feat = ae.decode(feat).reshape(h, w, -1)[points[0], points[1]]

        res.append(selected_feat)

    return res

def arg_parse():
    parser = ArgumentParser(description="prompt any label")
    parser.add_argument("--res_dir", type=Path)
    parser.add_argument("--ae_ckpt_path", type=Path)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--mode", type=str, choices=["train", "test"], default="test")
    parser.add_argument("--selected_point", type=tuple, default=(399, 399))
    parser.add_argument("--device", type=torch.device, default=torch.device("cuda"))
    parser.add_argument("--seed", type=int, default=1102)
    parser.add_argument("--model", type=str, choices=["open_clip", "clip", "ssv2a"], default="ssv2a")
    parser.add_argument("--size", type=int, choices=[512, 768, 1024], default=512)
    
    return parser.parse_args()

    

if __name__ == "__main__":
    args = arg_parse()
    seed_everything(args.seed)

    autoencoder = get_autoencoder(args.size)
    autoencoder.load_state_dict(torch.load(args.ae_ckpt_path.as_posix()))
    autoencoder = autoencoder.cuda().eval()

    feature_dir = args.res_dir / args.mode / "ours_None" / "renders_npy"
    compressed_features = torch.stack([
        torch.from_numpy(
            np.load(feat_path)
        ).float()[args.selected_point[0], args.selected_point[1], ...]
        for feat_path in sorted(feature_dir.glob("*.npy"))
    ], dim=0).to(args.device)

    with torch.no_grad():
        feats = autoencoder.decode(compressed_features)

    
    model = get_lang_model(args.model).to("cuda").eval()
    
    text_features = model.encode_text(["a photo of a car"])

    for feat in feats:
        cos_sim = torch.nn.functional.cosine_similarity(feat[None], text_features)
        print(f"cos_sim: {cos_sim}")



    # dataset_name = args.dataset_name
    # feature_dir = args.feat_dir
    # image_dir = args.image_dir
    # output_path = args.output_dir

    # autoencoder = DefaultCLIPAutoencoder()
    # autoencoder.load_state_dict(torch.load(args.ae_ckpt_path.as_posix()))
    # autoencoder = autoencoder.cuda().eval()
    
    

    # evaluate(feature_dir, image_dir, output_path, audio_path, autoencoder, mask_thresh)
    # evaluate(feat_dir, output_path, ae_ckpt_path, mask_thresh, args.encoder_dims, args.decoder_dims)
