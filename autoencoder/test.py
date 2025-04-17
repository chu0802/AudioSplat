import os
import numpy as np
import torch
import argparse
import shutil
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import Autoencoder_dataset
from model import get_model

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--model', type=str, choices=['open_clip', 'clip', 'audio_clip'], default='clip')
    return parser.parse_args()

if __name__ == '__main__':
    args = arg_parse()
    
    dataset_name = args.dataset_name
    dataset_path = args.dataset_path
    ckpt_path = f"ckpt/{dataset_name}/best_ckpt.pth"

    for mode in ["train", "test"]:

        data_dir = f"{dataset_path}/language_features_{mode}"
        output_dir = f"{dataset_path}/language_features_dim3_{mode}"
        os.makedirs(output_dir, exist_ok=True)
        
        # copy the segmentation map
        for filename in os.listdir(data_dir):
            if filename.endswith("_s.npy"):
                source_path = os.path.join(data_dir, filename)
                target_path = os.path.join(output_dir, filename)
                shutil.copy(source_path, target_path)


        checkpoint = torch.load(ckpt_path)
        train_dataset = Autoencoder_dataset(data_dir)

        test_loader = DataLoader(
            dataset=train_dataset, 
            batch_size=256,
            shuffle=False, 
            num_workers=16, 
            drop_last=False   
        )

        model = get_model(args.model).to("cuda:0")

        model.load_state_dict(checkpoint)
        model.eval()

        for idx, feature in tqdm(enumerate(test_loader)):
            data = feature.to("cuda:0")
            with torch.no_grad():
                outputs = model.encode(data).to("cpu").numpy()  
            if idx == 0:
                features = outputs
            else:
                features = np.concatenate([features, outputs], axis=0)

        os.makedirs(output_dir, exist_ok=True)
        start = 0
        
        for k,v in train_dataset.data_dic.items():
            path = os.path.join(output_dir, k)
            np.save(path, features[start:start+v])
            start += v
