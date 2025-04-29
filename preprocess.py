import os
import random
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import cv2

import torch

from features.base import BaseFeatureNetwork
from features.open_clip import OpenCLIPNetwork, OpenCLIPNetworkConfig
from features.clip import OpenAICLIPNetwork, OpenAICLIPNetworkConfig
from features.ssv2a import SSV2ANetwork, SSV2ANetworkConfig
from torchvision.transforms import ToTensor, Compose, Resize
from torchvision.utils import save_image


def save_numpy(save_path, feature, seg_map):
    if isinstance(save_path, Path):
        save_path = save_path.as_posix()
    save_path_s = save_path + '_s.npy'
    save_path_f = save_path + '_f.npy'
    np.save(save_path_s, seg_map)
    np.save(save_path_f, feature)

def get_features(seg_images, model):
    clip_embed = model.encode_image(seg_images.to("cuda"))
    clip_embed /= clip_embed.norm(dim=-1, keepdim=True)
    return clip_embed.detach().cpu().float().numpy()

def get_seg_img(mask, image):
    image = image.copy()
    # here I remove the masking operation.
    # image[mask['segmentation']==0] = np.array([0, 0,  0], dtype=np.uint8)
    x,y,w,h = np.int32(mask['bbox'])
    seg_img = image[y:y+h, x:x+w, ...]
    return seg_img

def pad_img(img):
    h, w, _ = img.shape
    l = max(w,h)
    pad = np.zeros((l,l,3), dtype=np.uint8)
    if h > w:
        pad[:,(h-w)//2:(h-w)//2 + w, :] = img
    else:
        pad[(w-h)//2:(w-h)//2 + h, :, :] = img
    return pad

def filter(keep: torch.Tensor, masks_result) -> None:
    keep = keep.int().cpu().numpy()
    result_keep = []
    for i, m in enumerate(masks_result):
        if i in keep: result_keep.append(m)
    return result_keep

def mask_nms(masks, scores, iou_thr=0.7, score_thr=0.1, inner_thr=0.2, **kwargs):
    """
    Perform mask non-maximum suppression (NMS) on a set of masks based on their scores.
    
    Args:
        masks (torch.Tensor): has shape (num_masks, H, W)
        scores (torch.Tensor): The scores of the masks, has shape (num_masks,)
        iou_thr (float, optional): The threshold for IoU.
        score_thr (float, optional): The threshold for the mask scores.
        inner_thr (float, optional): The threshold for the overlap rate.
        **kwargs: Additional keyword arguments.
    Returns:
        selected_idx (torch.Tensor): A tensor representing the selected indices of the masks after NMS.
    """

    scores, idx = scores.sort(0, descending=True)
    num_masks = idx.shape[0]
    
    masks_ord = masks[idx.view(-1), :]
    masks_area = torch.sum(masks_ord, dim=(1, 2), dtype=torch.float)

    iou_matrix = torch.zeros((num_masks,) * 2, dtype=torch.float, device=masks.device)
    inner_iou_matrix = torch.zeros((num_masks,) * 2, dtype=torch.float, device=masks.device)
    for i in range(num_masks):
        for j in range(i, num_masks):
            intersection = torch.sum(torch.logical_and(masks_ord[i], masks_ord[j]), dtype=torch.float)
            union = torch.sum(torch.logical_or(masks_ord[i], masks_ord[j]), dtype=torch.float)
            iou = intersection / union
            iou_matrix[i, j] = iou
            # select mask pairs that may have a severe internal relationship
            if intersection / masks_area[i] < 0.5 and intersection / masks_area[j] >= 0.85:
                inner_iou = 1 - (intersection / masks_area[j]) * (intersection / masks_area[i])
                inner_iou_matrix[i, j] = inner_iou
            if intersection / masks_area[i] >= 0.85 and intersection / masks_area[j] < 0.5:
                inner_iou = 1 - (intersection / masks_area[j]) * (intersection / masks_area[i])
                inner_iou_matrix[j, i] = inner_iou

    iou_matrix.triu_(diagonal=1)
    iou_max, _ = iou_matrix.max(dim=0)
    inner_iou_matrix_u = torch.triu(inner_iou_matrix, diagonal=1)
    inner_iou_max_u, _ = inner_iou_matrix_u.max(dim=0)
    inner_iou_matrix_l = torch.tril(inner_iou_matrix, diagonal=1)
    inner_iou_max_l, _ = inner_iou_matrix_l.max(dim=0)
    
    keep = iou_max <= iou_thr
    keep_conf = scores > score_thr
    keep_inner_u = inner_iou_max_u <= 1 - inner_thr
    keep_inner_l = inner_iou_max_l <= 1 - inner_thr
    
    # If there are no masks with scores above threshold, the top 3 masks are selected
    if keep_conf.sum() == 0:
        index = scores.topk(3).indices
        keep_conf[index, 0] = True
    if keep_inner_u.sum() == 0:
        index = scores.topk(3).indices
        keep_inner_u[index, 0] = True
    if keep_inner_l.sum() == 0:
        index = scores.topk(3).indices
        keep_inner_l[index, 0] = True
    keep *= keep_conf
    keep *= keep_inner_u
    keep *= keep_inner_l

    selected_idx = idx[keep]
    return selected_idx

def masks_update(masks_lvl, **kwargs):
    # remove redundant masks based on the scores and overlap rate between masks
    
    seg_pred =  torch.from_numpy(np.stack([m['segmentation'] for m in masks_lvl], axis=0))
    iou_pred = torch.from_numpy(np.stack([m['predicted_iou'] for m in masks_lvl], axis=0))
    stability = torch.from_numpy(np.stack([m['stability_score'] for m in masks_lvl], axis=0))

    scores = stability * iou_pred
    keep_mask_nms = mask_nms(seg_pred, scores, **kwargs)
    masks_lvl = filter(keep_mask_nms, masks_lvl)
    return masks_lvl

def mask2segmap(masks, image):
    seg_img_list = []
    seg_map = -np.ones(image.shape[:2], dtype=np.int32)
    for i in range(len(masks)):
        mask = masks[i]
        seg_img = get_seg_img(mask, image)
        pad_seg_img = Resize(224)(ToTensor()(pad_img(seg_img)))
        # pad_seg_img = ToTensor()(cv2.resize(pad_img(seg_img), (224,224)))
        seg_img_list.append(pad_seg_img)

        seg_map[masks[i]['segmentation']] = i
    seg_imgs = torch.stack(seg_img_list, axis=0).to("cuda")

    return seg_imgs, seg_map

def mask_processor(image, mask_generator, save_folder=None):
    masks = mask_generator.generate(image)
    masks = masks_update(masks, iou_thr=0.8, score_thr=0.7, inner_thr=0.5)
    
    seg_images, seg_map = mask2segmap(masks, image)
    
    if save_folder is not None:
        save_folder.mkdir(parents=True, exist_ok=True)
        for i, seg_img in enumerate(seg_images):
            save_image(seg_img, save_folder / f"seg_img_{i:03d}.png")
    
    seg_map = np.tile(seg_map, (4, 1, 1))
    
    return seg_images, seg_map


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


MODEL_DICT = {
    "clip": OpenAICLIPNetwork,
    "open_clip": OpenCLIPNetwork,
    "ssv2a": SSV2ANetwork,
}

CONFIG_DICT = {
    "clip": OpenAICLIPNetworkConfig,
    "open_clip": OpenCLIPNetworkConfig,
    "ssv2a": SSV2ANetworkConfig,
}

def get_model(model_name: str) -> BaseFeatureNetwork:
    return MODEL_DICT[model_name](CONFIG_DICT[model_name]())

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--resolution', type=int, default=-1)
    parser.add_argument('--sam_ckpt_path', type=str, default="ckpts/sam_vit_h_4b8939.pth")
    parser.add_argument('--model', type=str, choices=['clip', 'ssv2a'], default='ssv2a')
    parser.add_argument('--seed', type=int, default=1102)
    parser.add_argument('--device', type=str, default="cuda")
    
    return parser.parse_args()

if __name__ == '__main__':
    args = arg_parse()
    seed_everything(args.seed)

    torch.set_default_dtype(torch.float32)

    dataset_path = args.dataset_path
    sam_ckpt_path = args.sam_ckpt_path

    model = get_model(args.model).to(args.device).eval()

    sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt_path).to(args.device)
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.7,
        box_nms_thresh=0.7,
        stability_score_thresh=0.85,
        min_mask_region_area=100,
    )


    WARNED = False

    for mode in ['train', 'test']:
        print(f"Processing {mode} set")

        img_folder = Path(dataset_path) / mode

        if not img_folder.exists():
            print(f"Warning: {img_folder} does not exist")
            continue

        data_list = sorted(img_folder.iterdir())

        images = []
        for data_path in tqdm(data_list):
            image = cv2.cvtColor(cv2.imread(data_path), cv2.COLOR_BGR2RGB)

            h, w = image.shape[:2]
            if h >= 1080:
                scale = float(h / 1080)
                resolution = (int( w  / scale), int(h / scale))
                image = cv2.resize(image, resolution)

            save_folder = Path(dataset_path) / f'language_features_{mode}'
            save_folder.mkdir(parents=True, exist_ok=True)
            seg_images, seg_map = mask_processor(image, mask_generator, save_folder / "intermediate_results" / data_path.stem)

            image_features = get_features(seg_images, model)

            save_numpy(save_folder / data_path.stem, image_features, seg_map)
