#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import numpy as np
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

def get_min_max_btwn_gt_and_rendering(gt_feature, rendering_feature):
    gt_min_value = gt_feature.reshape(gt_feature.shape[0], -1).min(dim=1).values
    rendering_min_value = rendering_feature.reshape(rendering_feature.shape[0], -1).min(dim=1).values
    
    min_value = torch.concat([gt_min_value, rendering_min_value], dim=0).min(dim=0).values
    
    gt_max_value = gt_feature.reshape(gt_feature.shape[0], -1).max(dim=1).values
    rendering_max_value = rendering_feature.reshape(rendering_feature.shape[0], -1).max(dim=1).values
    max_value = torch.concat([gt_max_value, rendering_max_value], dim=0).max(dim=0).values
    
    return min_value, max_value

def feature_to_color(feature, min_value, max_value):
    # input feature size: (feature_dim, H, W)
    normalized_feature = (feature - min_value[..., None]) / (max_value[..., None] - min_value[..., None])
    return normalized_feature.reshape(feature.shape)
    

def render_set(model_path, source_path, name, iteration, views, gaussians, pipeline, background, args):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    render_npy_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_npy")
    gts_npy_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_npy")

    makedirs(render_npy_path, exist_ok=True)
    makedirs(gts_npy_path, exist_ok=True)
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        output = render(view, gaussians, pipeline, background, args)

        if not args.include_feature:
            rendering = output["render"]
        else:
            rendering = output["language_feature_image"]
            
        if not args.include_feature:
            gt = view.original_image[0:3, :, :]
            
        else:
            gt, mask = view.get_language_feature(feature_level=args.feature_level)

        min_value, max_value = get_min_max_btwn_gt_and_rendering(gt, rendering)
        np.save(os.path.join(render_npy_path, view.image_name + ".npy"),rendering.permute(1,2,0).cpu().numpy())
        np.save(os.path.join(gts_npy_path, view.image_name + ".npy"),gt.permute(1,2,0).cpu().numpy())
        torchvision.utils.save_image(feature_to_color(rendering, min_value, max_value), os.path.join(render_path, view.image_name + ".png"))
        torchvision.utils.save_image(feature_to_color(gt, min_value, max_value), os.path.join(gts_path, view.image_name + ".png"))
               
def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, args):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, shuffle=False)
        checkpoint = os.path.join(args.model_path, 'chkpnt30000.pth')
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, args, mode='test')
        
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, dataset.source_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, args)

        if not skip_test:
             render_set(dataset.model_path, dataset.source_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, args)

if __name__ == "__main__":
    # Set up command line argument parser
    
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--include_feature", action="store_true")

    args = get_combined_args(parser)
    args.eval = True

    print("Rendering " + args.model_path)

    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args)