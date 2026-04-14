#!/usr/bin/env python
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

import os
# Use GPU 1 only
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import json
import numpy as np
from PIL import Image
from gaussian_renderer import render, GaussianModel
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from scene.cameras import Camera
import sys
import cv2


def load_poses_from_json(pose_file):
    """
    Load camera poses from JSON file.
    Expected format:
    {
        "poses": [
            {
                "name": "00001",
                "R": [[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]],
                "T": [t1, t2, t3],
                "FoVx": 1.0,
                "FoVy": 0.8,
                "width": 1920,
                "height": 1080
            },
            ...
        ]
    }
    """
    with open(pose_file, 'r') as f:
        data = json.load(f)
    return data['poses']


def build_camera_from_pose(pose, data_device="cuda"):
    """
    Build a Camera object from a pose dictionary.
    """
    R = np.array(pose['R'])
    T = np.array(pose['T'])
    FoVx = pose['FoVx']
    FoVy = pose['FoVy']
    width = pose.get('width', 1920)
    height = pose.get('height', 1080)
    name = pose.get('name', 'custom')

    # Create a dummy white image since we don't have ground truth
    dummy_image = Image.fromarray(np.ones((height, width, 3), dtype=np.uint8) * 255)

    # Create camera with default parameters
    camera = Camera(
        resolution=(width, height),
        colmap_id=0,
        R=R,
        T=T,
        FoVx=FoVx,
        FoVy=FoVy,
        depth_params=None,
        image=dummy_image,
        invdepthmap=None,
        image_name=name,
        uid=0,
        data_device=data_device,
        train_test_exp=False,
        is_test_dataset=False,
        is_test_view=False
    )

    return camera


def render_set(model_path, output_dir, views, gaussians, pipeline, background, train_test_exp):
    """Render a set of views."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "rgb"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "depth"), exist_ok=True)

    for idx, view in enumerate(views):
        print(f"Rendering {idx+1}/{len(views)}: {view.image_name}")

        # For custom poses with train_test_exp, use identity exposure if not found
        if train_test_exp:
            # Check if pretrained_exposures exists and contains this camera
            if not hasattr(gaussians, 'pretrained_exposures') or gaussians.pretrained_exposures is None:
                # Create pretrained_exposures dict if needed
                gaussians.pretrained_exposures = {}

            if view.image_name not in gaussians.pretrained_exposures:
                # Use identity exposure for custom cameras
                with torch.no_grad():
                    exposure = torch.eye(4, 4, dtype=torch.float32, device="cuda")[:3, :]
                    gaussians.pretrained_exposures[view.image_name] = exposure
                    print(f"  Using default exposure for {view.image_name}")

        render_result = render(view, gaussians, pipeline, background,
                          use_trained_exp=train_test_exp)
        rendering = render_result["render"]
        depth = render_result["depth"]

        # Save RGB image
        rgb_path = os.path.join(output_dir, "rgb", f"{view.image_name}_rgb.png")
        torchvision.utils.save_image(rendering, rgb_path)
        print(f"  Saved: {rgb_path}")

        # Create mask from RGB: black regions = 0, non-black regions = 1
        rgb_np = (rendering.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        mask = (rgb_np > 10).any(axis=2).astype(np.float32)
        mask_tensor = torch.from_numpy(mask).to(depth.device)

        # Save depth map (normalize to 0-1 for visualization, then apply mask)
        depth_path = os.path.join(output_dir, "depth", f"{view.image_name}_depth.png")
        depth_min = depth.min()
        depth_max = depth.max()
        depth_normalized = (depth - depth_min) / (depth_max - depth_min + 1e-8)
        # Apply mask: set black regions in RGB to 0 in depth
        depth_masked = depth_normalized * mask_tensor
        torchvision.utils.save_image(depth_masked, depth_path)
        print(f"  Saved: {depth_path}")

        # Also save raw depth as .npy for precise values (apply mask to keep black regions as 0)
        npy_depth_path = os.path.join(output_dir, "depth", f"{view.image_name}_depth.npy")
        depth_raw_masked = depth * mask_tensor
        np.save(npy_depth_path, depth_raw_masked.cpu().numpy())
        print(f"  Saved: {npy_depth_path}")


def render_sets(dataset, iteration, pipeline, pose_file, output_dir):
    """Render images from custom camera poses."""
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)

        # Load the trained model directly without loading all cameras
        if iteration == -1:
            # Find the latest iteration
            point_cloud_path = os.path.join(dataset.model_path, "point_cloud")
            iterations = [int(d.split('_')[1]) for d in os.listdir(point_cloud_path) if d.startswith("iteration_")]
            iteration = max(iterations)

        model_path = os.path.join(dataset.model_path, "point_cloud", f"iteration_{iteration}", "point_cloud.ply")
        print(f"Loading model from: {model_path}")
        gaussians.load_ply(model_path, use_train_test_exp=dataset.train_test_exp)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # Load custom poses from JSON file
        print(f"Loading poses from: {pose_file}")
        poses = load_poses_from_json(pose_file)
        print(f"Loaded {len(poses)} poses")

        # Build cameras from poses (only the cameras we need to render)
        views = [build_camera_from_pose(pose, dataset.data_device) for pose in poses]

        # Render
        print(f"Rendering to: {output_dir}")
        render_set(dataset.model_path, output_dir, views, gaussians, pipeline, background, dataset.train_test_exp)

        print("\nDone!")


if __name__ == "__main__":
    parser = ArgumentParser(description="Render images from custom camera poses")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--pose_file", type=str, required=True,
                        help="Path to JSON file containing camera poses")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save rendered images")
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)

    print("Rendering " + args.model_path)

    # Initialize system state
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args),
                args.pose_file, args.output_dir)
