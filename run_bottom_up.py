#  1463  conda create --name openmmlab python=3.8 -y
#  1464  conda activate openmmlab
#  1465  conda install pytorch torchvision -c pytorch
#  1466  conda install pytorch torchvision cpuonly -c pytorch
#  1467  pip install -U openmim
#  1468  mim install mmcv-full
#  1470  pip install setuptools==58.2.0
#  1471  pip install mmpose

# python run_bottom_up.py \
#     configs_pose/body/2d_kpt_sview_rgb_img/associative_embedding/coco/hrnet_w32_coco_512x512.py \
#     https://download.openmmlab.com/mmpose/bottom_up/hrnet_w32_coco_512x512-bcb8c247_20200816.pth \
#     --img-root /Volumes/CrucialX8/aiplay/eyegol_errors/difficult \
#     --out-img-root vis_results_$POSE_MODEL --device=cpu

# POSE_MODEL=vipnas_res50_coco_256x192 && \
# configs_pose/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/$POSE_MODEL.py \
# https://download.openmmlab.com/mmpose/top_down/vipnas/$POSE_MODEL-cc43b466_20210624.pth \
# --img-root /Volumes/CrucialX8/aiplay/eyegol_errors/difficult --out-img-root vis_results_$POSE_MODEL --device=cpu

# Copyright (c) OpenMMLab. All rights reserved.
import os
import warnings
from argparse import ArgumentParser
from time import perf_counter as pc

import cv2
import mmcv

from mmpose.apis import (inference_bottom_up_pose_model, init_pose_model,
                         vis_pose_result)
from mmpose.datasets import DatasetInfo


def main():
    """Visualize the demo images."""
    parser = ArgumentParser()
    parser.add_argument('pose_config', help='Config file for pose')
    parser.add_argument('pose_checkpoint', help='Checkpoint file for pose')
    # parser.add_argument('--video-path', type=str, help='Video path')
    parser.add_argument('--img-root', type=str, default='', help='Image root')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show visualizations.')
    parser.add_argument(
        '--out-img-root',
        type=str,
        default='',
        help='root of the output img file. '
        'Default not saving the visualization images.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')
    parser.add_argument(
        '--pose-nms-thr',
        type=float,
        default=0.9,
        help='OKS threshold for pose NMS')
    parser.add_argument(
        '--radius',
        type=int,
        default=4,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')


    args = parser.parse_args()

    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.device.lower())

    dataset = pose_model.cfg.data['test']['type']
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
        assert (dataset == 'BottomUpCocoDataset')
    else:
        dataset_info = DatasetInfo(dataset_info)

    ############################################################################
    # read video
    # video = mmcv.VideoReader(args.video_path)
    # assert video.opened, f'Faild to load video file {args.video_path}'

    # if args.out_video_root == '':
    #     save_out_video = False
    # else:
    #     os.makedirs(args.out_video_root, exist_ok=True)
    #     save_out_video = True

    # if save_out_video:
    #     fps = video.fps
    #     size = (video.width, video.height)
    #     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #     videoWriter = cv2.VideoWriter(
    #         os.path.join(args.out_video_root,
    #                      f'vis_{os.path.basename(args.video_path)}'), fourcc,
    #         fps, size)

    # optional
    return_heatmap = False

    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None
    times_pose = []
    for file in os.listdir(args.img_root):
      print(os.path.basename(file), os.path.basename(file)[0] == ".")
      if (file.endswith(".jpg") or file.endswith(".JPG")) and os.path.basename(file)[0] != '.':

        image_name = os.path.join(args.img_root, file)
        # image = cv2.
        print("[Analyzing]:", image_name)
        img = cv2.imread(image_name)

        print('Running inference...')

        t0 = pc()
        pose_results, _ = inference_bottom_up_pose_model(
            pose_model,
            img,
            dataset=dataset,
            dataset_info=dataset_info,
            pose_nms_thr=args.pose_nms_thr,
            return_heatmap=return_heatmap,
            outputs=output_layer_names)
        inf_time = pc()-t0
        print(inf_time)
        if args.out_img_root == '':
            out_file = None
        else:
            os.makedirs(args.out_img_root, exist_ok=True)
            out_file = os.path.join(args.out_img_root, f"vis_{file}")

        # show the results
        vis_pose_result(
            pose_model,
            img,
            pose_results,
            radius=args.radius,
            thickness=args.thickness,
            dataset=dataset,
            dataset_info=dataset_info,
            kpt_score_thr=args.kpt_thr,
            out_file=out_file,
            show=False)
        
        times_pose.append(inf_time)
        # if args.show:
        #     cv2.imshow('Image', vis_frame)

        # if save_out_video:
        #     videoWriter.write(vis_frame)

        # if args.show and cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    print("MEAN POSE", sum(times_pose)/len(times_pose))

if __name__ == '__main__':
    main()