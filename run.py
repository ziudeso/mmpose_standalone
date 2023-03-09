# POSE_MODEL=vipnas_res50_coco_256x192 && \
# python run.py configs_detection/faster_rcnn/faster_rcnn_r50_fpn_fp16_1x_coco.py \
# https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
# configs_pose/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/$POSE_MODEL.py \
# https://download.openmmlab.com/mmpose/top_down/vipnas/vipnas_res50_coco_256x192-cc43b466_20210624.pth \
# --img-root /Volumes/CrucialX8/aiplay/eyegol_errors/difficult --out-img-root vis_results_$POSE_MODEL --device=cpu
# Copyright (c) OpenMMLab. All rights reserved.
import os
import warnings
from argparse import ArgumentParser
import cv2
from time import perf_counter as pc

from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         process_mmdet_results, vis_pose_result)
from mmpose.datasets import DatasetInfo
import mmdet
try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False


def main():
    """Visualize the demo images.

    Using mmdet to detect the human.
    """
    parser = ArgumentParser()
    parser.add_argument('det_config', help='Config file for detection')
    parser.add_argument('det_checkpoint', help='Checkpoint file for detection')
    parser.add_argument('pose_config', help='Config file for pose')
    parser.add_argument('pose_checkpoint', help='Checkpoint file for pose')
    parser.add_argument('--img-root', type=str, default='', help='Image root')
    parser.add_argument('--img', type=str, default='', help='Image file')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show img')
    parser.add_argument(
        '--out-img-root',
        type=str,
        default='',
        help='root of the output img file. '
        'Default not saving the visualization images.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--det-cat-id',
        type=int,
        default=1,
        help='Category id for bounding box detection model')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.3,
        help='Bounding box score threshold')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')
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

    assert has_mmdet, 'Please install mmdet to run the demo.'

    args = parser.parse_args()

    assert args.show or (args.out_img_root != '')
    # assert args.img != ''
    assert args.det_config is not None
    assert args.det_checkpoint is not None

    det_model = init_detector(
        args.det_config, args.det_checkpoint, device=args.device.lower())
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
    else:
        dataset_info = DatasetInfo(dataset_info)

    ############################################################################
    times_det = []
    times_pose = []
    
    for file in os.listdir(args.img_root):
      print(os.path.basename(file), os.path.basename(file)[0] == ".")
      if (file.endswith(".jpg") or file.endswith(".JPG")) and os.path.basename(file)[0] != '.':

        image_name = os.path.join(args.img_root, file)
        # image = cv2.
        print("[Analyzing]:", image_name)
        t0 = pc()
        # test a single image, the resulting box is (x1, y1, x2, y2)
        mmdet_results = inference_detector(det_model, image_name)

        # keep the person class bounding boxes.
        detections = process_mmdet_results(mmdet_results, args.det_cat_id)
        print(detections)
        # [{'bbox': array([233.39185   ,  17.408554  , 269.04776   , 106.4376    ,
        #   0.98680824], dtype=float32)}, {'bbox': array([  2.495374, 155.59207 ,  72.270065, 401.08768 ,   0.986602],
        # dtype=float32)}, {'bbox': array([1.0102945e+03, 2.8725622e+00, 1.0426963e+03, 8.2350174e+01,
        # 9.8401111e-01], dtype=float32)}, {'bbox': array([1.3481908e+03, 4.6174645e+00, 1.3877356e+03, 1.0545623e+02,
        # 9.8232031e-01], dtype=float32)}, {'bbox': array([517.8409   ,  22.358997 , 555.0935   , 115.57348  ,   0.9802571],
        # dtype=float32)}, {'bbox': array([707.0671   , 172.17702  , 770.7114   , 534.8565   ,   0.9699486],
        # dtype=float32)}, {'bbox': array([8.0073458e-01, 1.5897192e+02, 4.0864452e+01, 2.1046815e+02,
        # 1.2831143e-01], dtype=float32)}]

        det_time = pc()-t0
        print("time for detection", det_time)
        times_det.append(det_time)
        t0 = pc()

        pose_results, returned_outputs = inference_top_down_pose_model(
            pose_model,
            image_name,
            detections,
            bbox_thr=args.bbox_thr,
            format='xyxy',
            dataset=dataset,
            dataset_info=dataset_info,
            return_heatmap=False,
            outputs=None)

        if args.out_img_root == '':
            out_file = None
        else:
            os.makedirs(args.out_img_root, exist_ok=True)
            out_file = os.path.join(args.out_img_root, f"vis_{file}")

        # show the results
        vis_pose_result(
            pose_model,
            image_name,
            pose_results,
            dataset=dataset,
            dataset_info=dataset_info,
            kpt_score_thr=args.kpt_thr,
            radius=args.radius,
            thickness=args.thickness,
            show=args.show,
            out_file=out_file)

        poset_time = pc()-t0
        print("time for pose", poset_time)
        times_pose.append(poset_time)

    print("MEAN DET", sum(times_det)/len(times_det))
    print("MEAN POSE", sum(times_pose)/len(times_pose))
if __name__ == '__main__':
    main()
