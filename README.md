[ssd + hrnet48]
python run.py \
configs_detection/ssd/ssdlite_mobilenetv2_scratch_600e_coco.py \
https://download.openmmlab.com/mmdetection/v2.0/ssd/ssdlite_mobilenetv2_scratch_600e_coco/ssdlite_mobilenetv2_scratch_600e_coco_20210629_110627-974d9307.pth \
configs_pose/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py \
https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth \
--img-root /Users/francescodelogu/Downloads/eyegol_errors/difficult \
--out-img-root vis_results_hr_net48_ssd \
--device=cpu


[ssd500 + hrnet48]
python run.py \
configs_detection/ssd/ssd512_coco.py \
https://download.openmmlab.com/mmdetection/v2.0/ssd/ssd512_coco/ssd512_coco_20210803_022849-0a47a1ca.pth \
configs_pose/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py \
https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth \
--img-root /Users/francescodelogu/Downloads/eyegol_errors/difficult \
--out-img-root vis_results_hr_net48_ssd \
--device=cpu

[centernet + hrnet48]
python run.py \
configs_detection/centernet/centernet_resnet18_dcnv2_140e_coco.py \
https://download.openmmlab.com/mmdetection/v2.0/centernet/centernet_resnet18_dcnv2_140e_coco/centernet_resnet18_dcnv2_140e_coco_20210702_155131-c8cd631f.pth \
configs_pose/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py \
https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth \
--img-root /Users/francescodelogu/Downloads/eyegol_errors/difficult \
--out-img-root vis_results_hr_net48_centernet \
--device=cpu