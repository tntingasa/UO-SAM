import numpy as np
import torch
from torch.nn import functional as F

import os
import cv2
from PIL import Image
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import warnings
import pycocotools.mask as maskUtils

warnings.filterwarnings('ignore')

# from show import *
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator
from per_segment_anything import sam_model_registry, SamPredictor


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type=str, default='./data')
    parser.add_argument('--mask', type=str, default='./data')
    parser.add_argument('--outdir', type=str, default='persam')
    parser.add_argument('--ckpt', type=str, default='sam_vit_h_4b8939.pth')
    parser.add_argument('--thresholds', type=int, default=0.4)
    parser.add_argument('--sam_type', type=str, default='vit_h')

    args = parser.parse_args()
    return args


def main():
    args = get_arguments()
    print("Args:", args)

    images_path = args.data
    masks_path = args.mask
    output_path = './outputs/' + args.outdir

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    list1 = os.listdir(images_path)
    list1.sort(key=lambda x: int(x[1:]))
    for image_folder in list1:
        list2 = os.listdir(os.path.join(images_path, image_folder))
        list2.sort(key=lambda x: int(x[15:-5]))
        if not os.path.exists(os.path.join(images_path, image_folder)):
            os.makedirs(os.path.join(output_path, image_folder))
        for filename in list2:
            #save_path
            base = os.path.basename(filename)
            base = os.path.splitext(base)[0]
            save_base = os.path.join(output_path, image_folder, base+'.png')
            if os.path.exists(save_base):
                continue
            # read picture
            img_path = os.path.join(images_path, image_folder, filename)
            image = cv2.imread(img_path)
            if image is None:
                print(f"Could not load '{filename}' as an image, skipping...")
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            #read mask
            mask_path = os.path.join(masks_path, image_folder, filename[:-5] + '.png')
            mask = cv2.imread(mask_path)
            if cv2.countNonZero(mask) == 0:
                continue
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

            if image.shape != mask.shape:
                mask = mask.transpose((1, 0, 2))

            mask_lro = LRO(args,image,mask)
            final_mask = GRO(args,image,mask_lro)
            Image.fromarray(final_mask).save(save_base)

def LRO(args,image,mask):

    image_temp = image
    mask_temp = mask
    class_ids = np.unique(mask_temp[:, :, 0])
    if len(class_ids) == 1:
        return mask_temp
    else:
        mask_colors = np.zeros((mask_temp.shape[0], mask_temp.shape[1], 3)).astype(np.uint8)
        pos_mask_dictionary = {}
        for id in class_ids[1:]:
            mask_id = np.select([mask_temp[:, :, 0] == id], [mask_temp[:, :, 0] + 200], default=0)
            pos_mask = sum(sum(mask_id != 0)) / (mask_id.shape[0] * mask_id.shape[1])
            if pos_mask < 0.1:
                continue
            else:
                pos_mask_dictionary[id] = pos_mask
        sorted_items = sorted(pos_mask_dictionary.items(), key=lambda x: x[1], reverse=True)
        for items in sorted_items:
            id, pos = items
            ref_mask = np.zeros((mask_temp.shape[0], mask_temp.shape[1], 3)).astype(np.uint8)
            mask_id = np.select([mask_temp[:, :, 0] == id], [mask_temp[:, :, 0] + 200], default=0)
            ref_mask[:, :, 0] = torch.from_numpy(mask_id)
            print("======> Load SAM")
            if args.sam_type == 'vit_h':
                sam_type, sam_ckpt = 'vit_h', 'weights/sam_vit_h_4b8939.pth'
                sam = sam_model_registry[sam_type](checkpoint=sam_ckpt).cuda()
            elif args.sam_type == 'vit_t':
                sam_type, sam_ckpt = 'vit_t', 'weights/mobile_sam.pt'
                device = "cuda" if torch.cuda.is_available() else "cpu"
                sam = sam_model_registry[sam_type](checkpoint=sam_ckpt).to(device=device)
                sam.eval()

            predictor = SamPredictor(sam)

            print("======> Obtain Location Prior")

            ref_mask = predictor.set_image(image_temp, mask_temp)
            ref_feat = predictor.features.squeeze().permute(1, 2, 0)

            ref_mask = F.interpolate(ref_mask, size=ref_feat.shape[0: 2], mode="bilinear")
            ref_mask = ref_mask.squeeze()[0]

            target_feat = ref_feat[ref_mask > 0]
            target_embedding = target_feat.mean(0).unsqueeze(0)
            target_feat = target_embedding / target_embedding.norm(dim=-1, keepdim=True)

            predictor.set_image(image_temp)
            test_feat = predictor.features.squeeze()

            # Cosine similarity
            C, h, w = test_feat.shape
            test_feat = test_feat / test_feat.norm(dim=0, keepdim=True)
            test_feat = test_feat.reshape(C, h * w)
            sim = target_feat @ test_feat

            sim = sim.reshape(1, 1, h, w)
            sim = F.interpolate(sim, scale_factor=4, mode="bilinear")
            sim = predictor.model.postprocess_masks(
                sim,
                input_size=predictor.input_size,
                original_size=predictor.original_size).squeeze()

            # Positive-negative location prior
            topk_xy_i, topk_label_i, last_xy_i, last_label_i = point_selection(sim, topk=1)
            topk_xy = np.concatenate([topk_xy_i, last_xy_i], axis=0)
            topk_label = np.concatenate([topk_label_i, last_label_i], axis=0)

            # find largestConnectComponent #
            labeled_img, mcr = largestConnectComponent(mask)
            props = measure.regionprops(labeled_img)
            numPix = []
            for ia in range(len(props)):
                numPix += [props[ia].area]
            maxnum = max(numPix)
            index = numPix.index(maxnum)
            minr, minc, maxr, maxc = props[index].bbox
            input_box_1 = np.array([minc, minr, maxc, maxr])

            # First-step prediction
            masks, scores, logits, _ = predictor.predict(
                point_coords=topk_xy,
                point_labels=topk_label,
                box=input_box_1,
                multimask_output=True,
            )
            best_idx = np.argmax(scores)

            y, x = np.nonzero(masks[best_idx])
            if len(y) == 0 or len(x) == 0:
                continue
            else:
                x_min = x.min()
                x_max = x.max()
                y_min = y.min()
                y_max = y.max()
                input_box = np.array([x_min, y_min, x_max, y_max])

                # Second-step prediction
                masks, scores, logits, _ = predictor.predict(
                    point_coords=topk_xy,
                    point_labels=topk_label,
                    box=input_box[None, :],
                    mask_input=logits[best_idx: best_idx + 1, :, :],
                    multimask_output=False)

                final_mask = masks[0]
                pre_masks = np.zeros((final_mask.shape[0], final_mask.shape[1], 3), dtype=np.uint8)
                pre_masks[final_mask, :] = np.array([[200, 0, 0]], dtype=np.uint8)

            if mask_id.shape != pre_masks[..., 0].shape:
                print("Inconsistent image and label sizes!")
                mask_id = mask_id.transpose((1, 0))
                final_mask = final_mask.transpose((1, 0))
            sim_iou = mask_iou(mask_id, pre_masks[..., 0])

            if sim_iou < args.thresholds:
                return mask_temp
            else:
                mask_colors[final_mask, :] = np.array([[id, 0, 0]], dtype=np.uint8)

        if len(np.unique(mask_colors)) == 1:
            return mask_temp
        else:
            return mask_colors

def GRO(args,image,mask_lro):

    image_temp = image
    mask_temp = mask_lro
    if len(np.unique(mask_temp[:, :, 0])) == 1:
        return  mask_temp
    if args.sam_type == 'vit_h':
        sam_type, sam_ckpt = 'vit_h', 'weights/sam_vit_h_4b8939.pth'
        sam = sam_model_registry[sam_type](checkpoint=sam_ckpt).cuda()
    elif args.sam_type == 'vit_t':
        sam_type, sam_ckpt = 'vit_t', 'weights/mobile_sam.pt'
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = sam_model_registry[sam_type](checkpoint=sam_ckpt).to(device=device)
        sam.eval()

    mask_branch_model = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=64,  # 每边的采样点数量，用于从输入图像中生成分割掩模。
        # Foggy driving (zero-shot evaluate) is more challenging than other dataset, so we use a larger points_per_side
        pred_iou_thresh=0.86,  # 预测的 IoU 阈值，用于筛选生成的分割掩模。
        stability_score_thresh=0.92,  # 稳定性得分阈值，用于筛选生成的分割掩模。
        crop_n_layers=1,  # 裁剪操作的层数，用于生成分割掩模时进行局部感受野的操作。
        crop_n_points_downscale_factor=2,  # 裁剪操作中下采样因子，用于减小采样点数量
        min_mask_region_area=100,  # Requires open-cv to run post-processing 最小的掩模区域面积，用于在后处理中排除过小的掩模区域。
        output_mode='coco_rle',  # 输出模式，可能是 'coco_rle' 或其他格式，用于指定分割掩模的输出格式。
    )

    anns = {'annotations': mask_branch_model.generate(image_temp)}
    # print(ref_mask_temp.shape)
    class_ids = torch.from_numpy(mask_temp[:, :, 0])
    # print(class_ids.shape)
    semantc_mask = class_ids.clone()  # 创建副本
    anns['annotations'] = sorted(anns['annotations'], key=lambda x: x['area'], reverse=True)  # 从大到小的顺序
    for ann in anns['annotations']:
        valid_mask = torch.tensor(maskUtils.decode(ann['segmentation'])).bool()
        # print(valid_mask.shape)
        # get the class ids of the valid pixels
        if valid_mask.shape != class_ids.shape:
            valid_mask = valid_mask.transpose(1, 0)
        propose_classes_ids = class_ids[valid_mask]
        # print(propose_classes_ids.shape)
        num_class_proposals = len(torch.unique(propose_classes_ids))
        if num_class_proposals == 1:
            semantc_mask[valid_mask] = propose_classes_ids[0]
            continue
        top_1_propose_class_ids = torch.bincount(propose_classes_ids.flatten()).topk(1).indices
        semantc_mask[valid_mask] = top_1_propose_class_ids

        del valid_mask
        del propose_classes_ids
        del num_class_proposals
        del top_1_propose_class_ids
    semantc_mask = semantc_mask.unsqueeze(0).cpu().numpy()
    mask_colors = np.zeros((mask_temp.shape[0], mask_temp.shape[1], 3), dtype=np.uint8)
    mask_colors[:, :, 0] = semantc_mask
    if len(np.unique(mask_colors)) == 1:
        return mask_temp
    return mask_colors


def point_selection(mask_sim, topk=1):
    # Top-1 point selection
    w, h = mask_sim.shape
    topk_xy = mask_sim.flatten(0).topk(topk)[1]
    topk_x = (topk_xy // h).unsqueeze(0)
    topk_y = (topk_xy - topk_x * h)
    topk_xy = torch.cat((topk_y, topk_x), dim=0).permute(1, 0)
    topk_label = np.array([1] * topk)
    topk_xy = topk_xy.cpu().numpy()

    # Top-last point selection
    last_xy = mask_sim.flatten(0).topk(topk, largest=False)[1]
    last_x = (last_xy // h).unsqueeze(0)
    last_y = (last_xy - last_x * h)
    last_xy = torch.cat((last_y, last_x), dim=0).permute(1, 0)
    last_label = np.array([0] * topk)
    last_xy = last_xy.cpu().numpy()

    return topk_xy, topk_label, last_xy, last_label


def mask_iou(mask, pre_masks):

    intersection = np.logical_and(np.any(mask != 0, axis=-1), np.any(pre_masks != 0, axis=-1))
    union = np.logical_or(np.any(mask != 0, axis=-1), np.any(pre_masks != 0, axis=-1))
    iou = np.sum(intersection) / np.sum(union)

    return iou


if __name__ == "__main__":
    main()
