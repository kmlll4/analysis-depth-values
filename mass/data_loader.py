from pathlib import Path
import cv2
from imageio.v2 import imread
import numpy as np
from pycocotools import mask as pymask
import torch

class DatasetWrapper:
    def __init__(self, dataset, transforms, Rotater, WLGetter , CFG):
        self.dataset = dataset
        self.transforms = transforms
        self.rotater = Rotater
        self.wl_getter = WLGetter
        self.cfg = CFG

    def __getitem__(self, idx):
        pig = self.dataset.iloc[idx]

        # Segmentation Refinementを使用したか
        w_attr = "refined_w" if self.cfg["data"]["refined"] else "new_w"
        l_attr = "refined_l" if self.cfg["data"]["refined"] else "new_l"
        bbox_attr = "refined_bbox" if self.cfg["data"]["refined"] else "bbox"
        polygon_attr = "refined_polygon" if self.cfg["data"]["refined"] else "polygon"

        # print(idx)
        img_path = Path(pig['abs_path'])
        # print(img_path) #toDo
        # print(pig.floor_depth)
        frame_img = imread(img_path)
        frame_depth = imread(img_path.parent / img_path.name.replace("rgb", "depth").replace(".jpg", ".png"))

        x0, y0, w, h = getattr(pig, bbox_attr)

        onepig_img = frame_img[y0 : y0 + h, x0 : x0 + w]
        # 確認用
        # onepig_tmp_depth =  frame_depth[y0 : y0 + h, x0 : x0 + w]
        # print(onepig_depth)

        frame_depth[frame_depth > pig.floor_depth] = pig.floor_depth
        frame_depth = cv2.resize(frame_depth, (frame_img.shape[1], frame_img.shape[0]))
        onepig_depth = frame_depth[y0 : y0 + h, x0 : x0 + w]
        # print(onepig_depth) #ToDo
        
        onepig_depth = np.where(onepig_depth != 0, pig.floor_depth - onepig_depth.astype(int), 0)
        # onepig_depth = (onepig_depth > 0) * (pig.new_floor_depth - onepig_depth.astype(int))

        polygon = getattr(pig, polygon_attr)
        if type(polygon) == list:
            frame_mask = pymask.decode(polygon)[:, :, 0]
        else:
            frame_mask = pymask.decode(polygon._rle)

        onepig_mask = frame_mask[y0 : y0 + h, x0 : x0 + w]

        if self.cfg["train"]["resize_by_res_depth"]:
            target_h, target_w = map(int, (onepig_img.shape[0] * pig.resize_ratio, onepig_img.shape[1] * pig.resize_ratio))
            onepig_img = cv2.resize(onepig_img, (target_w, target_h))
            onepig_mask = cv2.resize(onepig_mask, (target_w, target_h))
            onepig_depth = cv2.resize(onepig_depth, (target_w, target_h))

        # depth正規化を追加
        onepig_depth[onepig_depth == 0] = onepig_depth[onepig_depth > 0].mean()
        onepig_depth *= onepig_mask

        _mean = onepig_depth[onepig_depth > 0].mean()
        _std = onepig_depth[onepig_depth > 0].std()
        _sigma = _mean + 2 * _std
        onepig_depth[onepig_depth > _sigma] = _mean
        onepig_depth *= onepig_mask

        # rotate
        theta = pig.theta
        onepig_img, onepig_mask, onepig_depth = self.rotater.rotate(onepig_img, onepig_mask, onepig_depth, theta)  # thetaが決まっている時の回転
        onepig_img, onepig_mask, onepig_depth = shrink(onepig_img, onepig_mask, onepig_depth)
        onepig_depth *= onepig_mask

        weight = pig.weight

        new_w = getattr(pig, w_attr)
        new_l = getattr(pig, l_attr)

        if self.cfg["model"]["num_features"] == 3:
            new_mm_squared = pig.new_mm_squared
            data = {"img": onepig_img, "depth": onepig_depth, "mask": onepig_mask, "w": new_w, "l": new_l, "mm_squared": new_mm_squared}
        elif self.cfg["model"]["num_features"] == 2:
            data = {"img": onepig_img, "depth": onepig_depth, "mask": onepig_mask, "w": new_w, "l": new_l, "resize_ratio": pig.resize_ratio}

        data = self.transforms(data)

        onepig_mask = data["mask"]
        data_list = [data[feat] for feat in self.cfg["model"]["input_feats"]]  # input_featsに書いてあるリストを使用する
        tensor = torch.cat(data_list)

        # mask inputs 豚以外の背景を0にする
        tensor = onepig_mask * tensor
        if self.cfg["model"]["num_features"] == 3:
            feats = torch.Tensor([data["mm_squared"], data["w"], data["l"]])
        elif self.cfg["model"]["num_features"] == 2:
            feats = torch.Tensor([data["w"], data["l"]])

        weight = torch.Tensor([weight])

        cutoff_ratio = torch.zeros(len(tensor))
        if "cutoff_ratio" in data.keys():
            cutoff_ratio = torch.FloatTensor([data["cutoff_ratio"]])

        return tensor.float(), feats.float(), weight.float(), cutoff_ratio

    def __len__(self):
        return len(self.dataset)

def shrink(rgb, mask, depth):
    # maskが丁度入る大きさにする
    mask_indices = np.where(mask)  # 0以外の値が入っているindexを取得

    xmin = np.min(mask_indices[1])
    ymin = np.min(mask_indices[0])
    xmax = np.max(mask_indices[1])
    ymax = np.max(mask_indices[0])

    shrinked_rgb = rgb[ymin:ymax, xmin:xmax]
    shrinked_mask = mask[ymin:ymax, xmin:xmax]
    shrinked_depth = depth[ymin:ymax, xmin:xmax]

    return shrinked_rgb, shrinked_mask, shrinked_depth