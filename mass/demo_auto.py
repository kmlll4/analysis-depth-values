import os
import sys

sys.path.append("..")
sys.path.append("/workspace/mass/ecopork-main")

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import numpy as np
from tqdm import tqdm
import pandas as pd
from pycocotools import mask as pymask
from datetime import datetime
import torch
import cv2
from imageio.v2 import imread, imwrite

from higher_hrnet.lib.predictor import Refine
from higher_hrnet.lib.transforms import InferenceTransform
from higher_hrnet.lib.clustering import SpatialClustering
from higher_hrnet.models.higher_hrnet import HigherHRNet

from abcmodel.lib.mass import transforms as T
from abcmodel.lib.datasets.cvat import JOINT_LABELS
from load_posture_model import load_posture_model

sys.path.append("/workspace/module/")
from rotater_wlgetter import RotaterInclKP, WLGetter
from anomaly_pig_filter import AnomalyPigFilter
from origin_filter import do_edge_filter, do_keypoint_filter, do_anomaly_depth_filter
from utils import shrink, show_result, show_result_with_judge

fontType = cv2.FONT_HERSHEY_SIMPLEX


class UV_map:
    def __init__(self, args) -> None:
        self.seg_key_model = HigherHRNet(num_keypoints=13, num_seg_classes=2, dim_tag=5).eval()
        self.seg_key_model.load_state_dict(torch.load(args.hr_model_path))
        self.seg_key_model = self.seg_key_model.cuda().eval()
        self.seg_key_model = Refine(self.seg_key_model, JOINT_LABELS, average_tag=True)

        self.inference_transform = InferenceTransform(input_size=480)
        self.clustering = SpatialClustering(threshold=0.05, min_pixels=20, margin=0.5)

    def find_peak(self, hms, thresh=0.1):
        hms[hms < thresh] = 0
        center = np.pad(hms, [[0, 0], [2, 2], [2, 2]])
        up = np.pad(hms, [[0, 0], [0, 4], [2, 2]])
        down = np.pad(hms, [[0, 0], [4, 0], [2, 2]])
        left = np.pad(hms, [[0, 0], [2, 2], [0, 4]])
        right = np.pad(hms, [[0, 0], [2, 2], [4, 0]])

        peak = (center > up) & (center > down) & (center > left) & (center > right)
        peak = peak[:, 2:-2, 2:-2]
        return peak * hms

    def iou(self, b1, b2):
        xmin = max(b1[0], b2[0])
        ymin = max(b1[1], b2[1])
        xmax = min(b1[2], b2[2])
        ymax = min(b1[3], b2[3])
        area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        inter = max((xmax - xmin), 0) * max((ymax - ymin), 0)
        union = area1 + area2 - inter
        return inter / union

    def calc_iou(self, bbox1, bbox2):
        return [[self.iou(b1, b2) for b1 in bbox2] for b2 in bbox1]

    def predict(self, img):
        x, t_inv = self.inference_transform(img)
        x = x.unsqueeze(0).cuda()

        seg_pred, hm_preds, tag_preds = self.seg_key_model(x)  # seg_pred = 耳 tag_preds = instance郡 hm:ヒートマップ
        hr_hm = hm_preds[1].cpu()
        seed = torch.sigmoid(tag_preds[0, -1]).cpu()
        instance_map = self.clustering(tag_preds)
        instance_map = instance_map.cpu().squeeze()

        ins_map = instance_map.numpy()
        ins_map = cv2.resize(ins_map, (640, 480), interpolation=cv2.INTER_NEAREST)
        seg = seg_pred.softmax(dim=1)[0, 1].cpu().numpy()
        hms = hm_preds[1][0].cpu().numpy()

        return ins_map, seg, hms


class MassPredict:
    def __init__(self, args, depth_mean, depth_std, weight_mean, weight_std) -> None:
        self.model = None
        self.depth_mean = depth_mean
        self.depth_std = depth_std
        self.weight_mean = weight_mean
        self.weight_std = weight_std
        self.floor_depth = None

    def set_floor_depth(self, param, x, y, w, h):
        a, b, c = param
        self.floor_depth = np.zeros(w * h).reshape(h, w)
        for i in range(0, w):
            for j in range(0, h):
                self.floor_depth[j, i] = a * (x + i) + b * (y + j) + c
        return self.floor_depth

    def load_mass_model(self, args):
        model = load_posture_model(args.model_name, args.input_channels, args.num_features, args.transfer, args.pretrained_weight)
        if args.model_name == "origin":
            model.module.load_state_dict(torch.load(args.pretrained_weight))

        else:
            model.load_state_dict(torch.load(args.pretrained_weight))

        model = model.cuda().eval()

        self.model = model

    def pred_mass(self, bbox, ori_rgb, ori_mask, img, depth, mask, kp, floor_depth):
        x, y, w, h = bbox

        height = np.where(depth != 0, floor_depth - depth.astype(int), 0)
        # depth正規化を追加
        height[height == 0] = height[height > 0].mean()
        height *= mask

        _mean = height[height > 0].mean()
        _std = height[height > 0].std()
        _sigma = _mean + 2 * _std
        height[height > _sigma] = _mean
        height *= mask

        # 回転させてnew_w, new_lの取得
        if kp_rotater.kp_check(kp):
            rotated_img, rotated_mask, rotated_depth, rotated_kp, theta = kp_rotater(img, mask, depth, kp)
            _, _, rotated_height, _ = kp_rotater.rotate(img, mask, height, kp, theta)
            rotate_method = "kp"

        else:
            # img, mask, depth, kp, theta  = ori_rotater(img, mask, depth, kp)
            # rotated_img, rotated_mask, rotated_depth, rotated_kp, theta  = seq_rotater(img, mask, depth, kp)
            rotated_img, rotated_mask, rotated_depth, rotated_kp, theta = ori_rotater(img, mask, depth, kp)
            _, _, rotated_height, _ = seq_rotater.rotate(img, mask, height, kp, theta)
            rotate_method = "origin"

        # 画像可視化用
        # visualized_mask = wl_getter.visualize_wl(rotated_mask)
        # visualized_mask = cv2.cvtColor(visualized_mask, cv2.COLOR_GRAY2RGB) * 120
        # cv2.putText(visualized_mask, rotate_method, (10, 30), fontType, 0.7, (0, 255, 0), 2)
        # # 回転前imgと回転後vis_maskを結合する
        # shape_img = img.shape[:2]
        # shape_mask = visualized_mask.shape[:2]
        # max_h = max(shape_img[0], shape_mask[0]) + 10
        # max_w = max(shape_img[1], shape_mask[1]) + 10
        # img_for_concat = cv2.copyMakeBorder(img, (max_h - shape_img[0]) // 2, (max_h - shape_img[0]) // 2, (max_w - shape_img[1]) // 2, (max_w - shape_img[1]) // 2, cv2.BORDER_CONSTANT, (0, 0, 0))
        # mask_for_concat = cv2.copyMakeBorder(visualized_mask, (max_h - shape_mask[0]) // 2, (max_h - shape_mask[0]) // 2, (max_w - shape_mask[1]) // 2, (max_w - shape_mask[1]) // 2, cv2.BORDER_CONSTANT, (0, 0, 0))
        # img_for_concat = cv2.resize(img_for_concat, [max_w, max_h])
        # mask_for_concat = cv2.resize(mask_for_concat, [max_w, max_h])
        # bbox_rgb = ori_rgb.copy()
        # cv2.rectangle(bbox_rgb, (x, y), (x + w, y + h), (255, 0, 0), 3)
        # bbox_rgb = cv2.resize(bbox_rgb, [int((max_h / bbox_rgb.shape[0]) * bbox_rgb.shape[1]), max_h])
        # vis_img = cv2.hconcat([bbox_rgb, img_for_concat, mask_for_concat])

        new_w, new_l = wl_getter(rotated_mask, rotated_depth, theta, [x, y, w, h])

        # img, mask, depth, kp = shrink(img, mask, depth, kp) # maskを利用して画像が丁度収まるようにshrinkさせる
        shrinked_img, shrinked_mask, shrinked_height, shrinked_kp = shrink(rotated_img, rotated_mask, rotated_height, rotated_kp)  # maskを利用して画像が丁度収まるようにshrinkさせる
        shrinked_height *= shrinked_mask  # 標準化する前に綺麗にする

        # sreフィルター用
        pig = {"shrinked_rgb": shrinked_img.copy(), "shrinked_mask": shrinked_mask.copy(), "shrinked_peak": shrinked_kp.copy(), "W": new_w, "L": new_l}

        data = {"img": shrinked_img, "depth": shrinked_height, "mask": shrinked_mask, "w": new_w, "l": new_l}

        data = T.without_mm_val_transforms(data)

        # mask inputs 豚以外の背景を0にする
        mask = data["mask"]
        data_list = [data[feat] for feat in args.input_feats]  # input_featsに書いてあるリストを使用する
        tensor = torch.cat(data_list)
        tensor *= mask

        if args.num_features == 3:
            feats = torch.Tensor([data["mm_squared"], data["w"], data["l"]])
        elif args.num_features == 2:
            feats = torch.Tensor([data["w"], data["l"]])

        tensor = tensor.float().cuda()
        feats = feats.float().cuda()

        tensor = tensor.unsqueeze(0)
        feats = feats.unsqueeze(0)

        tensor_hflip = torch.flip(tensor, dims=(-1,))
        tensor_vflip = torch.flip(tensor, dims=(-2,))
        tensor_vhflip = torch.flip(tensor, dims=(-1, -2))

        # add flip image and ensemble
        tta_tensor = torch.cat([tensor, tensor_hflip, tensor_vflip, tensor_vhflip], dim=0)
        tta_feats = feats.repeat_interleave(repeats=4, dim=0)
        assert len(tta_tensor) == len(tta_feats)
        assert tta_tensor.dim() == 4

        with torch.no_grad():
            pred = self.model(tta_tensor, tta_feats)
            pred = pred.view(4, pred.shape[1], 1)
            pred = pred.mean(axis=(0, 1))
            pred = pred.flatten() * self.weight_std + self.weight_mean
            pred = pred.cpu().detach().numpy()[0]

        return pred, pig, new_w, new_l

    def compensation(self, a, b, m):
        # a 勾配　b バイアス　m pred
        c = a * m - b
        m = m + c
        return m


def part_judge(part, necessary, hm_, flag):
    points = []
    for p in part:
        if hm_[p].max() > 0:
            points.append(p)
            if p in necessary:
                necessary.remove(p)

    # コメントはimao予想
    part_ok = False
    if flag == "head":
        if len(points) >= 2 and len(necessary) == 0:  # ２つ以上のpointが0以上なら、全てが0以上であるべき？そうでないならマスクが削られている？
            part_ok = True
    elif flag == "body":
        if len(points) >= 2 and len(necessary) <= 2:  # ２つ以上のpointが0以上なら、３つ以上は0以上であるべき？ そうでないならマスクが削られている？
            part_ok = True
    elif flag == "leg":
        if len(points) == 4:  # 全てのpointが0以上であるべき？ そうでないならマスクが削られている？
            part_ok = True
    return part_ok


def is_occlussion(hm_):
    head, head_necessary = [0, 1, 2, 3], [0, 1, 2, 3]
    body, body_necessary = [4, 5, 6, 7, 8], [4, 5, 6, 7, 8]
    leg, leg_necessary = [9, 10, 11, 12], [9, 10, 11, 12]

    head_ok = part_judge(head, head_necessary, hm_, "head")
    body_ok = part_judge(body, body_necessary, hm_, "body")
    leg_ok = part_judge(leg, leg_necessary, hm_, "leg")

    if not head_ok or not body_ok and not leg_ok:
        return True
    else:
        return False


def has_anomaly_value(depth_, mask_):
    depth_area = np.where(depth_ > 0, 1, 0).sum()
    mask_area = np.where(mask_ > 0, 1, 0).sum()
    d_mean = (depth_ * mask_).mean()
    d_std = (depth_ * mask_).std()
    sigma_u = d_mean + 3 * d_std
    sigma_l = d_mean - 3 * d_std
    d_upper = depth_[depth_ > sigma_u].sum()
    d_lower = depth_[(depth_ < sigma_l) & (depth_ > 0)].sum()

    if mask_area > depth_area or d_upper > 0 or d_lower > 0:
        return True
    else:
        return False


def iou(a, b):
    a_x1, a_y1, a_x2, a_y2 = a
    b_x1, b_y1, b_x2, b_y2 = b

    if a == b:
        return 1.0
    elif ((a_x1 <= b_x1 and a_x2 > b_x1) or (a_x1 >= b_x1 and b_x2 > a_x1)) and ((a_y1 <= b_y1 and a_y2 > b_y1) or (a_y1 >= b_y1 and b_y2 > a_y1)):
        intersection = (min(a_x2, b_x2) - max(a_x1, b_x1)) * (min(a_y2, b_y2) - max(a_y1, b_y1))
        union = (a_x2 - a_x1) * (a_y2 - a_y1) + (b_x2 - b_x1) * (b_y2 - b_y1) - intersection
        return intersection / union
    else:
        return 0.0


def run(args, weight_mean, weight_std, data_dir):

    # floor_depth = 2440.0
    floor_depth = 2425.0

    uvmap = UV_map(args)
    depth_mean = 0.3
    depth_std = 0.15
    mass = MassPredict(args, depth_mean, depth_std, weight_mean, weight_std)

    mass.load_mass_model(args)
    file_list = []

    for jpg_name in os.listdir(data_dir):
        if jpg_name.split("_")[0] == "rgb":
            file_list.append(jpg_name)

    file_list.sort()

    for i, filename in enumerate(tqdm(file_list)):
        rgb = imread(os.path.join(data_dir, filename))
        depth = imread(os.path.join(data_dir, filename.replace("rgb", "depth").replace("jpg", "png")))

        rgb = cv2.resize(rgb, (640, 480))

        depth[depth > floor_depth] = floor_depth
        depth = cv2.resize(depth, (640, 480))

        ins_map, seg, hms = uvmap.predict(rgb)
        hms = cv2.resize(hms.transpose(1, 2, 0), (640, 480)).transpose(2, 0, 1)
        peak = uvmap.find_peak(hms)

        bboxes = []

        for j in np.unique(ins_map):
            if j != 0:
                y, x = np.where(ins_map == j)
                xmin, ymin = np.min([x, y], axis=1)
                xmax, ymax = np.max([x, y], axis=1)
                bboxes.append([xmin, ymin, xmax, ymax])

        result = []
        for i, bbox in enumerate(bboxes):
            mid = np.unique(ins_map)[i + 1]
            mask = (ins_map == mid).astype(np.uint8)

            flag_edge_filter = do_edge_filter(mask)

            rle_mask = pymask.encode(np.asfortranarray(mask))
            x, y, w, h = map(int, pymask.toBbox(rle_mask))

            rgb_ = rgb[y : y + h, x : x + w]
            depth_ = depth[y : y + h, x : x + w]
            mask_ = mask[y : y + h, x : x + w]
            # hm_ = hms[:, y : y + h, x : x + w] * mask_
            kp_ = peak[:, y : y + h, x : x + w] * mask_

            ori_rgb = rgb.copy()
            ori_mask = mask.copy()

            mass.floor_depth = floor_depth
            pred, pig, pig_w, pig_l = mass.pred_mass([x, y, w, h], ori_rgb, ori_mask, rgb_, depth_, mask_, kp_, floor_depth)
            anomaly_judge, flag_anomaly_wl, flag_anomaly_mask, flag_incomplete_mask, _ = anomaly_pig_filter(pig)

            all_judge = flag_edge_filter or anomaly_judge
            result.append([[x, y, w, h], pred, all_judge])
            # result.append([[x, y, w, h], pred])

        # BBoxの描画と保存
        bbox_img = show_result_with_judge(rgb, result)

        imwrite(os.path.join(output_dir, filename), bbox_img)


if __name__ == "__main__":
    # used_mass_train_file = "/workspace/datas/sre_mass_datas_v1.0.0.pkl"
    used_mass_train_file = "/workspace/datas/sre_mass_datas_v1.1.0.pkl"
    # used_mass_train_file = '/workspace/datas/sre_mass_datas_v1.1.1.pkl'
    m_df = pd.read_pickle(used_mass_train_file)

    train_df = m_df[m_df["subset"] == "train"]
    weight_mean = train_df["weight"].mean()
    weight_std = train_df["weight"].std()

    now = datetime.now()

    parser = argparse.ArgumentParser(description="pig")
    # parser.add_argument('--num_features', type=int, default=3) # w, l, mm
    parser.add_argument("--num_features", type=int, default=2)  # w, l
    parser.add_argument("--input_channels", type=int, default=4)  # rgb + depth
    # parser.add_argument('--input_channels', type=int, default=3) # rgb
    # parser.add_argument('--input_channels', type=int, default=1) # depth
    parser.add_argument("--input_feats", nargs="*", type=str, default=["img", "depth"])
    # parser.add_argument('--input_feats', nargs="*", type=str, default=['img'])
    parser.add_argument("--arch", type=str, default="mobilenet, xception")
    # parser.add_argument("--pretrained_weight", type=str, default="/workspace/mass/ecopork-main/mass_models/2022100114_new_data/epoch-2093_val_MRE-0.0353_val_narrow_MRE-0.0313.pth")  # ori
    parser.add_argument("--pretrained_weight", type=str, default="/workspace/mass/ecopork-main/mass_models/epoch-305_val_MRE-0.0452_val_narrow_MRE-0.0431.pth")  # ori
    parser.add_argument("--hr_model_path", type=str, default="/workspace/pretrained_models/origin/hrnet/seg_key_model.pth")
    parser.add_argument("--transfer", action="store_true")  # default = False
    parser.add_argument("--model_name", type=str, default="origin")
    # parser.add_argument('--model_name', type=str, default='eff')
    # parser.add_argument('--model_name', type=str, default='hrnet')
    parser.add_argument("--save_img", action="store_true")  # default = False
    # parser.add_argument('--save_img', action='store_false') # default = True

    parser.add_argument("--csv_name", type=str, default="_sre.csv")
    # parser.add_argument('--csv_name', type=str, default='_sre_pos.csv')

    args = parser.parse_args()

    ori_rotater = RotaterInclKP("origin")
    kp_rotater = RotaterInclKP("keypoint")
    seq_rotater = RotaterInclKP("max_sequence")
    wl_getter = WLGetter(0.1, 2.0)
    anomaly_pig_filter = AnomalyPigFilter()

    data_dir = "/workspace/mass/ecopork-main/input/tmp/JF"
    output_dir = "/workspace/mass/ecopork-main/output/tmp"

    # continue
    run(args, weight_mean, weight_std, data_dir)
