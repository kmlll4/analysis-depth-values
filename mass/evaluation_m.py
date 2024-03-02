import os
import sys
from datetime import datetime
import numpy as np
import pandas as pd
import cv2
from sklearn.metrics import mean_absolute_percentage_error
from imageio.v2 import imread
from pycocotools import mask as pymask
import mlflow
import torch
import yaml
import math
from torch.utils.data import Sampler

sys.path.append("..")
from abcmodel.lib.mass import transforms as T
from load_posture_model import load_posture_model

sys.path.append("/workspace/module")

# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
torch.set_grad_enabled(False)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

LIGHTCAFE_M_DATA_DIR = "/workspace/ssd/smartfarming_gotech_datas/mass_train_datas/lightcafe_m_mass_train_datas/mass_train_data_rgb_depth"
LIGHTCAFE_GOTECH_DATA_DIR = "/workspace/ssd/smartfarming_gotech_datas/mass_train_datas/lightcafe_gotech_mass_train_datas/mass_train_data_rgb_depth"

CFG_FILE_PATH = "/workspace/mass/ecopork-main/configs/config_m.yml"
with open(CFG_FILE_PATH, "r") as yml:
    CFG = yaml.safe_load(yml)

# MLFLOW = False
MLFLOW = True
# SAVE_PREDICTION_SCORE = False
SAVE_PREDICTION_SCORE = True


class DatasetWrapper:
    def __init__(self, dataset, transforms):
        self.dataset = dataset
        self.transforms = transforms

    def __getitem__(self, idx):
        pig = self.dataset.iloc[idx]

        # Segmentation Refinementを使用したか
        w_attr = "refined_w" if CFG["data"]["refined"] else "w"
        l_attr = "refined_l" if CFG["data"]["refined"] else "l"
        bbox_attr = "refined_bbox" if CFG["data"]["refined"] else "bbox"
        polygon_attr = "refined_polygon" if CFG["data"]["refined"] else "polygon"

        if pig.data_sf_gotech == "sf":
            frame_img = imread(os.path.join(LIGHTCAFE_M_DATA_DIR, pig.filename))
        elif pig.data_sf_gotech == "gotech":
            frame_img = imread(os.path.join(LIGHTCAFE_GOTECH_DATA_DIR, pig.filename))

        x0, y0, w, h = getattr(pig, bbox_attr)

        onepig_img = frame_img[y0 : y0 + h, x0 : x0 + w]
        onepig_depth = pig.height  # onepig_depth = height

        polygon = getattr(pig, polygon_attr)
        if type(polygon) == list:
            frame_mask = pymask.decode(polygon)[:, :, 0]
        else:
            frame_mask = pymask.decode(polygon._rle)

        onepig_mask = frame_mask[y0 : y0 + h, x0 : x0 + w]

        # depth正規化を追加
        onepig_depth[onepig_depth == 0] = onepig_depth[onepig_depth > 0].mean()
        onepig_depth *= onepig_mask

        _mean = onepig_depth[onepig_depth > 0].mean()
        _std = onepig_depth[onepig_depth > 0].std()
        _sigma = _mean + 2 * _std
        onepig_depth[onepig_depth > _sigma] = _mean
        onepig_depth *= onepig_mask

        onepig_img, onepig_mask, onepig_depth = shrink(onepig_img, onepig_mask, onepig_depth)
        onepig_depth *= onepig_mask

        weight = pig.weight

        feat_w = getattr(pig, w_attr)
        feat_l = getattr(pig, l_attr)

        if pig.flipud:
            onepig_img = cv2.flip(onepig_img, 0)
            onepig_mask = cv2.flip(onepig_mask, 0)
            onepig_depth = cv2.flip(onepig_depth, 0)

        if CFG["model"]["num_features"] == 4:
            occulusion_info = pig.occulusion_info
            data = {"img": onepig_img, "depth": onepig_depth, "mask": onepig_mask, "w": feat_w, "l": feat_l, "occulusion": occulusion_info}
        if CFG["model"]["num_features"] == 3:
            new_mm_squared = pig.new_mm_squared
            data = {"img": onepig_img, "depth": onepig_depth, "mask": onepig_mask, "w": feat_w, "l": feat_l, "mm_squared": new_mm_squared}
        elif CFG["model"]["num_features"] == 2:
            data = {"img": onepig_img, "depth": onepig_depth, "mask": onepig_mask, "w": feat_w, "l": feat_l}

        data = self.transforms(data)

        onepig_mask = data["mask"]
        data_list = [data[feat] for feat in CFG["model"]["input_feats"]]  # input_featsに書いてあるリストを使用する
        tensor = torch.cat(data_list)

        # mask inputs 豚以外の背景を0にする
        tensor = onepig_mask * tensor
        if CFG["model"]["num_features"] == 2:
            feats = torch.Tensor([data["w"], data["l"]])
        elif CFG["model"]["num_features"] == 3:
            feats = torch.Tensor([data["mm_squared"], data["w"], data["l"]])
        elif CFG["model"]["num_features"] == 4:
            feats = torch.Tensor([data["w"], data["l"], *data["occulusion"]])  # [w, l, occulusion[0], occulusion[1]] になる

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


def get_oversampled_dataset(dataset: pd.DataFrame, sampler: Sampler):
    return dataset.iloc[list(sampler)]


def get_dataloader(m_df, CFG):
    train_df = m_df[m_df["subset_cv_0"] == "train"]
    val_df = m_df[m_df["subset_cv_0"] == "val"]
    test_df = m_df[m_df["subset_cv_0"] == "test"]
    print("訓練データ数: ", train_df.shape)
    print("検証データ数: ", val_df.shape)
    print("テストデータ数: ", test_df.shape)

    weight_mean = train_df["weight"].mean()
    weight_std = train_df["weight"].std()
    # 決め打ち
    # weight_mean = 83.38761122518822
    # weight_std = 22.55673669482818

    train_df.loc[:, "weight"] = (train_df["weight"] - weight_mean) / weight_std
    val_df.loc[:, "weight"] = (val_df["weight"] - weight_mean) / weight_std
    test_df.loc[:, "weight"] = (test_df["weight"] - weight_mean) / weight_std

    if CFG["model"]["num_features"] == 2:
        train_transforms = T.witouht_mm_train_transforms
        val_transforms = T.without_mm_val_transforms
    elif CFG["model"]["num_features"] == 3:
        train_transforms = T.new_train_transforms
        val_transforms = T.new_val_transforms
    elif CFG["model"]["num_features"] == 4:
        train_transforms = T.with_occ_train_transforms
        val_transforms = T.with_occ_val_transforms

    train_dataset = DatasetWrapper(
        dataset=train_df,
        transforms=train_transforms,
    )
    val_dataset = DatasetWrapper(
        dataset=val_df,
        transforms=val_transforms,
    )
    test_dataset = DatasetWrapper(
        dataset=test_df,
        transforms=val_transforms,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=CFG["train"]["batch_size"],
        num_workers=8,
        pin_memory=True,
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, num_workers=1, pin_memory=True, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, num_workers=1, pin_memory=True, shuffle=False)

    return train_loader, val_loader, test_loader, weight_mean, weight_std


def train(m_df):
    # get data loader
    train_loader, val_loader, test_loader, weight_mean, weight_std = get_dataloader(m_df, CFG)

    model = load_posture_model(
        CFG["model"]["encoder"],
        CFG["model"]["input_channels"],
        CFG["model"]["num_features"],
        CFG["train"]["transfer"],
        CFG["train"]["pretrained_weight"],
    )

    if CFG["model"]["encoder"] == "origin":
        model.module.load_state_dict(torch.load(CFG["val"]["pretrained_weight"]))
    else:
        model.load_state_dict(torch.load(CFG["val"]["pretrained_weight"]))
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG["train"]["init_lr"])

    # scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

    # loss function
    train_loss_fn = torch.nn.MSELoss()

    best_val_mre = np.inf
    best_val_narrow_mre = np.inf
    val_mres = []
    val_narrow_mres = []

    # eval
    model = model.eval()
    val_gts = []
    val_preds = []

    for (tensor, feats, gt, _,) in val_loader:
        _b_size = len(tensor)
        tensor = tensor.cuda()
        feats = feats.cuda()

        tensor = tensor.unsqueeze(1)
        feats = feats.unsqueeze(1)

        tensor_hflip = torch.flip(tensor, dims=(-1,))
        tensor_vflip = torch.flip(tensor, dims=(-2,))
        tensor_vhflip = torch.flip(tensor, dims=(-1, -2))

        # add flip image and ensemble
        tta_tensor = torch.cat([tensor, tensor_hflip, tensor_vflip, tensor_vhflip], dim=1)
        tta_feats = feats.repeat_interleave(repeats=4, dim=1)

        assert len(tta_tensor) == len(tta_feats) == _b_size

        tta_tensor = tta_tensor.view(_b_size * 4, *tta_tensor.shape[-3:])
        tta_feats = tta_feats.view(_b_size * 4, CFG["model"]["num_features"])

        assert tta_tensor.dim() == 4

        with torch.no_grad():
            if CFG["train"]["use_cutoff"]:
                pred, _ = model(tta_tensor, tta_feats)
            else:
                pred = model(tta_tensor, tta_feats)

            pred = pred.view(_b_size, 4, pred.shape[1], 1)
            pred_mean = pred.mean(axis=(1, 2))

            val_gts.extend((gt.flatten() * weight_std + weight_mean).tolist())
            val_preds.extend((pred_mean.flatten() * weight_std + weight_mean).tolist())

    val_gts = np.array(val_gts)
    val_preds = np.array(val_preds)
    val_df = m_df[m_df["subset_cv_0"] == "val"]

    for i, val_pred in enumerate(val_preds):
        # val_df.iat[i, 26] = val_pred
        val_df.at[val_df.index[i], "pred"] = val_pred

    pig_num = len(val_df)

    val_df.loc[:, "relative_error"] = abs(val_df["pred"] - val_df["weight"]) / val_df["weight"]
    val_df.loc[:, "not_abs_relative_error"] = (val_df["pred"] - val_df["weight"]) / val_df["weight"]
    val_df.loc[:, "abs_error"] = abs(val_df["pred"] - val_df["weight"])
    val_df.loc[:, "error"] = val_df["pred"] - val_df["weight"]

    mre = val_df["relative_error"].mean()
    val_df.at[val_df.index[0], "mre"] = mre

    mae = val_df["abs_error"].mean()
    val_df.at[val_df.index[0], "mae"] = mae

    rmse = math.sqrt(((val_df["pred"] - val_df["weight"]) ** 2).mean())
    val_df.at[val_df.index[0], "rmse"] = rmse

    four_error_ratio = (val_df["relative_error"] < 0.04).sum() / len(val_df)
    val_df.at[val_df.index[0], "four_error_ratio"] = four_error_ratio

    ten_error_ratio = (val_df["relative_error"] < 0.1).sum() / len(val_df)
    val_df.at[val_df.index[0], "ten_error_ratio"] = ten_error_ratio

    max_re = val_df["relative_error"].max()
    val_df.at[val_df.index[0], "max_re"] = max_re

    val_mre = mean_absolute_percentage_error(val_gts, val_preds)
    val_mres.append(val_mre)
    narrow = (val_gts >= 100) & (val_gts <= 120)
    val_narrow_mre = mean_absolute_percentage_error(val_gts[narrow], val_preds[narrow])
    val_narrow_mres.append(val_narrow_mre)

    if SAVE_PREDICTION_SCORE:
        val_df.to_excel("/workspace/mass/ecopork-main/output/evaluation.xlsx")
        val_df.to_pickle("/workspace/notebooks/input/evaluation.pkl")

    if MLFLOW:
        mlflow.log_metric("Narrow MRE", val_narrow_mre)
        mlflow.log_metric("Num of Pigs", pig_num)
        mlflow.log_metric("Mean Relative Error", mre)
        mlflow.log_metric("Mean Absolute Error", mae)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("Relative Error Ratio 0.04", four_error_ratio)
        mlflow.log_metric("Relative Error Ratio 0.1", ten_error_ratio)
        mlflow.log_metric("Max Relative Error", max_re)
        mlflow.log_artifact("/workspace/mass/ecopork-main/output/evaluation.xlsx")

    print(f"val MRE: {val_mre}, val narrow MRE: {val_narrow_mre}")


if __name__ == "__main__":

    now = datetime.now()

    pkl_path = os.path.join("/workspace/datas", CFG["data"]["dataset"])
    m_df = pd.read_pickle(pkl_path)

    if not CFG["data"]["with_occulusion"]:
        m_df = m_df[m_df["occulusion"] == False]

    print("データ数: ", m_df.shape)

    # 学習する姿勢のdataframeを抽出する
    # pos_dfs = []
    # for pos in CFG["train"]["by_posture"]:
    #     pos_df = m_df[m_df["posture"] == pos]
    #     pos_dfs.append(pos_df)
    # m_df = pd.concat(pos_dfs)

    # mlflow
    if MLFLOW:
        TRACKING_URI = "http://192.168.3.231:5000"
        mlflow.set_tracking_uri(TRACKING_URI)
        EXPERIMENT_NAME = "evaluation_mass_gotech"
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        if experiment is None:  # 当該Experiment存在しないとき、新たに作成
            experiment_id = mlflow.create_experiment(name=EXPERIMENT_NAME)
        else:  # 当該Experiment存在するとき、IDを取得
            experiment_id = experiment.experiment_id

        with mlflow.start_run(experiment_id=experiment_id) as run:
            mlflow.log_artifact(pkl_path)
            mlflow.log_artifact(CFG["val"]["pretrained_weight"])
            mlflow.log_artifact(CFG_FILE_PATH)
            train(m_df)
    else:
        train(m_df)
