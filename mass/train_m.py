import os
import sys
import io
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
import yaml
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
from imageio.v2 import imread
from pycocotools import mask as pymask
from typing import Tuple
import bisect
import mlflow
import torch

from torch.utils.data import Subset, Sampler, Dataset
from torch.utils.tensorboard import SummaryWriter
from abcmodel.lib.mass import transforms as T
from abcmodel.lib.mass.sampler import MinMaxImbalancedSampler, IndividualImbalancedSampler, get_oversampled_dataset
from load_posture_model import load_posture_model
import torch_optimizer as optim
from custom_loss import MseWithPos

sys.path.append("/workspace/module")

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
LIGHTCAFE_M_DATA_DIR = "/workspace/ssd/smartfarming_gotech_datas/mass_train_datas/lightcafe_m_mass_train_datas/mass_train_data_rgb_depth"
LIGHTCAFE_GOTECH_DATA_DIR = "/workspace/ssd/smartfarming_gotech_datas/mass_train_datas/lightcafe_gotech_mass_train_datas/mass_train_data_rgb_depth"

CFG_FILE_PATH = "/workspace/mass/ecopork-main/configs/config_m.yml"
with open(CFG_FILE_PATH, "r") as yml:
    CFG = yaml.safe_load(yml)

MLFLOW = True
# MLFLOW = False
# TB = True
TB = False

torch.set_grad_enabled(False)


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

        if CFG["model"]["num_features"] == 4 and CFG["data"]["with_occulusion"]:
            occulusion_info = pig.occulusion_info
            data = {"img": onepig_img, "depth": onepig_depth, "mask": onepig_mask, "w": feat_w, "l": feat_l, "occulusion": occulusion_info}
        if CFG["model"]["num_features"] == 3:
            new_mm_squared = pig.new_mm_squared
            data = {"img": onepig_img, "depth": onepig_depth, "mask": onepig_mask, "w": feat_w, "l": feat_l, "mm_squared": new_mm_squared}
        elif CFG["model"]["num_features"] == 2:
            data = {"img": onepig_img, "depth": onepig_depth, "mask": onepig_mask, "w": feat_w, "l": feat_l}

        data = self.transforms(data)  # Normalizeパラメータ注意

        onepig_mask = data["mask"]
        data_list = [data[feat] for feat in CFG["model"]["input_feats"]]  # input_featsに書いてあるリストを使用する
        tensor = torch.cat(data_list)

        # mask inputs 豚以外の背景を0にする
        tensor = onepig_mask * tensor

        if CFG["model"]["num_features"] == 2:
            feats = torch.Tensor([data["w"], data["l"]])
        elif CFG["model"]["num_features"] == 3:
            feats = torch.Tensor([data["mm_squared"], data["w"], data["l"]])
        elif CFG["model"]["num_features"] == 4 and CFG["data"]["with_occulusion"]:
            feats = torch.Tensor([data["w"], data["l"], *data["occulusion"]])  # [w, l, occulusion[0], occulusion[1]] になる

        weight = torch.Tensor([weight])

        cutoff_ratio = torch.zeros(len(tensor))
        if "cutoff_ratio" in data.keys():
            cutoff_ratio = torch.FloatTensor([data["cutoff_ratio"]])

        return tensor.float(), feats.float(), weight.float(), cutoff_ratio

    def __len__(self):
        return len(self.dataset)


class WeightOverSampler(Sampler):
    def __init__(self, m_df: pd.DataFrame, bins: Tuple[int], counts: Tuple[int]):
        """
        Args:
            m_df: dataframe subse==train.
            bins: bins for weight. arbitrary length but has to be same length as counts.
            counts: desired count for each bin.
        """

        super(WeightOverSampler, self).__init__(data_source=None)
        assert len(bins) == len(counts) + 1
        labels = [p.weight for idx, p in m_df.iterrows()]
        _, bins = np.histogram(labels, range=range, bins=bins)
        df = pd.Series(bisect.bisect_left(bins, label) for label in labels)
        weights = pd.Series(counts, index=np.arange(1, len(bins)))
        weights = weights[df]

        # self.weights = torch.DoubleTensor(weights.to_list())
        self.weights = torch.DoubleTensor((1 / weights).to_list())
        self.length = sum(counts)

    def __len__(self):
        return self.length

    def __iter__(self):
        return map(int, torch.multinomial(self.weights, self.length, replacement=True))


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


def get_dataloader(m_df):
    train_df = m_df[m_df["subset_cv_0"] == "train"]
    val_df = m_df[m_df["subset_cv_0"] == "val"]
    test_df = m_df[m_df["subset_cv_0"] == "test"]
    print("訓練データ数: ", train_df.shape)
    print("検証データ数: ", val_df.shape)
    print("テストデータ数: ", test_df.shape)

    # normalize
    weight_mean = train_df["weight"].mean()
    weight_std = train_df["weight"].std()

    if MLFLOW:
        mlflow.log_param("all_weight_mean", m_df["weight"].mean())
        mlflow.log_param("all_weight_std", m_df["weight"].std())
        mlflow.log_param("train_weight_mean", weight_mean)
        mlflow.log_param("train_weight_std", weight_std)

    if CFG["train"]["use_oversample"]:
        balanced_train_df = get_oversampled_dataset(train_df, IndividualImbalancedSampler(train_df, 20, 50))
        bins = [30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0]
        bins = bins + [float(i) for i in range(111, 125)] + [125.0, 130.0, 140.0]

        count, bins = np.histogram([p.weight for idx, p in balanced_train_df.iterrows()], bins=bins)
        print("bins: ", bins)
        print("count: ", count)

        oversampler = WeightOverSampler(balanced_train_df, bins=bins, counts=count)
        train_df = get_oversampled_dataset(dataset=balanced_train_df, sampler=oversampler)

    train_df.loc[:, "weight"] = (train_df["weight"] - weight_mean) / weight_std
    val_df.loc[:, "weight"] = (val_df["weight"] - weight_mean) / weight_std
    test_df.loc[:, "weight"] = (test_df["weight"] - weight_mean) / weight_std

    # 全てtransformを変更　train_transforms → new_train_transforms, val_transforms → new_val_transforms

    if CFG["model"]["num_features"] == 2:
        train_transforms = T.witouht_mm_train_transforms
        val_transforms = T.without_mm_val_transforms
    elif CFG["model"]["num_features"] == 3:
        train_transforms = T.new_train_transforms
        val_transforms = T.new_val_transforms
    elif CFG["model"]["num_features"] == 4 and CFG["data"]["with_occulusion"]:
        train_transforms = T.with_occ_train_transforms
        val_transforms = T.with_occ_val_transforms

    train_dataset = DatasetWrapper(dataset=train_df, transforms=train_transforms)
    val_dataset = DatasetWrapper(dataset=val_df, transforms=val_transforms)
    test_dataset = DatasetWrapper(dataset=test_df, transforms=val_transforms)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=CFG["train"]["batch_size"], num_workers=4, pin_memory=True, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, num_workers=1, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, num_workers=1, pin_memory=True)

    return train_loader, val_loader, test_loader, weight_mean, weight_std


def train(m_df):
    # get data loader
    train_loader, val_loader, test_loader, weight_mean, weight_std = get_dataloader(m_df)

    # create model
    model = load_posture_model(CFG["model"]["encoder"], CFG["model"]["input_channels"], CFG["model"]["num_features"], CFG["train"]["transfer"], CFG["train"]["pretrained_weight"])

    if CFG["train"]["restart"]:
        model.load_state_dict(torch.load(CFG["train"]["restart_weight"]))

    optimizer = torch.optim.Adam(model.parameters(), lr=CFG["train"]["init_lr"])
    # optimizer = torch.optim.Adam(model.parameters(), lr=CFG["train"]["init_lr"], weight_decay=CFG["train"]["weight_decay"]) # 荷重減衰追加
    # optimizer = optim.RAdam(model.parameters(), lr=CFG["train"]["init_lr"], weight_decay=CFG["train"]["weight_decay"])
    # optimizer = optim.RAdam(model.parameters(), lr=CFG["train"]["init_lr"])

    if not CFG["train"]["transfer"] and CFG["train"]["use_scheduler"]:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

    # loss function
    # train_loss_fn = MseWithPos()
    train_loss_fn = torch.nn.MSELoss()

    best_val_mre = np.inf
    best_val_narrow_mre = np.inf
    train_mres = []
    val_mres = []
    val_narrow_mres = []
    tot_epochs = CFG["train"]["tot_epochs"]

    if MLFLOW:
        mlflow.log_param("optimzer", optimizer)
        mlflow.log_param("train_loss_fn", train_loss_fn)
        mlflow.log_param("max_epoch", tot_epochs)

    # Training
    pbar = tqdm(range(tot_epochs))

    for epoch in pbar:
        if CFG["train"]["transfer"] and epoch == CFG["train"]["fix_epoch"]:
            if CFG["train"]["specific_layer"]:
                if CFG["model"]["encoder"] == "eff":
                    fine_tune_layer = ["features.4", "features.5", "features.6", "features.7", "features.8"]  # この名前が含まれている層はupdateをTrueにする
                elif CFG["model"]["encoder"] == "hrnet":
                    fine_tune_layer = ["stage4", "sre_layers"]  # この名前が含まれている層はupdateをTrueにする
                print("------new_update_layer------")
                for name, param in model.named_parameters():
                    if [s for s in fine_tune_layer if s in name]:
                        param.requires_grad = True
                        print(name)
                    else:
                        pass
                print("----------------------------")
            else:
                for param in model.parameters():
                    param.requires_grad = True

            if CFG["train"]["change_optimizer"]:
                # optimizer = optim.RAdam(model.parameters(), lr=CFG["train"]["ft_lr"], weight_decay=CFG["train"]["ft_weight_decay"])
                optimizer = optim.RAdam(model.parameters(), lr=CFG["train"]["ft_lr"])
                # optimizer = torch.optim.Adam(model.parameters(), lr=CFG["train"]["init_lr"])
            else:
                # optimizer = torch.optim.Adam(model.parameters(), lr=CFG["train"]["init_lr"], weight_decay=CFG["train"]["weight_decay"])
                optimizer = torch.optim.Adam(model.parameters(), lr=CFG["train"]["init_lr"])
            if CFG["train"]["use_scheduler"]:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
                if MLFLOW:
                    mlflow.log_param("scheduler", scheduler)

        print(f"epoch: {epoch} / {tot_epochs}")
        # train
        model = model.train()
        train_gts = []
        train_preds = []
        train_loss = []
        for step, (tensor, feats, gt, cr_gt) in tqdm(enumerate(train_loader)):
            # import pdb; pdb.set_trace()
            tensor = tensor.cuda()
            feats = feats.cuda()

            # because of Ensemble model, number of target is 2.
            if CFG["model"]["encoder"] == "origin":
                arch = CFG["model"]["arch"].split(",")
                target = torch.repeat_interleave(gt.unsqueeze(1), repeats=len(arch), dim=1).cuda()
                cr_gt = torch.repeat_interleave(cr_gt.unsqueeze(1), repeats=len(arch), dim=1).cuda()
            else:
                target = gt.cuda()

            with torch.enable_grad():
                if CFG["train"]["use_cutoff"]:
                    pred, cr_pred = model(tensor, feats)
                    cutoff_loss = torch.mean((cr_pred - cr_gt) ** 2)
                    loss = train_loss_fn(pred, target)
                    loss += 0.01 * cutoff_loss
                else:
                    pred = model(tensor, feats)
                    loss = train_loss_fn(pred, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            train_loss.append(loss.item())
            train_gts.extend((gt.flatten() * weight_std + weight_mean).tolist())
            train_preds.extend((pred.mean(dim=1).flatten() * weight_std + weight_mean).tolist())

        print("train loss: ", np.mean(train_loss))

        if CFG["train"]["transfer"] and epoch >= CFG["train"]["fix_epoch"] and CFG["train"]["use_scheduler"]:
            scheduler.step()
        elif not CFG["train"]["transfer"] and CFG["train"]["use_scheduler"]:
            scheduler.step()

        train_mre = mean_absolute_percentage_error(train_gts, train_preds)
        train_mres.append(train_mre)

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
        val_mre = mean_absolute_percentage_error(val_gts, val_preds)
        val_mres.append(val_mre)
        narrow = (val_gts >= 100) & (val_gts <= 120)
        val_narrow_mre = mean_absolute_percentage_error(val_gts[narrow], val_preds[narrow])
        val_narrow_mres.append(val_narrow_mre)

        if TB:
            writer.add_scalars("train_val_mre", {"train_mre": train_mre, "val_mre": val_mre, "val_narrow_mre": val_narrow_mre}, epoch)

        if MLFLOW:
            # mlflow_評価指標(Metrics)
            mlflow.log_metric("train_mre", train_mre, step=epoch)
            mlflow.log_metric("val_mre", val_mre, step=epoch)
            mlflow.log_metric("val_narrow_mre", val_narrow_mre, step=epoch)

        if epoch > CFG["train"]["save_from_x_epoch"] and val_mre < best_val_mre:
            best_val_mre = val_mre
            best_val_ls = [epoch, val_mre, val_narrow_mre]
            if CFG["model"]["encoder"] == "origin":
                torch.save(model.module.state_dict(), f"{base_dir}/epoch-{epoch}_val_MRE-{val_mre:.4f}_val_narrow_MRE-{val_narrow_mre:.4f}.pth")
            else:
                torch.save(model.state_dict(), f"{base_dir}/epoch-{epoch}_val_MRE-{val_mre:.4f}_val_narrow_MRE-{val_narrow_mre:.4f}.pth")

        if epoch > CFG["train"]["save_from_x_epoch"] and val_narrow_mre < best_val_narrow_mre:
            best_val_narrow_mre = val_narrow_mre
            best_val_narrow_ls = [epoch, val_mre, val_narrow_mre]
            if CFG["model"]["encoder"] == "origin":
                torch.save(model.module.state_dict(), f"{base_dir}/epoch-{epoch}_val_MRE-{val_mre:.4f}_val_narrow_MRE-{val_narrow_mre:.4f}.pth")
            else:
                torch.save(model.state_dict(), f"{base_dir}/epoch-{epoch}_val_MRE-{val_mre:.4f}_val_narrow_MRE-{val_narrow_mre:.4f}.pth")

        pbar.set_postfix({"train MRE": train_mre, "val MRE": val_mre, "val narrow MRE": val_narrow_mre, "best MRE": best_val_mre, "best narrow MRE": best_val_narrow_mre})

    if MLFLOW:
        # 最後のベストエポックのみmlflowへ保存
        mlflow.log_artifact(f"{base_dir}/epoch-{best_val_ls[0]}_val_MRE-{best_val_ls[1]:.4f}_val_narrow_MRE-{best_val_ls[2]:.4f}.pth")
        mlflow.log_artifact(f"{base_dir}/epoch-{best_val_narrow_ls[0]}_val_MRE-{best_val_narrow_ls[1]:.4f}_val_narrow_MRE-{best_val_narrow_ls[2]:.4f}.pth")


if __name__ == "__main__":
    now = datetime.now()

    pkl_path = os.path.join("/workspace/datas", CFG["data"]["dataset"])
    m_df = pd.read_pickle(pkl_path)

    if not CFG["data"]["with_occulusion"]:
        m_df = m_df[m_df["occulusion"] == False]

    print("データ数: ", m_df.shape)

    base_dir = os.path.join(CFG["train"]["train_logs"], f"{now.strftime('%Y%m%d%H%M%S')}")

    os.makedirs(base_dir, exist_ok=True)

    if TB:
        tb_dir = "/workspace/mass/ecopork-main/tb_logs/" + f"{now.strftime('%Y%m%d%H%M%S')}"
        writer = SummaryWriter(log_dir=tb_dir)

    if MLFLOW:
        # mlflow
        TRACKING_URI = "http://192.168.3.231:5000"
        mlflow.set_tracking_uri(TRACKING_URI)
        EXPERIMENT_NAME = "train_mass_gotech"

        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        if experiment is None:  # 当該Experiment存在しないとき、新たに作成
            experiment_id = mlflow.create_experiment(name=EXPERIMENT_NAME)
        else:  # 当該Experiment存在するとき、IDを取得
            experiment_id = experiment.experiment_id

        with mlflow.start_run(experiment_id=experiment_id) as run:
            # 実験条件(Parameters)
            mlflow.log_param("init_lr", CFG["train"]["init_lr"])
            mlflow.log_param("weight_decay", CFG["train"]["weight_decay"])
            mlflow.log_param("num_features", CFG["model"]["num_features"])
            mlflow.log_param("input_channels", CFG["model"]["input_channels"])
            mlflow.log_param("train_batch_size", CFG["train"]["batch_size"])
            mlflow.log_param("use_cutoff", CFG["train"]["use_cutoff"])
            mlflow.log_param("use_oversample", CFG["train"]["use_oversample"])
            mlflow.log_param("arch", CFG["model"]["arch"])
            mlflow.log_param("model_name", CFG["model"]["encoder"])
            mlflow.log_param("pretrained_weight", CFG["train"]["pretrained_weight"])
            mlflow.log_param("transfer", CFG["train"]["transfer"])
            mlflow.log_param("specific_layer", CFG["train"]["specific_layer"])
            mlflow.log_param("fix_epoch", CFG["train"]["fix_epoch"])
            mlflow.log_param("change_optimizer", CFG["train"]["change_optimizer"])
            mlflow.log_param("ft_lr", CFG["train"]["ft_lr"])
            mlflow.log_param("ft_weight_decay", CFG["train"]["ft_weight_decay"])
            mlflow.log_param("use_scheduler", CFG["train"]["use_scheduler"])

            mlflow.log_artifact(CFG_FILE_PATH)
            # pklファイル保存
            mlflow.log_artifact(pkl_path)
            train(m_df)
    # elif TB:
    # configファイル保存するように追記(予定)
    else:
        train(m_df)
