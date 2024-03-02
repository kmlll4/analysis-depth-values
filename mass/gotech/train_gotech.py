import os
import sys
sys.path.append('..')
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import argparse
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
from imageio.v2 import imread
from pycocotools import mask as pymask
from typing import Tuple
import bisect
import mlflow
import torch
torch.set_grad_enabled(False)
from torch.utils.data import Subset, Sampler, Dataset
from torch.utils.tensorboard import SummaryWriter

sys.path.append('/workspace/mass/ecopork-main')
from abcmodel.lib.datasets.cvat import Pig, CVATMassDataset, Polygon, JOINT_LABELS
from abcmodel.lib.mass import transforms as T
from abcmodel.lib.mass.sampler import (
    MinMaxImbalancedSampler, IndividualImbalancedSampler, 
    get_oversampled_dataset
)

from load_posture_model import load_posture_model
import torch_optimizer as optim
import json

DATA_DIR = '/workspace/annotation_datas'
OUTPUT_EXP = '/workspace/mass/ecopork-main/output'

class DatasetWrapper:
    def __init__(self, dataset, transforms, DATA_DIR):
        self.dataset = dataset
        self.transforms = transforms
        self.DATA_DIR = DATA_DIR

    def __getitem__(self, idx):
        pig = self.dataset.iloc[idx]
                
        img = imread(os.path.join(self.DATA_DIR, pig.img_path))
        x0, y0, w, h = pig.bbox
        img = img[y0:y0 + h, x0:x0 + w]

        # depth = imread(os.path.join(self.DATA_DIR, pig.img_path.replace('rgb', 'depth').replace('.jpg', '.png')))
        
        depth = pig.height # depth means height
        
        # depth[depth > pig.floor_depth] = pig.floor_depth
        # depth = cv2.resize(depth, (640, 480))
        # depth = depth[y0:y0 + h, x0:x0 + w]
        # depth = np.where(depth != 0, pig.floor_depth - depth.astype(int), 0)

        if type(pig.polygon) == list:
            mask = pymask.decode(pig.polygon)[:,:,0]
        else:
            mask = pymask.decode(pig.polygon._rle)
        
        mask = mask[y0:y0 + h, x0:x0 + w]

        # depth正規化を追加
        depth[depth == 0] = depth[depth > 0].mean()
        depth *= mask

        _mean = depth[depth > 0].mean()
        _std = depth[depth > 0].std()
        _sigma = _mean + 2 * _std
        depth[depth > _sigma] = _mean
        depth *= mask

        # rotate
        # theta = pig.theta
        # img, mask, depth = self.rotater.rotate(img, mask, depth, theta) # thetaが決まっている時の回転
        img, mask, depth = shrink(img, mask, depth)
        depth *= mask

        weight = pig.weight
        w = pig.w
        l = pig.l
            
        if args.num_features == 3:
            new_mm_squared = pig.new_mm_squared
            data = {'img': img, 'depth': depth, 'mask': mask, 
                    'w': w, 'l': l, 'mm_squared': new_mm_squared}
        elif args.num_features == 2:
            data = {'img': img, 'depth': depth, 'mask': mask, 
                    'w': w, 'l': l}

        data = self.transforms(data) # Normalizeパラメータ注意

        mask = data['mask']
        data_list = [data[feat] for feat in args.input_feats] # input_featsに書いてあるリストを使用する
        tensor = torch.cat(data_list)

        # mask inputs 豚以外の背景を0にする
        tensor = mask * tensor
        if args.num_features == 3:
            feats = torch.Tensor([data['mm_squared'], data['w'], data['l']])
        elif args.num_features == 2:
            feats = torch.Tensor([data['w'], data['l']])
            
        weight = torch.Tensor([weight])

        cutoff_ratio = torch.zeros(len(tensor))
        if 'cutoff_ratio' in data.keys():
            cutoff_ratio = torch.FloatTensor([data['cutoff_ratio']])
        
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


class IndividualImbalancedSampler(MinMaxImbalancedSampler):
    def __init__(self, m_df: pd.DataFrame, min_sample_per_individual: int, 
                max_sample_per_individual: int):
        """
        Args:
            dataset: dataset class
            min_sample_per_individual: minimum number of samples per individual.
            max_sample_per_individual: maximum number of samples per individual.
        """
        ids = np.array([pig.ID for idx, pig in m_df.iterrows()])
        super(IndividualImbalancedSampler, self).__init__(
            labels=ids, min_sample_per_individual=min_sample_per_individual,
            max_sample_per_individual=max_sample_per_individual)

def shrink(rgb, mask, depth):
    # maskが丁度入る大きさにする
    mask_indices = np.where(mask) # 0以外の値が入っているindexを取得
    
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


def get_dataloader(m_df, args):
    train_df = m_df[m_df["subset_cv_0"]=="train"]
    val_df = m_df[m_df["subset_cv_0"]=="val"]
    test_df = m_df[m_df["subset_cv_0"]=="test"]
    print('訓練データ数: ', train_df.shape)
    print('検証データ数: ', val_df.shape)
    print('テストデータ数: ', test_df.shape)

    # normalize
    weight_mean = train_df['weight'].mean()
    weight_std = train_df['weight'].std()
    
    if use_mlflow:
        mlflow.log_param('all_weight_mean', m_df['weight'].mean())
        mlflow.log_param('all_weight_std', m_df['weight'].std())
        mlflow.log_param('train_weight_mean', weight_mean)
        mlflow.log_param('train_weight_std', weight_std)

    if args.use_oversample:
        balanced_train_df = get_oversampled_dataset(
            train_df, IndividualImbalancedSampler(train_df, 20, 50))
        bins = [30., 40., 50., 60., 70., 80., 90., 100., 110.]
        bins = bins + [float(i) for i in range(111, 125)] + [125., 130., 140.]

        count, bins = np.histogram(
            [p.weight for idx, p in balanced_train_df.iterrows()], bins=bins)
        print("bins: ", bins)
        print("count: ", count)

        oversampler = WeightOverSampler(
            balanced_train_df, bins=bins, counts=count)
        train_df = get_oversampled_dataset(
            dataset=balanced_train_df, sampler=oversampler)

    train_df.loc[:, 'weight'] = (train_df['weight'] - weight_mean) / weight_std
    val_df.loc[:, 'weight'] = (val_df['weight'] - weight_mean) / weight_std
    test_df.loc[:, 'weight'] = (test_df['weight'] - weight_mean) / weight_std

    # 全てtransformを変更　train_transforms → new_train_transforms, val_transforms → new_val_transforms

    if args.num_features == 3:
        train_transforms = T.new_train_transforms
        val_transforms = T.new_val_transforms
    elif args.num_features == 2:
        train_transforms = T.witouht_mm_train_transforms
        val_transforms = T.without_mm_val_transforms

    train_dataset = DatasetWrapper(
        dataset=train_df, transforms=train_transforms, DATA_DIR=DATA_DIR)
    val_dataset = DatasetWrapper(
        dataset=val_df, transforms=val_transforms, DATA_DIR=DATA_DIR)
    test_dataset = DatasetWrapper(
        dataset=test_df, transforms=val_transforms, DATA_DIR=DATA_DIR)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size,
        num_workers=4, pin_memory=True, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=32, num_workers=1, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, num_workers=1, pin_memory=True)

    return train_loader, val_loader, test_loader, weight_mean, weight_std


def train(m_df, args):
    # get data loader
    train_loader, val_loader, test_loader, weight_mean, weight_std = get_dataloader(m_df, args)
    
    # create model
    model = load_posture_model(args.model_name, args.input_channels, args.num_features, args.transfer, args.pretrained_weight)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr, weight_decay=args.weight_decay) # 荷重減衰追加
    # optimizer = optim.RAdam(model.parameters(), lr=args.init_lr, weight_decay=args.weight_decay)
    # optimizer = optim.RAdam(model.parameters(), lr=args.init_lr)
    
    if not args.transfer and args.use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)    
        
    # loss function
    train_loss_fn = torch.nn.MSELoss()

    best_val_mre = np.inf
    best_val_narrow_mre = np.inf
    train_mres = []
    val_mres = []
    val_narrow_mres = []
    tot_epochs = 1000

    if use_mlflow:
        mlflow.log_param('optimzer', optimizer)
        mlflow.log_param('train_loss_fn', train_loss_fn)
        mlflow.log_param('max_epoch', tot_epochs)
    
    # Training
    pbar = tqdm(range(tot_epochs))

    for epoch in pbar:
        
        if args.transfer and epoch == args.fix_epoch:
            
            if args.specific_layer:
                if args.model_name == 'eff':
                    fine_tune_layer = ['features.4', 'features.5', 'features.6', 'features.7', 'features.8'] # この名前が含まれている層はupdateをTrueにする
                elif args.model_name == 'hrnet':
                    fine_tune_layer = ['stage4', 'sre_layers'] # この名前が含まれている層はupdateをTrueにする
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
                
            if args.change_optimizer:
                # optimizer = optim.RAdam(model.parameters(), lr=args.ft_lr, weight_decay=args.ft_weight_decay) 
                optimizer = optim.RAdam(model.parameters(), lr=args.ft_lr) 
                # optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr)
            else:
                # optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr, weight_decay=args.weight_decay)
                optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr)
            if args.use_scheduler:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
                if use_mlflow:
                    mlflow.log_param('scheduler', scheduler)
        

        print(f'epoch: {epoch} / {tot_epochs}')
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
            arch = args.arch.split(",")
            if args.model_name == 'origin':
                target = torch.repeat_interleave(gt.unsqueeze(1), repeats=len(arch), dim=1).cuda()
                cr_gt = torch.repeat_interleave(cr_gt.unsqueeze(1), repeats=len(arch), dim=1).cuda()
            else:
                target = gt.cuda()

            with torch.enable_grad():
                if args.use_cutoff:
                    pred, cr_pred = model(tensor, feats)
                    cutoff_loss = torch.mean((cr_pred - cr_gt)**2)                
                    loss = train_loss_fn(pred, target)

                    loss += 0.01 * cutoff_loss
                else:
                    pred = model(tensor, feats)                    
                    loss = train_loss_fn(pred, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            train_loss.append(loss.item())
            train_gts.extend((gt.flatten()*weight_std+weight_mean).tolist())
            train_preds.extend((pred.mean(dim=1).flatten()*weight_std+weight_mean).tolist())
        
        print('train loss: ', np.mean(train_loss))

        if args.transfer and epoch >= args.fix_epoch and args.use_scheduler:
            scheduler.step()
        elif not args.transfer and args.use_scheduler:
            scheduler.step()

        train_mre = mean_absolute_percentage_error(train_gts, train_preds)
        train_mres.append(train_mre)

        # eval
        model = model.eval()
        val_gts = []
        val_preds = []
        for tensor, feats, gt, _ in val_loader:
            _b_size = len(tensor)
            tensor = tensor.cuda()
            feats = feats.cuda()

            tensor = tensor.unsqueeze(1)
            feats = feats.unsqueeze(1)

            tensor_hflip = torch.flip(tensor, dims=(-1,))
            tensor_vflip = torch.flip(tensor, dims=(-2,))
            tensor_vhflip = torch.flip(tensor, dims=(-1, -2))

            # add flip image and ensemble
            tta_tensor = torch.cat([
                tensor, tensor_hflip, tensor_vflip, tensor_vhflip], dim=1)
            tta_feats = feats.repeat_interleave(repeats=4, dim=1)

            assert len(tta_tensor) == len(tta_feats) == _b_size

            tta_tensor = tta_tensor.view(_b_size*4, *tta_tensor.shape[-3:])
            tta_feats = tta_feats.view(_b_size*4, args.num_features)

            assert tta_tensor.dim() == 4

            with torch.no_grad():
                if args.use_cutoff:
                    pred, _ = model(tta_tensor, tta_feats)
                else:
                    pred = model(tta_tensor, tta_feats)

                pred = pred.view(_b_size, 4, pred.shape[1], 1)
                pred_mean = pred.mean(axis=(1,2))

                val_gts.extend((gt.flatten()*weight_std+weight_mean).tolist())
                val_preds.extend((pred_mean.flatten()*weight_std+weight_mean).tolist())

        val_gts, val_preds = np.array(val_gts), np.array(val_preds)
        val_mre = mean_absolute_percentage_error(val_gts, val_preds)
        val_mres.append(val_mre)
        narrow = (val_gts >= 100) & (val_gts <= 120)
        val_narrow_mre = mean_absolute_percentage_error(val_gts[narrow], val_preds[narrow])
        val_narrow_mres.append(val_narrow_mre)
        
        if use_tb:
            writer.add_scalars(f"train_val_mre", 
            {'train_mre':train_mre, 'val_mre':val_mre, 'val_narrow_mre':val_narrow_mre}, epoch)

        if use_mlflow:
            # mlflow_評価指標(Metrics)
            mlflow.log_metric('train_mre', train_mre, step = epoch)
            mlflow.log_metric('val_mre', val_mre, step = epoch)
            mlflow.log_metric('val_narrow_mre', val_narrow_mre, step = epoch)

        # if not args.debug and val_mre < best_val_mre and val_mre < args.save_loss_thresh:
        #     best_val_mre = val_mre
        #     best_val_ls  = [epoch, val_mre, val_narrow_mre]
        #     if args.model_name == 'origin':
        #         save_model = model.module
        #     torch.save(save_model.state_dict(), 
        #                f"{base_dir}/epoch-{epoch}_val_MRE-{val_mre:.4f}_val_narrow_MRE-{val_narrow_mre:.4f}.pth")

        # if not args.debug and val_narrow_mre < best_val_narrow_mre and val_mre < args.save_loss_thresh:
        #     best_val_narrow_mre = val_narrow_mre
        #     best_val_narrow_ls  = [epoch, val_mre, val_narrow_mre]
        #     if args.model_name == 'origin':
        #         save_model = model.module
        #     torch.save(save_model.module.state_dict(), 
        #                f"{base_dir}/epoch-{epoch}_val_MRE-{val_mre:.4f}_val_narrow_MRE-{val_narrow_mre:.4f}.pth")

        if not args.debug and val_mre < best_val_mre and val_mre < args.save_loss_thresh:
            best_val_mre = val_mre
            best_val_ls  = [epoch, val_mre, val_narrow_mre]
            if args.model_name == 'origin':
                torch.save(
                    model.module.state_dict(), 
                    f"{base_dir}/epoch-{epoch}_val_MRE-{val_mre:.4f}_val_narrow_MRE-{val_narrow_mre:.4f}.pth")
            else:
                torch.save(
                    model.state_dict(), 
                    f"{base_dir}/epoch-{epoch}_val_MRE-{val_mre:.4f}_val_narrow_MRE-{val_narrow_mre:.4f}.pth")

        if not args.debug and val_narrow_mre < best_val_narrow_mre and val_mre < args.save_loss_thresh:
            best_val_narrow_mre = val_narrow_mre
            best_val_narrow_ls  = [epoch, val_mre, val_narrow_mre]
            if args.model_name == 'origin':
                torch.save(
                    model.module.state_dict(), 
                    f"{base_dir}/epoch-{epoch}_val_MRE-{val_mre:.4f}_val_narrow_MRE-{val_narrow_mre:.4f}.pth")
            else:
                torch.save(
                    model.state_dict(), 
                    f"{base_dir}/epoch-{epoch}_val_MRE-{val_mre:.4f}_val_narrow_MRE-{val_narrow_mre:.4f}.pth")
        
        pbar.set_postfix({
            'train MRE': train_mre, 'val MRE': val_mre, 'val narrow MRE': val_narrow_mre, 
            'best MRE': best_val_mre, 'best narrow MRE': best_val_narrow_mre})

    if use_mlflow:
        # 最後のベストエポックのみmlflowへ保存
        mlflow.log_artifact(f"{base_dir}/epoch-{best_val_ls[0]}_val_MRE-{best_val_ls[1]:.4f}_val_narrow_MRE-{best_val_ls[2]:.4f}.pth")
        mlflow.log_artifact(f"{base_dir}/epoch-{best_val_narrow_ls[0]}_val_MRE-{best_val_narrow_ls[1]:.4f}_val_narrow_MRE-{best_val_narrow_ls[2]:.4f}.pth")

if __name__ == "__main__":

    now = datetime.now()

    use_mlflow = True
    # use_mlflow = False
    # use_tb = True
    use_tb = False
    
    pkl_path = '/workspace/datas/gotech_mass_datas_v1.0.0.pkl'
    
    m_df = pd.read_pickle(pkl_path)
    print('データ数: ', m_df.shape)
    
    parser = argparse.ArgumentParser(description='pig')
    parser.add_argument('--init_lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', default=1e-5)
    parser.add_argument('--num_features', type=int, default=2) # 2:w, l 
    parser.add_argument('--input_channels', type=int, default=4) # imao変更　ここでxceptionにinputする次元を決定する
    parser.add_argument('--input_feats', nargs="*", type=str, default=['img', 'depth'])
    parser.add_argument('--train_batch_size', type=int, default=40) # default 48
    parser.add_argument('--grad_accumulation_steps', type=int, default=1)
    parser.add_argument('--use_cutoff', action='store_true')
    parser.add_argument('--use_oversample', action='store_true') 
    # parser.add_argument('--use_scheduler', action='store_true')
    parser.add_argument('--use_scheduler', action='store_false') 

    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--arch', type=str, default='mobilenet, xception')
    
    parser.add_argument('--model_name', type=str, default='origin')
    # parser.add_argument('--model_name', type=str, default='eff')
    # parser.add_argument('--model_name', type=str, default='hrnet')
    # origin
    # parser.add_argument('--pretrained_weight', type=str, default='/workspace/posture_estimation/posture_estimation/posture_train_log/2022101943/epoch-464_val_loss-0.2434.pth')
    # eff
    # parser.add_argument('--pretrained_weight', type=str, default='/workspace/posture_estimation/posture_estimation/posture_train_log/2022102810/epoch-365_val_loss-0.1416.pth') # rgb
    parser.add_argument('--pretrained_weight', type=str, default='/workspace/posture_estimation/posture_estimation/posture_train_log/2022120848/epoch-322_val_loss-0.1660.pth') # rgb + Depth
    # hrnet
    # parser.add_argument('--pretrained_weight', type=str, default='/workspace/posture_estimation/posture_estimation/posture_train_log/2022110304/epoch-360_val_loss-0.1454.pth')
    # parser.add_argument('--pretrained_weight', type=str, default='/workspace/posture_estimation/epoch-473_val_loss-0.2005.pth') # rgb + depth
    parser.add_argument('--transfer', action='store_true')
    # parser.add_argument('--transfer', action='store_false')
    parser.add_argument('--specific_layer', action='store_true') 
    parser.add_argument('--fix_epoch', type=int, default=50) # 転移する場合は、その層を何epoch固定するか
    parser.add_argument('--change_optimizer', action='store_true')
    # parser.add_argument('--change_optimizer', action='store_false')
    parser.add_argument('--ft_lr', type=float, default=1e-4) # fix_epochが終わりFineTuningする際のlr  
    parser.add_argument('--ft_weight_decay', type=float, default=1e-5) # fix_epochが終わりFineTuningする際のweight_decay
    parser.add_argument('--posture_weight', nargs="*", type=float, default=[1.0, 1.0, 1.0]) # 姿勢別の損失重み
    parser.add_argument('--save_loss_thresh', type=float, default=0.11)
    
    parser.add_argument('--comment', type=str, default='gotech_v1.0.0')
    
    args = parser.parse_args()
    
    base_dir = os.path.join('mass_models', f"{now.strftime('%Y%m%d%M')}_{args.comment}")
    
    if (not args.debug) and (not os.path.exists("mass_models")):
        os.mkdir("mass_models")
    if (not args.debug) and (not os.path.exists(base_dir)):
        os.makedirs(base_dir, exist_ok=True)
    
    if use_tb:
        tb_dir = "/workspace/mass/ecopork-main/tb_logs/" + f"{now.strftime('%Y%m%d%M')}_{args.comment}"
        writer = SummaryWriter(log_dir=tb_dir)
    
    if use_mlflow:
        # mlflow
        TRACKING_URI = 'http://192.168.3.231:5000'
        mlflow.set_tracking_uri(TRACKING_URI)
        EXPERIMENT_NAME = 'gotech_train_mass'

        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        if experiment is None:  # 当該Experiment存在しないとき、新たに作成
            experiment_id = mlflow.create_experiment(name=EXPERIMENT_NAME)
        else: # 当該Experiment存在するとき、IDを取得
            experiment_id = experiment.experiment_id
        
        with mlflow.start_run(experiment_id=experiment_id) as run:
            # 実験条件(Parameters)
            mlflow.log_param('init_lr', args.init_lr)
            mlflow.log_param('weight_decay', args.weight_decay)
            mlflow.log_param('num_features', args.num_features)
            mlflow.log_param('input_channels', args.input_channels)
            mlflow.log_param('train_batch_size', args.train_batch_size)
            mlflow.log_param('grad_accumulation_steps', args.grad_accumulation_steps)
            mlflow.log_param('use_cutoff', args.use_cutoff)
            mlflow.log_param('use_oversample', args.use_oversample)
            mlflow.log_param('arch', args.arch)
            mlflow.log_param('model_name', args.model_name)
            mlflow.log_param('pretrained_weight', args.pretrained_weight)
            mlflow.log_param('transfer', args.transfer)
            mlflow.log_param('specific_layer', args.specific_layer)
            mlflow.log_param('fix_epoch', args.fix_epoch)
            mlflow.log_param('change_optimizer', args.change_optimizer)
            mlflow.log_param('ft_lr', args.ft_lr)
            mlflow.log_param('ft_weight_decay', args.ft_weight_decay)
            mlflow.log_param('posture_weight', args.posture_weight)
            mlflow.log_param('use_scheduler', args.use_scheduler)
            
            # pklファイル保存 
            mlflow.log_artifact(pkl_path)
            train(m_df, args)
    elif use_tb:
        with open(os.path.join(tb_dir, "params.json"), mode="w") as f:
            json.dump(args.__dict__, f, indent=4)
        train(m_df, args)
    else:
        train(m_df, args)