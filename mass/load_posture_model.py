# Posture関連
import torch
import argparse
from mass.abcmodel.models.mass_model import MassModelV3, EnsembleModel


def load_posture_model(model_name, input_channels, num_features, transfer, pretrained_weight=None, arch=['mobilenet', 'xception']):

    if model_name == 'origin':
        # 初期のmodelに姿勢推定で得られた重みを読み込む

        # モデル作成
        ensemble_models = []
        for a in arch:
            m = MassModelV3(
                backbone=a.strip(),
                input_channels=input_channels,
                num_features=num_features
            )
            ensemble_models.append(m)
        origin_model = EnsembleModel(ensemble_models)
        # origin_model = torch.nn.DataParallel(origin_model).cuda()

        # 重み読み込み
        # 各種dict
        if transfer:
            origin_model_dict = origin_model.state_dict()
            pretrained_dict = torch.load(pretrained_weight)

            # pretrained_ditcには含まれていない層のリスト
            new_layer_list = [k for k in origin_model_dict.keys() if k not in pretrained_dict]

            # origin_model_dictに入っている要素だけのpretrained_dictを作成する
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in origin_model_dict}
            origin_model_dict.update(pretrained_dict)  # 学習済み重みがあるdictをupdate
            origin_model.load_state_dict(origin_model_dict)  # load

            # 新しい層の重みだけupdateを行う
            params_to_update = []
            update_param_names = new_layer_list

            for name, param in origin_model.named_parameters():
                if name in update_param_names:
                    param.requires_grad = True
                    params_to_update.append(param)
                    print(name)
                else:
                    param.requires_grad = False
            print("-----------")
            # print(params_to_update)

        origin_model = torch.nn.DataParallel(origin_model).cuda()

        return origin_model

    elif model_name == 'eff':

        model = MassModelV3(
            backbone='efficientnet',
            input_channels=input_channels,
            num_features=num_features,
            transfer=transfer,
            pretrained_weight=pretrained_weight
        )
        return model.cuda()

    elif model_name == 'hrnet':

        model = MassModelV3(
            backbone='hrnet',
            input_channels=input_channels,
            num_features=num_features,
            transfer=transfer,
            pretrained_weight=pretrained_weight
        )
        return model.cuda()


# test
if __name__ == "__main__":

    # model_name = 'origin'
    model_name = 'eff'
    # model_name = 'hrnet'

    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--input_channels', type=int, default=4)
    parser.add_argument('--num_features', type=int, default=2)
    parser.add_argument('--arch', type=str, default='mobilenet, xception')
    parser.add_argument('--transfer', action='store_false')  # default = True

    # parser.add_argument('--model_name', type=str, default='origin')
    parser.add_argument('--model_name', type=str, default='eff')
    # parser.add_argument('--model_name', type=str, default='hrnet')
    # origin
    # parser.add_argument('--pretrained_weight', type=str, default='/workspace/posture_estimation/posture_estimation/posture_train_log/2022101943/epoch-464_val_loss-0.2434.pth')
    # eff
    parser.add_argument('--pretrained_weight', type=str, default='/workspace/posture_estimation/posture_estimation/posture_train_log/2022102018/epoch-243_val_loss-0.2363.pth')
    # hrnet
    # parser.add_argument('--pretrained_weight', type=str, default='/workspace/posture_estimation/posture_estimation/posture_train_log/2022102233/epoch-82_val_loss-0.2150.pth')

    args = parser.parse_args()

    model = load_posture_model(args.model_name, args.input_channels, args.num_features, args.transfer, args.pretrained_weight)
