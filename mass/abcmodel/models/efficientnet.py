import torch
import torch.nn as nn

from torchvision.models import efficientnet_v2_l, efficientnet_v2_m, efficientnet_v2_s


class Efficientnet(nn.Module):
    def __init__(self, input_channels, transfer, pretrained_weight):
        super(Efficientnet, self).__init__()

        self.model = efficientnet_v2_l()
        # self.model = efficientnet_v2_m()
        # self.model = efficientnet_v2_s()

        # 層の削除(このやり方だとkey名が変わってしまう)
        # self.model = nn.Sequential(*(list(self.model.children())[:-1])) # dropoutとlinearを除く

        if input_channels == 4:
            self.model.features[0][0] = nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1, bias=False)
            # self.model.features[0][0] = nn.Conv2d(4, 24, kernel_size=3, stride=2, padding=1, bias=False)

        # 層の入れ替え
        self.model.classifier[0] = nn.Identity()  # dropout → identity
        self.model.classifier[1] = nn.Identity()  # fc → identity

        if transfer:
            # 各種dict
            model_dict = self.model.state_dict()
            pretrained_dict = torch.load(pretrained_weight)

            # 新しく追加した層のリスト
            # new_layer_list = [k for k in model_dict.keys() if k not in pretrained_dict]  # 恒等関数は消える？

            # model_dictに入っている要素だけのpretrained_dictを作成する
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)  # 学習済み重みがあるdictをupdate
            self.model.load_state_dict(model_dict)  # load

            # fine_tune_layer = ['features.6', 'features.7', 'features.8'] # この名前が含まれている層はupdateをTrueにする
            # all_count = 0
            # update_layer_count = 0
            # for name, param in self.model.named_parameters():
            #     all_count += 1
            #     if [s for s in fine_tune_layer if s in name]:
            #         update_layer_count += 1
            # print("new_update_layer_num = " + str(update_layer_count))
            # print("all_eff_layer_num = " + str(all_count))

            # 重みのUpdateをFalseにする
            for name, param in self.model.named_parameters():
                param.requires_grad = False
                print(name)
            print("-----------")

    def forward(self, x):
        x = self.model(x)
        return x


# test
if __name__ == "__main__":
    eff = Efficientnet()
