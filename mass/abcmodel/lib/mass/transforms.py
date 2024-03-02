import numpy as np
import cv2
import random
from typing import Union, Callable, List, Tuple, Sequence, Dict

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F


class Compose:
    def __init__(self, transforms: List[Callable]):
        assert isinstance(transforms, (list, tuple))
        self.transforms = transforms

    def __repr__(self):
        return "Transforms composed of [ \n" + ", \n".join([transform.__class__.__name__ for transform in self.transforms]) + "\n]"

    def __call__(self, data: Dict) -> Dict:
        for transform in self.transforms:
            data = transform(data)
        return data


class Stochastic:
    def __init__(self, transform, p=0.5):
        self.transform = transform
        assert 0 <= p <= 1
        self.p = p

    def __call__(self, data):
        if np.random.random() < self.p:
            data = self.transform(data)
        return data


class ToTensor:
    @staticmethod
    def __call__(data):
        img = data["img"]
        depth = data["depth"]
        mask = data["mask"]
        data["img"] = F.to_tensor(img)  # (h, w, c) -> (c, h ,w)
        h, w = depth.shape[-2:]
        data["depth"] = torch.from_numpy(depth).view(-1, h, w)  # (h, w) -> (1, h, w)
        data["mask"] = torch.from_numpy(mask).unsqueeze(0)  # (h, w) -> (1, h, w)
        return data

# class ToTensorWithOcc:
#     @staticmethod
#     def __call__(data):
#         img = data["img"]
#         depth = data["depth"]
#         mask = data["mask"]
#         occulusion = data["occulusion"]
        
#         data["img"] = F.to_tensor(img)  # (h, w, c) -> (c, h ,w)
#         h, w = depth.shape[-2:]
#         data["depth"] = torch.from_numpy(depth).view(-1, h, w)  # (h, w) -> (1, h, w)
#         data["mask"] = torch.from_numpy(mask).unsqueeze(0)  # (h, w) -> (1, h, w)
#         data["occulusion"] = torch.from_numpy(occulusion).unsqueeze(0)  # (h, w) -> (1, h, w)
#         return data


# 新しいW,L,MMでの平均・標準偏差の数値を使用
# get_from_pkl.ipynbで計算している
class NormalizeNewWLMM:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std
        # depth here means height
        self.depth_mean = 0.3
        self.depth_std = 0.15
        # sre
        self.w_mean = 325.2555796499227
        self.l_mean = 1052.9265912677627
        self.w_std = 54.6550448909098
        self.l_std = 121.04773707855092

    def __call__(self, data):
        img, depth, mm_squared, w, l = data["img"], data["depth"], data["mm_squared"], data["w"], data["l"]
        img = F.normalize(img, mean=self.mean, std=self.std)
        depth = depth / 1000  # mm -> m
        depth = (depth - self.depth_mean) / self.depth_std
        mm_squared = (mm_squared - self.mm_squared_mean) / self.mm_squared_std
        w = (w - self.w_mean) / self.w_std
        l = (l - self.l_mean) / self.l_std
        data["img"], data["depth"], data["mm_squared"], data["w"], data["l"] = img, depth, mm_squared, w, l
        return data


class NormalizeWithoutMM:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std
        # depth here means height
        self.depth_mean = 0.3
        self.depth_std = 0.15

        # sre_mass_datas_v1.0.0
        # self.w_mean = 325.2555796499227
        # self.l_mean = 1052.9265912677627
        # self.w_std = 54.6550448909098
        # self.l_std = 121.04773707855092

        # sre_mass_datas_v1.0.0.b
        # self.w_mean = 322.9808416717679
        # self.l_mean = 1047.8624542051416
        # self.w_std = 53.515615800443754
        # self.l_std = 119.4599636401279

        # sre_mass_datas_v1.1.0
        # self.w_mean = 323.38341476082616
        # self.l_mean = 1050.0655004282264
        # self.w_std = 53.46560607947578
        # self.l_std = 119.94428887525501

        # sre_mass_datas_v1.1.1
        # self.w_mean = 324.4177917021983
        # self.l_mean = 1050.6917881280629
        # self.w_std = 53.87844938477692
        # self.l_std = 120.72323637499046

        # sre_mass_datas_v1.2.0_no_refinement
        # self.w_mean = 323.1858734945161
        # self.l_mean = 1047.7094068464025
        # self.w_std = 53.60293870623994
        # self.l_std = 118.72038384091769

        # sre_mass_datas_v1.2.0_with_refinement
        # self.w_mean = 332.663406209343
        # self.l_mean = 1060.1993333286512
        # self.w_std = 54.03452250484248
        # self.l_std = 122.27515713578559

        # sre_mass_datas_v1.2.1
        # self.w_mean = 332.85864410663675
        # self.l_mean = 1060.0336697170337
        # self.w_std = 54.297850155703145
        # self.l_std = 122.91788113298357

        # gotech_mass_datas_v1.0.0
        # self.w_mean = 328.7144075682382
        # self.l_mean = 1044.945099255583
        # self.w_std = 113.59666471821626
        # self.l_std = 347.9555463567175

        # sre_mass_datas_dropped_v1.3.0
        # self.w_mean = 335.99956519820114
        # self.l_mean = 1073.1959102655112
        # self.w_std = 53.06689074907414
        # self.l_std = 120.81063142502664

        # sre_mass_datas_m_v1.0.0
        # self.w_mean = 397.66073853234974
        # self.l_mean = 1109.1888981628
        # self.w_std = 196.89537742030595
        # self.l_std = 346.44564679769854

        # sre_mass_datas_m_v1.0.0 with occulusion
        # self.w_mean = 390.5186361830326
        # self.l_mean = 1101.5475160979177
        # self.w_std = 193.58125721969594
        # self.l_std = 345.7099357268214

        # # sre_mass_datas_gotech_v1.0.0
        # self.w_mean = 348.58716446056496
        # self.l_mean = 1118.2932361263852
        # self.w_std = 119.20425956721147
        # self.l_std = 355.70758289766763

        # sre_mass_datas_gotech_v1.0.1
        # self.w_mean = 387.08432497689245
        # self.l_mean = 1113.5693546013115
        # self.w_std = 175.88166198533713
        # self.l_std = 357.38714630306606

         # sre_mass_datas_dropped_v1.3.0
        # self.w_mean = 335.99956519820114
        # self.l_mean = 1073.1959102655112
        # self.w_std = 53.06689074907414
        # self.l_std = 120.81063142502664

        #  sre_mass_datas_dropped_v1.3.xxxx
        self.w_mean = 336.0124369557878
        self.l_mean = 1073.2730922282576
        self.w_std = 53.061150287393595
        self.l_std = 120.79544749025756


    def __call__(self, data):
        img, depth, w, l = data["img"], data["depth"], data["w"], data["l"]
        img = F.normalize(img, mean=self.mean, std=self.std)
        depth = depth / 1000  # mm -> m
        depth = (depth - self.depth_mean) / self.depth_std
        w = (w - self.w_mean) / self.w_std
        l = (l - self.l_mean) / self.l_std
        data["img"], data["depth"], data["w"], data["l"] = img, depth, w, l
        return data


class Normalize:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std
        # depth here means height
        self.depth_mean = 0.3
        self.depth_std = 0.15

        # araya
        # self.mm_squared_mean = 93503.83962203018
        # self.mm_squared_std = 42001.485982506514
        # self.w_mean = 461.9149284374276
        # self.l_mean = 387.747688094784
        # self.w_std = 162.475755689894
        # self.l_std = 131.98252992339118

        # for araya sre_mass_datas_v1.1.0
        self.mm_squared_mean = 92866.54865219438
        self.mm_squared_std = 41423.371711927655
        self.w_mean = 461.29078172740714
        self.l_mean = 385.7286505868813
        self.w_std = 161.61257838213706
        self.l_std = 131.31876030061392

    def __call__(self, data):
        img, depth, mm_squared, w, l = data["img"], data["depth"], data["mm_squared"], data["w"], data["l"]
        img = F.normalize(img, mean=self.mean, std=self.std)
        depth = depth / 1000  # mm -> m
        depth = (depth - self.depth_mean) / self.depth_std
        mm_squared = (mm_squared - self.mm_squared_mean) / self.mm_squared_std
        w = (w - self.w_mean) / self.w_std
        l = (l - self.l_mean) / self.l_std
        data["img"], data["depth"], data["mm_squared"], data["w"], data["l"] = img, depth, mm_squared, w, l
        return data


def normalize(img, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    return F.normalize(img, mean=mean, std=std)


class ColorJitter:
    def __init__(self, brightness=(0.2, 1.5), contrast=0.5, saturation=0.5, hue=0.5):
        self.jitter = T.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    def __call__(self, data):
        data["img"] = self.jitter(data["img"])  # (c, h, w)
        return data


class HorizontalFlip:
    @staticmethod
    def __call__(data):
        data["img"] = torch.flip(data["img"], dims=(-1,))
        data["depth"] = torch.flip(data["depth"], dims=(-1,))
        data["mask"] = torch.flip(data["mask"], dims=(-1,))
        return data


class VerticalFlip:
    @staticmethod
    def __call__(data):
        data["img"] = torch.flip(data["img"], dims=(-2,))
        data["depth"] = torch.flip(data["depth"], dims=(-2,))
        data["mask"] = torch.flip(data["mask"], dims=(-2,))
        return data


class PadToSquare:
    def __init__(self, size: int = 224):
        self.size = size

    def __call__(self, data):
        img, depth, mask = data["img"], data["depth"], data["mask"]
        assert img.shape[-2:] == depth.shape[-2:]
        img = self.pad(img)
        depth = self.pad(depth)
        mask = self.pad(mask)
        data["img"], data["depth"], data["mask"] = img, depth, mask
        return data

    @staticmethod
    def pad(img: torch.Tensor, size: int = 224):
        orig_size = img.shape[-2:]

        d_w = size - orig_size[1]
        d_h = size - orig_size[0]
        top, bottom = d_h // 2, d_h - (d_h // 2)
        left, right = d_w // 2, d_w - (d_w // 2)
        new = F.pad(img, padding=[left, top, right, bottom], padding_mode="constant", fill=0)

        assert new.shape[-2:] == (size, size), new.shape
        return new


class ResizeAndPadToSquare:
    def __init__(self, size: int = 224, img_fill_value=0, depth_fill_value=300):
        self.size = size
        self.img_fill_value = img_fill_value
        self.depth_fill_value = depth_fill_value

    def __call__(self, data):
        img, depth, mask = data["img"], data["depth"], data["mask"]
        assert img.shape[-2:] == depth.shape[-2:]
        img = self.resize_and_pad(img, fill_value=self.img_fill_value, interpolation=F.InterpolationMode.BILINEAR)
        depth = self.resize_and_pad(depth, fill_value=self.depth_fill_value, interpolation=F.InterpolationMode.BILINEAR)
        mask = self.resize_and_pad(mask, fill_value=0, interpolation=F.InterpolationMode.NEAREST)
        data["img"], data["depth"], data["mask"] = img, depth, mask
        return data

    @staticmethod
    def resize_and_pad(img: torch.Tensor, fill_value: Union[Sequence, float], interpolation: F.InterpolationMode, size: int = 224):
        orig_size = img.shape[-2:]
        ratio = size / max(orig_size)
        new_h, new_w = tuple([int(i * ratio) for i in orig_size])
        new = T.Resize(size=(new_h, new_w), interpolation=interpolation)(img)

        d_w = size - new_w
        d_h = size - new_h
        top, bottom = d_h // 2, d_h - (d_h // 2)
        left, right = d_w // 2, d_w - (d_w // 2)
        new = F.pad(new, padding=[left, top, right, bottom], padding_mode="constant", fill=fill_value)
        return new


# Lの大きさに応じてResizeする前処理
class ResizeAndPadToSquareByL:
    def __init__(self, size: int = 224, img_fill_value=0, depth_fill_value=300):
        self.size = size
        self.img_fill_value = img_fill_value
        self.depth_fill_value = depth_fill_value

    def __call__(self, data):
        img, depth, mask, resize_ratio = data["img"], data["depth"], data["mask"], data["resize_ratio"]
        assert img.shape[-2:] == depth.shape[-2:]
        img = self.resize_and_pad(img, fill_value=self.img_fill_value, interpolation=F.InterpolationMode.BILINEAR, resize_size=int(self.size * resize_ratio))
        depth = self.resize_and_pad(depth, fill_value=self.depth_fill_value, interpolation=F.InterpolationMode.BILINEAR, resize_size=int(self.size * resize_ratio))
        mask = self.resize_and_pad(mask, fill_value=0, interpolation=F.InterpolationMode.NEAREST, resize_size=int(self.size * resize_ratio))
        data["img"], data["depth"], data["mask"] = img, depth, mask
        return data

    @staticmethod
    def resize_and_pad(img: torch.Tensor, fill_value: Union[Sequence, float], interpolation: F.InterpolationMode, resize_size: int = 224, padding_size: int = 224):
        orig_size = img.shape[-2:]
        ratio = resize_size / max(orig_size)
        new_h, new_w = tuple([int(i * ratio) for i in orig_size])
        new = T.Resize(size=(new_h, new_w), interpolation=interpolation)(img)

        d_w = padding_size - new_w
        d_h = padding_size - new_h
        top, bottom = d_h // 2, d_h - (d_h // 2)
        left, right = d_w // 2, d_w - (d_w // 2)
        new = F.pad(new, padding=[left, top, right, bottom], padding_mode="constant", fill=fill_value)
        return new


# random_resize
class RandomResizeAndPadToSquare:
    def __init__(self, size: int = 224, img_fill_value=127 / 255, depth_fill_value=300):
        self.size = size
        self.img_fill_value = img_fill_value
        self.depth_fill_value = depth_fill_value

    def __call__(self, data):
        img, depth, mask = data["img"], data["depth"], data["mask"]
        assert img.shape[-2:] == depth.shape[-2:]
        img = self.resize_and_pad(img, fill_value=self.img_fill_value, interpolation=F.InterpolationMode.BILINEAR)
        depth = self.resize_and_pad(depth, fill_value=self.depth_fill_value, interpolation=F.InterpolationMode.BILINEAR)
        mask = self.resize_and_pad(mask, fill_value=0, interpolation=F.InterpolationMode.NEAREST)
        data["img"], data["depth"], data["mask"] = img, depth, mask
        return data

    @staticmethod
    def resize_and_pad(img: torch.Tensor, fill_value: Union[Sequence, float], interpolation: F.InterpolationMode, size: int = 224):
        orig_size = img.shape[-2:]
        max_ratio = size / max(orig_size)
        min_ratio = max_ratio * 0.8
        ratio = random.uniform(min_ratio, max_ratio)
        new_h, new_w = tuple([int(i * ratio) for i in orig_size])
        new = T.Resize(size=(new_h, new_w), interpolation=interpolation)(img)

        d_w = size - new_w
        d_h = size - new_h
        top, bottom = d_h // 2, d_h - (d_h // 2)
        left, right = d_w // 2, d_w - (d_w // 2)
        new = F.pad(new, padding=[left, top, right, bottom], padding_mode="constant", fill=fill_value)
        return new


class ResizeToSquare:
    def __init__(self, size: int = 224):
        self.size = [size, size]

    def __call__(self, data):
        img, depth, mask = data["img"], data["depth"], data["mask"]
        assert img.shape[-2:] == depth.shape[-2:]
        img = F.resize(img, size=self.size, interpolation=T.InterpolationMode.BILINEAR)
        depth = F.resize(depth, size=self.size, interpolation=T.InterpolationMode.BILINEAR)
        mask = F.resize(mask, size=self.size, interpolation=T.InterpolationMode.NEAREST)
        data["img"], data["depth"], data["mask"] = img, depth, mask
        return data


class RandomSafeRotate:
    """Rotate image, depth, mask together with w & l in such a way that the rotated image fits in the new image
    without cropping original image
    """

    def __init__(self, degrees=(-40, 40)):
        self.degree_min, self.degree_max = degrees

    def __call__(self, data):
        degree = random.uniform(self.degree_min, self.degree_max)
        img, depth, mask, w, l = data["img"], data["depth"], data["mask"], data["w"], data["l"]
        orig_h, orig_w, _ = img.shape
        mat, (new_h, new_w) = self.calculate_affine_matrix(degree, (orig_h, orig_w))
        img = cv2.warpAffine(img, mat, (new_w, new_h))
        depth = cv2.warpAffine(depth, mat, (new_w, new_h))
        mask = cv2.warpAffine(np.array(mask, dtype=np.uint8), mat, (new_w, new_h)).astype(np.bool_)
        w = w / orig_w * new_w
        l = l / orig_h * new_h
        data["img"], data["depth"], data["mask"], data["w"], data["l"] = img, depth, mask, w, l
        data["img"], data["depth"], data["mask"] = img, depth, mask
        return data

    @staticmethod
    def calculate_affine_matrix(degree, image_shape):
        orig_h, orig_w, *_ = image_shape
        rad = np.radians(degree)
        w_new = np.round(orig_w * abs(np.cos(rad)) + orig_h * abs(np.sin(rad)))
        h_new = np.round(orig_w * abs(np.sin(rad)) + orig_h * abs(np.cos(rad)))
        mat = cv2.getRotationMatrix2D((orig_w / 2, orig_h / 2), degree, 1)
        mat[0, 2] += -orig_w / 2 + w_new / 2
        mat[1, 2] += -orig_h / 2 + h_new / 2

        return mat, (int(h_new), int(w_new))


class NewRandomSafeRotate:
    """Rotate image, depth, mask together with w & l in such a way that the rotated image fits in the new image
    without cropping original image
    """

    def __init__(self, degrees=(-40, 40)):
        self.degree_min, self.degree_max = degrees

    def __call__(self, data):
        degree = random.uniform(self.degree_min, self.degree_max)
        img, depth, mask = data["img"], data["depth"], data["mask"]
        orig_h, orig_w, _ = img.shape
        mat, (new_h, new_w) = self.calculate_affine_matrix(degree, (orig_h, orig_w))
        img = cv2.warpAffine(img, mat, (new_w, new_h))
        depth = cv2.warpAffine(depth, mat, (new_w, new_h))
        mask = cv2.warpAffine(np.array(mask, dtype=np.uint8), mat, (new_w, new_h)).astype(np.bool_)
        data["img"], data["depth"], data["mask"] = img, depth, mask
        return data

    @staticmethod
    def calculate_affine_matrix(degree, image_shape):
        orig_h, orig_w, *_ = image_shape
        rad = np.radians(degree)
        w_new = np.round(orig_w * abs(np.cos(rad)) + orig_h * abs(np.sin(rad)))
        h_new = np.round(orig_w * abs(np.sin(rad)) + orig_h * abs(np.cos(rad)))
        mat = cv2.getRotationMatrix2D((orig_w / 2, orig_h / 2), degree, 1)
        mat[0, 2] += -orig_w / 2 + w_new / 2
        mat[1, 2] += -orig_h / 2 + h_new / 2

        return mat, (int(h_new), int(w_new))


class NewGaussianMaskSmoothing:  #
    def __init__(self, ksize=(7, 7), sigma_x=3, sigma_y=3):
        self.ksize = ksize
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y

    # mm_squareにおいては、GaussianMaskSmoothingが適応済のmaskから計算した値が既にpklファイルに格納されているのでここでは計算しない
    def __call__(self, data):
        mask = cv2.GaussianBlur(data["mask"], ksize=self.ksize, sigmaX=self.sigma_x, sigmaY=self.sigma_y)
        data["mask"] = mask
        return data


class GaussianMaskSmoothing:
    def __init__(self, ksize=(7, 7), sigma_x=3, sigma_y=3):
        self.ksize = ksize
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y

    def __call__(self, data):
        mask = cv2.GaussianBlur(data["mask"], ksize=self.ksize, sigmaX=self.sigma_x, sigmaY=self.sigma_y)
        mm_squared = data["w"] * data["l"] * (mask.sum() / np.prod(mask.shape))
        data["mask"], data["mm_squared"] = mask, mm_squared
        return data


class CropErasing:
    def __init__(self):
        self.transform = alb.ShiftScaleRotate(
            shift_limit=0.3,
            scale_limit=0,
            rotate_limit=0,
            p=0.5,
            border_mode=cv2.BORDER_CONSTANT,
        )
        # self.transform = alb.Cutout(p=1, max_h_size=20, max_w_size=20)

    def __call__(self, data):
        orig_mask = data["mask"]
        mask = self.transform(image=orig_mask)["image"]
        data["mask"] = mask

        cutoff_rate = np.sum(mask) / np.sum(orig_mask)
        data["cutoff_ratio"] = cutoff_rate

        return data


class OriginalErasing:
    def __init__(self):
        pass

    def __call__(self, data):
        img = data["img"]
        depth = data["depth"]
        mask = data["mask"]

        # sidecut erasing
        mask_shape = mask.shape
        rand = np.random.randint(4)
        if rand == 0:
            rand_cut = np.random.randint(1, mask_shape[0] // 2)
            mask = mask[rand_cut:, :]
            img = img[rand_cut:, :]
            depth = depth[rand_cut:, :]
        elif rand == 1:
            rand_cut = np.random.randint(1, mask_shape[0] // 2)
            mask = mask[:-rand_cut, :]
            img = img[:-rand_cut, :]
            depth = depth[:-rand_cut, :]
        elif rand == 2:
            rand_cut = np.random.randint(1, mask_shape[1] // 2)
            mask = mask[:, rand_cut:]
            img = img[:, rand_cut:]
            depth = depth[:, rand_cut:]
        elif rand == 3:
            rand_cut = np.random.randint(1, mask_shape[1] // 2)
            mask = mask[:, :-rand_cut]
            img = img[:, :-rand_cut]
            depth = depth[:, :-rand_cut]

        data["img"] = img
        data["depth"] = depth

        # trim erasing
        pad_x = mask.shape[0] // 2
        pad_y = mask.shape[1] // 2
        rand = np.random.randint(2)
        rand_pad = np.random.randint(pad_x, mask_shape[0], size=1)[0]
        if rand == 0:
            pad_x = (rand_pad, 0)
        elif rand == 1:
            pad_x = (0, rand_pad)

        rand = np.random.randint(2)
        rand_pad = np.random.randint(pad_y, mask_shape[1], size=1)[0]
        if rand == 0:
            pad_y = (rand_pad, 0)
        elif rand == 1:
            pad_y = (0, rand_pad)

        trim_mask = np.pad(mask, (pad_x, pad_y))

        if pad_x[0] > 0:
            trim_mask = trim_mask[: mask.shape[0]]
        if pad_x[1] > 0:
            trim_mask = trim_mask[-mask.shape[0] :]
        if pad_y[0] > 0:
            trim_mask = trim_mask[:, : mask.shape[1]]
        if pad_y[1] > 0:
            trim_mask = trim_mask[:, -mask.shape[1] :]

        trim_mask += 1
        trim_mask %= 2

        data["cutoff_ratio"] = np.sum(mask * trim_mask) / np.sum(data["mask"])
        data["img"] = img * trim_mask[:, :, np.newaxis]
        data["depth"] = depth * trim_mask
        data["mask"] = mask * trim_mask

        return data


class RandomBlur:
    def __init__(self, ksizes=(2, 50)):
        self.ksize_min, self.ksize_max = ksizes

    def __call__(self, data):
        ksize = random.randint(self.ksize_min, self.ksize_max)
        img = cv2.blur(data["img"], ksize=(ksize, ksize))
        data["img"] = img

        return data


class OcculusionToOnehot:
    def __init__(self, n_classes=4):
        self.n_classes = n_classes

    def __call__(self, data):

        occulusion = data["occulusion"]
        onehot_oocculusion = list(map(int, list(occulusion)))  # 文字列を2ビットのバイナリ表現に変換
        data["occulusion"] = onehot_oocculusion

        return data


train_transforms = Compose(
    [
        # OriginalErasing(),
        GaussianMaskSmoothing(),
        RandomSafeRotate(degrees=(-40, 40)),
        ToTensor(),
        ColorJitter(),
        Stochastic(VerticalFlip(), p=0.5),
        Stochastic(HorizontalFlip(), p=0.5),
        PadToSquare(),  # ResizeToSquare(),  # ResizeAndPadToSquare(),  # resize to square worked slightly better
        Normalize(),
    ]
)

val_transforms = Compose(
    [
        # OriginalErasing(),
        GaussianMaskSmoothing(),
        ToTensor(),
        PadToSquare(),  # ResizeToSquare(),  # ResizeAndPadToSquare(),  # resize to square worked slightly better
        Normalize(),
    ]
)

# new W, Lを使用し、mm_squaredを排除した前処理
witouht_mm_train_transforms = Compose(
    [
        # OriginalErasing(),
        NewGaussianMaskSmoothing(),  # GaussianMaskSmoothingを変更
        # RandomBlur(),
        NewRandomSafeRotate(degrees=(-40, 40)),
        ToTensor(),
        ColorJitter(),
        Stochastic(VerticalFlip(), p=0.5),
        Stochastic(HorizontalFlip(), p=0.5),
        # ResizeAndPadToSquareByL(),
        # ResizeAndPadToSquare(),
        PadToSquare(),
        # RandomResizeAndPadToSquare(),
        NormalizeWithoutMM(),
    ]
)

without_mm_val_transforms = Compose(
    [
        # OriginalErasing(),
        NewGaussianMaskSmoothing(),
        ToTensor(),
        # ResizeAndPadToSquareByL(),
        PadToSquare(),
        # ResizeAndPadToSquare(),
        NormalizeWithoutMM(),
    ]
)


# new W, Lを使用し、mm_squaredを排除した前処理
with_occ_train_transforms = Compose(
    [
        # OriginalErasing(),
        NewGaussianMaskSmoothing(),  # GaussianMaskSmoothingを変更
        # RandomBlur(),
        NewRandomSafeRotate(degrees=(-40, 40)),
        OcculusionToOnehot(),
        ToTensor(),
        ColorJitter(),
        Stochastic(VerticalFlip(), p=0.5),
        Stochastic(HorizontalFlip(), p=0.5),
        # ResizeAndPadToSquareByL(),
        ResizeAndPadToSquare(),
        # RandomResizeAndPadToSquare(),
        NormalizeWithoutMM(),
    ]
)

with_occ_val_transforms = Compose(
    [
        # OriginalErasing(),
        NewGaussianMaskSmoothing(),
        OcculusionToOnehot(),
        ToTensor(),
        # ResizeAndPadToSquareByL(),
        # PadToSquare(),
        ResizeAndPadToSquare(),
        NormalizeWithoutMM(),
    ]
)

exp_without_resize_transforms = Compose(
    [
        # OriginalErasing(),
        NewGaussianMaskSmoothing(),
        ToTensor(),
        PadToSquare(),  # ResizeToSquare(),  # ResizeAndPadToSquare(),  # resize to square worked slightly better
    ]
)

exp_with_resize_transforms = Compose(
    [
        # OriginalErasing(),
        NewGaussianMaskSmoothing(),
        ToTensor(),
        ResizeAndPadToSquare(),  # ResizeToSquare(),  # ResizeAndPadToSquare(),  # resize to square worked slightly better
    ]
)

exp_bbox_l_resize_transforms = Compose(
    [
        # OriginalErasing(),
        NewGaussianMaskSmoothing(),
        ToTensor(),
        ResizeAndPadToSquareByL(),  # ResizeToSquare(),  # ResizeAndPadToSquare(),  # resize to square worked slightly better
    ]
)
