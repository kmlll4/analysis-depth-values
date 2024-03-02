import numpy as np
import cv2
import torchvision.transforms.functional as F


class Normalize(object):
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    def __call__(self, image, mask, joints):
        image = F.normalize(image, mean=self.MEAN, std=self.STD)
        return image, mask, joints


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def get_affine_transform(center, scale, output_size, inv=False):
    src_w, src_h = scale
    dst_w, dst_h = output_size

    src_dir = get_dir([0, src_w * -0.5], 0)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros_like(src)
    src[0, :] = center
    src[1, :] = center + src_dir
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def resize_align(img, input_size, borderValue=(127, 127, 127), inv=True):
    h, w = img.shape[:2]
    center = np.array([int(w / 2.0 + 0.5), int(h / 2.0 + 0.5)])
    if w < h:
        w_resized = input_size
        h_resized = int((input_size * h / w + 63) // 64 * 64)
        scale_w = w
        scale_h = h_resized / w_resized * w
    else:
        h_resized = input_size
        w_resized = int((input_size * w / h + 63) // 64 * 64)
        scale_h = h
        scale_w = w_resized / h_resized * h

    size_scaled = (w_resized, h_resized)
    scale = np.array([scale_w, scale_h])

    trans = get_affine_transform(center, scale, size_scaled, inv=inv)
    img_resized = cv2.warpAffine(img, trans, size_scaled, borderValue=borderValue)

    return img_resized, trans


class InferenceTransform(Normalize):
    # def __init__(self, input_size=512):
    def __init__(self, input_size=224):
        self.input_size = input_size

    def __call__(self, image: np.ndarray):
        image, inv_t = resize_align(image, self.input_size)
        image = F.to_tensor(image)
        # Normalize
        image = F.normalize(image, mean=self.MEAN, std=self.STD)
        return image, inv_t
