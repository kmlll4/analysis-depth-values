import numpy as np
from typing import List
from collections import namedtuple
import torch
from pycocotools.coco import maskUtils
try:
    from cached_property import cached_property
except ImportError:
    cached_property = property


JOINT_LABELS = ('snout', 'head', 'r_ear', 'l_ear', 'r_shoulder', 'l_shoulder', 'r_hip', 
                'l_hip', 'tail', 'r_front_toe', 'l_front_toe', 'r_hind_toe', 'l_hind_toe')

Intrinsics = namedtuple('D455', ['fx', 'fy', 'ppx', 'ppy', 'coeffs'])(
    fx=378.51129150390625,
    fy=377.62335205078125,
    ppx=319.4342346191406,
    ppy=243.6131591796875,
    coeffs=[
        -0.05629967153072357,
        0.06505552679300308,
        -0.0009143425850197673,
        0.0002796284097712487,
        -0.02048596553504467
    ]
)


class Point:
    def __init__(self, point=None, x=None, y=None, z=None, label=None):
        if point is not None:
            self.label = point.get('label').replace(',', '')
            sep = ',' if ',' in point.get('points') else ';'
            try:
                self.x, self.y = map(float, point.get('points').split(sep))
            except ValueError:
                raise InvalidFormat
        elif (x is not None) and (y is not None) and (label is not None):
            self.x = x
            self.y = y
            self.label = label
        else:
            ValueError("Invalid argument format")
        self.z = z
        self.id = 0

    def __repr__(self):
        return f"point: {self.label} at {(self.x, self.y)}. depth: {self.z}"

    def isin(self, polygon):
        return bool(polygon.mask[int(self.y - 1), int(self.x - 1)])

    @property
    def deprojected_point(self):
        assert self.z is not None
        return deproject_pixel_to_point(pixel=[self.x, self.y], depth=self.z, distorted=True)


class BasePolygon:
    def __repr__(self):
        return f"Polygon: {self.rle}"

    @property
    def rle(self):
        raise NotImplementedError()

    @cached_property  # remove this for segmentation training due to memory leak
    def mask(self) -> np.ndarray:
        return maskUtils.decode(self.rle).squeeze()

    @property
    def area(self) -> np.ndarray:
        return maskUtils.area(self.rle).item()

    @property
    def bbox(self) -> np.ndarray:
        return maskUtils.toBbox(self.rle).squeeze()

    @property
    def center(self) -> np.ndarray:
        x0, y0, w, h = self.bbox
        return np.array([x0 + w * 0.5, y0 + h * 0.5])

    def iou(self, other):
        return maskUtils.iou([self.rle], [other.rle], [0])


class CVATPolygon(BasePolygon):
    def __init__(self, polygon, w=640, h=480, group_id=0):
        self.img_w, self.img_h = w, h
        self._polygon = polygon
        self.attributes = {}
        for attrib in self._polygon.iter('attribute'):
            self.attributes[attrib.get('name')] = attrib.text
        self.label = self._polygon.get('label')
        self.id = group_id

    @cached_property
    def rle(self):
        segmentation = list(chain.from_iterable(list(map(float, xy.split(',')))
                                                for xy in self._polygon.get('points').split(';')))
        return maskUtils.frPyObjects([segmentation], self.img_h, self.img_w)


def deproject_pixel_to_point(pixel, depth, distorted=True):
    # TODO: write test code
    x = (pixel[0] - Intrinsics.ppx) / Intrinsics.fx
    y = (pixel[1] - Intrinsics.ppy) / Intrinsics.fy
    if distorted:
        r2 = x**2 + y**2
        f = 1 + Intrinsics.coeffs[0]*r2 + Intrinsics.coeffs[1]*(r2**2) + Intrinsics.coeffs[4]*(r2**3)
        ux = x*f + 2*Intrinsics.coeffs[2]*x*y + Intrinsics.coeffs[3]*(r2 + 2*x*x)
        uy = y*f + 2*Intrinsics.coeffs[3]*x*y + Intrinsics.coeffs[2]*(r2 + 2*y*y)
        x = ux
        y = uy
    point = np.array([x, y, 1]) * depth
    return point


class TrainingPig:
    """
    TODO: make this subclass of Pig
    """
    def __init__(self, frame, polygon: CVATPolygon, points: List[Point] = None):
        self.frame = frame
        self.polygon = polygon
        self.points = points or list()
        depth_shape = self.frame.depth.shape
        for point in self.points:
            if (0 <= int(point.y) < depth_shape[0]) and (0 <= int(point.x) < depth_shape[1]):
                z = self.frame.depth[int(point.y), int(point.x)]
                setattr(point, 'z', z)
        self.id = self.polygon.attributes.get('ID')
        self.posture = self.polygon.attributes.get('posture')
        self.floor_depth = None
        self.weight = None

    def __repr__(self):
        return f"Pig in: {self.frame.filename}"

    @property
    def area(self):
        return self.polygon.area

    @property
    def bbox(self):
        return list(map(int, self.polygon.bbox))

    @property
    def mask(self):
        x0, y0, w, h = self.bbox
        return self.polygon.mask[y0:y0+h, x0:x0+w]

    @property
    def original_mask(self):
        return self.polygon.mask

    @property
    def img(self):
        x0, y0, w, h = self.bbox
        img = self.frame.img[y0:y0 + h, x0:x0 + w]
        return img

    @property
    def depth(self):
        x0, y0, w, h = self.bbox
        depth = self.frame.depth[y0:y0 + h, x0:x0 + w]
        # depth = depth * self.mask
        return depth

    @property
    def height(self):
        height = np.where(self.depth != 0, self.floor_depth - self.depth, 0)
        return height

    @property
    def h_mean(self):
        return self.height[self.mask==1].mean()

    @property
    def h_max(self):
        return self.height[self.mask==1].max()

    @property
    def h_std(self):
        return self.height[self.mask==1].std()

    @property
    def d_mean(self):
        return self.depth[self.mask==1].mean()

    @property
    def corners(self):
        x0, y0, w, h = self.bbox
        d_mean = self.d_mean
        tl = deproject_pixel_to_point(pixel=[x0, y0], depth=d_mean, distorted=True)
        br = deproject_pixel_to_point(pixel=[x0+w, y0+h], depth=d_mean, distorted=True)
        return tl, br

    @property
    def mm_squared(self):
        tl, br = self.corners
        w, h, _ = np.abs(tl - br)
        mm_squared = w * h * (self.mask.sum() / np.prod(self.mask.shape))
        return mm_squared


class Pig:
    def __init__(self, polygon: BasePolygon, img: np.ndarray, depth:np.ndarray, heatmap:np.ndarray, floor_depth: float):
        self.polygon = polygon
        self._img = img
        self._depth = depth
        self._hm = heatmap
        self.floor_depth = floor_depth

    @property
    def area(self):
        return self.polygon.area

    @property
    def bbox(self):
        return list(map(int, self.polygon.bbox))

    @property
    def mask(self):
        x0, y0, w, h = self.bbox
        return self.polygon.mask[y0:y0+h, x0:x0+w]

    @property
    def original_mask(self):
        return self.polygon.mask

    @property
    def img(self):
        x0, y0, w, h = self.bbox
        img = self._img[y0:y0 + h, x0:x0 + w]
        # img = np.where(self.mask[:, :, np.newaxis], img, self.img_fill_value)
        return img

    @property
    def depth(self):
        x0, y0, w, h = self.bbox
        depth = self._depth[y0:y0 + h, x0:x0 + w]
        # depth = depth * self.mask
        return depth

    @property
    def height(self):
        height = np.where(self.depth != 0, self.floor_depth - self.depth, 0)
        return height

    @property
    def heatmap(self):
        x0, y0, w, h = self.bbox
        heatmap = self._hm[:, y0:y0+h, x0:x0+w]
        return heatmap

    @property
    def h_mean(self):
        return self.height[self.mask==1].mean()

    @property
    def h_max(self):
        return self.height[self.mask==1].max()

    @property
    def h_std(self):
        return self.height[self.mask==1].std()

    @property
    def d_mean(self):
        return self.depth[self.mask==1].mean()

    @property
    def corners(self):
        x0, y0, w, h = self.bbox
        d_mean = self.d_mean
        tl = deproject_pixel_to_point(pixel=[x0, y0], depth=d_mean, distorted=True)
        br = deproject_pixel_to_point(pixel=[x0+w, y0+h], depth=d_mean, distorted=True)
        return tl, br

    @property
    def mm_squared(self):
        tl, br = self.corners
        w, h, _ = np.abs(tl - br)
        mm_squared = w * h * (self.mask.sum() / np.prod(self.mask.shape))
        return mm_squared

class BaseCVATDataset(torch.utils.data.Dataset):
    JOINT_LABELS = JOINT_LABELS

    def __init__(self, annFile, img_root, proper_format):
        from tqdm import tqdm
        root = ET.parse(annFile).getroot()
        img_root = Path(img_root)
        pbar = tqdm(list(root.iter('image')), leave=False)
        pbar.set_description(f"Processing {annFile}...")
        self.frames = [Frame(img, img_root, proper_format) for img in pbar]

    def __getitem__(self, idx):
        frame = self.frames[idx]
        img = frame.img
        depth = frame.depth
        keypoints = self._prepare_keypoints(frame)
        tag = self._prepare_tag(frame)
        return {'img': img, 'depth': depth, 'tag': tag, 'keypoints': keypoints}

    def __len__(self):
        return len(self.frames)

    def _prepare_keypoints(self, frame):
        keypoints = []
        for id in frame.unique_ids:
            points = np.zeros((len(self.JOINT_LABELS), 3))
            for pt in filter(lambda i: i.id == id, frame.points):
                joint_index = self.JOINT_LABELS.index(pt.label)
                points[joint_index] = np.array([pt.x, pt.y, 1])
            if np.sum(points):
                keypoints.append(points)
        keypoints = np.array(keypoints)
        return keypoints

    def _prepare_tag(self, frame):
        # 2 classes, 0 == background, 1 == pig, 2 == ear
        mask = np.zeros((3, frame.h, frame.w))
        for poly in frame.polygons:
            id = poly.id
            if poly.label == 'segmentation':
                mask[1] = np.where(poly.mask, id, mask[1])
            else:
                mask[2] = np.where(poly.mask, id, mask[2])
        return mask


class CVATDataset(BaseCVATDataset):
    def __init__(self,
                 annFile,
                 img_root,
                 lr_res = (128, 128),
                 hr_res = (256, 256),
                 transforms=None,
                 sigma=2.):
        """
        Args:
            annFile:
            img_root:
            lr_res: (lr_res:h, lr_res_w)
            hr_res: (hr_res_h, hr_res_w)
            transforms:
            sigma:
        """
        super(CVATDataset, self).__init__(annFile=annFile, img_root=img_root, proper_format=False)
        self.transforms = transforms
        self.lr_res = tuple(lr_res)
        self.hr_res = tuple(hr_res)
        self.lr_heatmap_generator = HeatmapGenerator(output_res=self.lr_res,
                                                     num_joints=len(self.JOINT_LABELS),
                                                     sigma=sigma)
        self.hr_heatmap_generator = HeatmapGenerator(output_res=self.hr_res,
                                                     num_joints=len(self.JOINT_LABELS),
                                                     sigma=sigma)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        img = frame.img.copy()
        keypoints = self._prepare_keypoints(frame)
        tag = self._prepare_tag(frame)

        if self.transforms is not None:
            img, tag, keypoints = self.transforms(img, tag, keypoints)

        mask = np.where(tag > 0, 1., 0.)
        '''
        for i, seg in enumerate(mask):
            seg = np.where(seg, i, seg)
        segmentation = seg.astype(np.long)
        '''
        segmentation = mask[2].astype(np.long) # only ear class
        lr_heatmap = self.lr_heatmap_generator(keypoints)
        hr_heatmap = self.hr_heatmap_generator(keypoints * 2)

        segmentation = torch.from_numpy(segmentation)
        lr_heatmap = torch.from_numpy(lr_heatmap)
        hr_heatmap = torch.from_numpy(hr_heatmap)
        tag = torch.from_numpy(tag)

        tag_2d = reduce(lambda x, y: torch.where(x > 0, x, y), tag)
        tag_2d = tag_2d.unsqueeze(dim=0)

        return img, segmentation, lr_heatmap, hr_heatmap, tag_2d, keypoints

    def append(self, other):
        self.frames += other.frames
        return self


class Polygon(BasePolygon):
    def __init__(self, binary_array: np.ndarray):
        self._rle = maskUtils.encode(np.asfortranarray(binary_array))

    @property
    def rle(self):
        return [self._rle]


class CVATMassDataset(BaseCVATDataset):
    def __init__(self, annFile, img_root, height_data: str, weight_data: str):
        super(CVATMassDataset, self).__init__(annFile=annFile, img_root=img_root, proper_format=True)
        import pandas as pd
        self.height_df = pd.read_excel(height_data)
        self.height_df["date"] = self.height_df["date"].dt.strftime("%Y%m%d")  # datetime to string
        self.weight_df = pd.read_excel(weight_data)
        self.pigs = []
        for frame in self.frames:
            date, camera, loc, time = frame.date, frame.camera, frame.loc, frame.time
            try:
                _df = self.height_df.query(f"date == '{date}' and camera == '{camera}' and pos == '{loc}' and time == {time}")
                floor_depth = _df["height"].iat[0]
            except IndexError:
                print(f"date == '{date}' and camera == '{camera}' and pos == '{loc}' and time == {time}")
                raise
            for gid in frame.unique_ids:
                if not gid:
                    continue
                polygon = next(filter(lambda i: (i.id == gid) & (i.label == 'segmentation'), frame.polygons))
                points = list(filter(lambda i: i.id == gid, frame.points))
                pig = TrainingPig(frame=frame, polygon=polygon, points=points)
                _df = self.weight_df.query(f"date=='{date}' and ID=='{pig.id}'")
                if pig.id not in pd.unique(_df["ID"]):
                    if pig.id is not None:
                        logging.warn(f"ID:{pig.id} in file:{frame.filename} not in dataframe")
                    continue
                weight = _df["weight"].iat[0]
                pig.floor_depth = floor_depth
                pig.weight = weight
                self.pigs.append(pig)

    def __len__(self):
        return len(self.pigs)

    def __getitem__(self, i):
        return self.pigs[i]

    def __repr__(self):
        return f"Pig Dataset. size = {len(self)}"
