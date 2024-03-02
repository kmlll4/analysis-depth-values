import tarfile
from typing import Tuple
from pathlib import Path
import cv2
import numpy as np


def read_tarfile(filename: str, depth_size: Tuple[int, int] = (240, 320), mode: 'str' = 'r:gz'):
    """
    Args:
        filename: tar.gz filename. must contain one .jpg and one .raw files.
        depth_size: depth image size
        mode: zip type.
    Returns:
        RGB image, depth
    """
    height, width = depth_size

    with tarfile.open(filename, mode=mode) as file:
        for member in file.members:
            byte = file.extractfile(member.name).read()
            member_name = Path(member.name)
            if member_name.suffix == '.jpg':
                bgr = cv2.imdecode(np.frombuffer(byte, np.uint8), cv2.IMREAD_COLOR)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            elif member_name.suffix == '.raw':
                depth = np.frombuffer(byte, np.uint16).reshape(height, width).astype(np.float32)
            else:
                print(member_name)
                raise ValueError(f"Invalid extension {member_name.suffix}")

    return rgb, depth
