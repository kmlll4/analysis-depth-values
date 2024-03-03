from pycocotools import mask

def decoder(rle):
    return mask.decode(rle)
    # cv2.imwrite("segmentation.png", dec_rle )