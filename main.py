from analysis_depth_values import pkl_loader, decoder
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import numpy as np

def path_replace(rgb_path):
    """
    """
    new_string = rgb_path.replace("rgb", "depth")
    png_path = new_string.replace("jpg", "png")
    return png_path

def main(data_root_path, pkl_path): 
    """
    """
    df = pkl_loader(pkl_path)
    output_df = pd.DataFrame(columns=['filename', 'bbox', 'brob', 'count_zero', 'erorr', 'mean_height', 'mean_height_nonzero', 'diff'])

    with tqdm(total=df['filename'].drop_duplicates().shape[0]) as pbar:
        for filename in df['filename'].drop_duplicates():
            rgb_image = cv2.imread(f"{data_root_path}/{filename}")
            h, w, _ = rgb_image.shape

            depth_path = path_replace(filename)
            depth_image = cv2.imread(f"{data_root_path}/{depth_path}", cv2.IMREAD_UNCHANGED)
            depth_image = cv2.resize(depth_image, (w, h))

            current_df = df[df['filename'] == filename]
            for _, row in current_df.iterrows():
                floor = row["floor_depth"]
                bbox = row["bbox"]
                rle = row["polygon"].__dict__["_rle"]
                mask = decoder(rle)

                one_pig = depth_image[mask!=0]

                ### height
                height = floor - one_pig

                ### count zero
                count_zero = len(one_pig) - np.count_nonzero(one_pig)
                error = (count_zero / len(one_pig))*100

                mean_height = round(height.mean(), 2)
                mean_height_nonzero = round(height[one_pig!=0].mean(), 2)
                new_row = {'filename': filename, 'bbox': bbox, 'brob': len(one_pig), 'count_zero':count_zero, 'erorr': round(error, 2),
                           'mean_height':mean_height, 'mean_height_nonzero':mean_height_nonzero, 'diff':mean_height-mean_height_nonzero}
                output_df.loc[len(output_df)] = new_row

            pbar.update(1)

    output_df.to_csv('data.csv', index=False)

if __name__ == "__main__":
    """
    """
    data_root_path = "./data/mass_train_data_rgb_depth"
    pkl_path = "./data/latest_from_xxx_abs_path_only_floor_depth.pkl"
    main(data_root_path, pkl_path)