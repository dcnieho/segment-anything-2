import compress_pickle
import numpy as np
import pandas as pd
import zipfile
import pathlib
import natsort
from tqdm import tqdm
from PIL import Image
import random
import cv2




if __name__ == "__main__":
    n_rand = 20
    root_dir = r"D:/OpenSFEDS/Static"
    base_path = pathlib.Path(f"//et-nas.humlab.lu.se/FLEX/datasets synthetic/OpenSFEDS/sam2/persubject_run/")
    subject_folders = list(pathlib.Path(root_dir).rglob("*.zip")) # should be a list of all zipfile paths
    subject_folders = natsort.natsorted(subject_folders)

    for zip_path in subject_folders:
        print(f'=== {zip_path.name} ===')
        in_dir = base_path / zip_path.name

        out_file = (base_path/zip_path.name).with_suffix('.tsv')
        if out_file.is_file():
            print(f"Already done. Skipping {zip_path.name}")
            continue

        in_file = in_dir / 'segments_0.pickle.gz'
        if not in_file.is_file():
            continue
        with zipfile.ZipFile(zip_path, 'r') as zipf:
            file_names = zipf.namelist()

            plot_frames = random.sample(file_names, n_rand)

            with open(in_file, 'rb') as handle:
                segments = compress_pickle.load(handle)

            output = []
            for idx in tqdm(segments, desc="frames"):
                mask = segments[idx][0].squeeze() > 0.5

                contours = list(cv2.findContours(np.uint8(mask), 0, 2)[0])
                contours.sort(key=cv2.contourArea, reverse=True)
                if contours:
                    M = cv2.moments(contours[0])
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                else:
                    cx=cy=np.nan
                output.append({
                    'zip': zip_path.name,
                    'file': segments[idx]['image_file'],
                    'cx': cx,
                    'cy': cy
                })

                if segments[idx]['image_file'] in plot_frames:
                    with zipf.open(segments[idx]['image_file']) as img_file:
                        img = Image.open(img_file).convert("RGB")

                    # add output mask
                    img = np.array(img)
                    blueImg = np.zeros(img.shape, img.dtype)
                    blueImg[:,:] = (0, 0, 255)
                    blueMask = cv2.bitwise_and(blueImg, blueImg, mask=np.uint8(mask))
                    img = cv2.addWeighted(blueMask, .6, img, .4, 0)
                    if not np.isnan(cx):
                        img = cv2.circle(img, (cx,cy), 1, (255, 0, 0), 3)
                    Image.fromarray(img).save(pathlib.Path(in_dir) / f'output_{segments[idx]["image_file"].split(".")[0]}.png')

            all_scores = pd.DataFrame.from_dict({i:s for i,s in enumerate(output)}, orient='index')
            all_scores.to_csv(out_file, index=False, na_rep='nan', float_format="%.8f", sep='\t')
                