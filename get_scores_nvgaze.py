import pandas as pd
import compress_pickle
import numpy as np
import zipfile
import pathlib
import natsort
from tqdm import tqdm
from PIL import Image



def get_scores(pred_mask, true_mask):
    blink_tracked = np.nan
    pupil_lost = False
    iou = np.nan

    if np.all(true_mask == False) and np.all(pred_mask == False):
        blink_tracked = True
    elif np.all(true_mask == False):
        blink_tracked = False
    elif np.all(pred_mask == False):  # Prediction lost track of the pupil
        pupil_lost = True
    else:
        intersection = np.logical_and(true_mask, pred_mask).sum()
        union = np.logical_or(true_mask, pred_mask).sum()

        if union > 0:
            iou = intersection / union

    scores = {
        'blink_detected': blink_tracked,
        'pupil_lost': pupil_lost,
        'iou': iou
    }
    return scores


def evaluate_segments(preds_path, labels_zip, save_csv_dir=None, save_csv_fname=None):

    with open(preds_path, 'rb') as handle:
        segments = compress_pickle.load(handle)

    all_scores = []
    for idx in tqdm(segments, desc="frames"):
        mask_pred = segments[idx][0].squeeze() > 0.5

        with labels_zip.open(f"type_maskWithoutSkin_frame_{(idx):04d}.png")  as eyeball_file:
            img = np.array(Image.open(eyeball_file).convert('RGB'))
            mask_true = (img[:, :, 1] > 150) & (img[:, :, 0] < 100) & (img[:, :, 2] < 100)

        with labels_zip.open(f"type_maskWithSkin_frame_{(idx):04d}.png")  as skin_file:
            img = np.array(Image.open(skin_file).convert('RGB'))
            skin_mask = (img[:, :, 0] > 150) & (img[:, :, 1] < 100) & (img[:, :, 2] < 100)

        mask_true = mask_true & ~skin_mask

        scores = get_scores(mask_pred, mask_true)

        all_scores.append({
            'zipfile': pathlib.Path(labels_zip.filename).parent.name,
            'frame': idx,
            'blink_detected': scores['blink_detected'],
            'pupil_lost': scores['pupil_lost'],
            'iou': scores['iou']
        })

    all_scores = pd.DataFrame.from_dict({i:s for i,s in enumerate(all_scores)}, orient='index')

    if save_csv_dir:
        savepath = save_csv_dir / save_csv_fname
        all_scores.to_csv(savepath, index=False, na_rep='nan', float_format="%.8f", mode='a', header=not savepath.is_file())

    return all_scores

def summarize_scores(all_scores):
    pupil_lost_rate = all_scores['pupil_lost'].mean()

    # calculate rate of blink detected (exclude nan)
    blink_detected_df = all_scores[all_scores['blink_detected'].notna()]
    blink_detected_count = blink_detected_df['blink_detected'].sum()
    blink_total_count = len(blink_detected_df)
    blink_detected_rate = blink_detected_count / blink_total_count if blink_total_count > 0 else None

    # calculate miou for cases where pupil isnt lost
    miou_not_lost = all_scores['iou'].mean()

    summary = {
        'pupil_lost_rate': pupil_lost_rate,
        'blink_detected_rate': blink_detected_rate,
        'miou_not_lost': miou_not_lost
    }
    return summary


if __name__ == "__main__":
    root_dir  = pathlib.Path('D:/nvgaze') # nvgaze directory
    base_path = pathlib.Path(f"//et-nas.humlab.lu.se/FLEX/datasets synthetic/nvidia/sam2/nvgaze/persubject_run2/")
    subject_folders = root_dir.rglob("*.zip") # should be a list of all zipfile paths
    subject_folders = natsort.natsorted(subject_folders)

    for zip_path in subject_folders:
        print(f'=== {zip_path.parent.name} ===')
        in_dir = base_path / zip_path.parent.name
        out_dir = base_path.parent / 'eval2'

        if (out_dir / f'{zip_path.parent.name}.csv').is_file():
            continue

        result_files = in_dir.glob("*.pickle.gz")
        with zipfile.ZipFile(zip_path, 'r') as zipf:
            for res_file in result_files:
                chunk_scores = evaluate_segments(res_file, zipf, save_csv_dir=out_dir, save_csv_fname=f'{zip_path.parent.name}.csv')
                score = summarize_scores(chunk_scores)
                bdr = "None" if score["blink_detected_rate"] is None else f'{score["blink_detected_rate"]:.3f}'
                print(f'pupil lost rate: {score["pupil_lost_rate"]:.3f}, blink correctly detected: {bdr}, mIoU (for tracked cases): {score["miou_not_lost"]:.3f}')
        scores = pd.read_csv(out_dir / f'{zip_path.parent.name}.csv')
        score = summarize_scores(chunk_scores)
        print('--total--')
        bdr = "None" if score["blink_detected_rate"] is None else f'{score["blink_detected_rate"]:.3f}'
        print(f'pupil lost rate: {score["pupil_lost_rate"]:.3f}, blink correctly detected: {bdr}, mIoU (for tracked cases): {score["miou_not_lost"]:.3f}')