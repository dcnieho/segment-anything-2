import pandas as pd
import numpy as np
import compress_pickle
import pathlib
import natsort
import cv2
import pebble
import os
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


def evaluate_segments(subject, preds_paths, gt_video, save_csv_dir=None, save_csv_fname=None, ori_stim_fold=False):

    if ori_stim_fold:
        mask_fold = save_csv_dir / 'masks'
        mask_fold.mkdir(parents=True, exist_ok=True)
        files = os.listdir(ori_stim_fold)
        files.sort(key=natsort.os_sort_keygen())
        blueImg = None
        redImg = None

    cap = cv2.VideoCapture(gt_video)
    frame_idx = -1
    next_chunk_idx = 0
    pred = None
    pred_max_fr = -1
    all_scores_list = []
    while True:
        ret, frame = cap.read()
        frame_idx+=1
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, mask_true = cv2.threshold(gray_frame, 127, 1, cv2.THRESH_BINARY)
        mask_true = mask_true>0

        if pred_max_fr<frame_idx:
            if next_chunk_idx>=len(preds_paths):
                break
            print(f'{subject}: chunk {next_chunk_idx}')
            pred = compress_pickle.load(preds_paths[next_chunk_idx])
            next_chunk_idx += 1
            pred_max_fr = max(pred.keys())

        mask_pred = pred[frame_idx][0].squeeze()
        
        scores = get_scores(mask_pred, mask_true)

        all_scores_list.append({
            'subject': subject,
            'frame': frame_idx,
            'blink_detected': scores['blink_detected'],
            'pupil_lost': scores['pupil_lost'],
            'iou': scores['iou']
        })

        if ori_stim_fold:
            with open(ori_stim_fold/files[frame_idx],'rb') as img_file:
                img = Image.open(img_file).convert("RGB")
            # add output mask
            img = np.array(img)
            if blueImg is None:
                blueImg = np.zeros(img.shape, img.dtype)
                blueImg[:,:] = (0, 0, 255)
                redImg = np.zeros(img.shape, img.dtype)
                redImg[:,:] = (255, 0, 0)
            blueMask = cv2.bitwise_and(blueImg, blueImg, mask=np.uint8(mask_true))
            redMask = cv2.bitwise_and(redImg, redImg, mask=np.uint8(mask_pred))
            bothMask = cv2.addWeighted(redMask, .6, blueMask, .4, 0)
            img = cv2.addWeighted(bothMask, .6, img, .4, 0)
            Image.fromarray(img).save(mask_fold / files[frame_idx])
    
        
    all_scores = pd.DataFrame.from_dict({i:s for i,s in enumerate(all_scores_list)}, orient='index')

    if save_csv_dir:
        savepath = save_csv_dir / save_csv_fname
        all_scores.to_csv(savepath, index=False, na_rep='nan', float_format="%.8f")

    return all_scores 


def summarize_scores(all_scores):
    pupil_lost_rate = all_scores['pupil_lost'].mean()
    
    # calculate rate of blink detected (exclude nan)
    blink_detected_df = all_scores[all_scores['blink_detected'].notna()]
    blink_detected_count = blink_detected_df['blink_detected'].sum()
    blink_total_count = len(blink_detected_df)
    blink_detected_rate = blink_detected_count / blink_total_count if blink_total_count > 0 else None

    # calculate miou for cases where pupil isnt lost
    iou_not_lost = all_scores['iou'].mean()

    summary = {
        'pupil_lost_rate': pupil_lost_rate,
        'blink_detected_rate': blink_detected_rate,
        'miou_not_lost': iou_not_lost
    }

    return summary


root_dir = pathlib.Path("//et-nas.humlab.lu.se/FLEX/datasets real/nvgaze_real_dataset_ar")
gt_dir   = pathlib.Path("//et-nas.humlab.lu.se/FLEX/datasets real/nvgaze_real_dataset_ar/TEyeDS/ANNOTATIONS")
base_path = pathlib.Path(f"//et-nas.humlab.lu.se/FLEX/datasets real/nvgaze_real_dataset_ar/sam2/persubject_run/")
do_plot_mask = False
def process(video):
    in_dir = base_path / video.stem
    in_gt = gt_dir / f'NVIDIAAR_{video.stem.lstrip("0")}_1.mp4pupil_seg_2D.mp4'
    out_dir = base_path.parent / 'eval'

    # if (out_dir/f'{video.stem}.csv').is_file():
    #     print(f"Already done. Skipping {video.stem}")
    #     return

    result_files = in_dir.glob("*.pickle.gz")
    result_files = natsort.natsorted(result_files)
    scores = evaluate_segments(video.stem, result_files, in_gt, save_csv_dir=out_dir, save_csv_fname=f'{video.stem}.csv', ori_stim_fold=root_dir/video.stem if do_plot_mask else None)
    score = summarize_scores(scores)
    bdr = "None" if score["blink_detected_rate"] is None else f'{score["blink_detected_rate"]:.3f}'
    print(f'{video.stem}: pupil lost rate: {score["pupil_lost_rate"]:.3f}, blink correctly detected: {bdr}, mIoU (for tracked cases): {score["miou_not_lost"]:.3f}')

if __name__ == "__main__":
    subject_folders = [f for f in root_dir.iterdir() if f.is_file() and f.suffix=='.zip']
    subject_folders = natsort.natsorted(subject_folders)

    with pebble.ProcessPool(max_workers=6) as pool:
        future = pool.map(process, subject_folders)
        future.result()
