import os
os.environ['TORCH_CUDNN_SDPA_ENABLED'] = '1'
import numpy as np
import torch
import pickle
import compress_pickle
import pandas as pd
import pathlib
from PIL import Image
import traceback
import logging
import gc
import natsort
import cv2
from sam2.build_sam import build_sam2_video_predictor


# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def propagate(predictor, inference_state, chunk_size, save_path=None, prompt=None):
    # run propagation throughout the video and collect the results in a dict
    video_segments = {}  # video_segments contains the per-frame segmentation results
    i = 0
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        i += 1
        video_segments[out_frame_idx] = {out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy() for i, out_obj_id in enumerate(out_obj_ids)}
        if prompt is not None and i==prompt['frame'] and save_path:
            img = inference_state["images"].get_frame(out_frame_idx)
            img = img.permute(1,2,0).numpy()
            img_min, img_max = img.min(), img.max()
            img = (img-img_min)/(img_max-img_min)
            img = Image.fromarray(np.uint8(img*255)).resize((inference_state["images"].video_width, inference_state["images"].video_height))
            img.save(pathlib.Path(save_path) / f'frame{out_frame_idx}.png')
            # add output mask
            img = np.array(img)
            blueImg = np.zeros(img.shape, img.dtype)
            blueImg[:,:] = (0, 0, 255)
            blueMask = cv2.bitwise_and(blueImg, blueImg, mask=np.uint8(video_segments[out_frame_idx][0].squeeze() > 0.5))
            img = cv2.addWeighted(blueMask, .6, img, .4, 0)
            if prompt is not None:
                p=[int(x) for x in prompt['prompt']['pupil']['points'].flatten()]
                img = cv2.circle(img, (p[0], p[1]), 1, (255, 0, 0), 3)
            Image.fromarray(img).save(pathlib.Path(save_path) / f'frame{out_frame_idx}_mask.png')
        if i%chunk_size == 0:
            yield video_segments
            video_segments.clear()
    return video_segments

def add_pupil_prompt(predictor, inference_state, prompts, ann_frame_index=0):
    ann_obj_id = 0  # give a unique id to each object we interact with (it can be any integers)

    points = prompts['pupil']['points']
    labels = prompts['pupil']['labels']
    box = prompts['pupil']['box']
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_index,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,
        box=box
    )

def retrieve_prompt_from_subject(video_path, gt_dir):
    # get specific subject prompt from video_dir
    gt_file = gt_dir / f'{video_path.name}pupil_eli.txt'
    gt = pd.read_csv(gt_file, sep=';')
    valid = gt['CENTER X'] != -1
    if valid.iloc[0]:
        fr_idx = 0
    else:
        # find first run of valid values that is long enough
        a = (valid.diff(1) != 0).astype('int').cumsum()
        a[~valid] = -1
        b = a.groupby(a).size()
        long_enough = np.where(np.logical_and(b>10,b.index!=-1))[0][0]
        fr_idxs = np.where(a==b.index[long_enough])[0]
        fr_idx = fr_idxs[4] # take a few frames in

    return {
        'prompt': {
            'pupil': {
                'points': np.array([[gt['CENTER X'].iloc[fr_idx], gt['CENTER Y'].iloc[fr_idx]]]),
                'labels': np.array([1]),
                'box': None
            },
            'iris': {
                'points': None,
                'labels': None,
                'box': None
            },
            'sclera': {
                'points': None,
                'labels': None,
                'box': None
            }
        },
        'frame': gt['FRAME'].iloc[fr_idx]-1  # TEyeDS frame numbers are 1-based, we use zero-based
    }


if __name__ == '__main__':
    from_sample = 'persubject_run'
    this_dataset = 'giw'

    # Output path for results and backup
    output_bin = pathlib.Path(f"//et-nas.humlab.lu.se/FLEX/datasets synthetic/nvidia/sam2/{this_dataset}/{from_sample}/") # will contain saved masks
    backup_bin = output_bin / 'backup'
    output_bin.mkdir(parents=True, exist_ok=True)
    backup_bin.mkdir(parents=True, exist_ok=True)

    # Path containing the videos (zip files or subdirectory of videos)
    root_dir = pathlib.Path(r"D:\GIW\TEyeDS\processed")
    gt_dir   = pathlib.Path(r"D:\GIW\TEyeDS\ANNOTATIONS")

    subject_folders = list(root_dir.rglob("*.mp4"))
    subject_folders = natsort.natsorted(subject_folders)

    # Set up logging to file and console
    logger = logging.getLogger()
    logger.setLevel(logging.ERROR)

    # File handler for logging to file
    file_handler = logging.FileHandler(backup_bin / 'log.txt')
    file_handler.setLevel(logging.ERROR)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    # Stream handler for logging to console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.ERROR)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)

    predictor = build_sam2_video_predictor("sam2_hiera_t.yaml", r"C:\Users\Dee\Desktop\sam2\sam2-dee\checkpoints\sam2_hiera_tiny.pt", device=device)
    offload_to_cpu = False
    chunk_size = 10000  # store to file once this many frames are processed
    cache_size = 200    # maximum number of input images to keep in memory
    image_feature_cache_size = 100
    for video_dir in subject_folders:
        try:
            this_output_path = output_bin / video_dir.name
            print(f"############## {this_output_path} ##############")
            this_output_path.mkdir(parents=True, exist_ok=True)

            savepath_videosegs = this_output_path / 'segments_0.pickle.gz'
            if os.path.exists(savepath_videosegs):
                print(f"Already done. Skipping {video_dir}")
                continue

            this_prompt = retrieve_prompt_from_subject(video_dir, gt_dir)
            frame_idx = this_prompt['frame']

            inference_state = predictor.init_state(video_path=str(video_dir)
                                                , offload_video_to_cpu=offload_to_cpu
                                                , offload_state_to_cpu=offload_to_cpu
                                                , async_loading_frames=True
                                                , image_cache_size=cache_size
                                                , image_feature_cache_size=image_feature_cache_size)

            add_pupil_prompt(predictor, inference_state, this_prompt['prompt'], ann_frame_index=frame_idx)

            for i,video_segments in enumerate(propagate(predictor, inference_state, chunk_size, this_output_path, this_prompt)):
                savepath_videosegs = this_output_path / f'segments_{i}.pickle.gz'
                with open(savepath_videosegs, 'wb') as handle:
                    compress_pickle.dump(video_segments, handle, pickler_kwargs={'protocol': pickle.HIGHEST_PROTOCOL})
                video_segments.clear()

            predictor.reset_state(inference_state)
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            predictor.reset_state(inference_state)
            gc.collect()
            torch.cuda.empty_cache()

            error_message = f'Failed: {video_dir} due to error.'
            logger.error(error_message)
            logger.error(f"An error occurred: {e}")
            logger.error("Error type: %s", type(e).__name__)
            traceback_details = traceback.format_exc()
            logger.error("Detailed traceback:\n%s", traceback_details)
