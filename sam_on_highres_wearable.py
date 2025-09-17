import os
os.environ['TORCH_CUDNN_SDPA_ENABLED'] = '1'
import numpy as np
import pandas as pd
import torch
import pickle
import compress_pickle
import pathlib
from PIL import Image
import traceback
import gc
import natsort
import cv2
import csv
import re
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

mask_clrs = ((0,0,255),(0,255,0),(255,0,0),(0,255,255))
sizes = (6,12,18,24)

def save_output_with_prompt(out_frame_idx, prompts, video_segments, save_path):
    img = inference_state["images"].get_frame(out_frame_idx)
    img = img.permute(1,2,0).cpu().numpy()
    img_min, img_max = img.min(), img.max()
    img = (img-img_min)/(img_max-img_min)
    img = Image.fromarray(np.uint8(img*255)).resize((inference_state["images"].video_width, inference_state["images"].video_height))
    #img.save(pathlib.Path(save_path) / f'frame{out_frame_idx}.png')
    # add output masks
    img = np.array(img)
    ori_img = img.copy()
    for i,oid in enumerate([i for i in video_segments[out_frame_idx] if isinstance(i,int)]):
        mask = np.uint8(video_segments[out_frame_idx][oid].squeeze() > 0.5)
        clrImg = np.zeros(img.shape, img.dtype)
        clrImg[:,:] = mask_clrs[i]
        clrMask = cv2.bitwise_and(clrImg, clrImg, mask=mask)
        # make image with just this object
        img2 = cv2.addWeighted(clrMask, .4, ori_img, .6, 0)
        #ori_img_masked = cv2.add(cv2.bitwise_or(ori_img,ori_img,mask=cv2.bitwise_not(mask*255)), cv2.bitwise_or(img2,img2,mask=mask))
        #Image.fromarray(ori_img_masked).save(pathlib.Path(save_path) / f'frame{out_frame_idx}_mask_obj{oid}.png')
        # make combined image
        img2 = cv2.addWeighted(clrMask, .4, img, .6, 0)
        img = cv2.add(cv2.bitwise_or(img,img,mask=cv2.bitwise_not(mask*255)), cv2.bitwise_or(img2,img2,mask=mask))
    if prompts is not None and out_frame_idx==prompts['frame']:
        for o in prompts:
            if o=='frame':
                continue
            pr = prompts[o]
            clr = mask_clrs[o]
            for c,l in zip(pr['coords'],pr['labels']):
                p = [int(x) for x in c]
                if l==1:
                    img = cv2.drawMarker(img, (p[0], p[1]), clr, cv2.MARKER_CROSS, 6, 2)
                    # img = cv2.circle(img, (p[0], p[1]), 2, clr, -1)
                else:
                    img = cv2.drawMarker(img, (p[0], p[1]), clr, cv2.MARKER_SQUARE, sizes[o], 2)
                    #img = cv2.rectangle(img, (p[0]-1, p[1]-1), (p[0]+1, p[1]+1), clr, -1)
    Image.fromarray(img).save(pathlib.Path(save_path) / f'frame{out_frame_idx}_mask_all.png')

has_saved = False
def propagate(predictor, inference_state, chunk_size, save_path=None, prompts=None, save_range=None, stop_early=False):
    global has_saved
    # run propagation throughout the video and collect the results in a dict
    # simplify prompts, only one prompt file here
    if prompts is not None:
        prompts = next(iter(prompts.values()))

    video_segments = {}  # video_segments contains the per-frame segmentation results
    if prompts is not None and prompts['frame']>0:
        # first do a reverse pass from the prompted frame
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, reverse=True):
            video_segments[out_frame_idx] = {out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy() for i, out_obj_id in enumerate(out_obj_ids)}
            video_segments[out_frame_idx]['image_file'] = inference_state['images'].img_paths[min(out_frame_idx,len(inference_state['images'].img_paths)-1)]
            if save_path and not has_saved and (save_range and out_frame_idx in save_range):
                save_output_with_prompt(out_frame_idx, prompts, video_segments, save_path)
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy() for i, out_obj_id in enumerate(out_obj_ids)}
        video_segments[out_frame_idx]['image_file'] = inference_state['images'].img_paths[min(out_frame_idx,len(inference_state['images'].img_paths)-1)]
        if save_path and not has_saved and (save_range and out_frame_idx in save_range):
            save_output_with_prompt(out_frame_idx, prompts, video_segments, save_path)

        if out_frame_idx>0 and out_frame_idx%chunk_size == 0:
            yield video_segments
            video_segments.clear()
            has_saved = True
            if stop_early:
                break
    yield video_segments

def extract_last_number(filename):
    match = re.search(r'_(\d+)\.png$', filename)
    if match:
        return int(match.group(1))
    return None

def load_prompts_from_folder(folder: pathlib.Path):
    # dict with key (full) filename, containing per object a list of coordinates and associated labels
    prompts: dict[pathlib.Path,list[int,int,tuple[int,int]]] = {}
    with open(folder / "cam1_R001.mp4_prompts.txt") as f:
        reader = csv.reader(f, delimiter="\t")
        pr = list(reader)
    # get associated frame
    prompt_imgs = list(folder.glob("cam1_R001.mp4_*.png"))
    if len(prompt_imgs)!=1:
        raise ValueError('wrong')
    prompt_img = extract_last_number(prompt_imgs[0].name)

    prompts[prompt_imgs[0]] = {'frame':None}
    ip = -1
    for i,p in enumerate(pr):
        label = int(p[3])   # 1 is positive prompt, 0 negative
        if label==1:
            ip+=1
            prompts[prompt_imgs[0]][ip] = {'coords':[], 'labels':[]}
        point_coord = tuple((int(x) for x in p[1:3]))
        prompts[prompt_imgs[0]]['frame'] = prompt_img
        prompts[prompt_imgs[0]][ip]['coords'].append(point_coord)
        prompts[prompt_imgs[0]][ip]['labels'].append(label)
    return prompts


if __name__ == '__main__':
    input_dir   = pathlib.Path(r"D:\PSA wearable\raw recordings 2\Basler")
    prompts_base = pathlib.Path(r"D:\PSA wearable\annotate_pupil_basler\prompts")
    output_base  = pathlib.Path(r"D:\PSA wearable\annotate_pupil_basler\segmentation")
    model = ('l','large') # ('t','tiny') # ('l','large')

    # Path containing the videos (zip files or subdirectory of videos)
    subject_folders = [pathlib.Path(f.path) for f in os.scandir(input_dir) if f.is_dir() and (pathlib.Path(prompts_base)/f.name).is_dir()]
    subject_folders = natsort.natsorted(subject_folders)

    predictor = build_sam2_video_predictor(f"configs/sam2.1/sam2.1_hiera_{model[0]}.yaml", f"checkpoints/sam2.1_hiera_{model[1]}.pt", device=device)
    offload_to_cpu = False
    chunk_size = 10000  # store to file once this many frames are processed
    cache_size = 200    # maximum number of input images to keep in memory
    image_feature_cache_size = 100
    for subject in subject_folders:
        #if "P2_i" not in subject.name and "P3_i" not in subject.name:
        #    continue
        print(f"############## {subject.name} ##############")
        video_files = [subject/"cam1_R001.mp4"]
        for i,video_file in enumerate(video_files):
            try:
                this_output_path = output_base / subject.name
                print(f"############## {this_output_path} ##############")
                this_output_path.mkdir(parents=True, exist_ok=True)

                savepath_videosegs = this_output_path / 'segments_0.pickle.gz'
                if os.path.exists(savepath_videosegs):
                    print(f"Already done. Skipping {subject.name}/{video_file.name}")
                    continue

                prompts = load_prompts_from_folder(prompts_base / subject.name)

                inference_state = predictor.init_state(video_path=str(video_file)
                                                    , offload_video_to_cpu=offload_to_cpu
                                                    , offload_state_to_cpu=offload_to_cpu
                                                    , image_cache_size=cache_size
                                                    , image_feature_cache_size=image_feature_cache_size
                                                    , separate_prompts=prompts)

                has_saved = False
                for i,video_segments in enumerate(propagate(predictor, inference_state, chunk_size, this_output_path, prompts, save_range=range(140), stop_early=False)):
                    savepath_videosegs = this_output_path / f'segments_{i}.pickle.gz'
                    with open(savepath_videosegs, 'wb') as handle:
                        compress_pickle.dump(video_segments, handle, pickler_kwargs={'protocol': pickle.HIGHEST_PROTOCOL})
                    video_segments.clear()

                predictor.reset_state(inference_state)
                gc.collect()
                torch.cuda.empty_cache()

            except Exception as e:
                if 'inference_state' in globals():
                    predictor.reset_state(inference_state)
                gc.collect()
                torch.cuda.empty_cache()

                error_message = f'Failed: {video_file} due to error.'
                print(error_message)
                print(f"An error occurred: {e}")
                print("Error type: %s", type(e).__name__)
                print("Detailed traceback:\n%s", traceback.format_exc())
