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

def propagate(predictor, inference_state, chunk_size, save_path=None, prompts=None, extra_output_mask_frames=0):
    # run propagation throughout the video and collect the results in a dict
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy() for i, out_obj_id in enumerate(out_obj_ids)}
        video_segments[out_frame_idx]['image_file'] = inference_state['images'].img_paths[min(out_frame_idx,len(inference_state['images'].img_paths)-1)]
        if out_frame_idx<(len(prompts) if prompts is not None else 0)+extra_output_mask_frames and save_path:
            img = inference_state["images"].get_frame(out_frame_idx)
            img = img.permute(1,2,0).cpu().numpy()
            img_min, img_max = img.min(), img.max()
            img = (img-img_min)/(img_max-img_min)
            img = Image.fromarray(np.uint8(img*255)).resize((inference_state["images"].video_width, inference_state["images"].video_height))
            img.save(pathlib.Path(save_path) / f'frame{out_frame_idx}.png')
            # add output masks
            img = np.array(img)
            ori_img = img.copy()
            for i,oid in enumerate([i for i in video_segments[out_frame_idx] if isinstance(i,int)]):
                mask = np.uint8(video_segments[out_frame_idx][oid].squeeze() > 0.5)
                clrImg = np.zeros(img.shape, img.dtype)
                clrImg[:,:] = mask_clrs[i]
                clrMask = cv2.bitwise_and(clrImg, clrImg, mask=mask)
                # make image with just this object
                img2 = cv2.addWeighted(clrMask, .6, ori_img, .4, 0)
                ori_img_masked = cv2.add(cv2.bitwise_or(ori_img,ori_img,mask=cv2.bitwise_not(mask*255)), cv2.bitwise_or(img2,img2,mask=mask))
                Image.fromarray(ori_img_masked).save(pathlib.Path(save_path) / f'frame{out_frame_idx}_mask_obj{oid}.png')
                # make combined image
                img2 = cv2.addWeighted(clrMask, .6, img, .4, 0)
                img = cv2.add(cv2.bitwise_or(img,img,mask=cv2.bitwise_not(mask*255)), cv2.bitwise_or(img2,img2,mask=mask))
            Image.fromarray(img).save(pathlib.Path(save_path) / f'frame{out_frame_idx}_mask_all.png')
            if prompts is not None and out_frame_idx<len(prompts):
                for o in prompts[video_segments[out_frame_idx]['image_file']]:
                    pr = prompts[video_segments[out_frame_idx]['image_file']][o]
                    clr = mask_clrs[o]
                    for c,l in zip(pr['coords'],pr['labels']):
                        p = [int(x) for x in c]
                        if l==1:
                            img = cv2.drawMarker(img, (p[0], p[1]), clr, cv2.MARKER_CROSS, 5, 1)
                            # img = cv2.circle(img, (p[0], p[1]), 2, clr, -1)
                        else:
                            img = cv2.drawMarker(img, (p[0], p[1]), clr, cv2.MARKER_SQUARE, 4, 1)
                            #img = cv2.rectangle(img, (p[0]-1, p[1]-1), (p[0]+1, p[1]+1), clr, -1)
                Image.fromarray(img).save(pathlib.Path(save_path) / f'frame{out_frame_idx}_mask_all_prompts.png')
        if out_frame_idx>0 and out_frame_idx%chunk_size == 0:
            yield video_segments
            video_segments.clear()
    yield video_segments

def load_prompts_from_folder(folder: pathlib.Path, N: int):
    # prompts are stored in text files, one per image load prompts for first N images (or less if there are less)
    prompt_files = list(folder.glob("*_prompts_all.txt"))
    prompt_files = natsort.natsorted(prompt_files)
    if N is not None:
        prompt_files = prompt_files[:N]
    # dict with key (full) filename, containing per object a list of coordinates and associated labels
    prompts: dict[pathlib.Path,list[int,int,tuple[int,int]]] = {}
    for fp in prompt_files:
        with open(fp) as f:
            reader = csv.reader(f, delimiter="\t")
            pr = list(reader)
        file = fp.with_name('_'.join(fp.stem.split('_')[:-2])+'.png')
        prompts[file] = {0:{'coords':[], 'labels':[]}, 1:{'coords':[], 'labels':[]}, 2:{'coords':[], 'labels':[]}, 3:{'coords':[], 'labels':[]}}
        for p in pr:
            obj_id = 0 if p[0]=='CR' else 1 if p[0]=='pupil' else 2 if p[0]=='iris' else 3
            label = int(p[3])   # 1 is positive prompt, 0 negative
            point_coord = tuple((int(x) for x in p[1:3]))
            prompts[file][obj_id]['coords'].append(point_coord)
            prompts[file][obj_id]['labels'].append(label)
    return prompts


if __name__ == '__main__':
    video_base   = pathlib.Path(r"D:\datasets")
    prompts_base = pathlib.Path(r"D:\prompts")
    output_base  = pathlib.Path(r"D:\output")
    dataset = '2023-04-25_1000Hz_100_EL' #'2023-09-12 1000 Hz many subjects' #
    N_prompts = 9
    model = ('l','large') # ('t','tiny') # ('l','large')

    # Path containing the videos (zip files or subdirectory of videos)
    input_dir = video_base / dataset
    subject_folders = [pathlib.Path(f.path) for f in os.scandir(input_dir) if f.is_dir() and not 'eyelink' in (p:=pathlib.Path(f.path)).stem]
    subject_folders = natsort.natsorted(subject_folders)

    predictor = build_sam2_video_predictor(f"configs/sam2.1/sam2.1_hiera_{model[0]}.yaml", f"checkpoints/sam2.1_hiera_{model[1]}.pt", device=device)
    offload_to_cpu = False
    chunk_size = 10000  # store to file once this many frames are processed
    cache_size = 200    # maximum number of input images to keep in memory
    image_feature_cache_size = 100
    for subject in subject_folders:
        print(f"############## {subject.name} ##############")
        video_files = list(subject.rglob("*.mp4"))
        video_files = natsort.natsorted(video_files)
        for i,video_file in enumerate(video_files):
            # if i>1:
            #    break
            try:
                this_output_path = output_base / f'{N_prompts}_prompt_frames_sclera_take2' / model[1] / dataset / subject.name / video_file.stem
                print(f"############## {this_output_path} ##############")
                this_output_path.mkdir(parents=True, exist_ok=True)

                savepath_videosegs = this_output_path / 'segments_0.pickle.gz'
                if os.path.exists(savepath_videosegs) and True:
                    print(f"Already done. Skipping {dataset}/{subject.name}/{video_file.name}")
                    continue

                prompts = load_prompts_from_folder(prompts_base / dataset / subject.name, N=N_prompts)

                inference_state = predictor.init_state(video_path=str(video_file)
                                                    , offload_video_to_cpu=offload_to_cpu
                                                    , offload_state_to_cpu=offload_to_cpu
                                                    , image_cache_size=cache_size
                                                    , image_feature_cache_size=image_feature_cache_size
                                                    , separate_prompts=prompts)

                for i,video_segments in enumerate(propagate(predictor, inference_state, chunk_size, this_output_path, prompts, 3)):
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
