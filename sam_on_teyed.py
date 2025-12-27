import os
os.environ['TORCH_CUDNN_SDPA_ENABLED'] = '1'
import numpy as np
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
    if prompts is not None and out_frame_idx in (p['frame'] for p in prompts.values()):
        prompt = [p for p in prompts.values() if p['frame']==out_frame_idx][0]
        for o in prompt:
            if o=='frame':
                continue
            pr = prompt[o]
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

def propagate(predictor, inference_state, chunk_size, prompts, save_path=None, save_range=None):
    # run propagation throughout the video and collect the results in a dict
    prompt_frames = sorted([p['frame'] for p in prompts.values()])
    segments = prompt_frames.copy()
    if segments[0]>0:
        segments = [0]+segments
    if segments[-1]<inference_state["num_frames"]-1:
        segments.append(inference_state["num_frames"])
    segments = [[segments[i],segments[i+1]-1] for i in range(0,len(segments)-1)]

    video_segments = {}  # video_segments contains the per-frame segmentation results
    skip_next_prompt = False
    for i,s in enumerate(segments):
        if i==0 and prompt_frames[0]>0:
            reverse = True
            to_prompt = s[1]+1
            skip_next_prompt = True
        else:
            reverse = False
            if skip_next_prompt:
                to_prompt = None
                skip_next_prompt = False
            else:
                to_prompt = s[0]
        if to_prompt is not None:
            add_prompt = [p for p in prompts.values() if p['frame']==to_prompt][0]
            for o in add_prompt:
                if o=='frame':
                    continue
                for c,l in zip(add_prompt[o]['coords'],add_prompt[o]['labels']):
                    predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=add_prompt['frame'],
                        obj_id=o,
                        points = np.array(c).reshape(-1,2),
                        labels = np.array([l]),  # 1 is positive click, 0 is negative click
                        clear_old_points=False
                    )
        if reverse:
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, start_frame_idx=s[1], max_frame_num_to_track=s[1]-s[0]+1, reverse=True):
                video_segments[out_frame_idx] = {out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy() for i, out_obj_id in enumerate(out_obj_ids)}
                video_segments[out_frame_idx]['image_file'] = inference_state['images'].img_paths[min(out_frame_idx,len(inference_state['images'].img_paths)-1)]
                if save_path and (out_frame_idx in prompt_frames or (save_range and out_frame_idx in save_range)):
                    save_output_with_prompt(out_frame_idx, prompts, video_segments, save_path)
            continue
        try:
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, start_frame_idx=s[0], max_frame_num_to_track=s[1]-s[0]+1):
                video_segments[out_frame_idx] = {out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy() for i, out_obj_id in enumerate(out_obj_ids)}
                video_segments[out_frame_idx]['image_file'] = inference_state['images'].img_paths[min(out_frame_idx,len(inference_state['images'].img_paths)-1)]
                if save_path and ((prompts is not None and out_frame_idx in prompt_frames) or (save_range and out_frame_idx in save_range)):
                    save_output_with_prompt(out_frame_idx, prompts, video_segments, save_path)

                if out_frame_idx>0 and out_frame_idx%chunk_size == 0:
                    yield video_segments
                    video_segments.clear()
        except Exception as e:
            num_frames = inference_state['images'].num_frames
            extra = ''
            if out_frame_idx > num_frames-10:
                yield video_segments
                # write file indicating frame completed and total number of frames
                completion_path = pathlib.Path(save_path) / 'segment_info.txt'
                with open(completion_path, 'w') as f:
                    f.write(f'{out_frame_idx}\n{num_frames}\n')
                extra = f', but state saved (frame {out_frame_idx} of {num_frames})'
            raise RuntimeError(f"Propagation failed{extra}") from e
    yield video_segments

def extract_last_number_and_fix_fname(filename):
    match = re.search(r'_frame(\d+)\.png$', filename)
    if match:
        original_num = int(match.group(1))
        corrected_num = original_num
        return corrected_num
    return None


def load_prompts_from_folder(folder: pathlib.Path, file_name: str):
    # dict with key (full) filename, containing per object a list of coordinates and associated labels
    prompts: dict[pathlib.Path,list[int,int,tuple[int,int]]] = {}
    prompt_files = list((folder).glob(file_name+".points.txt"))
    for pf in prompt_files:
        with open(pf) as f:
            reader = csv.reader(f, delimiter="\t")
            pr = list(reader)
        # get associated frame
        frame_files = list((folder).glob(file_name+"*_frame*.png"))
        if not frame_files:
            raise ValueError('missing prompt image file')   # actually it isn't used, but still, strange
        pim = frame_files[0].name
        prompt_img = extract_last_number_and_fix_fname(pim)

        prompts[pim] = {'frame': None, 0:{'coords':[], 'labels':[]}, 1:{'coords':[], 'labels':[]}, 2:{'coords':[], 'labels':[]}}
        for p in pr:
            obj_id = 0 if p[0]=='pupil' else 1 if p[0]=='iris' else 2
            label = int(p[3])   # 1 is positive prompt, 0 negative
            point_coord = tuple((int(x) for x in p[1:3]))
            prompts[pim]['frame'] = prompt_img
            prompts[pim][obj_id]['coords'].append(point_coord)
            prompts[pim][obj_id]['labels'].append(label)
    return prompts


if __name__ == '__main__':
    vid_dir = 'VIDEOS'
    gt_dir = 'ANNOTATIONS'
    input_dir    = pathlib.Path(r"\\et-nas.humlab.lu.se\FLEX\datasets real\TEyeD")
    prompts_base = pathlib.Path(r"\\et-nas.humlab.lu.se\FLEX\datasets real\TEyeD\prompts")
    output_base  = pathlib.Path(r"\\et-nas.humlab.lu.se\FLEX\2025 SAM2_3\TEyeD\output\SAM2_point_prompts")
    model = ('l','large') # ('t','tiny') # ('l','large')
    run_reversed = False

    # Path containing the videos (zip files or subdirectory of videos)
    datasets = [fp for f in os.scandir(input_dir) if (fp:=pathlib.Path(f.path)).is_dir() and all((fp/s).is_dir() for s in [vid_dir,gt_dir])]
    datasets = natsort.natsorted(datasets, reverse=run_reversed)

    predictor = build_sam2_video_predictor(f"configs/sam2.1/sam2.1_hiera_{model[0]}.yaml", f"checkpoints/sam2.1_hiera_{model[1]}.pt", device=device)
    offload_to_cpu = False
    chunk_size = 10000  # store to file once this many frames are processed
    cache_size = 200    # maximum number of input images to keep in memory
    image_feature_cache_size = 100
    for dataset in datasets:
        print(f"############## {dataset.name} ##############")
        video_files = list((dataset/vid_dir).glob("*.mp4"))
        video_files = natsort.natsorted(video_files, reverse=run_reversed)
        if not video_files:
            print(f"No video files found for subject {dataset.name}, skipping.")
            continue

        if not (prompts_base / dataset.name).exists():
            print(f"No prompts found for subject {dataset.name}, skipping.")
            continue

        for i,video_file in enumerate(video_files):
            if not video_file.exists():
                continue
            prompt_files = list((prompts_base / dataset.name).glob(video_file.stem+".points.txt"))
            if len(prompt_files)<1:
                print(f'prompts not found for video {video_file.name}, skipping.')
                continue
            try:
                this_output_path = output_base / dataset.name / video_file.stem
                print(f"############## {this_output_path} ##############")
                this_output_path.mkdir(parents=True, exist_ok=True)

                savepath_videosegs = this_output_path / 'segments_0.pickle.gz'
                if os.path.exists(savepath_videosegs):
                    print(f"Already done. Skipping {dataset.name}/{video_file.name}")
                    continue

                prompts = load_prompts_from_folder(prompts_base / dataset.name, video_file.stem)

                inference_state = predictor.init_state(video_path=str(video_file)
                                                    , offload_video_to_cpu=offload_to_cpu
                                                    , offload_state_to_cpu=offload_to_cpu
                                                    , image_cache_size=cache_size
                                                    , image_feature_cache_size=image_feature_cache_size)

                to_save = {*range(0,1200,10), *range(1200,1000000,100)}
                for i,video_segments in enumerate(propagate(predictor, inference_state, chunk_size, prompts, this_output_path, save_range=to_save)):
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
