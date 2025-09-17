import compress_pickle
import numpy as np
import pandas as pd
import pathlib
import natsort
from tqdm import tqdm
from PIL import Image
import cv2
import os
import math
from torchvision.io.video_reader import VideoReader
from cutcutcodec.core.io import read as cccread
import ffmpeg as _ffmpeg
_ffmpeg.add_to_path()
import subprocess


def openCVCircle(img, center_coordinates, radius, color, thickness, sub_pixel_fac):
    p = [np.round(x*sub_pixel_fac) for x in center_coordinates]
    if np.all([not math.isnan(x) and abs(x)<np.iinfo(np.intc).max for x in p]):
        p = tuple([int(x) for x in p])
        cv2.circle(img, p, int(np.round(radius*sub_pixel_fac)), color, thickness, lineType=cv2.LINE_AA, shift=int(math.log2(sub_pixel_fac)))

def get_frame_timestamps_ffprobe(vid_file: pathlib.Path) -> np.array:
    command = ['ffprobe',
               '-v', 'quiet',
               '-select_streams', 'v',
               '-of', 'compact=p=0:nk=1',
               '-show_entries', 'packet=pts_time',
               f'{vid_file}']
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = process.communicate()
    if err:
        print(err)
        return None

    # turn into numpy array and ensure sorted (packets are not necessarily stored in presentation order)
    frame_ts = [float(x) for x in np.sort(np.fromiter(map(float,out.split()), 'float'))]
    frame_idx = range(0, len(frame_ts))
    return dict(zip(frame_idx, frame_ts))

if __name__ == "__main__":
    plot_every = 100
    mask_clrs = ((0,0,255),(0,255,0),(255,0,0),(0,255,255))
    root_dir = pathlib.Path(r"D:\PSA wearable\raw recordings 2\Basler")
    base_path = pathlib.Path(r"D:\PSA wearable\annotate_pupil_basler\segmentation")
    subject_folders = [pathlib.Path(f.path) for f in os.scandir(root_dir) if f.is_dir() and (pathlib.Path(base_path)/f.name).is_dir()]
    subject_folders = natsort.natsorted(subject_folders)

    for subject in subject_folders:
        print(f"############## {subject.name} ##############")
        in_dir = base_path / subject.name

        out_file = (base_path/subject.name).with_suffix('.tsv')
        if out_file.is_file():
            print(f"Already done. Skipping {subject.name}")
            continue

        video_file = subject/"cam1_R001.mp4"
        video_stream = VideoReader(str(video_file), stream="video")
        video_data = video_stream.get_metadata()['video']
        if "fps" in video_data:
            video_fps = video_data['fps'][0]
        else:
            video_fps = video_data["framerate"][0]
        try:
            num_frames = video_stream.container.streams.video[0].frames
        except:
            num_frames = int(video_data['duration'][0] * video_fps)
        plot_frames = range(0,num_frames,plot_every)
        del video_stream
        frame_ts = get_frame_timestamps_ffprobe(video_file)
        
        
        with cccread(str(video_file)) as container:
            stream = container.out_select("video")[0]
            output = []
            result_files = in_dir.glob("*.pickle.gz")
            for res_file in tqdm(result_files, desc="files"):
                with open(res_file, 'rb') as handle:
                    segments = compress_pickle.load(handle)

                for idx in tqdm(segments, desc="frames"):
                    this_frame = {}
                    for obj in segments[idx]:
                        if not isinstance(obj,int):
                            continue
                        mask = segments[idx][obj].squeeze() > 0.5

                        contours = list(cv2.findContours(np.uint8(mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0])
                        contours.sort(key=cv2.contourArea, reverse=True)
                        if contours:
                            M = cv2.moments(contours[0])
                            cx = M['m10']/M['m00']
                            cy = M['m01']/M['m00']
                            area = cv2.contourArea(contours[0])
                        else:
                            cx=cy=np.nan
                        output.append({
                            'object': obj,
                            'frame': idx,
                            'cx': cx,
                            'cy': cy,
                            'area': area
                        })
                        this_frame[obj] = (cx,cy)

                    if idx in plot_frames:
                        frame = stream.snapshot(frame_ts[idx], (stream.height, stream.width))
                        img = frame.numpy()
                        ori_img = img.copy()
                        for i,obj in enumerate([i for i in segments[idx] if isinstance(i,int)]):
                            mask = np.uint8(segments[idx][obj].squeeze() > 0.5)
                            clrImg = np.zeros(img.shape, img.dtype)
                            clrImg[:,:] = mask_clrs[i]
                            clrMask = cv2.bitwise_and(clrImg, clrImg, mask=mask)
                            # make image with just this object
                            img2 = cv2.addWeighted(clrMask, .4, ori_img, .6, 0)
                            # make combined image
                            img2 = cv2.addWeighted(clrMask, .4, img, .6, 0)
                            img = cv2.add(cv2.bitwise_or(img,img,mask=cv2.bitwise_not(mask*255)), cv2.bitwise_or(img2,img2,mask=mask))
                            openCVCircle(img, this_frame[obj], 1, mask_clrs[i], 3, 8)
                        Image.fromarray(img).save(in_dir / f'output_frame_{idx}.png')

        all_scores = pd.DataFrame.from_dict({i:s for i,s in enumerate(output)}, orient='index')
        all_scores.to_csv(out_file, index=False, na_rep='nan', float_format="%.8f", sep='\t')
                