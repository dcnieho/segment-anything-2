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


kernel_pup  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15, 15))
def detect_pupil_from_thresholded(thresholded, size_limits=None, symmetry_tresh=0.5, fill_thresh=0.2, kernel=kernel_pup, window_name=None):
    # Compute center location of image
    im_height, im_width = thresholded.shape
    center_x, center_y = im_width/2, im_height/2

    # Close  holes in the pupil, e.g., created by the CR
    blobs = cv2.morphologyEx(thresholded,cv2.MORPH_OPEN,kernel)
    blobs = cv2.morphologyEx(blobs,cv2.MORPH_CLOSE,kernel)

    # Visualized blobs if windown name given
    if window_name:
        cv2.imshow(window_name, blobs)

    # Find countours of the detected blobs
    blob_contours, hierarchy  = cv2.findContours(blobs,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)

    # Find pupil but checking one blob at the time. Pupils are round, so
    # add checks for 'roundness' criteria
    # If serveral blobs are found, select the one
    # closest to the center
    '''
    For a blob to be a pupil candidate
    1. blob must have the right area
    2. must be circular
    '''

    pupil_detected = False
    old_distance_image_center = np.inf
    for i, cnt in enumerate(blob_contours):

        # Take convex hull of countour points to alleviate holes
        cnt = cv2.convexHull(cnt)

        # Only contours with enouth points are of interest
        if len(cnt) < 10:
            continue

        # Compute area and bounding rect around blob
        temp_area = cv2.contourArea(cnt)
        rect = cv2.boundingRect(cnt)
        x, y, width, height = rect
        radius = 0.25 * (width + height)
        r1,r2 = width,height
        if r1>r2:
            r1,r2 = r2,r1

        # Check area and roundness criteria
        area_condition = True if size_limits is None else (size_limits[0] <= temp_area <= size_limits[1])
        symmetry_condition = (abs(1 - float(r1)/float(r2)) <= symmetry_tresh)
        fill_condition = (abs(1 - (temp_area / (np.pi * radius**2))) <= fill_thresh)

        # If these criteria are fulfilled, a pupil is probably detected
        if area_condition and symmetry_condition and fill_condition:
            # Compute moments of blob
            moments = cv2.moments(cnt)

            # Compute blob center of gravity from the moments
            cx, cy = moments['m10']/moments['m00'], \
                     moments['m01']/moments['m00']

            # Compute distance blob - image center
            distance_image_center = np.sqrt((cx - center_x)**2 +
                                            (cy - center_y)**2)
            # Check if the current blob-center is closer
            # to the image center than the previous one
            if distance_image_center < old_distance_image_center:
                pupil_detected = True

                # Store pupil variables
                contour_points = cnt
                area = temp_area

                cx_best = cx
                cy_best = cy

                # Fit an ellipse to the contour, and compute its area
                ellipse = cv2.fitEllipse(contour_points.squeeze().astype('int'))
                (x_ellipse, y_ellipse), (MA, ma), angle = ellipse
                area_ellipse = np.pi / 4.0 * MA * ma

                diameter_ellipse = np.sqrt(area_ellipse / np.pi) * 2
                max_radius_ellipse = max([MA, ma])
                width_ellipse = getEllipseWidth({'ellipse' : ((x_ellipse, y_ellipse), (MA, ma), angle, area_ellipse)})

                old_distance_image_center = distance_image_center

    # If no potential pupil is found, due to e.g., blinks,
    # return nans
    if not pupil_detected:
        cx_best = np.nan
        cy_best = np.nan
        area = np.nan
        contour_points = np.nan
        x_ellipse = np.nan
        y_ellipse = np.nan
        MA = np.nan
        ma = np.nan
        angle = np.nan
        area_ellipse = np.nan
        diameter_ellipse = np.nan
        max_radius_ellipse = np.nan
        width_ellipse = np.nan

    pupil_features = {'cog':(cx_best, cy_best), 'area':area, 'contour_points': contour_points,
                      'ellipse' : ((x_ellipse, y_ellipse), (MA, ma), angle,
                                    area_ellipse, diameter_ellipse, max_radius_ellipse, width_ellipse)}
    return pupil_features

def flatten_output(input):
    output = {}
    output['frame'] = input['frame']
    output['c_x'] = input['cog'][0]
    output['c_y'] = input['cog'][1]
    output['c_area'] = input['area']
    output['e_x'] = input['ellipse'][0][0]
    output['e_y'] = input['ellipse'][0][1]
    output['e_MA'] = input['ellipse'][1][0]
    output['e_ma'] = input['ellipse'][1][1]
    output['e_ang'] = input['ellipse'][2]
    output['e_area'] = input['ellipse'][3]
    output['e_diam'] = input['ellipse'][4]
    output['e_max_radius'] = input['ellipse'][5]
    output['e_width'] = input['ellipse'][6]
    return output

def openCVCircle(img, center_coordinates, radius, color, thickness, sub_pixel_fac):
    p = [np.round(x*sub_pixel_fac) for x in center_coordinates]
    if np.all([not math.isnan(x) and abs(x)<np.iinfo(np.intc).max for x in p]):
        p = tuple([int(x) for x in p])
        cv2.circle(img, p, int(np.round(radius*sub_pixel_fac)), color, thickness, lineType=cv2.LINE_AA, shift=int(math.log2(sub_pixel_fac)))

def openCVEllipse(img, center_coordinates, axesLength, angle, color, thickness, sub_pixel_fac):
    p = [np.round(x * sub_pixel_fac) for x in center_coordinates]
    pal = [np.round(x * sub_pixel_fac) for x in axesLength]
    if np.all([not np.isnan(x) and abs(x) < np.iinfo(np.intc).max for x in p]):
        p = tuple([int(x) for x in p])
        pal = tuple([int(x) for x in pal])
        cv2.ellipse(img, p, pal, angle, 0, 360, color, thickness, lineType=cv2.LINE_AA,
                   shift=int(np.log2(sub_pixel_fac)))
        
def getEllipseLR(xc, yc, arad, brad, theta):
    t = np.linspace(0, 2 * np.pi, 2000)
    X = xc + arad * np.cos(t) * np.cos(theta) - brad * np.sin(t) * np.sin(theta)
    Y = yc + arad * np.cos(t) * np.sin(theta) + brad * np.sin(t) * np.cos(theta)

    return (np.min(X), np.max(X))

def getEllipseWidth(pupil_features):
    xl, xr = getEllipseLR(
        pupil_features['ellipse'][0][0],
        pupil_features['ellipse'][0][1],
        pupil_features['ellipse'][1][0] / 2,
        pupil_features['ellipse'][1][1] / 2,
        np.deg2rad(pupil_features['ellipse'][2])
    )
    return xr-xl

def drawEllipseLines(cimg, pupil_features, color):
    xl, xr = getEllipseLR(
        pupil_features['ellipse'][0][0],
        pupil_features['ellipse'][0][1],
        pupil_features['ellipse'][1][0] / 2,
        pupil_features['ellipse'][1][1] / 2,
        np.deg2rad(pupil_features['ellipse'][2])
    )
    xy1 = tuple(map(int, (xl, 0)))
    xy2 = tuple(map(int, (xl, cimg.shape[0])))
    cv2.line(cimg, xy1, xy2, color, 2)
    xy1 = tuple(map(int, (xr, 0)))
    xy2 = tuple(map(int, (xr, cimg.shape[0])))
    cv2.line(cimg, xy1, xy2, color, 2)

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
    root_dir = pathlib.Path(r"C:\Users\Dee\Desktop\pupilValidation\basler")
    base_path = pathlib.Path(r"C:\Users\Dee\Desktop\pupilValidation\annotate_pupil_basler\segmentation")
    subject_folders = [pathlib.Path(f.path) for f in os.scandir(root_dir) if f.is_dir() and (pathlib.Path(base_path)/f.name).is_dir()]
    subject_folders = natsort.natsorted(subject_folders, reverse=True)

    for subject in subject_folders:
        in_dir_base = base_path / subject.name

        vids = []
        for v in range(1,7):
            vids.extend([f"cam1_R00{v}",f"cam2_R00{v}"])
        for v in vids:
            video_file = subject/f'{v}.mp4'
            if not video_file.exists():
                continue
            print(f"############## {subject.name}, {video_file.name} ##############")
            in_dir = in_dir_base/v
            out_file = (base_path/f'{subject.name}_{v}').with_suffix('.tsv')
            if out_file.is_file():
                print(f"Already done. Skipping {subject.name}")
                continue

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

                            if True:
                                out = detect_pupil_from_thresholded(np.uint8(mask))
                                out['frame'] = idx
                                output.append(out)

                                this_frame[obj] = out
                            else:
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
                                this_frame[obj] = {'cog':(cx,cy)}

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
                                if 'cog' in this_frame[obj]:
                                    openCVCircle(img, this_frame[obj]['cog'], 1, mask_clrs[i], 3, 8)
                                if 'ellipse' in this_frame[obj]:
                                    if not (np.isnan(this_frame[obj]['ellipse'][0][0]) | np.isnan(this_frame[obj]['ellipse'][0][1])):
                                        openCVEllipse(img,
                                                        (this_frame[obj]['ellipse'][0][0],
                                                        this_frame[obj]['ellipse'][0][1]),
                                                        (this_frame[obj]['ellipse'][1][0] / 2,
                                                        this_frame[obj]['ellipse'][1][1] / 2),
                                                        this_frame[obj]['ellipse'][2],
                                                        mask_clrs[i],
                                                        2,
                                                        8)
                                        drawEllipseLines(img, this_frame[obj], mask_clrs[i])
                            Image.fromarray(img).save(in_dir / f'output_frame_{idx}.png')

            all_scores = pd.DataFrame.from_dict({i:s for i,s in enumerate([flatten_output(x) for x in output])}, orient='index')
            all_scores.to_csv(out_file, index=False, na_rep='nan', float_format="%.8f", sep='\t')
                