# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import warnings
import natsort
import zipfile
from collections import OrderedDict

from PIL import Image
import torch
from torchvision.io.video_reader import VideoReader
from torchvision import transforms

def get_sdpa_settings():
    if torch.cuda.is_available():
        old_gpu = torch.cuda.get_device_properties(0).major < 7
        # only use Flash Attention on Ampere (8.0) or newer GPUs
        use_flash_attn = torch.cuda.get_device_properties(0).major >= 8
        if not use_flash_attn:
            warnings.warn(
                "Flash Attention is disabled as it requires a GPU with Ampere (8.0) CUDA capability.",
                category=UserWarning,
                stacklevel=2,
            )
        # keep math kernel for PyTorch versions before 2.2 (Flash Attention v2 is only
        # available on PyTorch 2.2+, while Flash Attention v1 cannot handle all cases)
        pytorch_version = tuple(int(v) for v in torch.__version__.split(".")[:2])
        if pytorch_version < (2, 2):
            warnings.warn(
                f"You are using PyTorch {torch.__version__} without Flash Attention v2 support. "
                "Consider upgrading to PyTorch 2.2+ for Flash Attention v2 (which could be faster).",
                category=UserWarning,
                stacklevel=2,
            )
        math_kernel_on = pytorch_version < (2, 2) or not use_flash_attn
    else:
        old_gpu = True
        use_flash_attn = False
        math_kernel_on = True

    return old_gpu, use_flash_attn, math_kernel_on


def get_connected_components(mask):
    """
    Get the connected components (8-connectivity) of binary masks of shape (N, 1, H, W).

    Inputs:
    - mask: A binary mask tensor of shape (N, 1, H, W), where 1 is foreground and 0 is
            background.

    Outputs:
    - labels: A tensor of shape (N, 1, H, W) containing the connected component labels
              for foreground pixels and 0 for background pixels.
    - counts: A tensor of shape (N, 1, H, W) containing the area of the connected
              components for foreground pixels and 0 for background pixels.
    """
    from sam2 import _C

    return _C.get_connected_componnets(mask.to(torch.uint8).contiguous())


def mask_to_box(masks: torch.Tensor):
    """
    compute bounding box given an input mask

    Inputs:
    - masks: [B, 1, H, W] masks, dtype=torch.Tensor

    Returns:
    - box_coords: [B, 1, 4], contains (x, y) coordinates of top left and bottom right box corners, dtype=torch.Tensor
    """
    B, _, h, w = masks.shape
    device = masks.device
    xs = torch.arange(w, device=device, dtype=torch.int32)
    ys = torch.arange(h, device=device, dtype=torch.int32)
    grid_xs, grid_ys = torch.meshgrid(xs, ys, indexing="xy")
    grid_xs = grid_xs[None, None, ...].expand(B, 1, h, w)
    grid_ys = grid_ys[None, None, ...].expand(B, 1, h, w)
    min_xs, _ = torch.min(torch.where(masks, grid_xs, w).flatten(-2), dim=-1)
    max_xs, _ = torch.max(torch.where(masks, grid_xs, -1).flatten(-2), dim=-1)
    min_ys, _ = torch.min(torch.where(masks, grid_ys, h).flatten(-2), dim=-1)
    max_ys, _ = torch.max(torch.where(masks, grid_ys, -1).flatten(-2), dim=-1)
    bbox_coords = torch.stack((min_xs, min_ys, max_xs, max_ys), dim=-1)

    return bbox_coords


class LRUCache:
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key, default=None):
        if key not in self.cache:
            return default
        # Move the key to the end to show that it was recently used
        self.cache.move_to_end(key)
        return self.cache[key]

    def __getitem__(self, key):
        return self.get(key)

    def put(self, key, value):
        # Insert the item or update the existing one
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        # If the cache exceeds the capacity, pop the first (least recently used) item
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

    def __setitem__(self, key, value):
        self.put(key, value)

    def __contains__(self, key):
        return key in self.cache

    def pop(self, key, default=None):
        return self.cache.pop(key, default)

    def clear(self):
        self.cache.clear()


class VideoFrameLoader:
    def __init__(
        self,
        img_paths,
        image_size,
        img_mean,
        img_std,
        extra_frames,
        offload_to_cpu,
        compute_device,
        cache_size=100,
    ):
        """
        Initialize the video frame loader with image paths or zip file, image size, mean, std, and caching options.
        """
        self.src_type, img_paths = img_paths
        self._img_paths = img_paths
        # caller-facing img_paths
        img_paths = [img_paths] if not isinstance(img_paths,list) else img_paths
        if extra_frames is not None:
            self.img_paths = extra_frames+img_paths
        else:
            self.img_paths = img_paths
        # output size
        self.image_size = image_size
        # input size
        self.video_width = None
        self.video_height= None
        self.img_mean = img_mean
        self.img_std = img_std
        self.offload_to_cpu = offload_to_cpu
        self.device = compute_device
        self.cache_size = cache_size

        self.extra_frames = extra_frames

        self.num_frames = None
        self.zip_file = None
        self.video_stream = None
        self.video_data = None
        self.video_fps = None
        self.video_frame_index = None
        if self.src_type=='video':
            self.video_stream = VideoReader(self._img_paths, stream="video")
            self.video_data = self.video_stream.get_metadata()['video']
            if "fps" in self.video_data:
                self.video_fps = self.video_data['fps'][0]
            else:
                self.video_fps = self.video_data["framerate"][0]
            try:
                self.num_frames = self.video_stream.container.streams.video[0].frames
            except:
                self.num_frames = int(self.video_data['duration'][0] * self.video_fps)

            self.video_frame_index = -1
            tforms = [
                transforms.Resize((image_size, image_size)),
                transforms.Lambda(transforms.functional.convert_image_dtype),   # NB: this divides by 255 also
                transforms.Normalize(mean=img_mean, std=img_std),
            ]
        else:
            if self.src_type=='zip':
                self.zip_file_path, self._img_paths = self._img_paths
                self.zip_file = zipfile.ZipFile(self.zip_file_path, 'r')
            self.num_frames = len(self._img_paths)
            tforms = [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),  # NB: this divides by 255 also
                transforms.Normalize(mean=img_mean, std=img_std),
            ]

        # Initialize image transformations
        self.transform = transforms.Compose(tforms)

        if self.extra_frames is not None:
            self.num_extra_frames = len(self.extra_frames)
            self.num_frames += self.num_extra_frames
            self.transform_extra = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),  # NB: this divides by 255 also
                transforms.Normalize(mean=img_mean, std=img_std),
            ])


        # Create an LRU cache for frames
        self.frame_cache = LRUCache(capacity=self.cache_size)
        # load first frame. Also sets video dimensions
        self.get_frame(0)

    def _load_frame(self, idx):
        """Internal method to load and preprocess a frame."""
        if self.extra_frames is not None:
            if idx<self.num_extra_frames:
                img = Image.open(self.extra_frames[idx]).convert("RGB")
                self.video_width  = img.width
                self.video_height = img.height
                return self.transform_extra(img)
            else:
                idx = idx-self.num_extra_frames
        if self.video_stream is not None:
            if self.video_frame_index + 1 == idx:
                img_dict = self.video_stream.__next__()
            else:
                timestamp = idx / self.video_fps
                self.video_stream = self.video_stream.seek(timestamp)
                img_dict = self.video_stream.__next__()
                # Seek to the correct frame
                while abs(timestamp - img_dict['pts']) > (1 / self.video_fps):
                    img_dict = self.video_stream.__next__()
            self.video_frame_index = idx
            img = img_dict['data']
            self.video_height, self.video_width = img.shape[1:]
        else:
            if self.zip_file is not None:
                # Handle zip file case
                with self.zip_file.open(self._img_paths[idx]) as img_file:
                    img = Image.open(img_file).convert("RGB")
            else:
                img = Image.open(self._img_paths[idx]).convert("RGB")
            self.video_width  = img.width
            self.video_height = img.height

        return self.transform(img)

    def get_frame(self, idx):
        """Fetch a frame using the LRU cache or load it if it's not cached."""
        # Check if frame is in cache
        cached_frame = self.frame_cache.get(idx)
        if cached_frame is not None:
            return cached_frame

        # Load the frame if it's not in cache
        frame = self._load_frame(idx)

        # Add the frame to the cache
        self.frame_cache.put(idx, frame)

        # Move to device if not offloading to CPU
        if not self.offload_to_cpu:
            frame = frame.to(self.device)

        return frame


def load_video_frames_with_cache(
    video_path,
    image_size,
    offload_video_to_cpu,
    img_fname_contains=None,    # str that filenames are filtered on
    separate_prompts=None,
    cache_size=100,
    img_mean=(0.485, 0.456, 0.406),
    img_std=(0.229, 0.224, 0.225),
    compute_device=torch.device("cuda")
):
    """
    Load video frames from a directory of JPEG files or a zipfile containing JPEG files with LRU cache for high efficiency.
    The frames are resized to image_size x image_size and normalized.
    """

    # Check if the input is a directory
    img_extensions = ('.jpg', '.jpeg', '.png')
    vid_extensions = (".mp4", ".avi", ".mov")
    src_type = None
    if isinstance(video_path, str):
        if os.path.isdir(video_path):
            # Directory of image files
            file_names = os.listdir(video_path)
            src_type = 'image_list'
        elif zipfile.is_zipfile(video_path):
            with zipfile.ZipFile(video_path, 'r') as zipf:
                file_names = zipf.namelist()
            src_type = 'zip'
        elif os.path.isfile(video_path) and video_path.lower().endswith(vid_extensions):
            src_type = 'video'
        else:
            raise NotImplementedError(
                "Only JPEG and PNG frames in directories or zip files, or MP¤, AVI and MOV video files are supported. Use ffmpeg to extract frames from other videos if needed."
            )
    else:
        raise NotImplementedError(
            "Unsupported input type for video_path. Expected a string containing a directory path, zip file path or video file path."
        )

    # filter out the images
    if src_type=='video':
        img_paths = video_path
    else:
        frame_names = [name for name in file_names if name.lower().endswith(img_extensions) and (not img_fname_contains or img_fname_contains in name)]
        # sort
        frame_names.sort(key=natsort.os_sort_keygen())
        # package
        if src_type=='zip':
            img_paths = (video_path, frame_names)
        else:
            img_paths = [os.path.join(video_path, frame_name) for frame_name in frame_names]

        # Ensure there are frames available
        num_frames = len(frame_names)
        if num_frames == 0:
            raise RuntimeError(f"No images found in {video_path}")

    # if we have extra frames for prompts, set up loading for those too
    extra_frames = None
    if separate_prompts is not None:
        extra_frames = [fr for fr in separate_prompts]

    # Initialize VideoFrameLoader with LRU cache
    img_paths = (src_type,img_paths)
    frame_loader = VideoFrameLoader(
        img_paths=img_paths,
        image_size=image_size,
        img_mean=img_mean,
        img_std=img_std,
        extra_frames=extra_frames,
        offload_to_cpu=offload_video_to_cpu,
        compute_device=compute_device,
        cache_size=cache_size,  # Set the cache size dynamically
    )

    # Return the frame loader and the total number of frames
    return frame_loader


def fill_holes_in_mask_scores(mask, max_area):
    """
    A post processor to fill small holes in mask scores with area under `max_area`.
    """
    # Holes are those connected components in background with area <= self.max_area
    # (background regions are those with mask scores <= 0)
    assert max_area > 0, "max_area must be positive"

    input_mask = mask
    try:
        labels, areas = get_connected_components(mask <= 0)
        is_hole = (labels > 0) & (areas <= max_area)
        # We fill holes with a small positive mask score (0.1) to change them to foreground.
        mask = torch.where(is_hole, 0.1, mask)
    except Exception as e:
        # Skip the post-processing step on removing small holes if the CUDA kernel fails
        warnings.warn(
            f"{e}\n\nSkipping the post-processing step due to the error above. You can "
            "still use SAM 2 and it's OK to ignore the error above, although some post-processing "
            "functionality may be limited (which doesn't affect the results in most cases; see "
            "https://github.com/facebookresearch/sam2/blob/main/INSTALL.md).",
            category=UserWarning,
            stacklevel=2,
        )
        mask = input_mask

    return mask


def concat_points(old_point_inputs, new_points, new_labels):
    """Add new points and labels to previous point inputs (add at the end)."""
    if old_point_inputs is None:
        points, labels = new_points, new_labels
    else:
        points = torch.cat([old_point_inputs["point_coords"], new_points], dim=1)
        labels = torch.cat([old_point_inputs["point_labels"], new_labels], dim=1)

    return {"point_coords": points, "point_labels": labels}
