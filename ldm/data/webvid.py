import os
import numpy as np
import PIL
from torchvision import transforms

from webdataset import WebLoader
from torch.utils.data import DataLoader

from video2dataset.dataloader import get_video_dataset

SHARDS=["/workspace/stable-diffusion/datasets/webvid_train/{00000..00007}.tar", "/workspace/stable-diffusion/datasets/webvid_train/{00009..00018}.tar",]
# SHARDS=["/workspace/stable-diffusion/datasets/webvid_train/{00009..00019}.tar",]
#SHARDS="/workspace/stable-diffusion/datasets/webvid/{00000..00003}.tar"
# SHARDS="dataset/00000.tar"
# SHARDS = "dataset/{00000..00004}.tar"
# SHARDS = "/data2/songtao.liu/WebVid/2M/videos/val-wb/val-{000000..000004}.tar"

def preprocess(frame):
    frame = frame.float().clamp(min=0, max=255) / 127.5 - 1.
    return frame

def get_video_dataloader():
    decoder_kwargs = {
        "n_frames": 8,  # get 8 frames from each video
        # "fps": 20,  # downsample to 10 FPS
        "fps": "sample",  # downsample to 10 FPS
        "min_fps": 4,
        "max_fps": 30,
        "num_threads": 12,  # use 12 threads to decode the video
        # "pad_frames": 8,
    }
    resize_size = crop_size = 256
    batch_size = 4

    dset = get_video_dataset(
        urls=SHARDS,
        batch_size=batch_size,
        decoder_kwargs=decoder_kwargs,
        resize_size=resize_size,
        crop_size=crop_size,
        custom_transforms={"mp4":preprocess},
        # enforce_additional_keys=[],
    #)
    ).shuffle(2000).to_tuple("mp4", "txt", "__key__").with_epoch(300)
    #).shuffle(2000).to_tuple("mp4", "__key__").with_epoch(300)

    num_workers = 8  # 8 dataloader workers

    dl = WebLoader(dset, batch_size=None, num_workers=num_workers, pin_memory=True).unbatched().shuffle(2000).batched(4)
    # dl = WebLoader(dset, batch_size=None, num_workers=num_workers)
    # dl = DataLoader(dset, batch_size=4, num_workers=num_workers)

    # for idx, sample in enumerate(dl):
        # print("~~~~~~~~~~~~~~~~~~~", idx)
        # video_batch = sample["mp4"]
        # video_batch = sample[0]
        # print(video_batch.shape)  # torch.Size([32, 8, 256, 256, 3])
        # print(sample[1])  # torch.Size([32, 8, 256, 256, 3])

        # TODO: need to add option for text/metadata preprocessing (tokenization etc.)
        # text_batch = sample["txt"]
        # print(text_batch[0])

    return dl
