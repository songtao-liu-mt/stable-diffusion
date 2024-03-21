import os
import numpy as np
import PIL
from torchvision import transforms

from webdataset import WebLoader
from torch.utils.data import DataLoader

from video2dataset.dataloader import get_video_dataset

# SHARDS=["/workspace/stable-diffusion/datasets/webvid_train/{00000..00007}.tar", "/workspace/stable-diffusion/datasets/webvid_train/{00009..00018}.tar",]
# SHARDS=["/data2/songtao.liu/WebVid/2M/dataset_train/{00044..00053}.tar",]
# SHARDS=["/data2/songtao.liu/WebVid/2M/dataset/{00000..00004}.tar",]
# SHARDS=["/juicefs/songtao.liu/data/WebVid/train-wd/train-{0000000..0002422}.tar"]
# SHARDS=["/data2/songtao.liu/WebVid/2M/dataset_train/00046.tar",]
#SHARDS="/workspace/stable-diffusion/datasets/webvid/{00000..00003}.tar"
# SHARDS="dataset/00000.tar"
# SHARDS = "dataset/{00000..00004}.tar"
# SHARDS = "/data2/songtao.liu/WebVid/2M/videos/train-wb/train-{0000000..0000055}.tar"

def preprocess(frame):
    frame = frame.float().clamp(min=0, max=255) / 127.5 - 1.
    return frame

def get_video_dataloader(test=False):
    if test:
        decoder_kwargs = {
            "n_frames": 8,  # get 8 frames from each video
            "fps": 10,  # downsample to 10 FPS
            "num_threads": 4,  # use 12 threads to decode the video
            "fix_sample": True,
            "pad_frames": 8,
        }
    else:
        decoder_kwargs = {
            "n_frames": 8,  # get 8 frames from each video
            # "fps": 20,  # downsample to 10 FPS
            "fps": "sample",  # downsample to 10 FPS
            "min_fps": 8,
            "max_fps": 30,
            "num_threads": 4,  # use 12 threads to decode the video
            # "pad_frames": 8,
        }
    resize_size = crop_size = 256
    batch_size = 2
    if test:
        SHARDS=["/data2/songtao.liu/WebVid/2M/dataset/{00000..00004}.tar",]
        batch_size = 4
        dset = get_video_dataset(
            urls=SHARDS,
            batch_size=batch_size,
            decoder_kwargs=decoder_kwargs,
            resize_size=resize_size,
            crop_size=crop_size,
            custom_transforms={"mp4":preprocess},
            # enforce_additional_keys=[],
        # )
        ).to_tuple("mp4", "txt", "__key__")
        # ).shuffle(5000).to_tuple("mp4", "__key__").with_epoch(300)

        num_workers = 4  # 8 dataloader workers

        dl = WebLoader(dset, batch_size=None, num_workers=num_workers, pin_memory=True).unbatched().batched(batch_size)
        # dl = WebLoader(dset, batch_size=None, num_workers=num_workers)
        # dl = DataLoader(dset, batch_size=4, num_workers=num_workers)

        # for idx, sample in enumerate(dl):
            # print("~~~~~~~~~~~~~~~~~~~", idx)
            # # video_batch = sample["mp4"]
            # video_batch = sample[0]
            # print(video_batch.shape)  # torch.Size([32, 8, 256, 256, 3])
            # print(sample[1])  # torch.Size([32, 8, 256, 256, 3])

            # TODO: need to add option for text/metadata preprocessing (tokenization etc.)
            # text_batch = sample["txt"]
            # print(text_batch[0])
    else:
        SHARDS=["/juicefs/songtao.liu/data/WebVid/train-wd/train-{0000000..0002422}.tar"]

        dset = get_video_dataset(
            urls=SHARDS,
            batch_size=batch_size,
            decoder_kwargs=decoder_kwargs,
            resize_size=resize_size,
            crop_size=crop_size,
            custom_transforms={"mp4":preprocess},
            # enforce_additional_keys=[],
        #)
        ).shuffle(5000).to_tuple("mp4", "txt", "__key__").with_epoch(300)
        # ).shuffle(5000).to_tuple("mp4", "__key__").with_epoch(300)

        num_workers = 4  # 8 dataloader workers

        dl = WebLoader(dset, batch_size=None, num_workers=num_workers, pin_memory=True).unbatched().shuffle(5000).batched(2)
        # dl = WebLoader(dset, batch_size=None, num_workers=num_workers)
        # dl = DataLoader(dset, batch_size=4, num_workers=num_workers)

        # for idx, sample in enumerate(dl):
            # print("~~~~~~~~~~~~~~~~~~~", idx)
            # # video_batch = sample["mp4"]
            # video_batch = sample[0]
            # print(video_batch.shape)  # torch.Size([32, 8, 256, 256, 3])
            # print(sample[1])  # torch.Size([32, 8, 256, 256, 3])

            # TODO: need to add option for text/metadata preprocessing (tokenization etc.)
            # text_batch = sample["txt"]
            # print(text_batch[0])

    return dl
