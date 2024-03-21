"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import os.path as osp
import random

import decord
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class WebvidDataset(Dataset):

    def __init__(self,
                 vis_root,
                 ann_path,
                 n_sample_frames=8,
                 sample_rate=3,
                 width=512,
                 height=512,
                 is_image=False):
        """
        vis_root (string): Root directory of video (e.g. webvid_eval/video/)
        ann_root (string): Root directory of video (e.g. webvid_eval/annotations/)
        split (string): val or test
        """
        # 分开读取 train and val
        df = pd.read_csv(ann_path)

        self.annotation = df
        self.length = len(self.annotation)
        self.vis_root = vis_root
        self.n_sample_frames = n_sample_frames
        self.sample_stride = sample_rate
        self.sample_start_idx = 0
        self.width = width
        self.height = height
        self.is_image = is_image

        assert self.width == self.height, f"test sample's width and height should be the same"
        self.pixel_transforms = transforms.Compose([
            transforms.Resize(self.width),
            transforms.CenterCrop((self.height, self.width)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5],
                                 inplace=True),
        ])

    def _get_video_path(self, sample):
        rel_video_fp = os.path.join(str(sample['videoid']) + '.mp4')
        full_video_fp = os.path.join(self.vis_root, rel_video_fp)
        return full_video_fp

    def __getitem__(self, index):
        while True:
            try:
                pixel_values, text = self.get_batch(index)
                break

            except Exception as e:
                print(e)
                index = random.randint(0, self.length - 1)

        pixel_values = self.pixel_transforms(pixel_values)
        # 取出第一帧作为 reference image, 先将它 unnormalize, 为之后送到 clip feature
        # extractor 做准备，因为后者使用不同的 normalize value
        # if self.is_image:
            # ref_image = self.unnormalize(pixel_values)
        # else:
            # ref_image = self.unnormalize(pixel_values[0])

        sample = dict(mp4=pixel_values,
                      txt=text,)
                      # ref_image=ref_image)
        return sample

    def unnormalize(self, image_tensor):
        return (image_tensor + 1) / 2.0 * 255.

    def __len__(self):
        return len(self.annotation)

    def get_batch(self, idx, test=False):
        sample = self.annotation.iloc[idx]
        sample_dict = sample.to_dict()
        video_id = sample_dict['videoid']

        if 'name' in sample_dict.keys():
            text = sample_dict['name'].strip()
        else:
            raise NotImplementedError("Un-supported text annotation format.")

        # fetch video
        video_path = self._get_video_path(sample_dict)
        video_reader = decord.VideoReader(video_path)
        video_length = len(video_reader)
        if not self.is_image:
            clip_length = min(video_length,
                              (self.n_sample_frames - 1) * self.sample_stride +
                              1)
            if test:
                start_idx = 0
            else:
                start_idx = random.randint(0, video_length - clip_length)
            batch_index = np.linspace(start_idx,
                                      start_idx + clip_length - 1,
                                      self.n_sample_frames,
                                      dtype=int)
        else:
            batch_index = [random.randint(0, video_length - 1)]

        pixel_values = torch.from_numpy(
            video_reader.get_batch(batch_index).asnumpy()).permute(
                0, 3, 1, 2).contiguous()
        pixel_values = pixel_values / 255.
        del video_reader

        if self.is_image:
            pixel_values = pixel_values[0]

        return pixel_values, text


if __name__ == "__main__":
    dataset = WebvidDataset(
        vis_root="/data2/songtao.liu/WebVid/2M/videos/train",
        ann_path="/data2/songtao.liu/WebVid/2M/valid_2M_train.csv",
        n_sample_frames=16,
        width=256,
        height=256)
    import ipdb; ipdb.set_trace()
    print(dataset[0]["images"].shape, dataset[0]["prompt_ids"],
          dataset[0]["id"])
