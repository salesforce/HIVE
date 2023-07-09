'''
 * Copyright (c) 2023 Salesforce, Inc.
 * All rights reserved.
 * SPDX-License-Identifier: Apache License 2.0
 * For full license text, see LICENSE.txt file in the repo root or http://www.apache.org/licenses/
 * By Shu Zhang
 * Modified from InstructPix2Pix repo: https://github.com/timothybrooks/instruct-pix2pix
 * Copyright (c) 2023 Timothy Brooks, Aleksander Holynski, Alexei A. Efros.  All rights reserved.
'''

from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision
from einops import rearrange
from PIL import Image
from torch.utils.data import Dataset
import jsonlines
from collections import deque


class EditDataset(Dataset):
    def __init__(
        self,
        path_instructpix2pix: str,
        path_hive_0: str,
        path_hive_1: str,
        path_hive_2: str,
        split: str = "train",
        splits: tuple[float, float, float] = (0.9, 0.05, 0.05),
        min_resize_res: int = 256,
        max_resize_res: int = 256,
        crop_res: int = 256,
        flip_prob: float = 0.0,
    ):
        assert split in ("train", "val", "test")
        assert sum(splits) == 1
        self.path_instructpix2pix = path_instructpix2pix
        self.path_hive_0 = path_hive_0
        self.path_hive_1 = path_hive_1
        self.path_hive_2 = path_hive_2
        self.min_resize_res = min_resize_res
        self.max_resize_res = max_resize_res
        self.crop_res = crop_res
        self.flip_prob = flip_prob
        self.seeds = []
        self.instructions = []
        self.source_imgs = []
        self.edited_imgs = []
        # load instructpix2pix dataset
        with open(Path(self.path_instructpix2pix, "seeds.json")) as f:
            seeds = json.load(f)
        split_0, split_1 = {
            "train": (0.0, splits[0]),
            "val": (splits[0], splits[0] + splits[1]),
            "test": (splits[0] + splits[1], 1.0),
        }[split]

        idx_0 = math.floor(split_0 * len(seeds))
        idx_1 = math.floor(split_1 * len(seeds))
        seeds = seeds[idx_0:idx_1]

        for seed in seeds:
            seed = deque(seed)
            seed.appendleft('')
            seed.appendleft('instructpix2pix')
            self.seeds.append(list(seed))


        # load HIVE dataset first part

        cnt = 0
        with jsonlines.open(Path(self.path_hive_0, "training_cycle.jsonl")) as reader:
            for ll in reader:
                self.instructions.append(ll['instruction'])
                self.source_imgs.append(ll['source_img'])
                self.edited_imgs.append(ll['edited_img'])
                self.seeds.append(['hive_0', '', '', [cnt]])
                cnt += 1

        # load HIVE dataset second part
        with open(Path(self.path_hive_1, "seeds.json")) as f:
            seeds = json.load(f)
        for seed in seeds:
            seed = deque(seed)
            seed.appendleft('hive_1')
            self.seeds.append(list(seed))
        # load HIVE dataset third part
        with open(Path(self.path_hive_2, "seeds.json")) as f:
            seeds = json.load(f)
        for seed in seeds:
            seed = deque(seed)
            seed.appendleft('hive_2')
            self.seeds.append(list(seed))

    def __len__(self) -> int:
        return len(self.seeds)

    def __getitem__(self, i: int) -> dict[str, Any]:

        name_0, name_1, name_2, seeds = self.seeds[i]
        if name_0 == 'instructpix2pix':
            propt_dir = Path(self.path_instructpix2pix, name_2)
            seed = seeds[torch.randint(0, len(seeds), ()).item()]
            with open(propt_dir.joinpath("prompt.json")) as fp:
                prompt = json.load(fp)["edit"]
            image_0 = Image.open(propt_dir.joinpath(f"{seed}_0.jpg"))
            image_1 = Image.open(propt_dir.joinpath(f"{seed}_1.jpg"))
        elif name_0 == 'hive_1':
            propt_dir = Path(self.path_hive_1, name_1, name_2)
            seed = seeds[torch.randint(0, len(seeds), ()).item()]
            with open(propt_dir.joinpath("prompt.json")) as fp:
                prompt = json.load(fp)["instruction"]
            image_0 = Image.open(propt_dir.joinpath(f"{seed}_0.jpg"))
            image_1 = Image.open(propt_dir.joinpath(f"{seed}_1.jpg"))
        elif name_0 == 'hive_2':
            propt_dir = Path(self.path_hive_2, name_1, name_2)
            seed = seeds[torch.randint(0, len(seeds), ()).item()]
            with open(propt_dir.joinpath("prompt.json")) as fp:
                prompt = json.load(fp)["instruction"]
            image_0 = Image.open(propt_dir.joinpath(f"{seed}_0.jpg"))
            image_1 = Image.open(propt_dir.joinpath(f"{seed}_1.jpg"))
        else:
            j = seeds[0]
            image_0 = Image.open(self.source_imgs[j])
            image_1 = Image.open(self.edited_imgs[j])
            prompt = self.instructions[j]

        reize_res = torch.randint(self.min_resize_res, self.max_resize_res + 1, ()).item()
        image_0 = image_0.resize((reize_res, reize_res), Image.Resampling.LANCZOS)
        image_1 = image_1.resize((reize_res, reize_res), Image.Resampling.LANCZOS)

        image_0 = rearrange(2 * torch.tensor(np.array(image_0)).float() / 255 - 1, "h w c -> c h w")
        image_1 = rearrange(2 * torch.tensor(np.array(image_1)).float() / 255 - 1, "h w c -> c h w")

        crop = torchvision.transforms.RandomCrop(self.crop_res)
        flip = torchvision.transforms.RandomHorizontalFlip(float(self.flip_prob))
        image_0, image_1 = flip(crop(torch.cat((image_0, image_1)))).chunk(2)

        return dict(edited=image_1, edit=dict(c_concat=image_0, c_crossattn=prompt))

