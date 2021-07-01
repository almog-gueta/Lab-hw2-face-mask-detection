"""
Here, we create a custom dataset
"""
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import List
import torchvision.transforms as transforms
from PIL import Image
import os
import json


DATA_TRANSFORMS = {
    'train' : transforms.Compose([
        transforms.Resize(256) ,
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5244, 0.4904, 0.4781],[0.2642, 0.2608, 0.2561])
    ]),
    'eval' : transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5244, 0.4904, 0.4781],[0.2642, 0.2608, 0.2561])
    ])
}



class FaceMaskDataset(Dataset):
    def __init__(self, image_dir, img_size, phase='train',is_predict=False):
        """
        Args:
            image_dir (string): Directory with all the images.
            phase (string): train or eval (For transformations).
        """
        self.image_dir = image_dir
        self.transform = DATA_TRANSFORMS[phase]
        self.img_size = img_size
        self.data = self.get_entries()
        self.is_predict=is_predict
        # self.images_paths = glob(f'{self.image_dir}/*.jpg')


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        filename, image_id, bbox, proper_mask = self.data[idx]
        # change bboc to fit YOLO format
        # bbox = prep_box_to_yolo(self.img_size, bbox) #YOLO

        # Load image
        image = Image.open(os.path.join(self.image_dir, filename))
        orig_width, orig_height = image.size
        image = self.transform(image)
        img_width, img_height =self.img_size,self.img_size
        if not self.is_predict:
            bbox = self.resize_bbox(orig_width, orig_height, img_width, img_height, bbox)
            item = (image, bbox, proper_mask)
        else:
            bbox,w_scale,h_scale = self.resize_bbox(orig_width, orig_height, img_width, img_height, bbox)
            item = (image, bbox, proper_mask,w_scale,h_scale)
        return item

    def get_entries(self) -> List:
        """
        Parse a directory with images.
        :param image_dir: Path to directory with images.
        :return: A list with (filename, image_id, bbox, proper_mask) for every image in the image_dir.
        """
        example_filenames = os.listdir(self.image_dir)
        data = []
        for filename in example_filenames:
            image_id, bbox, proper_mask = filename.strip(".jpg").split("__")
            # check validity of bbox
            bbox = json.loads(bbox)  # convert string '[x,y,w,h]' to list [x,y,w,h]
            x, y, w, h = bbox
            if x < 0 or y < 0 or w < 0 or h < 0 or w + h == 0:  # skip image if box is 0 or negative w/h
                continue
            img_width, img_height = self.img_size, self.img_size
            if x + w > img_width:
                # print(f'corrected width for image {image_id} from {w} to {width}')
                w = img_width - x
            if y + h > img_height:
                # print(f'corrected height for image {image_id} from {h} to {height}')
                h = img_height - y
            bbox = torch.tensor([x, y, w, h])
            proper_mask = 1 if proper_mask.lower() == "true" else 0
            data.append((filename, image_id, bbox, proper_mask))
        return data


    def resize_bbox(self,origin_w, origin_h, new_w, new_h, bbox):
        w_scale = new_w / origin_w
        h_scale = new_h / origin_h
        x = int(np.round(bbox[0] * w_scale))
        y = int(np.round(bbox[1] * h_scale))
        w = int(np.round(bbox[2] * w_scale))
        h = int(np.round(bbox[3] * h_scale))
        if not self.is_predict:
            return torch.tensor([x, y, w, h])
        else:
            return torch.tensor([x, y, w, h]),w_scale,h_scale







