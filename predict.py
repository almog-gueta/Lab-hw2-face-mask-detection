import os
import argparse
import numpy as np
import pandas as pd
from dataset import FaceMaskDataset
from torch.utils.data import DataLoader
import torch
import random
from my_utils import data_utils
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection import FasterRCNN
import gdown

def main():
    # Parsing script arguments
    parser = argparse.ArgumentParser(description='Process input')
    parser.add_argument('input_folder', type=str, help='Input folder path, containing images')
    args = parser.parse_args()

    dataset=FaceMaskDataset(image_dir=args.input_folder, img_size=224,phase='eval')
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=data_utils.collate_fn)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #####

    #load saved model
    model=download_model()
    model.to(device)
    model.eval()
    proper_mask_preds = []
    bbox_preds = []

    with torch.no_grad():
        for images, _ in data_loader:
            images = list(image.to(device) for image in images)
            predictions, _ = model(images)
            for idx,pred in enumerate(predictions):
                if len(pred['labels']) > 0:
                    mask_pred = int(pred['labels']) == 2 #convert label from 2,1 labels to True,False
                    proper_mask_preds.append(mask_pred)
                else:  # Return false label if empty label
                    proper_mask_preds.append(False)
                try:
                    bbox = pred['boxes'][0].tolist()
                except IndexError:
                    img_w = images[idx].shape[1]
                    img_h = images[idx].shape[2]
                    # randomly sample x1,y1
                    bbox = [random.randint(0, img_w), random.randint(0, img_h)]
                    # randomly sample x2,y2 so the box will be in format [x1,y1,x2,y2]
                    bbox.extend([random.randint(bbox[0], img_w), random.randint(bbox[1], img_h)])

                #convert format [x1,y1,x2,y2] to [x1,y1,w,h]
                bbox[2] = bbox[2] - bbox[0]
                bbox[3] = bbox[3] - bbox[1]
                bbox_preds.append(bbox)

    # get predictions to df
    files=data_loader.dataset.filenames
    prediction_df = pd.DataFrame(zip(files, *np.array(bbox_preds, dtype=int).transpose(), proper_mask_preds),
                                 columns=['filename', 'x', 'y', 'w', 'h', 'proper_mask'])
    #### save predictions as csv
    prediction_df.to_csv("prediction.csv", index=False, header=True)

def download_model():
    module_path = os.path.dirname(os.path.realpath(__file__))
    gdrive_file_id = '1LlNibC0_qhp9RbYRSiEonJnYGxQWZ7oa'
    url = f'https://drive.google.com/uc?id={gdrive_file_id}'
    weights_path = os.path.join(module_path, 'faster_rcnn.pth.tar')
    gdown.download(url, weights_path, quiet=False)

    backbone = 'resnext50_32x4d'
    backbone = resnet_fpn_backbone(backbone, pretrained=False)
    model = FasterRCNN(backbone, num_classes=3, min_size=224, max_size=224, box_detections_per_img=1)
    model.load_state_dict(torch.load(weights_path)['model_state'])
    return model


if __name__ == '__main__':
    main()