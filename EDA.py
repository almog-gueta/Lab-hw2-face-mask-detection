import os
import pandas as pd
import numpy as np
import torchvision
import json
import hydra
from omegaconf import DictConfig, OmegaConf
from PIL import Image
import random
import matplotlib.pyplot as plt
import torch
from IPython.display import display


def get_data(filenames):
    data = []
    for filename in filenames:
        image_id, bbox, proper_mask = filename.strip(".jpg").split("__")
        # check validity of bbox
        bbox = json.loads(bbox)  # convert string '[x,y,w,h]' to list [x,y,w,h]
        x, y, w, h = bbox
        proper_mask = 1 if proper_mask.lower() == "true" else 0
        data.append((filename, image_id, bbox, x, y, w, h, proper_mask))
    return data

def read_data(image_dir):
    filenames = os.listdir(image_dir)
    data=get_data(filenames)
    data = pd.DataFrame(data, columns=['file_name', 'image_id', 'bbox',"x", "y", "w", "h",'is_proper_mask'])
    data["box_area"] =data["w"] * data["h"]
    data["type"] = image_dir.split("/")[4]
    data["image_width"] = data.apply(lambda x: torchvision.io.read_image(os.path.join(image_dir, x.loc["file_name"])).shape[1], axis=1)
    data["image_height"] = data.apply(lambda x: torchvision.io.read_image(os.path.join(image_dir, x.loc["file_name"])).shape[2] , axis=1)
    return data

def show_images(image_dir):
    filenames = random.sample(os.listdir(image_dir), 64)
    transformer = torchvision.transforms.Resize([224,244])
    images = [transformer(torchvision.io.read_image(os.path.join(image_dir, filename))) for filename in filenames]
    imgs = torchvision.utils.make_grid(images)
    aux_show(imgs)


def aux_show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False, figsize=(15, 15))
    for i, img in enumerate(imgs):
        img = img.detach()
        img = torchvision.transforms.functional.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

def show_images_and_bboxes(data, image_dir):
    filenames = random.sample(os.listdir(image_dir), 3000)
    data = get_data(filenames)
    images_with_bbox = []
    counter = 0
    for idx, (filename, image_id, bbox, x, y, w, h, proper_mask) in enumerate(data):
        img = torchvision.io.read_image(os.path.join(image_dir, filename))
        if img.shape == torch.Size([3, 224, 224]):
            bbox[2] = bbox[0] + bbox[2]
            bbox[3] = bbox[1] + bbox[3]
            bbox = [bbox]
            bbox = torch.tensor(bbox)
            color = "green" if proper_mask else "red"
            images_with_bbox.append(torchvision.utils.draw_bounding_boxes(img, bbox, width=5, colors=[color]))
            counter += 1
        if counter > 63:
            break
    imgs = torchvision.utils.make_grid(images_with_bbox)
    aux_show(imgs)

def show_imag_w_bbox(img,filename):
    image_id, bbox, proper_mask = filename.strip(".jpg").split("__")
    bbox = json.loads(bbox)
    bbox[2] = bbox[0] + bbox[2]
    bbox[3] = bbox[1] + bbox[3]
    bbox = [bbox]
    bbox = torch.tensor(bbox)
    color = "green" if proper_mask else "red"
    images_with_bbox=torchvision.utils.draw_bounding_boxes(img, bbox, width=5, colors=[color])
    # imgs = torchvision.utils.make_grid(images_with_bbox)
    aux_show(images_with_bbox)

def get_file_name(total_data,list_name):
        return total_data[total_data["image_id"]==list_name]["file_name"].values[0]


@hydra.main(config_path="cfg", config_name='cfg')
def main(cfg: DictConfig):
    print("starting")
    train_dir=cfg['main']['paths']['train']
    test_dir=cfg['main']['paths']['test']
    train_data=read_data(train_dir)
    test_data=read_data(test_dir)
    total_data = pd.concat([train_data ,test_data])

    # print("get labels distribution")
    # #get labels distribution
    # train_test_mask_dist = total_data.groupby(["type", "is_proper_mask"]).count()
    # train_test_mask_dist=train_test_mask_dist.rename(columns={"image_id": "count"})
    # train_test_mask_dist=train_test_mask_dist[["count"]]
    # ax_1 =train_test_mask_dist.unstack().plot(kind='bar',colormap="Spectral",rot=0)
    # ax_1.set_xlabel("dataset type",fontsize=15)
    # ax_1.set_ylabel("Amount of images",fontsize=15)
    # L = plt.legend()
    # L.get_texts()[0].set_text('No proper mask')
    # L.get_texts()[1].set_text('Proper mask')
    # # ax_1.legend(loc="upper right")
    # ax_1.set_title("The amount of images per mask label in each dataset", fontdict={'fontsize': 15, 'fontweight': 'medium'})

    display(total_data[["box_area", "image_width", "image_height"]].describe().round())
    print("show random photos with bboxes")
    #show random photos with bboxes

    # show_images_and_bboxes(train_data,train_dir)
    # show_images_and_bboxes(test_data,test_dir)

    print("show problematic images")
    #show problematic images
    fig, ax = plt.subplots()

    #finished
    # negative_h_ids=["004828","009266"] #train,test
    # negative_h_train_image = torchvision.io.read_image(os.path.join(train_dir, get_file_name(total_data,negative_h_ids[0])))
    # show_imag_w_bbox(negative_h_train_image,get_file_name(total_data,negative_h_ids[0]))
    # negative_h_test_image = torchvision.io.read_image(os.path.join(train_dir, get_file_name(total_data,negative_h_ids[1])))
    # show_imag_w_bbox(negative_h_test_image,get_file_name(total_data,negative_h_ids[1]))
    # plt.show()

    # plt.imshow(np.asarray(negative_h_train_image))
    #
    # plt.imshow(np.asarray(negative_h_test_image))
    # plt.show()

#finishe
    # out_boundries_ids=["004465","009889","017621","017304"]#train,train,test,test
    # out_boundries_1_train_image = torchvision.io.read_image(os.path.join(train_dir,get_file_name(total_data,out_boundries_ids[0])))
    # out_boundries_2_train_image = Image.open(os.path.join(train_dir,get_file_name(total_data,out_boundries_ids[1])))
    # out_boundries_1_test_image = torchvision.io.read_image(os.path.join(test_dir, get_file_name(total_data,out_boundries_ids[2])))
    # out_boundries_2_test_image = Image.open(os.path.join(test_dir, get_file_name(total_data,out_boundries_ids[3])))
    # show_imag_w_bbox(out_boundries_1_train_image,get_file_name(total_data,out_boundries_ids[0]))
    # show_imag_w_bbox(out_boundries_1_test_image,get_file_name(total_data,out_boundries_ids[2]))
    # plt.show()
    # plt.imshow(out_boundries_2_train_image)
    # plt.show()
    # plt.imshow(out_boundries_1_test_image)
    # plt.show()
    # plt.imshow(out_boundries_2_test_image)
    # plt.show()



#finished
    # small_imgs=["012180","011651"]#train,train
    # small_imgs_train_image = Image.open(os.path.join(train_dir, get_file_name(total_data,small_imgs[0])))
    # small_imgs_test_image = Image.open(os.path.join(train_dir, get_file_name(total_data,small_imgs[1])))
    # plt.imshow(small_imgs_train_image)
    # plt.show()
    # plt.imshow(small_imgs_test_image)
    # plt.show()

    no_area_imgs=["008710","016148"]#train,test
    img1=get_file_name(total_data, no_area_imgs[0])
    ing2=get_file_name(total_data,no_area_imgs[1])
    no_area_imgs_train_image = torchvision.io.read_image(os.path.join(train_dir,img1))
    no_area_imgs_test_image = torchvision.io.read_image(os.path.join(test_dir,ing2 ))
    show_imag_w_bbox(no_area_imgs_train_image, img1)
    show_imag_w_bbox(no_area_imgs_test_image, ing2)

    # plt.imshow(no_area_imgs_train_image)
    plt.show()
    # plt.imshow(no_area_imgs_test_image)
    # plt.show()


if __name__ == '__main__':
    main()



