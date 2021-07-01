import torch


# from dataset import FaceMaskDataset

# def images_mean_std_RGB():
#     # calc the mean and std for each RGB channel over train_set. Used for torchvision.transform on the data images
#     mean = torch.zeros(3)
#     std = torch.zeros(3)
#     train_dataset = FaceMaskDataset(image_dir='/home/student/data/train', phase='train', img_size=224)
#
#     for inputs, bbox, proper_mask in train_dataset:
#         for i in range(3):
#             mean[i] += inputs[ i, :, :].mean()
#             std[i] += inputs[ i, :, :].std()
#
#     mean.div_(len(train_dataset))
#     std.div_(len(train_dataset))
#     print('mean: ', mean, 'std: ', std)


def prep_box_to_yolo(img_size, box):
    """

    :param box: [x_left, y_up, w, h]
    :return: box: [x_center, y_center, w, h]
    """
    x_center, y_center = corners2center(box)

    # normalized values
    x_center, y_center, w, h = normalize_box(img_size, x_center, y_center, box[2], box[3])
    return torch.tensor([x_center, y_center, w, h])


def corners2center(box):
    x_left, y_up, w, h = box
    return x_left + (w / 2), y_up + (h / 2)


def normalize_box(img_size, x, y, w, h):
    return x / img_size, y / img_size, w / img_size, h / img_size


# from normalized to pixels
# targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels

def prep_box_to_rcnn(box):
    xmin, ymin, w, h = box
    xmax, ymax = xmin + w, ymin + h
    return [xmin, ymin, xmax, ymax]


# for rcnn dataloader
def collate_fn(batch):
    return list(zip(*batch))
