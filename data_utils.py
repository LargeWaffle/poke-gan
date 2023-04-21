import torch
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.io import read_image, ImageReadMode


def prepare_data(dataroot, image_size, batch_size, workers, augmentation=False):
    # Create the datasets

    transformations = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    data = dset.ImageFolder(root=dataroot, transform=transformations)

    if augmentation:
        transformations = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        augmented = dset.ImageFolder(root=dataroot, transform=transformations)

        data = torch.utils.data.ConcatDataset([data, augmented])

    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                             shuffle=True, num_workers=workers)

    # noinspection PyTypeChecker
    print("Dataset length", len(dataloader.dataset))
    len_ds = len(dataloader)
    status_step = len_ds // 2

    print("len", len_ds, "stat", status_step)
    return dataloader
