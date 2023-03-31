import torch
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.io import read_image, ImageReadMode


def custom_loader(path):
    img = read_image(path, mode=ImageReadMode.RGB)
    img = transforms.ToPILImage()(img.squeeze(0))
    return img


def prepare_data(dataroot, image_size, batch_size, workers):
    # Create the datasets

    transformations = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    vanilla = dset.ImageFolder(root=dataroot, transform=transformations, loader=custom_loader)

    transformations = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    augmented = dset.ImageFolder(root=dataroot, transform=transformations, loader=custom_loader)

    datasets = torch.utils.data.ConcatDataset([vanilla, augmented])

    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(datasets, batch_size=batch_size,
                                             shuffle=True, num_workers=workers)

    # noinspection PyTypeChecker
    print("Dataset length", len(dataloader.dataset))

    return dataloader
