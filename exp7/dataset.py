"""Load images from a directory using torchvision ImageFolder."""
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def make_loader(data_dir, image_size=64, batch_size=16, num_workers=2, limit_images=0):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    if limit_images and limit_images > 0:
        import torch
        dataset = torch.utils.data.Subset(dataset, range(min(limit_images, len(dataset))))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, drop_last=True, pin_memory=True)
