import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
from .config import BATCH_SIZE, NUM_WORKERS

def get_dataloaders(batch_size = BATCH_SIZE, val_fraction = 0.1):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    train_tfm = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean,std),
    ])

    full_train = datasets.CIFAR10(root="./data", train=True, download=True, transform=train_tfm)
    test_set = datasets.CIFAR10(root="./data", train=False, download=True, transform=test_tfm)
    n_val = int(len(full_train) * val_fraction)
    n_train = len(full_train) - n_val
    train_set, val_set = random_split(full_train, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)

    return train_loader, val_loader, test_loader