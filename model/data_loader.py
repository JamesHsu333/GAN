import random
import os

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# Define a training image loader that specifies transforms on images
train_transformer = transforms.Compose([
    transforms.ToTensor(),   # Transform it into a torch tensor
    transforms.Normalize((0.5,), (0.5,))
])

# Define a evaluation image loader that specifies transforms on images
eval_transformer = transforms.Compose([
    transforms.ToTensor(),   # Transform it into a torch tensor
    transforms.Normalize((0.5,), (0.5,))
])

train_set = datasets.MNIST(
    root='data/mnist',
    train=True,
    download=True,
    transform=train_transformer,
)

test_set = datasets.MNIST(
    root='data/mnist',
    train=False,
    download=True,
    transform=eval_transformer,
)

class GetDataset(Dataset):
    def __init__(self, data_dir, transform):
        self.filenames = os.listdir(data_dir)
        self.filenames = [os.path.join(data_dir, f) for f in self.filenames if f.endswith('.jpg')]

        self.labels = [int(os.path.split(filename)[-1][0]) for filename in self.filenames]
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image = Image.open(self.filenames[idx])
        image = self.transform(image)
        return image, self.labels[idx]

def fetch_dataloader(types, data_dir, params):
    dataloaders={}

    for split in ['train', 'val', 'test']:
        if split in types:
            path = os.path.join(data_dir, "{}_signs".format(split))

            if split == 'train':
                dl = DataLoader(GetDataset(path, train_transformer), batch_size=params.batch_size, shuffle=True,
                                            num_workers=params.num_workers,
                                            pin_memory=params.cuda)
            else:
                dl = DataLoader(GetDataset(path, eval_transformer), batch_size=params.batch_size, shuffle=True,
                                            num_workers=params.num_workers,
                                            pin_memory=params.cuda)
            dataloaders[split] = dl
    return dataloaders

def mnist_dataloader(types, params):
    dataloaders={}

    for split in ['train', 'val', 'test']:
        if split in types:
            if split == 'train':
                dl =  DataLoader(train_set, batch_size=params.batch_size, shuffle=True,
                                    num_workers=params.num_workers,
                                    pin_memory=params.cuda)
            else:
                dl = DataLoader(test_set, batch_size=params.batch_size, shuffle=True,
                                    num_workers=params.num_workers,
                                    pin_memory=params.cuda)
            dataloaders[split] = dl
    return dataloaders