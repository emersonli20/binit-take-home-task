import torch
import timm
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import Subset
import umap.umap_ as umap
from matplotlib import pyplot as plt
import typer
import os
from tqdm import tqdm

def main(directory: str):
    # create model
    m = timm.create_model('resnet50', pretrained=True)

    # create training dataset
    transform = transforms.ToTensor()
    trainset = CIFAR10(root='./data', train=True, download=True, transform=transform)

    # choose 2 classes in CIFAR10
    # automobile
    idx1 = torch.tensor(trainset.targets) == 1
    # frog
    idx2 = torch.tensor(trainset.targets) == 6
    train_mask = idx1 | idx2
    train_indices = train_mask.nonzero().reshape(-1)

    # take 1500 image subset of CIFAR10 that only contains automobile and frog
    train_subset = Subset(trainset, train_indices)

    labels = []
    images, label = train_subset[0]
    labels.append(label)

    images = torch.unsqueeze(images, dim=0)
    for i in tqdm(range(1,1500)):
        new_image, label = train_subset[i]
        labels.append(label)
        new_image = torch.unsqueeze(new_image, dim=0)
        images = torch.cat((images, new_image), dim=0)

    # extract features from penultimate layer
    o = m.forward_features(images)
    o_flat = torch.flatten(o, start_dim=1)
    data = o_flat.detach().numpy()

    # reduce the features to a 2d embedding space
    reducer = umap.UMAP(n_neighbors=7, min_dist=0.025, random_state=42)
    reducer.fit(data)
    embedding = reducer.transform(data)

    # plot 2d embedding space
    plt.scatter(embedding[:,0], embedding[:,1], c=labels)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "embeddings.png")
    plt.savefig(path)

if __name__ == "__main__":
    typer.run(main)
