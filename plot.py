import torch
import timm
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import Subset
import umap
from matplotlib import pyplot as plt
import typer
import os

def main(directory: str):
    m = timm.create_model('resnet50', pretrained=True)

    transform = transforms.ToTensor()

    # create training dataset
    trainset = CIFAR10(root='./data', train=True, download=True, transform=transform)

    idx1 = torch.tensor(trainset.targets) == 1
    idx8 = torch.tensor(trainset.targets) == 8
    train_mask = idx1 | idx8
    train_indices = train_mask.nonzero().reshape(-1)

    train_subset = Subset(trainset, train_indices)

    image, label = train_subset[0]
    image = torch.unsqueeze(image, dim=0)
    image2, label2 = train_subset[1]
    image2 = torch.unsqueeze(image2, dim=0)

    images, _ = train_subset[0]
    images = torch.unsqueeze(images, dim=0)
    for i in range(1,100):
        new_image, _ = train_subset[i]
        new_image = torch.unsqueeze(new_image, dim=0)
        images = torch.cat((images, new_image), dim=0)


    o = m.forward_features(images)

    o_flat = torch.flatten(o, start_dim=1)

    data = o_flat.detach().numpy()

    reducer = umap.UMAP(random_state=42)
    reducer.fit(data)

    embedding = reducer.transform(data)

    plt.scatter(embedding[:,0], embedding[:,1])
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "embeddings.png")
    plt.savefig(path)

if __name__ == "__main__":
    typer.run(main)
