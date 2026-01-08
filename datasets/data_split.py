import pickle
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
def split_data():
    from torch.utils.data import Subset
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform_train)
    train_dataloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform_test)
    test_dataloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    total_samples = len(trainset)
    subset_size = int(total_samples * 0.1)
    num_classes = 100
    samples_per_class = 50

    subset1_indices = []
    for digit in range(num_classes):
        digit_indices = [i for i, (_, label) in enumerate(trainset) if label == digit]
        sampled_indices = torch.randperm(len(digit_indices))[:samples_per_class]
        subset1_indices.extend(digit_indices[idx] for idx in sampled_indices)

    # Subset2 from remaining indices
    remaining_indices = list(set(range(total_samples)) - set(subset1_indices))
    subset2_randperm = torch.randperm(len(remaining_indices))[:subset_size]
    subset2_indices = [remaining_indices[i] for i in subset2_randperm]

    # Subset3 is whatever remains from subset1 and subset2
    used_indices = set(subset1_indices) | set(subset2_indices)
    subset3_indices = list(set(range(total_samples)) - used_indices)
    subset1 = Subset(trainset, subset1_indices)

    # Convert subset2_indices to valid list of indices
    subset2 = Subset(trainset, np.array(remaining_indices)[list(subset2_randperm.numpy())])

    # Create Subset 3 from remaining indices
    subset3 = Subset(trainset, list(set(range(total_samples)) - set(subset1_indices) - set(subset2.indices)))

    with open('data/train_subset_cifar100.pickle', 'wb') as handle:
        pickle.dump(subset1, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('data/test_subset_cifar100.pickle', 'wb') as handle:
        pickle.dump(subset2, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('data/disjointsubset_cifar100.pickle', 'wb') as handle:
        pickle.dump(subset3, handle, protocol=pickle.HIGHEST_PROTOCOL)


